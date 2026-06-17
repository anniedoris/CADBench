"""Run end-to-end VLM inference + evaluation on a CADBench split.

This mirrors the flow in clean_run.ipynb as a single CLI invocation:

  1. Load a HuggingFace dataset split.
  2. Build a VLMProcessor that either renders the ground-truth STEPs/STLs
     into images or reads a precomputed image field directly from the dataset.
  3. Run inference against an OpenAI-compatible chat completions endpoint via
     APIInferenceEngine.
  4. Optionally score the generations with the CADBench Evaluator.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

from datasets import load_dataset

from CADBench import pretty_print_metrics
from CADBench.Eval import Evaluator
from CADBench.Inference import APIInferenceEngine
from CADBench.Processing import VLMProcessor


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert CAD engineer and 3D modeling specialist with deep "
    "knowledge of mechanical design, geometric modeling, and CadQuery."
)
DEFAULT_USER_PROMPT = """Generate CadQuery Python code to create this 3D CAD model
based on the provided image.

Requirements:
- Use CadQuery syntax
- Create a variable called 'result' containing the final geometry
- Include all necessary imports
- Use parametric dimensions where appropriate
- The code must be executable and create valid solid geometry
- The last line must be: result = <your_final_geometry>
- Do NOT include show_object, exportStep, or any display/export calls
- ALWAYS use method chaining for Workplane operations -- never call methods like moveTo, lineTo, close, extrude, box, circle etc. without capturing the return value. Every Workplane method returns a new object; chain them or reassign, like:
  result = (cq.Workplane()
      .moveTo(0, 0)
      .lineTo(10, 0)
      .close()
      .extrude(5))

Return ONLY the Python code, no explanations."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a VLM through an OpenAI-compatible API on a CADBench split, "
            "then optionally evaluate the generated code."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default="DeCoDELab/CADBench",
        help="HuggingFace dataset id.",
    )
    dataset_group.add_argument(
        "--split",
        type=str,
        default="bench0",
        help="Dataset split to run on.",
    )

    proc_group = parser.add_argument_group("processor (VLMProcessor)")
    proc_group.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Rendered image resolution (height and width).",
    )
    proc_group.add_argument(
        "--render-backend",
        type=str,
        default="blender",
        choices=["blender", "OCC"],
        help="Rendering backend.",
    )
    proc_group.add_argument(
        "--render-style",
        type=str,
        default="mesh",
        choices=["CAD", "mesh"],
        help="Rendering style. OCC backend requires 'CAD'.",
    )
    proc_group.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt fed to the VLM.",
    )
    proc_group.add_argument(
        "--user-prompt",
        type=str,
        default=DEFAULT_USER_PROMPT,
        help="User prompt fed alongside the rendered image.",
    )
    proc_group.add_argument(
        "--image-type",
        type=str,
        default="png",
        choices=["png", "jpeg"],
        help="Image encoding format used in the request payload.",
    )
    proc_group.add_argument(
        "--image-field",
        type=str,
        default=None,
        help=(
            "Optional dataset image column to use directly, such as "
            "'singleview_image'. If set, the script skips local rendering "
            "and sends that image field to the model."
        ),
    )
    proc_group.add_argument(
        "--id-field",
        type=str,
        default="file_id",
        help=(
            "Dataset column to use as the source identifier. The output JSONL "
            "still stores this value under the 'file_id' key."
        ),
    )

    inf_group = parser.add_argument_group("inference (APIInferenceEngine)")
    inf_group.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name/identifier passed to the chat completions API.",
    )
    inf_group.add_argument(
        "--api-endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL of the OpenAI-compatible inference server.",
    )
    inf_group.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key. If unset the engine substitutes a placeholder.",
    )
    inf_group.add_argument(
        "--inference-num-workers",
        type=int,
        default=16,
        help="Number of parallel inference workers.",
    )

    sampling_group = parser.add_argument_group(
        "sampling (forwarded to chat.completions.create)"
    )
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Only forwarded if explicitly set.",
    )
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top_p. Only forwarded if explicitly set.",
    )
    sampling_group.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Presence penalty. Only forwarded if explicitly set.",
    )
    sampling_group.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help="Frequency penalty. Only forwarded if explicitly set.",
    )
    sampling_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens to generate. Only forwarded if explicitly set.",
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="top_k sampling, placed inside extra_body.",
    )

    thinking_group = sampling_group.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="thinking",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Set extra_body.chat_template_kwargs.enable_thinking=True "
            "(for Qwen-style thinking models)."
        ),
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="thinking",
        action="store_const",
        const=False,
        help="Set extra_body.chat_template_kwargs.enable_thinking=False.",
    )

    sampling_group.add_argument(
        "--extra-body-json",
        type=str,
        default=None,
        help=(
            "Raw JSON string merged into extra_body. Explicit --top-k / "
            "thinking flags override matching keys."
        ),
    )

    out_group = parser.add_argument_group("outputs")
    out_group.add_argument(
        "--generated-out",
        type=str,
        default="generated.jsonl",
        help="JSONL file to write inference results to.",
    )
    out_group.add_argument(
        "--metrics-out",
        type=str,
        default="metrics.json",
        help="Path to write the summary metrics dictionary.",
    )
    out_group.add_argument(
        "--logs-out",
        type=str,
        default="full_metric_logs.json",
        help="Path to write the per-item evaluation logs.",
    )

    eval_group = parser.add_argument_group("evaluator")
    eval_group.add_argument(
        "--skip-eval",
        "--jsonl-only",
        dest="skip_eval",
        action="store_true",
        help="Only run inference; do not score the results.",
    )
    eval_group.add_argument(
        "--eval-num-workers",
        type=int,
        default=16,
        help="Number of parallel evaluator workers.",
    )
    eval_group.add_argument(
        "--variable-name",
        type=str,
        default="result",
        help="Python variable name expected to hold the resulting geometry.",
    )
    eval_group.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Per-item code execution timeout, in seconds.",
    )

    misc_group = parser.add_argument_group("misc")
    misc_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars from both inference and evaluation.",
    )
    misc_group.add_argument(
        "--no-print",
        action="store_true",
        help="Skip the final pretty_print_metrics call.",
    )

    return parser.parse_args()


def build_api_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Assemble the kwargs forwarded to chat.completions.create()."""
    kwargs: Dict[str, Any] = {}

    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.presence_penalty is not None:
        kwargs["presence_penalty"] = args.presence_penalty
    if args.frequency_penalty is not None:
        kwargs["frequency_penalty"] = args.frequency_penalty
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens

    extra_body: Dict[str, Any] = {}
    if args.extra_body_json:
        try:
            parsed = json.loads(args.extra_body_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--extra-body-json is not valid JSON: {exc}")
        if not isinstance(parsed, dict):
            raise SystemExit("--extra-body-json must be a JSON object.")
        extra_body.update(parsed)

    if args.top_k is not None:
        extra_body["top_k"] = args.top_k

    if args.thinking is not None:
        chat_template_kwargs = extra_body.get("chat_template_kwargs", {})
        if not isinstance(chat_template_kwargs, dict):
            raise SystemExit(
                "extra_body.chat_template_kwargs must be a JSON object."
            )
        chat_template_kwargs["enable_thinking"] = args.thinking
        extra_body["chat_template_kwargs"] = chat_template_kwargs

    if extra_body:
        kwargs["extra_body"] = extra_body

    return kwargs


def write_jsonl(rows: list, path: str) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()

    print(f"Loading dataset {args.dataset} (split={args.split})...", file=sys.stderr)
    dataset = load_dataset(args.dataset, split=args.split)

    processor = VLMProcessor(
        dataset=dataset,
        resolution=args.resolution,
        render_backend=args.render_backend,
        render_style=args.render_style,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        image_type=args.image_type,
        image_field=args.image_field,
        id_field=args.id_field,
    )

    api_kwargs = build_api_kwargs(args)

    api = APIInferenceEngine(
        processor,
        model=args.model,
        num_workers=args.inference_num_workers,
        api_endpoint=args.api_endpoint,
        api_key=args.api_key,
        verbose=not args.quiet,
        **api_kwargs,
    )

    print(f"Running inference with model={args.model}...", file=sys.stderr)
    results = api.run()

    write_jsonl(results, args.generated_out)
    print(f"Wrote {len(results)} generations to {args.generated_out}", file=sys.stderr)

    if args.skip_eval:
        print("--skip-eval set; done.", file=sys.stderr)
        return

    evaluator = Evaluator(
        dataset=dataset,
        num_workers=args.eval_num_workers,
        verbose=not args.quiet,
        variable_name=args.variable_name,
        timeout=args.timeout,
    )

    metrics, per_item = evaluator.run(results)

    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote summary metrics to {args.metrics_out}", file=sys.stderr)

    with open(args.logs_out, "w") as f:
        json.dump(per_item, f, indent=2)
    print(f"Wrote per-item logs to {args.logs_out}", file=sys.stderr)

    if not args.no_print:
        pretty_print_metrics(metrics)


if __name__ == "__main__":
    main()
