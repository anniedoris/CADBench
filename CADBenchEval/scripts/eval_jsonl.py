"""Evaluate a JSONL of generated CADQuery code against a CADBench split.

Reads a JSONL file containing rows of at least ``{"file_id": ..., "generated": ...}``
(extra keys such as ``raw_response`` are tolerated and ignored), runs the
CADBench Evaluator against the specified dataset split, and writes the summary
metrics and per-item logs to disk.
"""

import argparse
import json
import os
import sys

from datasets import load_dataset

from CADBench import pretty_print_metrics
from CADBench.Eval import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a JSONL of generated code against a CADBench split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "jsonl",
        type=str,
        help="Path to the input JSONL file with one generation per line.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default="DeCoDELab/CADBench",
        help="HuggingFace dataset id holding the ground-truth split.",
    )
    dataset_group.add_argument(
        "--split",
        type=str,
        default="bench0",
        help="Dataset split to evaluate against.",
    )

    eval_group = parser.add_argument_group("evaluator")
    eval_group.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel worker processes used by the evaluator.",
    )
    eval_group.add_argument(
        "--variable-name",
        type=str,
        default="solid",
        help="Python variable name expected to hold the resulting geometry.",
    )
    eval_group.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Per-item code execution timeout, in seconds.",
    )

    output_group = parser.add_argument_group("outputs")
    output_group.add_argument(
        "--metrics-out",
        type=str,
        default="metrics.json",
        help="Path to write the summary metrics dictionary.",
    )
    output_group.add_argument(
        "--logs-out",
        type=str,
        default="full_metric_logs.json",
        help="Path to write the per-item evaluation logs.",
    )

    misc_group = parser.add_argument_group("misc")
    misc_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress evaluator progress bars.",
    )
    misc_group.add_argument(
        "--no-print",
        action="store_true",
        help="Skip the final pretty_print_metrics call.",
    )

    return parser.parse_args()


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Failed to parse JSON on line {line_no} of {path}: {exc}"
                )
    return rows


def main() -> None:
    args = parse_args()

    print(f"Loading dataset {args.dataset} (split={args.split})...", file=sys.stderr)
    dataset = load_dataset(args.dataset, split=args.split)

    print(f"Loading generations from {args.jsonl}...", file=sys.stderr)
    outputs = load_jsonl(args.jsonl)
    print(f"  Loaded {len(outputs)} rows.", file=sys.stderr)

    evaluator = Evaluator(
        dataset=dataset,
        num_workers=args.num_workers,
        verbose=not args.quiet,
        variable_name=args.variable_name,
        timeout=args.timeout,
    )

    metrics, per_item = evaluator.run(outputs)

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
