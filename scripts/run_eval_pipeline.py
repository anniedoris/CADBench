#!/usr/bin/env python3

"""Run the full CADBench metric evaluation pipeline in one command.

Steps:
  1. metric_compute_batch_gen.py  — generates run_batch_metrics.sh
  2. run_batch_metrics.sh         — executes per-JSONL metric computation
  3. collect_metrics.py           — aggregates results into CSV / summary
"""

import argparse
import subprocess
import sys
from pathlib import Path

CADBENCH_EVAL_DIR = Path(__file__).resolve().parent.parent / "CADBenchEval"


def run(cmd: list[str], cwd: Path) -> None:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full CADBench eval pipeline.")
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to the model output directory (e.g. tested_models/my_model)",
    )
    parser.add_argument(
        "--variable-name",
        required=True,
        help="CadQuery variable name used in generated code (e.g. 'result')",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Number of parallel workers for metric computation (default: 20)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip JSONLs that already have completed metrics "
            "(a <stem>_results/<stem>_per_label_metrics.txt file)."
        ),
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        sys.exit(f"--model_dir is not a directory: {model_dir}")

    if not CADBENCH_EVAL_DIR.is_dir():
        sys.exit(f"CADBenchEval directory not found: {CADBENCH_EVAL_DIR}")

    print(f"Model directory : {model_dir}")
    print(f"CADBenchEval dir: {CADBENCH_EVAL_DIR}")

    # Step 1: generate run_batch_metrics.sh
    gen_cmd = [
        sys.executable, "scripts/metric_compute_batch_gen.py",
        "--json_dir", str(model_dir),
        "--variable-name", args.variable_name,
        "--num-workers", str(args.num_workers),
    ]
    if args.skip_existing:
        gen_cmd.append("--skip-existing")
    run(gen_cmd, cwd=CADBENCH_EVAL_DIR)

    # Step 2: run the generated batch metrics script
    run(["bash", "scripts/run_batch_metrics.sh"], cwd=CADBENCH_EVAL_DIR)

    # Step 3: collect and aggregate results
    run(
        [sys.executable, "scripts/collect_metrics.py", "--results_json", str(model_dir)],
        cwd=CADBENCH_EVAL_DIR,
    )

    print(f"\nDone. Results written alongside JSONL files in {model_dir}")


if __name__ == "__main__":
    main()
