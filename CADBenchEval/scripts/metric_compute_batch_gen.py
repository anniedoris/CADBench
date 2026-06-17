#!/usr/bin/env python3

"""Generate a shell script to evaluate JSONL files in a directory tree.

Recursively scans `--json_dir` for `*.jsonl` files, infers the bench split
from each filename, auto-detects the python variable name holding the result
geometry from a sample generation (unless `--variable-name` is supplied), and
writes one `python -m scripts.eval_jsonl_modified ...` line per file to
`scripts/run_batch_metrics.sh`.

Use `--skip-existing` to omit JSONLs whose results were already produced.
"""

import argparse
import ast
import json
import os
import shlex
from pathlib import Path
import re


SPLIT_NAMES = ("benchB", "benchF", "benchE", "benchA", "benchM", "benchO")
SPLIT_ORDER = {split: index for index, split in enumerate(SPLIT_NAMES)}

# Boundary on both sides allows the split token to be flanked by `_`, `.`, or
# the start/end of the filename stem (e.g. `benchB.jsonl`, `benchO.jsonl`).
SPLIT_PATTERN = re.compile(
    r"(?:^|[_.])(" + "|".join(SPLIT_NAMES) + r")(?=[_.]|$)"
)

DEFAULT_VARIABLE_NAME = "solid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate run_batch_metrics.sh for all JSONL files in a directory tree."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        required=True,
        help="Root directory to scan recursively for *.jsonl files.",
    )
    parser.add_argument(
        "--variable-name",
        type=str,
        default=None,
        help=(
            "Variable name holding the result. If omitted, the script "
            "auto-detects it per file by parsing the last assignment in a "
            "sample generation (falls back to 'solid')."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Number of workers passed through to scripts.eval_jsonl_modified.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip JSONLs that already have completed per-label metrics.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path to write the generated bash script. Defaults to "
            "scripts/run_batch_metrics.sh next to this script."
        ),
    )
    return parser.parse_args()


def infer_split(jsonl_path: Path) -> str:
    match = SPLIT_PATTERN.search(jsonl_path.stem)
    if match is None:
        raise ValueError(
            f"Could not infer split from filename: {jsonl_path.name} "
            f"(expected a bench split token like benchB, benchE, etc.)"
        )
    return match.group(1)


def _last_top_level_assignment(generated: str) -> str | None:
    """Return the name of the last top-level assignment in `generated`."""
    try:
        tree = ast.parse(generated)
    except SyntaxError:
        return None
    for node in reversed(tree.body):
        if isinstance(node, ast.Assign):
            target = node.targets[-1]
            if isinstance(target, ast.Name):
                return target.id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return node.target.id
    return None


def _regex_last_assignment(generated: str) -> str | None:
    pattern = re.compile(r"^([A-Za-z_]\w*)\s*=(?!=)")
    for line in reversed(generated.splitlines()):
        if not line or line[0].isspace():
            continue
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        match = pattern.match(stripped)
        if match:
            return match.group(1)
    return None


def detect_variable_name(jsonl_path: Path, max_samples: int = 5) -> str:
    """Peek at the first few rows and infer the result variable name."""
    seen = 0
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            generated = row.get("generated", "")
            if not generated:
                continue
            name = _last_top_level_assignment(generated) or _regex_last_assignment(generated)
            if name:
                return name
            seen += 1
            if seen >= max_samples:
                break
    return DEFAULT_VARIABLE_NAME


def shell_quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def has_completed_metrics(jsonl_path: Path) -> bool:
    base_name = jsonl_path.stem
    results_dir = jsonl_path.parent / f"{base_name}_results"
    per_label_txt = results_dir / f"{base_name}_per_label_metrics.txt"
    return per_label_txt.is_file()


def resolve_json_dir(json_dir_arg: str, cadbench_eval_dir: Path) -> Path:
    raw_path = Path(json_dir_arg).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    candidates = [
        (Path.cwd() / raw_path).resolve(),
        (cadbench_eval_dir / raw_path).resolve(),
        (cadbench_eval_dir.parent / raw_path).resolve(),
    ]

    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.is_dir():
            return candidate

    return candidates[0]


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    cadbench_eval_dir = script_dir.parent.resolve()
    json_dir = resolve_json_dir(args.json_dir, cadbench_eval_dir)

    if not json_dir.is_dir():
        raise SystemExit(f"--json_dir is not a directory: {json_dir}")

    jsonl_paths = sorted(path for path in json_dir.rglob("*.jsonl") if path.is_file())
    if not jsonl_paths:
        raise SystemExit(f"No .jsonl files found in: {json_dir}")

    pending: list[tuple[Path, str, str]] = []
    skipped_existing = 0
    skipped_unknown_split: list[Path] = []

    for jsonl_path in jsonl_paths:
        if args.skip_existing and has_completed_metrics(jsonl_path):
            skipped_existing += 1
            continue
        try:
            split = infer_split(jsonl_path)
        except ValueError:
            skipped_unknown_split.append(jsonl_path)
            continue
        variable_name = args.variable_name or detect_variable_name(jsonl_path)
        pending.append((jsonl_path, split, variable_name))
    pending.sort(
        key=lambda item: (
            item[0].parent.as_posix(),
            SPLIT_ORDER[item[1]],
            item[0].name,
        )
    )

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else script_dir / "run_batch_metrics.sh"
    )

    lines = [
        "#!/bin/bash",
        "",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'CADBENCH_EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"',
        "",
        'cd "$CADBENCH_EVAL_DIR"',
        "",
    ]

    for jsonl_path, split, variable_name in pending:
        jsonl_arg = os.path.relpath(jsonl_path, start=cadbench_eval_dir)
        command = [
            "python",
            "-m",
            "scripts.eval_jsonl_modified",
            jsonl_arg,
            "--split",
            split,
            "--num-workers",
            str(args.num_workers),
            "--variable-name",
            variable_name,
        ]
        lines.append(shell_quote_command(command))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_path.chmod(0o755)

    print(
        f"Wrote {len(pending)} commands to {output_path} "
        f"(skipped {skipped_existing} completed, "
        f"{len(skipped_unknown_split)} unknown-split)"
    )
    if skipped_unknown_split:
        print("Files with unrecognized split:")
        for path in skipped_unknown_split:
            print(f"  {path}")


if __name__ == "__main__":
    main()
