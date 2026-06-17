#!/usr/bin/env python3

"""Collect selected per-label metrics into compact CSV files and summaries."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


LABEL_ORDER = {"easy": 0, "medium": 1, "hard": 2}
EXPECTED_SUFFIX = "_per_label_metrics.json"
COLLECTED_METRICS_SUFFIX = "_collected_metrics.csv"
RUN_DIR_RE = re.compile(r"r\d+$")
BENCH_NAME_RE = re.compile(r"(?:^|_)(bench[^_]+)(?:_|$)")
METRIC_FIELDNAMES = [
    "aligned_iou_adjusted_median",
    "aligned_chamfer_distance_success_only_median",
    "aligned_surface_iou_adjusted_median",
    "vsr",
    "token_count_success_only_median",
    "total_operations_success_only_median",
]
SUMMARY_AGGREGATE_METRICS = [
    "aligned_iou_adjusted_median",
    "aligned_chamfer_distance_success_only_median",
    "aligned_surface_iou_adjusted_median",
    "vsr",
    "token_count_success_only_median",
    "total_operations_success_only_median",
]
INTEGER_SUMMARY_METRICS = {
    "token_count_success_only_median",
    "total_operations_success_only_median",
}
NO_HEATCELL_METRICS = {
    "token_count_success_only_median",
    "total_operations_success_only_median",
}
CHAMFER_HEATCELL_METRICS = {
    "aligned_chamfer_distance_success_only_median",
}
SUMMARY_COLUMN_ORDER: list[tuple[str, str | None]] = [
    ("benchB", "easy"),
    ("benchF", "easy"),
    ("benchE", "easy"),
    ("benchA", "easy"),
    ("benchB", "medium"),
    ("benchF", "medium"),
    ("benchE", "medium"),
    ("benchA", "medium"),
    ("benchB", "hard"),
    ("benchF", "hard"),
    ("benchE", "hard"),
    ("benchA", "hard"),
    ("benchM", None),
    ("benchO", None),
]
SUMMARY_COLUMN_SET = set(SUMMARY_COLUMN_ORDER)
SUMMARY_COLUMN_WEIGHTS = {
    summary_key: 3 if summary_key[0] in {"benchM", "benchO"} else 1
    for summary_key in SUMMARY_COLUMN_ORDER
}
SUMMARY_FAMILY_ORDER = [
    "benchB",
    "benchF",
    "benchE",
    "benchA",
    "benchM",
    "benchO",
]
SUMMARY_FAMILY_SET = set(SUMMARY_FAMILY_ORDER)
SUMMARY_FAMILY_MEMBERS = {
    "benchB": [
        ("benchB", "easy"),
        ("benchB", "medium"),
        ("benchB", "hard"),
    ],
    "benchF": [
        ("benchF", "easy"),
        ("benchF", "medium"),
        ("benchF", "hard"),
    ],
    "benchE": [
        ("benchE", "easy"),
        ("benchE", "medium"),
        ("benchE", "hard"),
    ],
    "benchA": [
        ("benchA", "easy"),
        ("benchA", "medium"),
        ("benchA", "hard"),
    ],
    "benchM": [("benchM", None)],
    "benchO": [("benchO", None)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract selected fields from one or more *_per_label_metrics.json "
            "files into CSV."
        )
    )
    parser.add_argument(
        "--results_json",
        type=str,
        required=True,
        help=(
            "Path to a JSON file ending in _per_label_metrics.json, or a "
            "directory to search recursively for matching files."
        ),
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default=None,
        help="Optional output CSV path. Defaults next to the JSON input.",
    )
    return parser.parse_args()


def derive_csv_out(results_json: Path, csv_out_arg: str | None) -> Path:
    if csv_out_arg is not None:
        return Path(csv_out_arg).expanduser().resolve()
    out_name = results_json.name.replace(
        EXPECTED_SUFFIX, COLLECTED_METRICS_SUFFIX
    )
    return results_json.with_name(out_name)


def get_nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            joined = " -> ".join(keys)
            raise KeyError(f"Missing key path: {joined}")
        current = current[key]
    return current


def label_sort_key(label: str) -> tuple[int, str]:
    return (LABEL_ORDER.get(label.lower(), 99), label.lower())


def collect_rows(results_json: Path) -> list[dict[str, Any]]:
    with results_json.open("r", encoding="utf-8") as f:
        metrics_by_label = json.load(f)

    if not isinstance(metrics_by_label, dict) or not metrics_by_label:
        raise SystemExit(f"No per-label metrics found in: {results_json}")

    rows: list[dict[str, Any]] = []
    for label in sorted(metrics_by_label, key=label_sort_key):
        label_metrics = metrics_by_label[label]

        aligned_iou_adjusted_count = float(
            get_nested(label_metrics, "Aligned IoU", "adjusted", "count")
        )
        aligned_iou_success_count = float(
            get_nested(label_metrics, "Aligned IoU", "success_only", "count")
        )
        vsr = (
            aligned_iou_success_count / aligned_iou_adjusted_count
            if aligned_iou_adjusted_count > 0
            else 0.0
        )

        rows.append(
            {
                "split": label,
                "aligned_iou_adjusted_median": get_nested(
                    label_metrics, "Aligned IoU", "adjusted", "median"
                ),
                "aligned_chamfer_distance_success_only_median": get_nested(
                    label_metrics,
                    "Aligned Chamfer Distance",
                    "success_only",
                    "median",
                ),
                "aligned_surface_iou_adjusted_median": get_nested(
                    label_metrics,
                    "Aligned Surface IoU",
                    "adjusted",
                    "median",
                ),
                "vsr": vsr,
                "token_count_success_only_median": get_nested(
                    label_metrics, "token_count", "success_only", "median"
                ),
                "total_operations_success_only_median": get_nested(
                    label_metrics,
                    "total_operations",
                    "success_only",
                    "median",
                ),
            }
        )

    return rows


def write_csv(csv_out: Path, rows: list[dict[str, Any]]) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", *METRIC_FIELDNAMES]
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_parse_bench_name(path: Path) -> str | None:
    match = BENCH_NAME_RE.search(path.stem)
    if match is None:
        return None
    return match.group(1)


def find_results_jsons(results_path: Path) -> list[Path]:
    if results_path.is_file():
        if not results_path.name.endswith(EXPECTED_SUFFIX):
            raise SystemExit(
                f"--results_json must end with {EXPECTED_SUFFIX}: {results_path}"
            )
        return [results_path]

    if results_path.is_dir():
        matches = sorted(results_path.rglob(f"*{EXPECTED_SUFFIX}"))
        if not matches:
            raise SystemExit(
                f"No files ending in {EXPECTED_SUFFIX} found under: {results_path}"
            )
        current_matches = [
            path
            for path in matches
            if maybe_parse_bench_name(path) in SUMMARY_FAMILY_SET
        ]
        ignored_count = len(matches) - len(current_matches)
        if ignored_count:
            print(
                f"Ignored {ignored_count} metrics file(s) with non-current "
                "bench names."
            )
        if not current_matches:
            raise SystemExit(
                "No current benchmark metrics found under "
                f"{results_path}; expected bench names "
                f"{', '.join(SUMMARY_FAMILY_ORDER)}."
            )
        return current_matches

    raise SystemExit(f"--results_json is not a file or directory: {results_path}")


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def find_summary_roots(results_path: Path, csv_paths: list[Path]) -> list[Path]:
    if not results_path.is_dir() or not csv_paths:
        return []

    child_dirs = sorted(
        child
        for child in results_path.iterdir()
        if child.is_dir() and any(is_relative_to(csv_path, child) for csv_path in csv_paths)
    )
    if not child_dirs:
        return [results_path]

    if any(RUN_DIR_RE.fullmatch(child.name) for child in child_dirs):
        return [results_path]

    if all(child.name.startswith("model_output_") for child in child_dirs):
        return [results_path]

    return child_dirs


def parse_bench_name(csv_path: Path) -> str:
    bench_name = maybe_parse_bench_name(csv_path)
    if bench_name is None:
        raise SystemExit(f"Could not parse bench name from CSV path: {csv_path}")
    return bench_name


def get_run_key(modality_root: Path, csv_path: Path, has_run_dirs: bool) -> str:
    if not has_run_dirs:
        return "__single_run__"

    relative_csv_path = csv_path.relative_to(modality_root)
    if not relative_csv_path.parts:
        raise SystemExit(f"Unexpected CSV path under modality root: {csv_path}")

    run_name = relative_csv_path.parts[0]
    if not RUN_DIR_RE.fullmatch(run_name):
        raise SystemExit(
            f"Expected run directory like r1/r2/... under {modality_root}, "
            f"but found {csv_path}"
        )
    return run_name


def normalize_summary_key(bench_name: str, split: str) -> tuple[str, str | None]:
    normalized_split = split.strip().lower() or None
    if bench_name in {"benchM", "benchO"}:
        return (bench_name, None)
    return (bench_name, normalized_split)


def format_number(value: float, integer: bool = False) -> str:
    if integer:
        return str(int(round(value)))
    return f"{value:.3f}"


def format_heatcell(
    mean_value: float,
    std_value: float | None,
    integer: bool = False,
    command: str = "heatcell",
) -> str:
    if std_value is None:
        return f"\\{command}{{{format_number(mean_value, integer=integer)}}}"
    return (
        f"\\{command}[{format_number(std_value, integer=integer)}]"
        f"{{{format_number(mean_value, integer=integer)}}}"
    )


def summarize_run_values(
    run_values: list[float],
    *,
    has_run_dirs: bool,
    summary_key: tuple[str, str | None],
    metric_name: str,
    modality_root: Path,
) -> tuple[float | None, float | None]:
    if not run_values:
        print(
            f"Missing summary value for {metric_name} at {summary_key!r} "
            f"under {modality_root}"
        )
        return None, None

    if has_run_dirs:
        average_value = statistics.fmean(run_values)
        std_value = statistics.stdev(run_values) if len(run_values) > 1 else 0.0
        return average_value, std_value

    if len(run_values) != 1:
        raise SystemExit(
            "Found multiple values for a single-run summary at "
            f"{summary_key!r} under {modality_root}"
        )
    return run_values[0], None


def compute_weighted_mean(weighted_values: list[tuple[float, int]]) -> float | None:
    if not weighted_values:
        return None

    total_weight = sum(weight for _, weight in weighted_values)
    if total_weight <= 0:
        return None

    return sum(value * weight for value, weight in weighted_values) / total_weight


def compute_family_mean(
    family_values: dict[tuple[str, str | None], float],
    *,
    family_name: str,
    metric_name: str,
    modality_root: Path,
) -> float | None:
    member_keys = SUMMARY_FAMILY_MEMBERS[family_name]
    missing_keys = [
        summary_key for summary_key in member_keys if summary_key not in family_values
    ]
    if missing_keys:
        print(
            f"Missing family summary value for {metric_name} at {family_name} "
            f"under {modality_root}; missing {missing_keys!r}"
        )
        return None

    return statistics.fmean(family_values[summary_key] for summary_key in member_keys)


def collect_summary_metrics(
    modality_root: Path, csv_paths: list[Path]
) -> tuple[dict[str, dict[tuple[str, str | None], dict[str, float]]], bool]:
    has_run_dirs = any(
        RUN_DIR_RE.fullmatch(child.name)
        for child in modality_root.iterdir()
        if child.is_dir()
    )
    summary_values: dict[str, dict[tuple[str, str | None], dict[str, float]]] = {
        metric_name: {} for metric_name in METRIC_FIELDNAMES
    }

    for csv_path in sorted(csv_paths):
        bench_name = parse_bench_name(csv_path)
        run_key = get_run_key(modality_root, csv_path, has_run_dirs)

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise SystemExit(f"CSV has no header: {csv_path}")

            missing_metrics = [
                metric_name
                for metric_name in METRIC_FIELDNAMES
                if metric_name not in reader.fieldnames
            ]
            if missing_metrics:
                raise SystemExit(
                    f"CSV is missing expected metric columns {missing_metrics}: {csv_path}"
                )

            for row in reader:
                summary_key = normalize_summary_key(bench_name, row.get("split", ""))
                if summary_key not in SUMMARY_COLUMN_SET:
                    print(
                        "Skipping unexpected bench/split combination "
                        f"{summary_key!r} from {csv_path}"
                    )
                    continue

                for metric_name in METRIC_FIELDNAMES:
                    run_values = summary_values[metric_name].setdefault(summary_key, {})
                    if run_key in run_values:
                        raise SystemExit(
                            "Duplicate summary value for "
                            f"{metric_name} at {summary_key!r} in {csv_path}"
                        )
                    run_values[run_key] = float(row[metric_name])

    return summary_values, has_run_dirs


def write_results_summary(summary_out: Path, modality_root: Path, csv_paths: list[Path]) -> None:
    summary_values, has_run_dirs = collect_summary_metrics(modality_root, csv_paths)
    lines: list[str] = []
    mean_values_by_metric: dict[str, list[tuple[float, int]]] = {
        metric_name: [] for metric_name in METRIC_FIELDNAMES
    }
    cell_mean_values_by_metric: dict[
        str, dict[tuple[str, str | None], float]
    ] = {metric_name: {} for metric_name in METRIC_FIELDNAMES}

    for metric_name in METRIC_FIELDNAMES:
        cells: list[str] = []
        integer_metric = metric_name in INTEGER_SUMMARY_METRICS
        if metric_name in NO_HEATCELL_METRICS:
            heatcell_command = "noheatcell"
        elif metric_name in CHAMFER_HEATCELL_METRICS:
            heatcell_command = "heatcellcd"
        else:
            heatcell_command = "heatcell"
        for summary_key in SUMMARY_COLUMN_ORDER:
            run_values = list(summary_values[metric_name].get(summary_key, {}).values())
            average_value, std_value = summarize_run_values(
                run_values,
                has_run_dirs=has_run_dirs,
                summary_key=summary_key,
                metric_name=metric_name,
                modality_root=modality_root,
            )
            if average_value is None:
                cells.append("NA")
                continue

            mean_values_by_metric[metric_name].append(
                (average_value, SUMMARY_COLUMN_WEIGHTS[summary_key])
            )
            cell_mean_values_by_metric[metric_name][summary_key] = average_value
            cells.append(
                format_heatcell(
                    average_value,
                    std_value,
                    integer=integer_metric,
                    command=heatcell_command,
                )
            )

        aggregate_value = compute_weighted_mean(mean_values_by_metric[metric_name])
        if aggregate_value is None:
            cells.append("NA")
        else:
            cells.append(
                "\\textbf{"
                + format_heatcell(
                    aggregate_value,
                    None,
                    integer=integer_metric,
                    command=heatcell_command,
                )
                + "}"
            )

        lines.append(f"{metric_name}: {' & '.join(cells)}")

    lines.append("")
    for metric_name in SUMMARY_AGGREGATE_METRICS:
        weighted_values = mean_values_by_metric[metric_name]
        if not weighted_values:
            print(
                "Missing all summary values for "
                f"{metric_name} under {modality_root}"
            )
            lines.append(f"average_{metric_name}_across_all_benches: NA")
            continue
        mean_of_means = compute_weighted_mean(weighted_values)
        if mean_of_means is None:
            lines.append(f"average_{metric_name}_across_all_benches: NA")
            continue
        integer_metric = metric_name in INTEGER_SUMMARY_METRICS
        lines.append(
            f"average_{metric_name}_across_all_benches: "
            f"{format_number(mean_of_means, integer=integer_metric)}"
        )

    lines.append("")
    for metric_name in SUMMARY_AGGREGATE_METRICS:
        integer_metric = metric_name in INTEGER_SUMMARY_METRICS
        family_values = cell_mean_values_by_metric[metric_name]
        for family_name in SUMMARY_FAMILY_ORDER:
            family_mean = compute_family_mean(
                family_values,
                family_name=family_name,
                metric_name=metric_name,
                modality_root=modality_root,
            )
            if family_mean is None:
                lines.append(f"average_{metric_name}_for_{family_name}: NA")
                continue
            lines.append(
                f"average_{metric_name}_for_{family_name}: "
                f"{format_number(family_mean, integer=integer_metric)}"
            )

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_json).expanduser().resolve()
    results_jsons = find_results_jsons(results_path)

    if len(results_jsons) > 1 and args.csv_out is not None:
        raise SystemExit("--csv_out can only be used when processing a single JSON.")

    generated_csvs: list[Path] = []
    for results_json in results_jsons:
        csv_out = derive_csv_out(results_json, args.csv_out)
        rows = collect_rows(results_json)
        write_csv(csv_out, rows)
        generated_csvs.append(csv_out)
        print(f"Wrote {len(rows)} rows to {csv_out}")

    for summary_root in find_summary_roots(results_path, generated_csvs):
        modality_csvs = [
            csv_path
            for csv_path in generated_csvs
            if is_relative_to(csv_path, summary_root)
        ]
        if not modality_csvs:
            continue

        summary_out = summary_root / "results_summary.txt"
        write_results_summary(summary_out, summary_root, modality_csvs)
        print(f"Wrote summary to {summary_out}")


if __name__ == "__main__":
    main()
