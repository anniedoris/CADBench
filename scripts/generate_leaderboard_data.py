#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "docs" / "static" / "leaderboard-data.json"
MODALITIES_LATEX_PATH = ROOT / "graph" / "plots" / "modalities_latex.txt"
METRIC_SPECS = {
    "iou": {
        "source_key": "average_aligned_iou_adjusted_median_across_all_benches",
    },
    "siou": {
        "source_key": "average_aligned_surface_iou_adjusted_median_across_all_benches",
    },
    "vsr": {
        "source_key": "average_vsr_across_all_benches",
    },
    "cd": {
        "source_key": "average_aligned_chamfer_distance_success_only_median_across_all_benches",
    },
    "token_count": {
        "source_key": "average_token_count_success_only_median_across_all_benches",
    },
    "op_count": {
        "source_key": "average_total_operations_success_only_median_across_all_benches",
    },
}

BENCH_METRIC_KEY_TEMPLATES = {
    "iou":         "average_aligned_iou_adjusted_median_for_{suffix}",
    "siou":        "average_aligned_surface_iou_adjusted_median_for_{suffix}",
    "cd":          "average_aligned_chamfer_distance_success_only_median_for_{suffix}",
    "vsr":         "average_vsr_for_{suffix}",
    "token_count": "average_token_count_success_only_median_for_{suffix}",
    "op_count":    "average_total_operations_success_only_median_for_{suffix}",
}

BENCH_ORDER = [
    "B0_easy", "B0F_easy", "B1A_easy", "B1B_easy",
    "B0_medium", "B0F_medium", "B1A_medium", "B1B_medium",
    "B0_hard", "B0F_hard", "B1A_hard", "B1B_hard",
]

BENCH_SLIDES = [
    {"slide_index": 0, "bench_suffix": "bench0",  "family_name": "CAD-Base"},
    {"slide_index": 1, "bench_suffix": "bench0F", "family_name": "CAD-Fusion"},
    {"slide_index": 2, "bench_suffix": "bench1A", "family_name": "CAD-Extrude"},
    {"slide_index": 3, "bench_suffix": "bench1B", "family_name": "CAD-All-Ops"},
    {"slide_index": 4, "bench_suffix": "bench2",  "family_name": "CAD-Mechanical"},
    {"slide_index": 5, "bench_suffix": "bench3",  "family_name": "CAD-Organic"},
]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    color: str


MESH_MODELS = [
    ModelSpec("cadfit", "CADFit", "#08306b"),
    ModelSpec("cadrecode", "CAD-Recode", "#9e8cc5"),
    ModelSpec("cadevolve", "CADEvolve", "#74c6c6"),
    ModelSpec("cadrille_pc", "Cadrille", "#6baed6"),
]

IMAGE_MODELS = [
    ModelSpec("claude4.7", "Claude Opus 4.7", "#005824"),
    ModelSpec("gemini3.1", "Gemini 3.1 Pro", "#2ca25f"),
    ModelSpec("gpt5.4", "GPT-5.4", "#8c510a"),
    ModelSpec("kimi_2.6", "Kimi K2.6", "#c8d400"),
    ModelSpec("qwen3.59b", "Qwen 3.59B", "#65a30d"),
    ModelSpec("qwen3.527b", "Qwen 3.527B", "#f59e0b"),
    ModelSpec("cadcoder", "CAD-Coder", "#e08010"),
]


def parse_mesh_noisy_iou() -> dict[str, dict[str, float]]:
    """Parse noisy-mesh IoU and delta for mesh models from the first block of modalities_latex.txt.

    Returns a dict mapping model label (e.g. "CADFit") to {"noisy": value, "delta": value}.
    The first block ends at the first blank line.
    """
    delta_re = re.compile(r"\\deltacell\{([+-]?[0-9.]+)\}")
    text = MODALITIES_LATEX_PATH.read_text(encoding="utf-8")
    result: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            break  # end of mesh section
        parts = [p.strip() for p in stripped.split("&")]
        if len(parts) < 4:
            continue
        try:
            noisy = float(parts[2])
        except ValueError:
            continue
        delta_match = delta_re.search(parts[3])
        delta = float(delta_match.group(1)) if delta_match else None
        entry: dict[str, float] = {"noisy": noisy}
        if delta is not None:
            entry["delta"] = delta
        result[parts[0]] = entry
    return result


IMAGE_MODALITY_NAME_MAP = {
    "Qwen3.5 27B": "Qwen 3.527B",
    "Qwen3.5 9B": "Qwen 3.59B",
}


def parse_image_modality_iou() -> dict[str, dict[str, float]]:
    """Parse photorealistic and multiview IoU + deltas from the second block of modalities_latex.txt.

    Format per line: name & default & photo & \\deltacell{photo_delta} & multi & \\deltacell{multi_delta}
    Returns {normalized_label: {"photo": v, "photo_delta": v, "multi": v, "multi_delta": v}}.
    The second block is the first non-empty block after skipping the mesh block.
    """
    delta_re = re.compile(r"\\deltacell\{([+-]?[0-9.]+)\}")
    text = MODALITIES_LATEX_PATH.read_text(encoding="utf-8")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) < 2:
        return {}
    result: dict[str, dict[str, float]] = {}
    for line in blocks[1].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [p.strip() for p in stripped.split("&")]
        if len(parts) < 5:
            continue
        raw_name = parts[0]
        name = IMAGE_MODALITY_NAME_MAP.get(raw_name, raw_name)
        try:
            photo = float(parts[2])
            multi = float(parts[4])
        except ValueError:
            continue
        entry: dict[str, float] = {"photo": photo, "multi": multi}
        photo_delta_m = delta_re.search(parts[3]) if len(parts) > 3 else None
        multi_delta_m = delta_re.search(parts[5]) if len(parts) > 5 else None
        if photo_delta_m:
            entry["photo_delta"] = float(photo_delta_m.group(1))
        if multi_delta_m:
            entry["multi_delta"] = float(multi_delta_m.group(1))
        result[name] = entry
    return result


def parse_split_row(summary_text: str, row_key: str) -> dict[str, float]:
    """Parse a LaTeX heatcell row into a dict keyed by BENCH_ORDER."""
    pattern = re.compile(rf"^{re.escape(row_key)}:\s*(.+)$", re.MULTILINE)
    match = pattern.search(summary_text)
    if not match:
        raise ValueError(f"Missing row '{row_key}' in results summary")
    values = re.findall(r"\{([0-9.]+)\}", match.group(1))
    if len(values) < len(BENCH_ORDER):
        raise ValueError(
            f"Expected at least {len(BENCH_ORDER)} values for '{row_key}', got {len(values)}"
        )
    return {key: float(values[i]) for i, key in enumerate(BENCH_ORDER)}


def build_complexity_iou(models: list[ModelSpec], path_segment: str) -> dict[str, dict[str, float]]:
    result = {}
    for model in models:
        summary_path = ROOT / "tested_models" / model.key / path_segment / "results_summary.txt"
        result[model.key] = parse_split_row(
            summary_path.read_text(encoding="utf-8"), "aligned_iou_adjusted_median"
        )
    return result


def parse_summary_value(summary_text: str, metric_key: str) -> float:
    pattern = re.compile(rf"^{re.escape(metric_key)}:\s*([0-9]*\.?[0-9]+)\s*$", re.MULTILINE)
    match = pattern.search(summary_text)

    if not match:
        raise ValueError(f"Missing metric '{metric_key}' in results summary")

    return float(match.group(1))


def build_metric_entries(
    models: list[ModelSpec], path_segment: str, source_key: str
) -> list[dict[str, object]]:
    entries = []

    for model in models:
        summary_path = ROOT / "tested_models" / model.key / path_segment / "results_summary.txt"
        summary_text = summary_path.read_text(encoding="utf-8")
        entries.append(
            {
                "name": model.label,
                "value": parse_summary_value(summary_text, source_key),
                "color": model.color,
            }
        )

    return entries


def build_metric_payload(metric_key: str) -> dict[str, object]:
    source_key = METRIC_SPECS[metric_key]["source_key"]
    mesh_entries = build_metric_entries(MESH_MODELS, "mesh", source_key)

    if metric_key == "iou":
        noisy_iou_map = parse_mesh_noisy_iou()
        for entry in mesh_entries:
            data = noisy_iou_map.get(entry["name"])
            if data is not None:
                entry["noisy_value"] = data["noisy"]
                if "delta" in data:
                    entry["noisy_delta"] = data["delta"]

    image_entries = build_metric_entries(IMAGE_MODELS, "singleview", source_key)

    if metric_key == "iou":
        image_modality_map = parse_image_modality_iou()
        for entry in image_entries:
            data = image_modality_map.get(entry["name"])
            if data is not None:
                for field in ("photo", "photo_delta", "multi", "multi_delta"):
                    if field in data:
                        entry[f"{field}_value" if field in ("photo", "multi") else field] = data[field]

    return {
        "source_key": source_key,
        "mesh_entries": mesh_entries,
        "image_entries": image_entries,
    }


def build_bench_slide_payload(bench_suffix: str) -> dict[str, object]:
    metrics = {}
    for metric_key, template in BENCH_METRIC_KEY_TEMPLATES.items():
        source_key = template.format(suffix=bench_suffix)
        metrics[metric_key] = {
            "source_key": source_key,
            "mesh_entries": build_metric_entries(MESH_MODELS, "mesh", source_key),
            "image_entries": build_metric_entries(IMAGE_MODELS, "singleview", source_key),
        }
    return {"metrics": metrics}


def main() -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_patterns": {
            "mesh": "tested_models/{model_name}/mesh/results_summary.txt",
            "image": "tested_models/{model_name}/singleview/results_summary.txt",
        },
        "metrics": {
            metric_key: build_metric_payload(metric_key)
            for metric_key in METRIC_SPECS
        },
        "bench_slides": {
            str(slide["slide_index"]): build_bench_slide_payload(slide["bench_suffix"])
            for slide in BENCH_SLIDES
        },
        "complexity": {
            "bench_complexities": json.loads(
                (ROOT / "graph" / "bench_complexities.json").read_text(encoding="utf-8")
            ),
            "iou": {
                "mesh": build_complexity_iou(MESH_MODELS, "mesh"),
                "image": build_complexity_iou(IMAGE_MODELS, "singleview"),
            },
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
