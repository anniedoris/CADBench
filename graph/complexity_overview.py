import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
TESTED_MODELS_DIR = SCRIPT_DIR.parent / "tested_models"
PLOTS_DIR = SCRIPT_DIR / "plots"

BENCH_ORDER = [
    "benchB_easy",
    "benchF_easy",
    "benchE_easy",
    "benchA_easy",
    "benchB_medium",
    "benchF_medium",
    "benchE_medium",
    "benchA_medium",
    "benchB_hard",
    "benchF_hard",
    "benchE_hard",
    "benchA_hard",
]

with (SCRIPT_DIR / "bench_complexities.json").open(encoding="utf-8") as _f:
    BENCH_COMPLEXITIES = json.load(_f)

MESH_SERIES = [
    ("cadfit", "mesh", "CADFit", {"color": "#08306b", "marker": "^"}, True),
    ("cadrecode", "mesh", "CAD-Recode", {"color": "#9e8cc5", "marker": "^"}, True),
    ("cadevolve", "mesh", "CADEvolve", {"color": "#74c6c6", "marker": "^"}, True),
    ("cadrille_pc", "mesh", "Cadrille", {"color": "#6baed6", "marker": "^"}, True),
]

IMAGE_SERIES = [
    ("claude4.7", "singleview", "Claude Opus 4.7", {"color": "#005824"}, True),
    ("gemini3.1", "singleview", "Gemini 3.1 Pro", {"color": "#2ca25f"}, True),
    ("gpt5.4", "singleview", "GPT-5.4", {"color": "#8c510a"}, True),
    ("kimi_2.6", "singleview", "Kimi K2.6", {"color": "#c8d400"}, False),
    ("cadcoder", "singleview", "CAD-Coder", {"color": "#e08010", "marker": "^"}, True),
]

METRIC_PLOTS = [
    (
        "aligned_iou_adjusted_median",
        "IoU Score on Split",
        "cadfit_mesh_complexity.png",
    )
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", choices=["extrude", "all", "total"], default="extrude")
    return parser.parse_args()


def load_bench_metric(model, modality, metric_name):
    path = TESTED_MODELS_DIR / model / modality / "results_summary.txt"
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{metric_name}:"):
                values = re.findall(r"\{([0-9.]+)\}", line)[: len(BENCH_ORDER)]
                if len(values) != len(BENCH_ORDER):
                    raise ValueError(
                        f"Expected {len(BENCH_ORDER)} values for {metric_name} in {path}, "
                        f"found {len(values)}"
                    )
                return dict(zip(BENCH_ORDER, [float(v) for v in values]))
    raise ValueError(f"{metric_name} not found in {path}")


def get_benchmark_complexities(ops):
    if ops == "total":
        return {**BENCH_COMPLEXITIES["extrude"], **BENCH_COMPLEXITIES["all"]}
    return BENCH_COMPLEXITIES[ops]


def get_label_offsets(ops):
    if ops == "all":
        return {"benchA_easy": -6, "benchF_medium": 8}
    if ops == "total":
        return {"benchB_medium": 4, "benchA_easy": -8}
    return {"benchB_medium": 4}


def load_series(metric_name, series_specs):
    loaded = []
    for model, modality, label, style, annotate in series_specs:
        loaded.append(
            (load_bench_metric(model, modality, metric_name), label, style, annotate)
        )
    return loaded


def plot_model(ax, data, benchmark_complexities, label, **kwargs):
    keys = sorted(
        (key for key in data if key in benchmark_complexities),
        key=lambda key: benchmark_complexities[key],
    )
    x = [benchmark_complexities[key] for key in keys]
    y = [data[key] for key in keys]
    ax.plot(
        x,
        y,
        marker=kwargs.pop("marker", "o"),
        label=label,
        linewidth=2.5,
        **kwargs,
    )


def make_tick_labels(benchmark_complexities):
    ticks = sorted(set(benchmark_complexities.values()))
    return ticks, [str(tick) for tick in ticks]


def annotate_splits(
    ax,
    all_model_data,
    benchmark_complexities,
    find="min",
    xoffsets=None,
    yoffsets=None,
):
    abbrev = {"easy": "$_{L}$", "medium": "$_{M}$", "hard": "$_{H}$"}
    bench_names = {"benchB": "B", "benchF": "F", "benchE": "E", "benchA": "A"}
    base_yoffset = -5 if find == "min" else 5
    va = "top" if find == "min" else "bottom"
    xoffsets = xoffsets or {}
    yoffsets = yoffsets or {}
    for key, x in benchmark_complexities.items():
        prefix, diff = key.rsplit("_", 1)
        label = f"{bench_names.get(prefix, prefix)}{abbrev[diff]}"
        values = [data[key] for data in all_model_data if key in data]
        if not values:
            continue
        y = min(values) if find == "min" else max(values)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(xoffsets.get(key, 0), base_yoffset + yoffsets.get(key, 0)),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=11,
        )


def configure_axis(ax, tick_vals, tick_labels, ylabel, title):
    ax.set_xscale("log")
    ax.set_xlabel("Split Median Face Count")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    for value in tick_vals:
        ax.axvline(value, color="lightgray", linestyle=":", linewidth=0.8, zorder=0)


def print_total_axis_order(benchmark_complexities):
    abbrev = {"easy": "$_{L}$", "medium": "$_{M}$", "hard": "$_{H}$"}
    bench_names = {"benchB": "B", "benchF": "F", "benchE": "E", "benchA": "A"}
    by_faces = defaultdict(list)
    for key, faces in sorted(benchmark_complexities.items(), key=lambda item: item[1]):
        prefix, diff = key.rsplit("_", 1)
        by_faces[faces].append(f"{bench_names[prefix]}{abbrev[diff]}")
    parts = [
        "/".join(labels) + f" ({faces})"
        for faces, labels in sorted(by_faces.items())
    ]
    print("x-axis ordering: " + ", ".join(parts))


def apply_tick_label_adjustments(fig, axes, ops):
    for ax in axes:
        for tick in ax.xaxis.get_major_ticks():
            if abs(tick.get_loc() - 12) < 0.1:
                x_shift = -6 / 72 if ops == "all" else 4 / 72
                offset = mtransforms.ScaledTranslation(x_shift, 0, fig.dpi_scale_trans)
                tick.label1.set_transform(tick.label1.get_transform() + offset)
            if ops == "all" and abs(tick.get_loc() - 13) < 0.1:
                offset = mtransforms.ScaledTranslation(8 / 72, 0, fig.dpi_scale_trans)
                tick.label1.set_transform(tick.label1.get_transform() + offset)
            if ops == "total" and (
                abs(tick.get_loc() - 13) < 0.1 or abs(tick.get_loc() - 79) < 0.5
            ):
                tick.label1.set_text("")
                tick.label1.set_visible(False)


def no_marker(handles):
    handles_copy = [copy.copy(handle) for handle in handles]
    for handle in handles_copy:
        handle.set_marker("none")
    return handles_copy


def add_legends(ax1, ax2):
    mesh_handles, mesh_labels = ax1.get_legend_handles_labels()
    image_handles, image_labels = ax2.get_legend_handles_labels()

    mesh_order = [
        mesh_labels.index(label)
        for label in ["CADFit", "CADEvolve", "Cadrille", "CAD-Recode"]
    ]
    image_order = [
        image_labels.index(label)
        for label in [
            "Gemini 3.1 Pro",
            "Claude Opus 4.7",
            "GPT-5.4",
            "Kimi K2.6",
            "CAD-Coder",
        ]
    ]

    ax1.legend(
        no_marker([mesh_handles[index] for index in mesh_order]),
        [mesh_labels[index] for index in mesh_order],
        loc="lower left",
    )

    dummy = Line2D([], [], color="none")
    ordered_image_handles = no_marker([image_handles[index] for index in image_order])
    ordered_image_labels = [image_labels[index] for index in image_order]
    ax2.legend(
        ordered_image_handles[:2] + [dummy] + ordered_image_handles[2:],
        ordered_image_labels[:2] + [""] + ordered_image_labels[2:],
        ncol=2,
    )


def build_plot(metric_name, ylabel, output_name, ops, benchmark_complexities, label_offsets):
    mesh_series = load_series(metric_name, MESH_SERIES)
    image_series = load_series(metric_name, IMAGE_SERIES)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 3.2))
    tick_vals, tick_labels = make_tick_labels(benchmark_complexities)

    for data, label, style, _ in mesh_series:
        plot_model(ax1, data, benchmark_complexities, label, **style)
    configure_axis(ax1, tick_vals, tick_labels, ylabel, "Mesh-to-CAD Models")

    if ops != "total":
        annotate_splits(
            ax1,
            [data for data, _, _, annotate in mesh_series if annotate],
            benchmark_complexities,
            find="min",
            xoffsets=label_offsets,
            yoffsets={},
        )

    for data, label, style, _ in image_series:
        plot_model(ax2, data, benchmark_complexities, label, **style)
    configure_axis(ax2, tick_vals, tick_labels, ylabel, "Image-to-CAD Models")

    if ops != "total":
        annotate_splits(
            ax2,
            [data for data, _, _, annotate in image_series if annotate],
            benchmark_complexities,
            find="max",
            xoffsets=label_offsets,
        )

    apply_tick_label_adjustments(fig, (ax1, ax2), ops)
    add_legends(ax1, ax2)

    plt.tight_layout()
    output_path = PLOTS_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    args = parse_args()
    benchmark_complexities = get_benchmark_complexities(args.ops)
    label_offsets = get_label_offsets(args.ops)

    plt.rcParams.update({"font.size": 12})

    if args.ops == "total":
        print_total_axis_order(benchmark_complexities)

    for metric_name, ylabel, output_name in METRIC_PLOTS:
        build_plot(
            metric_name,
            ylabel,
            output_name,
            args.ops,
            benchmark_complexities,
            label_offsets,
        )


if __name__ == "__main__":
    main()
