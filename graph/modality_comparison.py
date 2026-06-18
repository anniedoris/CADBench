import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "tested_models")
PLOTS_ROOT = Path(__file__).resolve().parent / "plots"
METRIC_KEY = "average_aligned_iou_adjusted_median_across_all_benches"
VSR_KEY = "average_vsr_across_all_benches"


def color_tick(ax, value, color="red"):
    for tick, tick_val in zip(ax.yaxis.get_major_ticks(), ax.get_yticks()):
        if abs(tick_val - value) < 1e-9:
            tick.label1.set_color(color)
            tick.label1.set_fontweight("bold")


def extract_metric(model, modality, key=METRIC_KEY):
    path = os.path.join(RESULTS_ROOT, model, modality, "results_summary.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if line.startswith(key):
                return float(line.split(":")[1].strip())
    return None


MODALITY_LABELS = {
    "mesh": "Clean Mesh",
    "noisy_mesh": "Noisy Mesh",
    "singleview": "Single-View",
    "pbr": "Photorealistic",
    "multiview": "Multi-View",
}

MESH_MODEL_STYLES = {
    "cadfit":     {"label": "CADFit",     "color": "#08306b"},
    "cadrecode":  {"label": "CAD-Recode", "color": "#9e8cc5"},
    "cadevolve":  {"label": "CADEvolve",  "color": "#74c6c6"},
    "cadrille_pc":{"label": "Cadrille",   "color": "#6baed6"},
}


X_POSITIONS_MESH = [0.15, 0.85]
X_POSITIONS_IMAGE = [0.1, 0.5, 0.9]


def plot_mesh_subplot(ax, models, modalities):
    for model in models:
        values = [extract_metric(model, m) for m in modalities]
        if all(v is None for v in values):
            continue
        style = MESH_MODEL_STYLES.get(model, {})
        ax.plot(X_POSITIONS_MESH[:len(modalities)], values, marker="^",
                label=style.get("label", model),
                color=style.get("color", None),
                linewidth=2.5, markersize=8)

    ax.set_title("Mesh-to-CAD Models", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("IoU", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_xticks(X_POSITIONS_MESH[:len(modalities)])
    ax.set_xticklabels([MODALITY_LABELS.get(m, m) for m in modalities], fontsize=13)
    ax.set_ylim(-0.2, 1.1)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.tick_params(axis="y", labelsize=13)
    import copy
    handles, labels = ax.get_legend_handles_labels()
    handles = [copy.copy(h) for h in handles]
    for h in handles:
        h.set_marker("none")
    ax.legend(handles, labels, fontsize=13, labelspacing=0.1, loc="lower left")


IMAGE_MODEL_STYLES = {
    "claude4.7":  {"label": "Claude Opus 4.7", "color": "#005824"},
    "gemini3.1":  {"label": "Gemini 3.1 Pro",  "color": "#2ca25f"},
    "gpt5.4":     {"label": "GPT-5.4",          "color": "#8c510a"},
    "cadcoder":   {"label": "CAD-Coder",         "color": "#e08010", "marker": "^"},
    "kimi_2.6":   {"label": "Kimi K2.6",         "color": "#c8d400"},
    "qwen3.527b": {"label": "Qwen3.5 27B",       "color": "#807dba"},
    "qwen3.59b":  {"label": "Qwen3.5 9B",        "color": "#bcbddc"},
}


def plot_image_subplot(ax, models, modalities):
    x_pos = X_POSITIONS_IMAGE[:len(modalities)]
    for model in models:
        values = [extract_metric(model, m) for m in modalities]
        if all(v is None for v in values):
            continue
        style = IMAGE_MODEL_STYLES.get(model, {})
        xs = [x_pos[i] for i, v in enumerate(values) if v is not None]
        ys = [v for v in values if v is not None]
        ax.plot(xs, ys, marker=style.get("marker", "o"),
                label=style.get("label", model),
                color=style.get("color", None),
                linewidth=2.5, markersize=8)

    ax.set_title("Image-to-CAD Models", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("IoU", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([MODALITY_LABELS.get(m, m) for m in modalities], fontsize=13)
    ax.set_ylim(-0.15, 1.05)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.tick_params(axis="y", labelsize=13)
    import copy
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles = [copy.copy(h) for h in handles]
    for h in handles:
        h.set_marker("none")
    dummy = Line2D([], [], color="none")
    ax.legend(handles[:2] + [dummy] + handles[2:], labels[:2] + [""] + labels[2:], ncol=2, fontsize=13)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_models", nargs="+", default=["gemini3.1", "claude4.7", "gpt5.4", "kimi_2.6", "cadcoder"])
    parser.add_argument("--table_only_image_models", nargs="+", default=["qwen3.527b", "qwen3.59b"])
    parser.add_argument("--mesh_models", nargs="+", default=["cadfit", "cadevolve", "cadrille_pc", "cadrecode"])
    parser.add_argument("--modality_list_image_model", nargs="+", default=["singleview", "pbr", "multiview"])
    parser.add_argument("--modality_list_mesh_model", nargs="+", default=["mesh", "noisy_mesh"])
    args = parser.parse_args()

    # Print values
    groups = [
        ("image_models", args.image_models + args.table_only_image_models, args.modality_list_image_model),
        ("mesh_models", args.mesh_models, args.modality_list_mesh_model),
    ]
    for group_name, models, modalities in groups:
        if not models:
            continue
        print(f"[{group_name}]")
        for model in models:
            print(f"  {model}:")
            for modality in modalities:
                value = extract_metric(model, modality)
                if value is None:
                    print(f"    {modality}: NOT FOUND")
                else:
                    print(f"    {modality}: {value:.4f}")
        print()

    _, (ax_mesh, ax_image) = plt.subplots(1, 2, figsize=(12, 3.2), gridspec_kw={"width_ratios": [2, 3]})

    plot_mesh_subplot(ax_mesh, args.mesh_models, args.modality_list_mesh_model)

    plot_image_subplot(ax_image, args.image_models, args.modality_list_image_model)

    plt.tight_layout()
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_ROOT / "modality_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    latex_path = PLOTS_ROOT / "modalities_latex.txt"
    with open(latex_path, "w") as f:
        for model in args.mesh_models:
            style = MESH_MODEL_STYLES.get(model, {})
            label = style.get("label", model)
            clean_iou = extract_metric(model, "mesh")
            noisy_iou = extract_metric(model, "noisy_mesh")
            clean_vsr = extract_metric(model, "mesh", key=VSR_KEY)
            noisy_vsr = extract_metric(model, "noisy_mesh", key=VSR_KEY)

            def fmt(clean, noisy):
                if clean is not None and noisy is not None:
                    d = noisy - clean
                    s = f"+{d:.3f}" if d >= 0 else f"{d:.3f}"
                    return f"{clean:.3f} & {noisy:.3f} & \\deltacell{{{s}}}"
                return "-- & -- & --"

            f.write(f"{label} & {fmt(clean_iou, noisy_iou)} & {fmt(clean_vsr, noisy_vsr)} \\\\\n")

        f.write("\n")

        all_image_models = args.image_models + args.table_only_image_models
        for model in all_image_models:
            style = IMAGE_MODEL_STYLES.get(model, {})
            label = style.get("label", model)
            sv = extract_metric(model, "singleview")
            pbr = extract_metric(model, "pbr")
            mv = extract_metric(model, "multiview")

            def val(v):
                return f"{v:.3f}" if v is not None else "--"

            def delta(base, other):
                if base is not None and other is not None:
                    d = other - base
                    s = f"+{d:.3f}" if d >= 0 else f"{d:.3f}"
                    return f"\\deltacell{{{s}}}"
                return "--"

            f.write(f"{label} & {val(sv)} & {val(pbr)} & {delta(sv, pbr)} & {val(mv)} & {delta(sv, mv)} \\\\\n")

        f.write("\n")

        for model in all_image_models:
            style = IMAGE_MODEL_STYLES.get(model, {})
            label = style.get("label", model)
            sv = extract_metric(model, "singleview", key=VSR_KEY)
            pbr = extract_metric(model, "pbr", key=VSR_KEY)
            mv = extract_metric(model, "multiview", key=VSR_KEY)
            f.write(f"{label} & {val(sv)} & {val(pbr)} & {delta(sv, pbr)} & {val(mv)} & {delta(sv, mv)} \\\\\n")

    print(f"Saved {latex_path}")


if __name__ == "__main__":
    main()
