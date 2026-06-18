"""
Collect per-prediction metrics from all *_logs.json files under tested_models/.

Output columns:
  sample_id, model, split, modality, iou, cd, siou
"""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTED_MODELS_DIR = REPO_ROOT / "tested_models"
OUTPUT_DIR = REPO_ROOT / "graph" / "plots"

# Order matters: check noisy_mesh before mesh to avoid substring match
KNOWN_MODALITIES = ["noisy_mesh", "mesh", "multiview", "singleview", "pbr"]

SPLIT_RE = re.compile(r"(bench\d+[A-Z]?)")


def extract_metadata(log_path: Path):
    rel_parts = log_path.relative_to(TESTED_MODELS_DIR).parts
    model = rel_parts[0]

    modality = None
    for part in rel_parts[1:]:
        for m in KNOWN_MODALITIES:
            if part == m:
                modality = m
                break
        if modality:
            break

    # Fall back to searching the full path string (e.g. cadrille_ourimages)
    if modality is None:
        path_str = "/".join(rel_parts)
        for m in KNOWN_MODALITIES:
            if re.search(rf"(?<![a-z]){re.escape(m)}(?![a-z])", path_str):
                modality = m
                break

    match = SPLIT_RE.search(log_path.stem)
    split = match.group(1) if match else None

    return model, modality, split


def collect(tested_models_dir: Path = TESTED_MODELS_DIR) -> pd.DataFrame:
    rows = []
    log_files = sorted(tested_models_dir.rglob("*_logs.json"))

    for log_path in log_files:
        model, modality, split = extract_metadata(log_path)

        with log_path.open() as f:
            records = json.load(f)

        for rec in records:
            if rec.get("status") != 1:
                continue
            rows.append(
                {
                    "sample_id": rec.get("file_id"),
                    "model": model,
                    "split": split,
                    "modality": modality,
                    "iou": rec.get("Aligned IoU"),
                    "cd": rec.get("Aligned Chamfer Distance"),
                    "siou": rec.get("Aligned Surface IoU"),
                }
            )

    return pd.DataFrame(rows, columns=["sample_id", "model", "split", "modality", "iou", "cd", "siou"])


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR / "stats_df.csv"

    df = collect()
    print(f"Collected {len(df):,} rows from {TESTED_MODELS_DIR}")
    print(f"Models:     {sorted(df['model'].unique())}")
    print(f"Modalities: {sorted(df['modality'].dropna().unique())}")
    print(f"Splits:     {sorted(df['split'].dropna().unique())}")
    print(df.head())

    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    print(f"\nRows used for correlation: {len(df):,}")
    corr = df[["iou", "cd", "siou"]].corr(method="spearman")
    print("\nSample-level Spearman correlations:")
    print(corr)

    corr.index = corr.columns = ["IoU", "CD", "SIoU"]

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="white")
    masked_corr = np.ma.array(corr.to_numpy(), mask=mask)
    image = ax.imshow(masked_corr, vmin=-1, vmax=1, cmap=cmap)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    for row in range(corr.shape[0]):
        for col in range(corr.shape[1]):
            if not mask[row, col]:
                ax.text(
                    col,
                    row,
                    f"{corr.iloc[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                )

    ax.set_aspect("equal")
    fig.colorbar(image, ax=ax)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center")
    ax.tick_params(labelsize=16)
    ax.figure.axes[-1].tick_params(labelsize=14)  # colorbar ticks
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "metric_correlation_heatmap.png"
    plt.savefig(plot_path)
    print(f"Saved heatmap to {plot_path}")

    axis_bounds = {
        "iou":  (0, 1),
        "siou": (0, 1),
        "cd":   (0, 1),
    }

    pairs = [
        ("iou",  "siou", "IoU",  "SIoU"),
        ("iou",  "cd",   "IoU",  "CD"),
        ("siou", "cd",   "SIoU", "CD"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, pairs):
        ax.scatter(df[xcol], df[ycol], s=1, alpha=0.05, linewidths=0)
        ax.set_xlim(*axis_bounds[xcol])
        ax.set_ylim(*axis_bounds[ycol])
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(labelsize=16)
        ax.set_box_aspect(1)

    plt.tight_layout()
    scatter_path = OUTPUT_DIR / "metric_pairwise_scatter.png"
    plt.savefig(scatter_path, dpi=150)
    print(f"Saved pairwise scatter to {scatter_path}")

    df["siou_minus_iou"] = df["siou"] - df["iou"]
    df_valid = df[df["iou"].between(0, 1)]
    cols = ["sample_id", "model", "split", "modality", "iou", "siou", "cd", "siou_minus_iou"]
    n = 10

    print(f"\n--- Top 50: High SIoU, Low IoU (siou - iou largest, iou > 0) ---")
    print(df_valid[df_valid["iou"] > 0].nlargest(50, "siou_minus_iou")[cols].to_string(index=False))

    print(f"\n--- Top {n}: High IoU, Low SIoU (siou - iou smallest) ---")
    print(df_valid.nsmallest(n, "siou_minus_iou")[cols].to_string(index=False))

    print(f"\nValid samples used: {len(df):,}")
