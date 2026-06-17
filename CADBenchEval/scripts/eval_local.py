#!/usr/bin/env python3
"""Evaluate generated STL meshes against ground truth STLs using CADBench metrics."""

import argparse
import os
import sys
import json
import trimesh
import numpy as np
from math import nan
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from CADBench.Eval._main import perform_evaluation, print_results_tabulate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated STL meshes against ground truth STLs using CADBench metrics.")
    parser.add_argument("--gt_dir", required=True, help="Directory of ground truth STLs, named <file_id>.stl")
    parser.add_argument("--generated_dir", required=True, help="Directory of generated STLs, named <file_id>.stl")
    parser.add_argument("--output_dir", default=None, help="Directory to write eval_results.json (default: --generated_dir)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel worker processes (default: 8)")
    return parser.parse_args()


def _evaluate_one(file_id, gt_path, gen_path):
    if not os.path.exists(gen_path):
        return {
            "file_id": file_id,
            "Aligned IoU": 0.0,
            "Aligned Chamfer Distance": nan,
            "Aligned Surface IoU": 0.0,
            "Naive IoU": 0.0,
            "Naive Chamfer Distance": nan,
            "Naive Surface IoU": 0.0,
            "status": 0,
            "details": "Generated mesh not found",
        }

    gt_mesh = trimesh.load(gt_path)
    gen_mesh = trimesh.load(gen_path)
    result = perform_evaluation(code="", ground_truth=gt_mesh, generated_mesh=gen_mesh)
    result["file_id"] = file_id
    return result


def main():
    args = parse_args()
    output_dir = args.output_dir or args.generated_dir
    os.makedirs(output_dir, exist_ok=True)

    gt_files = sorted(f for f in os.listdir(args.gt_dir) if f.endswith(".stl"))
    results = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                _evaluate_one,
                os.path.splitext(gt_file)[0],
                os.path.join(args.gt_dir, gt_file),
                os.path.join(args.generated_dir, gt_file),
            )
            for gt_file in gt_files
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            results.append(future.result())

    results.sort(key=lambda r: r["file_id"])

    # Aggregate
    successful = [r for r in results if r["status"] == 1]
    n_total = len(results)
    n_valid = len(successful)

    if n_valid == 0:
        print("\nNo successful evaluations.")
        return

    summary = {
        "Mean": {
            "Aligned IoU": float(np.mean([r["Aligned IoU"] for r in successful])),
            "Naive IoU": float(np.mean([r["Naive IoU"] for r in successful])),
            "Aligned Chamfer Distance": float(np.mean([r["Aligned Chamfer Distance"] for r in successful])),
            "Naive Chamfer Distance": float(np.mean([r["Naive Chamfer Distance"] for r in successful])),
            "Aligned Surface IoU": float(np.mean([r["Aligned Surface IoU"] for r in successful])),
            "Naive Surface IoU": float(np.mean([r["Naive Surface IoU"] for r in successful])),
        },
        "Median": {
            "Aligned IoU": float(np.median([r["Aligned IoU"] for r in successful])),
            "Naive IoU": float(np.median([r["Naive IoU"] for r in successful])),
            "Aligned Chamfer Distance": float(np.median([r["Aligned Chamfer Distance"] for r in successful])),
            "Naive Chamfer Distance": float(np.median([r["Naive Chamfer Distance"] for r in successful])),
            "Aligned Surface IoU": float(np.median([r["Aligned Surface IoU"] for r in successful])),
            "Naive Surface IoU": float(np.median([r["Naive Surface IoU"] for r in successful])),
        },
        "Valid Syntax Rate (%)": n_valid / n_total * 100,
    }

    print(f"\n{'='*60}")
    print(f"Results: {n_valid}/{n_total} successful")
    print(f"{'='*60}")
    print_results_tabulate(summary)

    # Save per-model results and summary
    output_path = os.path.join(output_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump({"per_model": results, "summary": summary}, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
