from math import nan
import pymeshlab
from .execution import get_mesh_safe
from .execution._utils import suppress_output_os, suppress_output
from .geometry import align_shapes, normalize_shape_bounding_box, iou, chamfer_distance, surface_iou, alpha_wrap
from .code_metrics import compute_code_metrics
import trimesh
from multiprocessing import Queue, Process
from typing import Dict, Any, List
from datasets import Dataset
from tqdm.auto import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError as FuturesTimeoutError, as_completed, ProcessPoolExecutor
import trimesh
from io import BytesIO
import numpy as np
from tabulate import tabulate

def _file_id_at(args):
    dataset, idx = args
    return idx, dataset[idx]["file_id"]

def print_results_tabulate(results):
    # Separate nested dictionaries (stats) from flat key-values (overall metrics)
    stat_names = [k for k, v in results.items() if isinstance(v, dict)]
    flat_stats = {k: v for k, v in results.items() if not isinstance(v, dict)}
    
    # Get all unique metric names (e.g. Aligned IoU, etc.) preserving order
    metric_names = list(dict.fromkeys(
        metric for stat in stat_names for metric in results[stat]
    ))
    
    # Build main table data
    table_data = []
    for metric in metric_names:
        row = [metric]
        for stat in stat_names:
            val = results[stat].get(metric, "-")
            row.append(f"{val:.4f}" if isinstance(val, (float, int)) else val)
        table_data.append(row)
        
    print("Geometry Metrics:")
    print(tabulate(table_data, headers=["Metric"] + stat_names, tablefmt="grid"))
    
    # Print flat metrics
    if flat_stats:
        print("\nValidity Metrics:")
        print(tabulate(flat_stats.items(), headers=["Metric", "Value (%)"], tablefmt="grid", floatfmt=".4f"))

def perform_evaluation(code: str, ground_truth: trimesh.Trimesh, variable_name: str = "solid", generated_mesh: trimesh.Trimesh = None) -> Dict[str, Any]:
    with suppress_output_os(), suppress_output():
        # Compute code metrics
        code_metrics = compute_code_metrics(code) if code else {
            "token_count": 0,
            "line_count": 0,
            "total_operations": 0,
            "operations_by_category": {}
        }
        
        if generated_mesh is None:
            try:
                generated_mesh = get_mesh_safe(code, variable_name)
            except Exception as e:
                return {
                    "Aligned IoU": 0.0,
                "Aligned Chamfer Distance": nan,
                "Aligned Surface IoU": 0.0,
                "Naive IoU": 0.0,
                "Naive Chamfer Distance": nan,
                "Naive Surface IoU": 0.0,
                "token_count": code_metrics["token_count"],
                "line_count": code_metrics["line_count"],
                "total_operations": code_metrics["total_operations"],
                "status": 0,
                "details": f"Code Execution Error: {e}",
            }

        if generated_mesh is None:
            return {
                "Aligned IoU": 0.0,
                "Aligned Chamfer Distance": nan,
                "Aligned Surface IoU": 0.0,
                "Naive IoU": 0.0,
                "Naive Chamfer Distance": nan,
                "Naive Surface IoU": 0.0,
                "token_count": code_metrics["token_count"],
                "line_count": code_metrics["line_count"],
                "total_operations": code_metrics["total_operations"],
                "status": 0,
                "details": "Code Execution Error: get_mesh_safe returned None",
            }
        else:
            #bbox normalize first
            try:
                # Alpha-wrap both inputs so all volumetric work downstream
                # (IoU, mass-property alignment) operates on guaranteed
                # watertight, non-self-intersecting outer shells.
                ground_truth   = alpha_wrap(ground_truth)
                generated_mesh = alpha_wrap(generated_mesh)

                ground_truth = normalize_shape_bounding_box(ground_truth)
                iou_score = iou(generated_mesh, ground_truth)
                cd_score = chamfer_distance(generated_mesh, ground_truth)
                generated_mesh = normalize_shape_bounding_box(generated_mesh)
                bbox_iou = max(iou(generated_mesh, ground_truth), iou_score)
                bbox_chamfer = min(chamfer_distance(generated_mesh, ground_truth), cd_score)
                bbox_surface_iou = surface_iou(generated_mesh, ground_truth)
                
                try:
                    aligned_mesh, aligned_iou = align_shapes(generated_mesh, ground_truth)
                    aligned_chamfer = chamfer_distance(aligned_mesh, ground_truth)
                    aligned_surface_iou = surface_iou(aligned_mesh, ground_truth)
                except Exception:
                    aligned_iou = bbox_iou
                    aligned_chamfer = bbox_chamfer
                    aligned_surface_iou = bbox_surface_iou
                return {
                    "Aligned IoU": float(aligned_iou),
                    "Aligned Chamfer Distance": float(aligned_chamfer),
                    "Aligned Surface IoU": float(aligned_surface_iou),
                    "Naive IoU": float(bbox_iou),
                    "Naive Chamfer Distance": float(bbox_chamfer),
                    "Naive Surface IoU": float(bbox_surface_iou),
                    "token_count": code_metrics["token_count"],
                    "line_count": code_metrics["line_count"],
                    "total_operations": code_metrics["total_operations"],
                    "status": 1,
                    "details": "Success"
                }
            
            except Exception as e:
                return {
                    "Aligned IoU": 0.0,
                    "Aligned Chamfer Distance": nan,
                    "Aligned Surface IoU": 0.0,
                    "Naive IoU": 0.0,
                    "Naive Chamfer Distance": nan,
                    "Naive Surface IoU": 0.0,
                    "token_count": code_metrics["token_count"],
                    "line_count": code_metrics["line_count"],
                    "total_operations": code_metrics["total_operations"],
                    "status": 0,
                    "details": str(e)
                }


class Evaluator:
    def __init__(self,
                 dataset: Dataset,
                 num_workers: int = 16,
                 verbose: bool = True,
                 variable_name: str = "solid",
                 timeout: int = 5):
        self.num_workers = num_workers
        self.verbose = verbose
        self.dataset = dataset
        self.variable_name = variable_name
        self.timeout = timeout
        id_dict = {}
        n = len(self.dataset)
        with ProcessPoolExecutor(max_workers=max(1, self.num_workers)) as executor:
            for idx, fid in tqdm(
                executor.map(_file_id_at, ((self.dataset, i) for i in range(n))),
                total=n,
                desc="Building ID Dictionary",
                disable=not self.verbose,
            ):
                id_dict[fid] = idx
        self.id_dict = id_dict
        
    def _process_item(self, output: Dict[str, Any]) -> Dict[str, Any]:
        file_id = output["file_id"]
        idx = self.id_dict[file_id]
        gt_mesh = trimesh.load(BytesIO(self.dataset[idx]["stl"]), file_type="stl")
        result = perform_evaluation(output["generated"], gt_mesh, self.variable_name)
        return {"file_id": file_id, **result}

    def run(self,
            outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        # check if all ids are present from id_dict in outputs
        ids_in_outputs = [output["file_id"] for output in outputs]
        if not all(id in ids_in_outputs for id in self.id_dict.keys()):
            print("WARNING: Some ids are not present in the outputs")
        
        results = []
        with ProcessPool(max_workers=self.num_workers) as pool:
            future_to_output = {
                pool.schedule(self._process_item, args=(output,), timeout=self.timeout): output
                for output in outputs
            }

            for future in tqdm(as_completed(future_to_output), total=len(future_to_output), desc="Processing", disable=not self.verbose):
                output = future_to_output[future]
                file_id = output["file_id"]
                code = output.get("generated", "")
                # Compute code metrics for error cases
                code_metrics = compute_code_metrics(code) if code else {
                    "token_count": 0,
                    "line_count": 0,
                    "total_operations": 0,
                    "operations_by_category": {}
                }
                
                try:
                    results.append(future.result())
                except FuturesTimeoutError:
                    results.append({
                        "file_id": file_id,
                        "Aligned IoU": 0.0,
                        "Aligned Chamfer Distance": nan,
                        "Aligned Surface IoU": 0.0,
                        "Naive IoU": 0.0,
                        "Naive Chamfer Distance": nan,
                        "Naive Surface IoU": 0.0,
                        "token_count": code_metrics["token_count"],
                        "line_count": code_metrics["line_count"],
                        "total_operations": code_metrics["total_operations"],
                        "status": 2,
                        "details": "Code Execution Timeout",
                    })
                except Exception as e:
                    results.append({
                        "file_id": file_id,
                        "Aligned IoU": 0.0,
                        "Aligned Chamfer Distance": nan,
                        "Aligned Surface IoU": 0.0,
                        "Naive IoU": 0.0,
                        "Naive Chamfer Distance": nan,
                        "Naive Surface IoU": 0.0,
                        "token_count": code_metrics["token_count"],
                        "line_count": code_metrics["line_count"],
                        "total_operations": code_metrics["total_operations"],
                        "status": 0,
                        "details": f"Worker error: {e}",
                    })
                
        Aligned_IoU_success_only = [result["Aligned IoU"] for result in results if result["status"] == 1]
        IoU_success_only = [result["Naive IoU"] for result in results if result["status"] == 1]
        Aligned_IoU_all = [result["Aligned IoU"] for result in results]
        IoU_all = [result["Naive IoU"] for result in results]
        
        Aligned_Chamfer_Distance_success_only = [result["Aligned Chamfer Distance"] for result in results if result["status"] == 1]
        Naive_Chamfer_Distance_success_only = [result["Naive Chamfer Distance"] for result in results if result["status"] == 1]
        
        Aligned_Surface_IoU_success_only = [result["Aligned Surface IoU"] for result in results if result["status"] == 1]
        Naive_Surface_IoU_success_only = [result["Naive Surface IoU"] for result in results if result["status"] == 1]
        Aligned_Surface_IoU_all = [result["Aligned Surface IoU"] for result in results]
        Naive_Surface_IoU_all = [result["Naive Surface IoU"] for result in results]
        
        # Code metrics
        token_counts_all = [result["token_count"] for result in results]
        line_counts_all = [result["line_count"] for result in results]
        operation_counts_all = [result["total_operations"] for result in results]
        
        
        n_valid = sum(1 for result in results if result["status"] == 1)
        n_total = len(results)
        
        n_timeout = sum(1 for result in results if result["status"] == 2)
        
        return {
            "Mean": {
                "Aligned IoU": float(np.mean(Aligned_IoU_success_only)),
                "Naive IoU": float(np.mean(IoU_success_only)),
                "Aligned Chamfer Distance": float(np.mean(Aligned_Chamfer_Distance_success_only)),
                "Naive Chamfer Distance": float(np.mean(Naive_Chamfer_Distance_success_only)),
                "Aligned Surface IoU": float(np.mean(Aligned_Surface_IoU_success_only)),
                "Naive Surface IoU": float(np.mean(Naive_Surface_IoU_success_only))
            },
            "Median": {
                "Aligned IoU": float(np.median(Aligned_IoU_success_only)),
                "Naive IoU": float(np.median(IoU_success_only)),
                "Aligned Chamfer Distance": float(np.median(Aligned_Chamfer_Distance_success_only)),
                "Naive Chamfer Distance": float(np.median(Naive_Chamfer_Distance_success_only)),
                "Aligned Surface IoU": float(np.median(Aligned_Surface_IoU_success_only)),
                "Naive Surface IoU": float(np.median(Naive_Surface_IoU_success_only))
            },
            "Std": {
                "Aligned IoU": float(np.std(Aligned_IoU_success_only)),
                "Naive IoU": float(np.std(IoU_success_only)),
                "Aligned Chamfer Distance": float(np.std(Aligned_Chamfer_Distance_success_only)),
                "Naive Chamfer Distance": float(np.std(Naive_Chamfer_Distance_success_only)),
                "Aligned Surface IoU": float(np.std(Aligned_Surface_IoU_success_only)),
                "Naive Surface IoU": float(np.std(Naive_Surface_IoU_success_only))
            },
            "Adjusted Mean": {
                "Aligned IoU": float(np.mean(Aligned_IoU_all)),
                "Naive IoU": float(np.mean(IoU_all)),
                "Aligned Surface IoU": float(np.mean(Aligned_Surface_IoU_all)),
                "Naive Surface IoU": float(np.mean(Naive_Surface_IoU_all))
            },
            "Adjusted Median": {
                "Aligned IoU": float(np.median(Aligned_IoU_all)),
                "Naive IoU": float(np.median(IoU_all)),
                "Aligned Surface IoU": float(np.median(Aligned_Surface_IoU_all)),
                "Naive Surface IoU": float(np.median(Naive_Surface_IoU_all))
            },
            "Adjusted Std": {
                "Aligned IoU": float(np.std(Aligned_IoU_all)),
                "Naive IoU": float(np.std(IoU_all)),
                "Aligned Surface IoU": float(np.std(Aligned_Surface_IoU_all)),
                "Naive Surface IoU": float(np.std(Naive_Surface_IoU_all))
            },
            "VSR": (n_valid / n_total) * 100,
            "Timeout Rate": (n_timeout / n_total) * 100,
            "Code Metrics Mean": {
                "Token Count": float(np.mean(token_counts_all)),
                "Line Count": float(np.mean(line_counts_all)),
                "Operation Count": float(np.mean(operation_counts_all))
            },
            "Code Metrics Median": {
                "Token Count": float(np.median(token_counts_all)),
                "Line Count": float(np.median(line_counts_all)),
                "Operation Count": float(np.median(operation_counts_all))
            }
        }, results