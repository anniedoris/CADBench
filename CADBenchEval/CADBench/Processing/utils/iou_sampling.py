import os
import numpy as np
import trimesh
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalize_mesh(mesh, box_min=-1, box_max=1):
    bb_min = mesh.bounds[0]
    bb_max = mesh.bounds[1]
    center = 0.5 * (bb_min + bb_max)
    max_dim = float((bb_max - bb_min).max())
    if max_dim > 0:
        scale = (box_max - box_min) / max_dim
        mesh.apply_translation(-center)
        mesh.apply_scale(scale)
    return mesh
# ---------------------------
# Utility: load mesh -> triangles tensor on GPU
# ---------------------------
def load_mesh_as_triangles(mesh, device: str = "cuda"):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    # Ensure triangles
    verts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(mesh.faces),    dtype=torch.long,   device=device)
    tris  = verts[faces]  # (F, 3, 3)
    return tris, mesh.bounds  # bounds: (min, max) in numpy

# ---------------------------
# GPU Möller–Trumbore ray/triangle intersections (batched)
# Counts intersections per point along +X direction
# ---------------------------
@torch.inference_mode()
def count_ray_intersections(points: torch.Tensor,
                            tris: torch.Tensor,
                            direction: torch.Tensor,
                            tri_batch: int = 65536,
                            verbose: bool = False) -> torch.Tensor:
    """
    points:    (P, 3) float32 cuda
    tris:      (T, 3, 3) float32 cuda
    direction: (3,) unit direction, e.g. +X on the same device/dtype as tris
    tri_batch: number of triangles per chunk to balance mem/speed

    returns: intersections per point (P,) int32
    """
    assert points.is_cuda and tris.is_cuda and direction.is_cuda
    P = points.shape[0]
    T = tris.shape[0]

    counts = torch.zeros(P, dtype=torch.int32, device=points.device)

    # normalize direction (just in case)
    direction = direction / (direction.norm() + 1e-12)

    tri_iter = tqdm(range(0, T, tri_batch), desc="Triangles (chunks)") if verbose else range(0, T, tri_batch)
    for t0 in tri_iter:
        t1 = min(t0 + tri_batch, T)
        tri_chunk = tris[t0:t1]                 # (Tb,3,3)
        v0 = tri_chunk[:, 0, :]                 # (Tb,3)
        v1 = tri_chunk[:, 1, :]
        v2 = tri_chunk[:, 2, :]

        e1 = v1 - v0                            # (Tb,3)
        e2 = v2 - v0                            # (Tb,3)

        # Cross needs matching shapes: expand direction to (Tb,3)
        d_exp = direction.view(1, 3).expand(e2.shape[0], 3)  # (Tb,3)
        pvec = torch.cross(d_exp, e2, dim=-1)                # (Tb,3)
        det  = (e1 * pvec).sum(dim=-1)                       # (Tb,)

        # Filter degenerate/parallel triangles
        det_mask = det.abs() > 1e-12
        if not det_mask.any():
            continue

        inv_det = torch.zeros_like(det)
        inv_det[det_mask] = 1.0 / det[det_mask]

        # Optional: drop truly degenerate triangles (zero area)
        area2 = torch.linalg.norm(torch.cross(e1, e2, dim=-1), dim=-1)  # ~2*area
        good_tris = det_mask & (area2 > 1e-20)
        if not good_tris.any():
            continue

        v0 = v0[good_tris]
        e1 = e1[good_tris]
        e2 = e2[good_tris]
        pvec = pvec[good_tris]
        inv_det = inv_det[good_tris]
        Tb = v0.shape[0]

        # Heuristic point batch size: tune for your GPU
        pt_batch = max(8192, 1048576 // max(1, Tb // 8))

        pt_iter = tqdm(range(0, P, pt_batch), leave=False, desc="  Points (batches)") if verbose else range(0, P, pt_batch)
        for p0 in pt_iter:
            p1 = min(p0 + pt_batch, P)
            o   = points[p0:p1].unsqueeze(1)    # (Pb,1,3)
            v0b = v0.unsqueeze(0)               # (1,Tb,3)
            e1b = e1.unsqueeze(0)               # (1,Tb,3)
            e2b = e2.unsqueeze(0)               # (1,Tb,3)
            pvb = pvec.unsqueeze(0)             # (1,Tb,3)
            inv = inv_det.unsqueeze(0)          # (1,Tb)

            tvec = o - v0b                      # (Pb,Tb,3)
            # u = dot(tvec, pvec) * inv_det
            u = (tvec * pvb).sum(dim=-1) * inv  # (Pb,Tb)

            qvec = torch.cross(tvec, e1b, dim=-1)        # (Pb,Tb,3)
            # v = dot(direction, qvec) * inv_det
            v = (qvec * direction.view(1, 1, 3)).sum(dim=-1) * inv  # (Pb,Tb)

            # t = dot(e2, qvec) * inv_det
            t = (e2b * qvec).sum(dim=-1) * inv  # (Pb,Tb)

            cond = (
                (u >= 0.0) & (u <= 1.0) &
                (v >= 0.0) & (u + v <= 1.0) &
                (t > 1e-8)  # in front of origin
            )

            counts[p0:p1] += cond.sum(dim=1).to(torch.int32)

    return counts


def plot_containment_points(points: torch.Tensor, 
                          inside_a: torch.Tensor, 
                          inside_b: torch.Tensor,
                          mesh_path_a: str,
                          mesh_path_b: str, 
                          iou: float,
                          intersection: torch.Tensor,
                          union: torch.Tensor,
                          b_only: torch.Tensor,
                          output_dir: str = None,
                          verbose: bool = False):
    """Plot the sampled points colored by containment in each mesh."""
    
    if verbose:
        print("[INFO] Creating containment plots...")
    
    # Convert to CPU numpy arrays for plotting
    points_cpu = points.cpu().numpy()
    inside_a_cpu = inside_a.cpu().numpy()
    inside_b_cpu = inside_b.cpu().numpy()
    
    # Subsample points for visualization (plotting millions of points is slow)
    max_plot_points = 50000
    if len(points_cpu) > max_plot_points:
        indices = np.random.choice(len(points_cpu), max_plot_points, replace=False)
        points_cpu = points_cpu[indices]
        inside_a_cpu = inside_a_cpu[indices]
        inside_b_cpu = inside_b_cpu[indices]
        if verbose:
            print(f"[INFO] Subsampled to {max_plot_points} points for visualization")
    
    # Define point categories
    intersection_mask = inside_a_cpu & inside_b_cpu
    a_only_mask = inside_a_cpu & ~inside_b_cpu
    b_only_mask = inside_b_cpu & ~inside_a_cpu
    outside_mask = ~inside_a_cpu & ~inside_b_cpu
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Base names for the meshes
    name_a = os.path.splitext(os.path.basename(mesh_path_a))[0]
    name_b = os.path.splitext(os.path.basename(mesh_path_b))[0]
    
    # 1. Plot points inside mesh A
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    inside_points_a = points_cpu[inside_a_cpu]
    outside_points_a = points_cpu[~inside_a_cpu]
    
    if len(inside_points_a) > 0:
        ax.scatter(inside_points_a[:, 0], inside_points_a[:, 1], inside_points_a[:, 2], 
                  c='red', s=1, alpha=0.6, label=f'Inside {name_a}')
    if len(outside_points_a) > 0:
        ax.scatter(outside_points_a[:, 0], outside_points_a[:, 1], outside_points_a[:, 2], 
                  c='lightgray', s=0.5, alpha=0.3, label=f'Outside {name_a}')
    
    ax.set_title(f'Points Inside Mesh A ({name_a})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'points_mesh_a_{name_a}.png'), dpi=150, bbox_inches='tight')
        if verbose:
            print(f"[INFO] Saved mesh A plot to {output_dir}/points_mesh_a_{name_a}.png")
    else:
        plt.show()
    plt.close()
    
    # 2. Plot points inside mesh B
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    inside_points_b = points_cpu[inside_b_cpu]
    outside_points_b = points_cpu[~inside_b_cpu]
    
    if len(inside_points_b) > 0:
        ax.scatter(inside_points_b[:, 0], inside_points_b[:, 1], inside_points_b[:, 2], 
                  c='blue', s=1, alpha=0.6, label=f'Inside {name_b}')
    if len(outside_points_b) > 0:
        ax.scatter(outside_points_b[:, 0], outside_points_b[:, 1], outside_points_b[:, 2], 
                  c='lightgray', s=0.5, alpha=0.3, label=f'Outside {name_b}')
    
    ax.set_title(f'Points Inside Mesh B ({name_b})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'points_mesh_b_{name_b}.png'), dpi=150, bbox_inches='tight')
        if verbose:
            print(f"[INFO] Saved mesh B plot to {output_dir}/points_mesh_b_{name_b}.png")
    else:
        plt.show()
    plt.close()
    
    # 3. Overlapped plot showing all categories
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot different categories with different colors
    if np.any(intersection_mask):
        intersection_points = points_cpu[intersection_mask]
        ax.scatter(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], 
                  c='purple', s=2, alpha=0.8, label=f'Intersection ({np.sum(intersection_mask)} pts)')
    
    if np.any(a_only_mask):
        a_only_points = points_cpu[a_only_mask]
        ax.scatter(a_only_points[:, 0], a_only_points[:, 1], a_only_points[:, 2], 
                  c='red', s=1, alpha=0.6, label=f'A only ({np.sum(a_only_mask)} pts)')
    
    if np.any(b_only_mask):
        b_only_points = points_cpu[b_only_mask]
        ax.scatter(b_only_points[:, 0], b_only_points[:, 1], b_only_points[:, 2], 
                  c='blue', s=1, alpha=0.6, label=f'B only ({np.sum(b_only_mask)} pts)')
    
    if np.any(outside_mask):
        outside_points = points_cpu[outside_mask]
        # Sample outside points for cleaner visualization
        if len(outside_points) > 10000:
            sample_indices = np.random.choice(len(outside_points), 10000, replace=False)
            outside_points = outside_points[sample_indices]
        ax.scatter(outside_points[:, 0], outside_points[:, 1], outside_points[:, 2], 
                  c='lightgray', s=0.3, alpha=0.2, label=f'Outside both ({np.sum(outside_mask)} pts)')
    
    ax.set_title(f'IoU Analysis: {name_a} vs {name_b}\nIoU = {iou:.4f} | Intersection: {int(intersection.item())} | Union: {int(union.item())} | B-only: {int(b_only.item())}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'iou_analysis_{name_a}_vs_{name_b}.png'), 
                   dpi=150, bbox_inches='tight')
        if verbose:
            print(f"[INFO] Saved IoU analysis plot to {output_dir}/iou_analysis_{name_a}_vs_{name_b}.png")
    else:
        plt.show()
    plt.close()
    
    if verbose:
        print(f"[INFO] Plotting complete. Categories:")
        print(f"  - Intersection (purple): {np.sum(intersection_mask)} points")
        print(f"  - A only (red): {np.sum(a_only_mask)} points") 
        print(f"  - B only (blue): {np.sum(b_only_mask)} points")
        print(f"  - Outside both (gray): {np.sum(outside_mask)} points")

# ---------------------------
# IoU via Monte Carlo sampling + GPU ray casting
# ---------------------------
@torch.inference_mode()
def mesh_iou_gpu_ray(points,
                     inside_a,
                     mesh_path_a: str,
                     mesh_path_b: str,
                     num_samples: int = 1_000_000,
                     points_batch: int = 1310,
                     tri_batch: int = 3055,
                     device: str = "cuda",
                     seed: int = 42,
                     alpha: float = 1,
                     verbose: bool = False,
                     plot_points: bool = False,
                     plot_output_dir: str = None,
                     normalize: bool = False) -> float:
    assert torch.cuda.is_available(), "CUDA not available"
    torch.manual_seed(seed)

    points = torch.tensor(points, dtype=torch.float32, device=device)
    inside_a = torch.tensor(inside_a, dtype=torch.bool, device=device)

    mesh_b = trimesh.load(mesh_path_b, force='mesh', process=False)
    tris_b, bounds_b = load_mesh_as_triangles(mesh_b, device=device)

    # Ray direction (+X) and epsilon shift to avoid edge cases
    direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)

    if verbose:
        print("[INFO] Counting intersections for mesh B...")
    inter_b = torch.zeros(num_samples, dtype=torch.int32, device=device)
    pt_iter_b = tqdm(range(0, num_samples, points_batch), desc="B: points (chunks)") if verbose else range(0, num_samples, points_batch)
    for p0 in pt_iter_b:
        p1 = min(p0 + points_batch, num_samples)
        inter_b[p0:p1] = count_ray_intersections(points[p0:p1], tris_b, direction, tri_batch=tri_batch, verbose=verbose)

    if verbose:
        print("[INFO] Computing occupancy & IoU (GPU)...")
    inside_b = (inter_b % 2 == 1)  # inside reconstructed (mesh B)

    intersection = torch.logical_and(inside_a, inside_b).sum(dtype=torch.int64)  # in both
    union        = torch.logical_or(inside_a, inside_b).sum(dtype=torch.int64)   # in either
    
    # Points in reconstructed but NOT in ground truth
    b_only = torch.logical_and(inside_b, ~inside_a).sum(dtype=torch.int64)

    # if b_only > alpha * intersection.float(): #0.01 * intersection.float() * (iou>0.1) + 10 * intersection.float() * (iou<=0.1):
    #     if verbose:
    #         print(f"[RESULT] Poor reconstruction detected: B-only={int(b_only.item())} > intersection={int(intersection.item())}")
    #         print(f"[RESULT] IoU = 0.000000 (penalty for excessive extension)")
    #     return 0.0

    iou = (intersection.float() / (union.float() + 1e-9)).item()

    if verbose:
        print(f"[RESULT] IoU = {iou:.6f} | intersection={int(intersection.item())} union={int(union.item())} B-only={int(b_only.item())}")
    
    # Plotting functionality
    if plot_points:
        plot_containment_points(points, inside_a, inside_b, mesh_path_a, mesh_path_b, 
                              iou, intersection, union, b_only, plot_output_dir, verbose)
    
    return iou
# def mesh_iou_gpu_ray(mesh_path_a: str,
#                      mesh_path_b: str,
#                      num_samples: int = 1_000_000,
#                      points_batch: int = 1310,
#                      tri_batch: int = 3055,
#                      device: str = "cuda",
#                      seed: int = 42,
#                      alpha: float = 1,
#                      verbose: bool = False,
#                      plot_points: bool = False,
#                      plot_output_dir: str = None,
#                      normalize: bool = False) -> float:
#     assert torch.cuda.is_available(), "CUDA not available"
#     torch.manual_seed(seed)

#     mesh_a = trimesh.load(mesh_path_a, force='mesh', process=False)
#     if normalize:
#         mesh_a = normalize_mesh(mesh_a)
#     mesh_b = trimesh.load(mesh_path_b, force='mesh', process=False)

#     if verbose:
#         print(f"[INFO] Loading meshes:\n  A = {mesh_path_a}\n  B = {mesh_path_b}")
#     tris_a, bounds_a = load_mesh_as_triangles(mesh_a, device=device)
#     tris_b, bounds_b = load_mesh_as_triangles(mesh_b, device=device)

#     # Union AABB
#     a_min, a_max = np.array(bounds_a[0]), np.array(bounds_a[1])
#     b_min, b_max = np.array(bounds_b[0]), np.array(bounds_b[1])
#     bb_min = torch.tensor(np.minimum(a_min, b_min), dtype=torch.float32, device=device)
#     bb_max = torch.tensor(np.maximum(a_max, b_max), dtype=torch.float32, device=device)

#     if verbose:
#         print(f"[INFO] Union AABB min={bb_min.tolist()}, max={bb_max.tolist()}")

#     # Sample points uniformly in AABB (GPU)
#     if verbose:
#         print("[INFO] Sampling points (GPU)...")
#     points = torch.rand((num_samples, 3), device=device) * (bb_max - bb_min) + bb_min

#     # Ray direction (+X) and epsilon shift to avoid edge cases
#     direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
#     points = points - 1e-7 * direction  # epsilon shift

#     if verbose:
#         print("[INFO] Counting intersections for mesh A...")
#     inter_a = torch.zeros(num_samples, dtype=torch.int32, device=device)
#     pt_iter_a = tqdm(range(0, num_samples, points_batch), desc="A: points (chunks)") if verbose else range(0, num_samples, points_batch)
#     for p0 in pt_iter_a:
#         p1 = min(p0 + points_batch, num_samples)
#         inter_a[p0:p1] = count_ray_intersections(points[p0:p1], tris_a, direction, tri_batch=tri_batch, verbose=verbose)

#     if verbose:
#         print("[INFO] Counting intersections for mesh B...")
#     inter_b = torch.zeros(num_samples, dtype=torch.int32, device=device)
#     pt_iter_b = tqdm(range(0, num_samples, points_batch), desc="B: points (chunks)") if verbose else range(0, num_samples, points_batch)
#     for p0 in pt_iter_b:
#         p1 = min(p0 + points_batch, num_samples)
#         inter_b[p0:p1] = count_ray_intersections(points[p0:p1], tris_b, direction, tri_batch=tri_batch, verbose=verbose)

#     if verbose:
#         print("[INFO] Computing occupancy & IoU (GPU)...")
#     inside_a = (inter_a % 2 == 1)  # inside ground truth (mesh A)
#     inside_b = (inter_b % 2 == 1)  # inside reconstructed (mesh B)

#     intersection = torch.logical_and(inside_a, inside_b).sum(dtype=torch.int64)  # in both
#     union        = torch.logical_or(inside_a, inside_b).sum(dtype=torch.int64)   # in either
    
#     # Points in reconstructed but NOT in ground truth
#     b_only = torch.logical_and(inside_b, ~inside_a).sum(dtype=torch.int64)

#     iou = (intersection.float() / (union.float() + 1e-9)).item()

#     # Check if reconstructed mesh extends too much beyond ground truth
#     # If more points are in B-only than in intersection, return score of 0
#     # if b_only > alpha * intersection.float(): #0.01 * intersection.float() * (iou>0.1) + 10 * intersection.float() * (iou<=0.1):
#     #     if verbose:
#     #         print(f"[RESULT] Poor reconstruction detected: B-only={int(b_only.item())} > intersection={int(intersection.item())}")
#     #         print(f"[RESULT] IoU = 0.000000 (penalty for excessive extension)")
#     #     return 0.0

    
#     if verbose:
#         print(f"[RESULT] IoU = {iou:.6f} | intersection={int(intersection.item())} union={int(union.item())} B-only={int(b_only.item())}")
    
#     # Plotting functionality
#     if plot_points:
#         plot_containment_points(points, inside_a, inside_b, mesh_path_a, mesh_path_b, 
#                               iou, intersection, union, b_only, plot_output_dir, verbose)
    
#     return iou

import torch
import trimesh
import numpy as np
from scipy.spatial import cKDTree

def cd(gt_path, gen_path, num_samples=2048, seed=42, delta=0.02):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load meshes
    mesh_a = trimesh.load(gt_path, force='mesh', process=False)
    mesh_a = normalize_mesh(mesh_a)

    mesh_b = trimesh.load(gen_path, force='mesh', process=False)

    # Sample points on surfaces
    points_a, _ = trimesh.sample.sample_surface(mesh_a, num_samples)
    points_b, _ = trimesh.sample.sample_surface(mesh_b, num_samples)

    # KD-tree for fast nearest-neighbor search
    tree_b = cKDTree(points_b)

    # Distance from each point in A to closest point in B
    dists, _ = tree_b.query(points_a, k=1)

    # Count points within delta
    count_within_delta = np.sum(dists <= delta)
    fraction_within_delta = count_within_delta / num_samples

    return round(fraction_within_delta,4)




# ---------------------------
# CLI example
# ---------------------------
if __name__ == "__main__":
    # Set your files:
    A = "test_20.stl"  # replace with your path
    B = "test_20_rec.stl"           # replace with your path

    # Tune num_samples for accuracy/time; points_batch & tri_batch for memory
    mesh_iou_gpu_ray(
        A, B,
        num_samples=100_000,     # reduced for faster plotting
        points_batch=131072,     # point chunk size (GPU memory)
        tri_batch=65536,         # triangle chunk size (GPU memory)
        device="cuda:1",
        seed=0,
        verbose=True,
        plot_points=True,        # Enable plotting
        plot_output_dir="iou_plots"  # Save plots to this directory
    )