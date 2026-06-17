import trimesh
import torch
import sys, os
import numpy as np
import pymeshlab as pml
from typing import Tuple
from ...Processing.utils.pc_utils import process_mesh as mesh_to_points
from ...Processing.utils.iou_sampling import load_mesh_as_triangles, count_ray_intersections
from scipy.spatial import cKDTree


def alpha_wrap(mesh: trimesh.Trimesh,
               alpha_fraction: float = 1e-3,
               offset_fraction: float = 1e-4) -> trimesh.Trimesh:
    """CGAL `alpha_wrap_3` via pymeshlab — guaranteed watertight,
    non-self-intersecting outer wrap. Apply at input boundaries before any
    volumetric work (IoU, mass properties)."""
    ms = pml.MeshSet()
    ms.add_mesh(pml.Mesh(
        vertex_matrix=np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.ascontiguousarray(mesh.faces, dtype=np.int32),
    ))
    last_err = None
    for kwargs in (
        {"alpha_fraction": alpha_fraction, "offset_fraction": offset_fraction},
        {"alpha_relative": alpha_fraction, "offset_relative": offset_fraction},
        {},
    ):
        try:
            ms.apply_filter("generate_alpha_wrap", **kwargs)
            cm = ms.current_mesh()
            v = np.asarray(cm.vertex_matrix(), dtype=np.float64)
            f = np.asarray(cm.face_matrix(), dtype=np.int64)
            return trimesh.Trimesh(vertices=v, faces=f, process=False)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"alpha_wrap failed: {last_err}")


def compute_mass_properties(mesh : trimesh.Trimesh) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute mass properties such as volume (interpreted as mass for unit density)
    and the center of mass of the given shape."""
    mass = mesh.mass
    center_of_mass = mesh.center_mass
    matrix_of_inertia = mesh.moment_inertia
    P = trimesh.inertia.principal_axis(matrix_of_inertia)[1].T[:,::-1]
    
    return mass, center_of_mass, matrix_of_inertia, P

def normalize_shape_bounding_box(shape : trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalize the shape to fit in [-1, 1] along the largest dimension."""
    bbox = shape.bounds
    center = (bbox[0] + bbox[1]) / 2
    shape = shape.apply_translation(-center)
    scale = float(2 / (bbox[1] - bbox[0]).max())
    shape = shape.apply_scale(scale)
    return shape

def align_shapes(source : trimesh.Trimesh, target : trimesh.Trimesh) -> Tuple[trimesh.Trimesh, float]:
    """Align source to target using the center of mass and the principal axes of inertia. also return normalized IOU"""
    
    m1, c1, mat1, v1 = compute_mass_properties(target)
    m2, c2, mat2, v2 = compute_mass_properties(source)

    s1 = np.sqrt(np.linalg.trace(mat1)/m1)
    s2 = np.sqrt(np.linalg.trace(mat2)/m2)

    translation_vector = -c1
    shape_1 = target.copy()
    shape_1 = shape_1.apply_translation(translation_vector)
    shape_1 = shape_1.apply_scale(1/s1)
    
    translation_vector = -c2
    shape_2 = source.copy()
    shape_2 = shape_2.apply_translation(translation_vector)
    shape_2 = shape_2.apply_scale(1/s2)

    Rs = np.zeros((4,3,3))
    Rs[0] = v1 @ v2.T

    for i in range(3):
        # all possible 2 out of 3 permutations
        alignment = 1 - 2 * np.array([i>0, (i+1)%2, i%3<=1])
        Rs[i+1] = v1 @ (alignment[None,:] * v2).T
    
    best_IOU = iou(shape_1, shape_2)
    best_T = np.eye(3)
    best_T = np.pad(best_T, ((0,1),(0,1)), mode='constant', constant_values=0)
    best_T[3,3] = 1
    
    for i in range(4):
        R = Rs[i]
        R = np.pad(R, ((0,1),(0,1)), mode='constant', constant_values=0)
        R[3,3] = 1
        shape_2_aligned = shape_2.copy()
        shape_2_aligned = shape_2_aligned.apply_transform(R)
        
        IOU = iou(shape_1, shape_2_aligned)
        
        if IOU > best_IOU and IOU <= 1.0:
            best_IOU = IOU
            best_T = R
        
    # rotation to align 2 on 1
    R = best_T
    shape_2 = shape_2.apply_transform(R)
    shape_2 = shape_2.apply_scale(s1)
    translation_vector = c1
    shape_2 = shape_2.apply_translation(translation_vector)
    
    return shape_2, best_IOU

def _sampling_iou(shape_1: trimesh.Trimesh, shape_2: trimesh.Trimesh,
                  num_samples: int = 500_000, device: str = "cuda") -> float:
    """Fallback IoU via Monte Carlo sampling + GPU ray casting from iou_sampling.py."""
    try:
        if not torch.cuda.is_available():
            device = "cpu"

        torch.manual_seed(42)
        tris_a, bounds_a = load_mesh_as_triangles(shape_1, device=device)
        tris_b, bounds_b = load_mesh_as_triangles(shape_2, device=device)

        bb_min = torch.tensor(np.minimum(bounds_a[0], bounds_b[0]), dtype=torch.float32, device=device)
        bb_max = torch.tensor(np.maximum(bounds_a[1], bounds_b[1]), dtype=torch.float32, device=device)

        points = torch.rand((num_samples, 3), device=device) * (bb_max - bb_min) + bb_min
        direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)

        inter_a = count_ray_intersections(points, tris_a, direction)
        inter_b = count_ray_intersections(points, tris_b, direction)

        inside_a = (inter_a % 2 == 1)
        inside_b = (inter_b % 2 == 1)

        intersection = (inside_a & inside_b).sum(dtype=torch.int64)
        union = (inside_a | inside_b).sum(dtype=torch.int64)

        return float(intersection / (union + 1e-9))
    except Exception:
        return 0.0


def iou(shape_1 : trimesh.Trimesh, shape_2 : trimesh.Trimesh) -> float:
    """Compute the Intersection Over Union (IOU) of two shapes.
    Falls back to GPU sampling IoU if boolean operations fail."""
    try:
        intersection = shape_1.intersection(shape_2)
        union = shape_1.union(shape_2)
        V_I = intersection.mass
        V_U = union.mass
        return V_I / V_U
    except Exception as e:
        print(f"Error in iou: {e}")
        return _sampling_iou(shape_1, shape_2)

def pc_chamfer_distance(pc_1 : np.ndarray, pc_2 : np.ndarray) -> float:
    tree_1 = cKDTree(pc_1)
    tree_2 = cKDTree(pc_2)
    
    distances, _ = tree_1.query(pc_2, k=1)
    distances_2, _ = tree_2.query(pc_1, k=1)
    
    return np.mean(distances) + np.mean(distances_2)

def chamfer_distance(shape_1 : trimesh.Trimesh, shape_2 : trimesh.Trimesh, n_points : int = 10000) -> float:
    
    pc_1 = mesh_to_points(shape_1,num_points=n_points)[0]
    pc_2 = mesh_to_points(shape_2,num_points=n_points)[0]
    
    return pc_chamfer_distance(pc_1, pc_2)

def surface_iou(shape_1 : trimesh.Trimesh, shape_2 : trimesh.Trimesh, n_points : int = 10000, threshold : float = 0.02) -> float:
    """Compute Surface IoU as the percentage of points with Chamfer Distance < threshold.
    
    Args:
        shape_1: First mesh
        shape_2: Second mesh
        n_points: Number of points to sample from each mesh
        threshold: Distance threshold for considering a point as matching (default: 0.02)
    
    Returns:
        Percentage of points (0-1) with distance < threshold
    """
    pc_1 = mesh_to_points(shape_1, num_points=n_points)[0]
    pc_2 = mesh_to_points(shape_2, num_points=n_points)[0]
    
    tree_1 = cKDTree(pc_1)
    tree_2 = cKDTree(pc_2)
    
    # Get nearest neighbor distances from pc_2 to pc_1
    distances_2to1, _ = tree_1.query(pc_2, k=1)
    # Get nearest neighbor distances from pc_1 to pc_2
    distances_1to2, _ = tree_2.query(pc_1, k=1)
    
    # Count points with distance < threshold
    points_below_threshold_2to1 = np.sum(distances_2to1 < threshold)
    points_below_threshold_1to2 = np.sum(distances_1to2 < threshold)
    
    # Average the percentage from both directions
    total_points = n_points * 2
    total_below_threshold = points_below_threshold_2to1 + points_below_threshold_1to2
    
    return float(total_below_threshold / total_points)

def ICP_alignment(source : trimesh.Trimesh, target : trimesh.Trimesh, n_points : int = 10000, max_iterations : int = 100) -> Tuple[np.ndarray, float]:
    src_pts = mesh_to_points(source, num_points=n_points)[0]
    tgt_pts = mesh_to_points(target, num_points=n_points)[0]

    tree = cKDTree(tgt_pts)

    R_total = np.eye(3)
    t_total = np.zeros(3)

    pts = src_pts.copy()
    prev_assoc = None

    for iteration in range(max_iterations):
        _, idx = tree.query(pts, k=1)

        if prev_assoc is not None and np.array_equal(idx, prev_assoc):
            break
        prev_assoc = idx

        matched = tgt_pts[idx]

        src_centroid = pts.mean(axis=0)
        tgt_centroid = matched.mean(axis=0)

        H = (pts - src_centroid).T @ (matched - tgt_centroid)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T

        t = tgt_centroid - R @ src_centroid

        pts = (R @ pts.T).T + t

        R_total = R @ R_total
        t_total = R @ t_total + t

    T = np.eye(4)
    T[:3, :3] = R_total
    T[:3, 3] = t_total

    dists, _ = tree.query(pts, k=1)
    chamfer = np.mean(dists)

    return T, chamfer, iteration