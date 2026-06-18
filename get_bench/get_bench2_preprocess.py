import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path
import numpy as np
import trimesh
from tqdm import tqdm
import pymeshlab

MIN_VOLUME = 1e-9

_NON_MANIFOLD_FILTERS = [
    "meshing_repair_non_manifold_edges",
    "meshing_repair_non_manifold_vertices",
    "meshing_remove_non_manifold_edges",
    "meshing_remove_non_manifold_vertices",
]

def try_repair_watertight(input_path):
    """
    Attempt to repair a mesh to watertight using PyMeshLab.

    Returns a trimesh.Trimesh on success, or None if repair failed or the
    result is not watertight / has non-positive volume.
    """
    try:

        mesh = trimesh.load(str(input_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) and hasattr(mesh, "geometry"):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_in  = str(Path(tmp) / "in.ply")
            tmp_out = str(Path(tmp) / "out.ply")

            mesh.export(tmp_in)

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(tmp_in)

            ms.meshing_remove_duplicate_vertices()
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_unreferenced_vertices()
            ms.meshing_remove_null_faces()

            try: ms.meshing_merge_close_vertices(threshold=1e-6)
            except Exception: pass

            try: ms.meshing_close_holes(maxholesize=1000)
            except Exception: pass

            for fn in _NON_MANIFOLD_FILTERS:
                if hasattr(ms, fn):
                    try: getattr(ms, fn)()
                    except Exception: pass

            ms.save_current_mesh(tmp_out)
            repaired = trimesh.load(tmp_out, force="mesh")

        if not repaired.is_watertight:
            return None
        if repaired.volume <= MIN_VOLUME:
            return None
        return repaired

    except ImportError:
        print("pymeshlab is not installed; skipping repair.")
        return None
    except Exception:
        return None

def canonical_mesh_hash(mesh, decimals=6):
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices(digits_vertex=decimals)

    vertices = np.round(mesh.vertices, decimals=decimals)
    faces = mesh.faces.copy()

    # Canonicalize vertex ordering so equivalent meshes with different
    # vertex orderings produce the same face indices before hashing.
    lex_order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    remap = np.empty(len(lex_order), dtype=np.int64)
    remap[lex_order] = np.arange(len(lex_order), dtype=np.int64)
    vertices = vertices[lex_order]
    faces = remap[faces]

    # sort vertex indices within each face
    faces = np.sort(faces, axis=1)

    # sort faces lexicographically
    faces = faces[np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))]

    payload = vertices.tobytes() + faces.tobytes()
    return hashlib.sha256(payload).hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Check OBJ files for watertightness.")
    parser.add_argument("--mesh_dir", required=True, help="Directory containing .obj files.")
    args = parser.parse_args()

    mesh_dir = Path(args.mesh_dir)
    out_dir = mesh_dir.parent / (mesh_dir.name + "_dedup")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    obj_files = sorted(mesh_dir.rglob("*.obj"))
    print(f"Found {len(obj_files)} .obj files in {mesh_dir}")

    utils_dir = Path("data/bench2_utils")
    utils_dir.mkdir(parents=True, exist_ok=True)
    not_watertight_file = utils_dir / "not_watertight.txt"
    not_singlebody_file = utils_dir / "not_singlebody.txt"
    duplicates_file = utils_dir / "duplicates.txt"

    not_watertight_names = []
    not_singlebody_names = []
    hash_to_names = {}

    cleaned_count = 0
    seen_hashes = {}
    for obj_path in tqdm(obj_files):
        mesh = trimesh.load(str(obj_path))

        if not isinstance(mesh, trimesh.Trimesh):
            print(f"  Skipping {obj_path.name}: loaded as {type(mesh).__name__}, not Trimesh")
            continue

        # Check that mesh is watertight. If not, try to repair it
        if not mesh.is_watertight:
            mesh = try_repair_watertight(obj_path)
            if mesh is None:
                not_watertight_names.append(obj_path.stem)
                continue

        # Only keep meshes that have single connected components
        components = mesh.split(only_watertight=True)
        if len(components) != 1:
            not_singlebody_names.append(obj_path.stem)
            continue

        # Remove duplicates
        mesh_hash = canonical_mesh_hash(mesh)
        if mesh_hash not in hash_to_names:
            hash_to_names[mesh_hash] = []
            dest = out_dir / obj_path.relative_to(mesh_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(dest))
            cleaned_count += 1
        else:
            print(f"  Skipping {obj_path.name}: duplicate")
        hash_to_names[mesh_hash].append(obj_path.stem)

    # Save problematic samples to files for debugging
    not_watertight_file.write_text("\n".join(not_watertight_names))
    not_singlebody_file.write_text("\n".join(not_singlebody_names))
    duplicates_file.write_text("\n".join(",".join(names) for names in hash_to_names.values()))

    print(f"{cleaned_count}/{len(obj_files)} cleaned meshes saved to {out_dir}")
    print(f"Not watertight: {len(not_watertight_names)}, not single-body: {len(not_singlebody_names)}, duplicate groups: {sum(1 for n in hash_to_names.values() if len(n) > 1)}")
    

if __name__ == "__main__":
    main()
