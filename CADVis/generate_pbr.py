import numpy as np
from PIL import Image
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import trimesh

# --- 1. FORCE EGL PLATFORM ---
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from CADVis.PyR.Material import AutoPBR
from CADVis.PyR.Rendering import render_shape
import pyrender

def get_metallic_material(color=[0.7, 0.7, 0.7, 1.0], metallic=1.0, roughness=0.2):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=color,
        metallicFactor=metallic,
        roughnessFactor=roughness)

def process_single_file(filename, input_folder, output_folder, surface_materials_path, part_materials_path,
                       min_metallic, max_metallic, min_roughness, max_roughness, min_color, max_color):
    """
    Worker function to render a single file (STEP or STL).
    """
    try:
        input_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"{base_name}.png")

        # Get material lists
        surface_materials = [f for f in os.listdir(surface_materials_path) if os.path.isdir(os.path.join(surface_materials_path, f))]
        part_materials = [f for f in os.listdir(part_materials_path) if os.path.isdir(os.path.join(part_materials_path, f))]

        surf_mat = AutoPBR(os.path.join(surface_materials_path, np.random.choice(surface_materials)))

        color = list(np.random.uniform(min_color, max_color, 3)) + [1.0]
        metallic = np.random.uniform(min_metallic, max_metallic)
        roughness = np.random.uniform(min_roughness, max_roughness)
        model_mat = get_metallic_material(color=color, metallic=metallic, roughness=roughness)

        # --- Handle STEP or STL ---
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.step', '.stp']:
            shape_or_mesh = input_path  # let render_shape handle STEP
        elif ext == '.stl':
            mesh = trimesh.load(input_path, force='mesh')
            shape_or_mesh = (np.array(mesh.vertices), np.array(mesh.faces))
        else:
            return f"[ERROR] {filename}: Unsupported file type."

        image, _ = render_shape(
            shape_or_mesh,
            ground_material=surf_mat,
            model_material=model_mat,
            light_intensity=1.0,
            mesh_uv_scale=4,
            model_color="#{:06x}".format(np.random.randint(0, 0xFFFFFF))
        )
        image.save(output_path)
        return f"[SUCCESS] {filename} -> {output_path}"

    except Exception as e:
        return f"[ERROR] {filename}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Parallel CAD Rendering Script")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input folder containing STEP/STL files")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to folder where images will be saved")
    parser.add_argument("--surf_mats", type=str, default="./Materials/Surfaces/", help="Path to surface materials")
    parser.add_argument("--part_mats", type=str, default="./Materials/Part/", help="Path to part materials")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of parallel processes (defaults to CPU count)")
    parser.add_argument("--min_metallic", type=float, default=0.05, help="Minimum metallic value (0.0-1.0)")
    parser.add_argument("--max_metallic", type=float, default=0.95, help="Maximum metallic value (0.0-1.0)")
    parser.add_argument("--min_roughness", type=float, default=0.05, help="Minimum roughness value (0.0-1.0)")
    parser.add_argument("--max_roughness", type=float, default=0.95, help="Maximum roughness value (0.0-1.0)")
    parser.add_argument("--min_color", type=float, default=0.0, help="Minimum color channel value (0.0-1.0)")
    parser.add_argument("--max_color", type=float, default=1.0, help="Maximum color channel value (0.0-1.0)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Accept .step, .stp, .stl
    files_to_process = [f for f in os.listdir(args.input) if f.lower().endswith(('.step', '.stp', '.stl'))]
    if not files_to_process:
        print(f"No STEP or STL files found in {args.input}")
        return

    print(f"Processing {len(files_to_process)} files using {args.workers or 'all'} cores...")

    worker_func = partial(
        process_single_file,
        input_folder=args.input,
        output_folder=args.output,
        surface_materials_path=args.surf_mats,
        part_materials_path=args.part_mats,
        min_metallic=args.min_metallic,
        max_metallic=args.max_metallic,
        min_roughness=args.min_roughness,
        max_roughness=args.max_roughness,
        min_color=args.min_color,
        max_color=args.max_color
    )

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in tqdm(executor.map(worker_func, files_to_process), total=len(files_to_process), desc="Rendering"):
            results.append(res)

    for res in results:
        print(res)

if __name__ == "__main__":
    main()