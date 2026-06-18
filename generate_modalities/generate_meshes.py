import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue
import multiprocessing as mp

import cadquery as cq
import numpy as np
from tqdm import tqdm
import trimesh


SUPPORTED_INPUT_EXTS = {".step", ".stp", ".obj", ".glb", ".gltf"}


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--mesh_tolerance", type=float, default=0.001)
parser.add_argument("--angular_tolerance", type=float, default=0.1)
parser.add_argument("--timeout", type=int, default=3000)
args = parser.parse_args()


def _normalize_vertices_to_unit_box(vertices):
    if len(vertices) == 0:
        return vertices

    vertices = np.asarray(vertices, dtype=np.float64)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_extent = 0.5 * (maxs - mins)
    scale = half_extent.max()

    if scale <= 0:
        return vertices - center

    return (vertices - center) / scale


def _normalize_trimesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    vertices = _normalize_vertices_to_unit_box(mesh.vertices)
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)


def _export_step_with_cadquery(input_path, output_path):
    model = cq.importers.importStep(str(input_path))
    shape = model.val()
    shape.exportStl(
        str(output_path),
        tolerance=args.mesh_tolerance,
        angularTolerance=args.angular_tolerance,
    )
    mesh = trimesh.load(output_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type after STL export for {input_path}: {type(mesh)}")
    mesh = _normalize_trimesh(mesh)
    mesh.export(output_path)


def _export_mesh_with_trimesh(input_path, output_path):
    mesh = trimesh.load(input_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type for {input_path}: {type(mesh)}")
    mesh = _normalize_trimesh(mesh)
    mesh.export(output_path)


def __process_file_and_save(input_path, output_path, queue=None):
    try:
        ext = os.path.splitext(input_path)[1].lower()
        if ext in {".step", ".stp"}:
            _export_step_with_cadquery(input_path, output_path)
        elif ext in {".obj", ".glb", ".gltf"}:
            _export_mesh_with_trimesh(input_path, output_path)
        else:
            raise ValueError(f"Unsupported input file extension: {ext}")

        if queue is not None:
            queue.put(True)
    except Exception as exc:
        print(f"Error processing {input_path}: {exc}")
        if queue is not None:
            queue.put(False)


def _process_file_and_save(input_path, output_path, timeout=30):
    queue = Queue()
    process = Process(target=__process_file_and_save, args=(input_path, output_path, queue))
    process.start()

    try:
        result = queue.get(timeout=timeout)
    except Exception:
        process.kill()
        process.join()
        return 2

    process.join()
    return int(result)


def main():
    all_files = []
    for root, _, files in os.walk(args.data_path):
        for file_name in files:
            if os.path.splitext(file_name)[1].lower() in SUPPORTED_INPUT_EXTS:
                all_files.append(os.path.join(root, file_name))
    all_files = sorted(all_files)

    output_dir = f"data/meshes/meshes_{os.path.basename(os.path.normpath(args.data_path))}"
    os.makedirs(output_dir, exist_ok=True)

    file_output_pairs = []
    for input_path in all_files:
        part_name = os.path.splitext(os.path.basename(input_path))[0] + ".stl"
        output_path = os.path.join(output_dir, part_name)
        file_output_pairs.append((input_path, output_path))

    processing_fails = 0
    timeout_fails = 0
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp.get_context("fork")) as executor:
        future_to_file = {
            executor.submit(_process_file_and_save, input_path, output_path, timeout=args.timeout): input_path
            for input_path, output_path in file_output_pairs
        }
        prog = tqdm(as_completed(future_to_file), total=len(all_files))
        for future in prog:
            try:
                result = future.result()
                if result == 0:
                    processing_fails += 1
                elif result == 2:
                    timeout_fails += 1
            except Exception:
                processing_fails += 1
            prog.set_postfix(processing_fails=processing_fails, timeout_fails=timeout_fails)


if __name__ == "__main__":
    main()
