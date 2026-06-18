import pymeshlab
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import multiprocessing

def _run_remesh_worker(input_path, temp_output):
    try:
        worker_ms = pymeshlab.MeshSet()
        worker_ms.load_new_mesh(input_path)
        worker_ms.meshing_isotropic_explicit_remeshing()
        worker_ms.save_current_mesh(temp_output)
    except:
        pass 

def remesh_single_file(filename, input_dir, output_dir, noise, smoothing, timeout_val):
    input_path = os.path.abspath(os.path.join(input_dir, filename))
    
    # Filename handling
    name_part, ext_part = os.path.splitext(filename)
    noisy_filename = f"{name_part}_noisy{ext_part}"
    output_path = os.path.abspath(os.path.join(output_dir, noisy_filename))
    
    temp_remesh_path = os.path.abspath(os.path.join(output_dir, f"temp_remesh_{filename}"))

    try:
        ms = pymeshlab.MeshSet()
        remesh_executed = False

        # --- TIMEOUT GUARDED REMESHING ---
        p = multiprocessing.Process(target=_run_remesh_worker, args=(input_path, temp_remesh_path))
        p.start()
        p.join(timeout=timeout_val)

        if p.is_alive():
            p.terminate() 
            p.join()
            ms.load_new_mesh(input_path)
        elif os.path.exists(temp_remesh_path):
            ms.load_new_mesh(temp_remesh_path)
            os.remove(temp_remesh_path)
            remesh_executed = True
        else:
            ms.load_new_mesh(input_path)

        # --- NOISE & SMOOTHING ---
        if ms.current_mesh().vertex_number() == 0:
            return "error", f"{filename} is empty"

        m = ms.current_mesh()
        vertices = m.vertex_matrix()
        noise_vals = np.random.normal(0, noise, vertices.shape)
        noisy_vertices = vertices + noise_vals

        new_mesh = pymeshlab.Mesh(vertex_matrix=noisy_vertices, face_matrix=m.face_matrix())
        ms.add_mesh(new_mesh)

        if smoothing:
            ms.apply_filter('apply_coord_laplacian_smoothing', stepsmoothnum=1, cotangentweight=True)

        ms.save_current_mesh(output_path)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            msg = f"{noisy_filename} (Remesh Timed Out)" if not remesh_executed else None
            return "success", msg
        else:
            return "error", f"{noisy_filename} saved as 0 bytes"
    
    except Exception as e:
        return "error", f"{filename} failed: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--skip", "-s", action="store_true")
    parser.add_argument("--workers", "-w", type=int, default=os.cpu_count())
    parser.add_argument("--noise", "-n", type=float, default=0.001)
    parser.add_argument("--smoothing", "-sm", action="store_true")
    parser.add_argument("--timeout", "-t", type=int, default=300) 
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Get list of all STL files
    all_files = [f for f in os.listdir(args.input) if f.lower().endswith('.stl')]
    files_to_process = []
    skipped_count = 0

    # --- PRE-PROCESS SKIP LOGIC ---
    for f in all_files:
        name_part, ext_part = os.path.splitext(f)
        expected_output = os.path.join(args.output, f"{name_part}_noisy{ext_part}")
        
        if args.skip and os.path.exists(expected_output):
            skipped_count += 1
        else:
            files_to_process.append(f)

    print(f"Total files: {len(all_files)} | Skipping: {skipped_count} | To process: {len(files_to_process)}")

    stats = {"success": 0, "error": 0}
    logs = []

    if not files_to_process:
        print("No new files to process.")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                remesh_single_file, f, args.input, args.output, 
                args.noise, args.smoothing, args.timeout
            ): f for f in files_to_process
        }

        with tqdm(total=len(files_to_process), desc="Processing", unit="file") as pbar:
            for future in as_completed(futures):
                result, detail = future.result()
                stats[result] += 1
                if detail:
                    logs.append(detail)
                pbar.update(1)

    print(f"\nSucceeded: {stats['success']} | Skipped: {skipped_count} | Errors: {stats['error']}")
    if logs:
        print("\n--- Processing Notes ---")
        for log in logs:
            print(log)

if __name__ == "__main__":
    main()