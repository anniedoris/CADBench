import argparse
import glob
import hashlib
import json
import os
import sys
import h5py
import cadquery as cq
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# cadlib is in get_bench/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "get_bench"))
from cadlib.visualize import vec2CADsolid
from OCC.Extend.DataExchange import write_step_file


def process_h5(args):
    h5_path, output_dir = args
    try:
        with h5py.File(h5_path, 'r') as fp:
            out_vec = fp["vec"][:].astype(float)
        out_shape = vec2CADsolid(out_vec)
        base_name = os.path.splitext(os.path.basename(h5_path))[0]
        save_path = os.path.join(output_dir, base_name + ".step")
        write_step_file(out_shape, save_path)
        return h5_path, None
    except Exception as e:
        return h5_path, str(e)


def main():
    parser = argparse.ArgumentParser(description="Convert DeepCAD .h5 vec files to STEP files.")
    parser.add_argument("--input_data_dir", required=True, help="Directory containing .h5 CAD vec files.")
    parser.add_argument("--output_data_dir", required=True, help="Directory to write .step files.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_dedup", action="store_true", default=False, help="Skip duplicate vec hash filtering.")
    parser.add_argument("--no_filter_multibody", action="store_true", default=False, help="Skip removing multi-body shapes.")
    parser.add_argument("--split_json", type=str, default=None, help="Path to split JSON (e.g. filtered_data.json); only generates STEP files for entries in the 'test' split.")
    args = parser.parse_args()

    os.makedirs(args.output_data_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(args.input_data_dir, "**", "*.h5"), recursive=True))
    print(f"Found {len(h5_files)} .h5 files in {args.input_data_dir}")

    if args.split_json is not None:
        with open(args.split_json) as f:
            split_data = json.load(f)
        test_ids = set(split_data["test"])
        print(f"Test split contains {len(test_ids)} entries.")
        h5_files = [p for p in h5_files if os.path.splitext(os.path.relpath(p, args.input_data_dir))[0] in test_ids]
        print(f"Filtered to {len(h5_files)} files matching test split from {args.split_json}")

    # Deduplicate by vec hash
    if not args.no_dedup:
        print("Deduplicating by vec hash...")
        seen_hashes = set()
        unique_files = []
        for path in tqdm(h5_files, desc="Hashing"):
            try:
                with h5py.File(path, 'r') as fp:
                    vec_hash = hashlib.sha256(fp["vec"][:].astype(float).tobytes()).hexdigest()
                if vec_hash not in seen_hashes:
                    seen_hashes.add(vec_hash)
                    unique_files.append(path)
            except Exception as e:
                print(f"  Hash error {path}: {e}")
        print(f"  {len(h5_files) - len(unique_files)} duplicates removed, {len(unique_files)} unique files.")
        h5_files = unique_files

    # Convert to STEP in parallel
    errors = 0
    tasks = [(p, args.output_data_dir) for p in h5_files]
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_h5, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Converting to STEP"):
            _, err = future.result()
            if err:
                errors += 1

    num_steps = len(glob.glob(os.path.join(args.output_data_dir, "*.step")))
    print(f"STEP files written: {num_steps} ({errors} errors)")

    # Remove multi-body shapes
    if not args.no_filter_multibody:
        removed = 0
        step_files = glob.glob(os.path.join(args.output_data_dir, "*.step"))
        for step_file in tqdm(step_files, desc="Filtering multi-body"):
            try:
                cq_model = cq.importers.importStep(step_file)
                if len(cq_model.solids().vals()) > 1:
                    os.remove(step_file)
                    removed += 1
            except Exception as e:
                print(f"  Error processing {step_file}: {e}")
                os.remove(step_file)
                removed += 1
        print(f"Removed {removed} multi-body/invalid files.")
        print(f"Final STEP count: {len(glob.glob(os.path.join(args.output_data_dir, '*.step')))}")


if __name__ == "__main__":
    main()
