import argparse
import os
import shutil
import sys
from glob import glob
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "get_bench"))
from utils import run_metric_parallel, is_single_body


def main():
    parser = argparse.ArgumentParser(description="Copy single-body .glb files from input_dir to output_dir.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory to search for .glb files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to copy single-body .glb files into.")
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    glb_files = sorted([
        os.path.join(dirpath, fname)
        for dirpath, _, filenames in os.walk(args.input_dir)
        for fname in filenames
        if fname.lower().endswith(".glb")
    ])
    print(f"Found {len(glb_files)} .glb files in {args.input_dir}")

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    results = run_metric_parallel(is_single_body, glb_files, num_workers=args.num_workers)

    for file_path, is_single in tqdm(results, desc="Copying single-body GLBs", total=len(glb_files)):
        if is_single:
            shutil.copy(file_path, args.output_dir)

    final_count = len(glob(os.path.join(args.output_dir, "*.glb")))
    print(f"Done. {final_count} single-body GLB files copied to {args.output_dir}")


if __name__ == "__main__":
    main()
