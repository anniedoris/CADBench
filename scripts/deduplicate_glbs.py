import os
import shutil
import hashlib
import argparse
from tqdm import tqdm


def file_hash(path, chunk_size=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def collect_glbs(directories):
    paths = []
    for d in directories:
        for dirpath, _, filenames in os.walk(d):
            for fname in filenames:
                if fname.lower().endswith(".glb"):
                    paths.append(os.path.join(dirpath, fname))
    return paths


def deduplicate(input_dirs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_files = collect_glbs(input_dirs)
    print(f"Found {len(all_files)} .glb files across {len(input_dirs)} directories.")

    seen_hashes = {}
    duplicates = 0

    for path in tqdm(all_files, desc="Deduplicating"):
        h = file_hash(path)
        if h in seen_hashes:
            duplicates += 1
            continue
        seen_hashes[h] = path

        # Preserve filename; handle collisions by appending a counter
        fname = os.path.basename(path)
        dest = os.path.join(output_dir, fname)
        counter = 1
        while os.path.exists(dest):
            stem, ext = os.path.splitext(fname)
            dest = os.path.join(output_dir, f"{stem}_{counter}{ext}")
            counter += 1

        shutil.copy2(path, dest)

    unique = len(seen_hashes)
    print(f"Done. {unique} unique files copied, {duplicates} duplicates skipped.")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Deduplicate .glb files from multiple directories.")
    parser.add_argument("input_dirs", nargs="+", help="Input directories to search for .glb files.")
    parser.add_argument("--output_dir", type=str, default="data/objaverse_dedup", help="Output directory for deduplicated files.")
    args = parser.parse_args()

    deduplicate(args.input_dirs, args.output_dir)


if __name__ == "__main__":
    main()
