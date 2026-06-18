"""Compare two benchmark directories and quantify file set differences per split."""
import argparse
import os


def get_file_stems(directory):
    stems = set()
    for fname in os.listdir(directory):
        stem, ext = os.path.splitext(fname)
        stems.add(stem)
    return stems


def compare_split(split, dir_a, dir_b, subdirs=("steps", "images")):
    print(f"\n{'='*50}")
    print(f"Split: {split}")
    print(f"{'='*50}")
    for subdir in subdirs:
        path_a = os.path.join(dir_a, split, subdir)
        path_b = os.path.join(dir_b, split, subdir)
        if not os.path.isdir(path_a):
            print(f"  [{subdir}] missing in {dir_a}")
            continue
        if not os.path.isdir(path_b):
            print(f"  [{subdir}] missing in {dir_b}")
            continue
        stems_a = get_file_stems(path_a)
        stems_b = get_file_stems(path_b)
        only_a = stems_a - stems_b
        only_b = stems_b - stems_a
        common = stems_a & stems_b
        print(f"  [{subdir}]")
        print(f"    {os.path.basename(dir_a)}: {len(stems_a)} files")
        print(f"    {os.path.basename(dir_b)}: {len(stems_b)} files")
        print(f"    In common:              {len(common)}")
        print(f"    Only in {os.path.basename(dir_a):12s}: {len(only_a)}")
        print(f"    Only in {os.path.basename(dir_b):12s}: {len(only_b)}")
        overlap_pct = 100 * len(common) / len(stems_a | stems_b) if stems_a | stems_b else 0
        print(f"    Overlap (Jaccard):      {overlap_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Quantify file set differences between two bench directories.")
    parser.add_argument("dir_a", help="First benchmark directory (e.g. data/bench0)")
    parser.add_argument("dir_b", help="Second benchmark directory (e.g. data/bench0old)")
    parser.add_argument("--splits", nargs="+", default=None, help="Splits to compare (default: all subdirs of dir_a)")
    args = parser.parse_args()

    splits = args.splits or sorted(
        d for d in os.listdir(args.dir_a) if os.path.isdir(os.path.join(args.dir_a, d))
    )
    for split in splits:
        compare_split(split, args.dir_a, args.dir_b)


if __name__ == "__main__":
    main()
