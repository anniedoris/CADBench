import argparse
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="List file_ids for a given split/label in CADBench.")
    parser.add_argument("--split", required=True, help="Dataset split (e.g. benchB)")
    parser.add_argument("--label", required=True, help="Label to filter on (e.g. easy, medium, hard)")
    parser.add_argument("--repo", default="DeCoDELab/CADBench", help="HuggingFace dataset repo (default: DeCoDELab/CADBench)")
    parser.add_argument("--out", default=None, help="Output .txt path (default: <split>_<label>_file_ids.txt)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(f"{args.split}_{args.label}_file_ids.txt")

    ds = load_dataset(args.repo, split=args.split)
    file_ids = [row["file_id"] for row in ds if row.get("label") == args.label]

    out_path.write_text("\n".join(file_ids) + "\n" if file_ids else "")
    print(f"Found {len(file_ids)} file_id(s) with label='{args.label}' in split='{args.split}'.")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
