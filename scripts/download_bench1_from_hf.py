import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Added 'singleview_image' to the recognized formats
FORMATS = ("step", "stl", "obj", "singleview_image")

SPLITS = {
    "bench0":  "bench0",
    "bench1A": "bench1A",
    "bench1B": "bench1B",
}


def main():
    parser = argparse.ArgumentParser(description="Download CADBench files from HuggingFace to disk.")
    parser.add_argument("--split",  required=True, choices=list(SPLITS), help="Dataset split.")
    parser.add_argument("--out",    required=True, help="Output directory")
    parser.add_argument("--repo",   default="DeCoDELab/CADBench", help="HuggingFace dataset repo")
    parser.add_argument("--format", default="step", choices=(*FORMATS, "all"), help="File format(s) to save")
    parser.add_argument("--limit",  type=int, default=None, help="Only download the first N rows")
    args = parser.parse_args()

    formats = list(FORMATS) if args.format == "all" else [args.format]
    hf_split = SPLITS[args.split]

    print(f"Loading {args.repo} split={hf_split} ({args.split}) ...")
    ds = load_dataset(args.repo, split=hf_split, streaming=bool(args.limit))

    root_out = Path(args.out)
    root_out.mkdir(parents=True, exist_ok=True)

    written = 0
    rows = ds.take(args.limit) if args.limit else ds
    
    for row in tqdm(rows, desc="Saving", total=args.limit):
        label = row.get("label") or "unlabeled"
        
        for fmt in formats:
            data = row.get(fmt)
            if data is None:
                continue

            # 1. Define the format-specific subfolder name
            # We use 'png' as the folder name instead of 'singleview_image' for brevity
            fmt_folder_name = "png" if fmt == "singleview_image" else fmt
            
            # 2. Construct path: out / label / format / file
            target_dir = root_out / label / fmt_folder_name
            target_dir.mkdir(parents=True, exist_ok=True)

            # 3. Handle naming and saving
            if fmt == "singleview_image":
                file_path = target_dir / f"{row['file_id']}_0.png"
                if hasattr(data, 'save'):
                    data.save(file_path)
                else:
                    file_path.write_bytes(data)
            else:
                file_path = target_dir / f"{row['file_id']}.{fmt}"
                file_path.write_bytes(data)
            
            written += 1

    print(f"\nDone. {written} file(s) written to {args.out}")


if __name__ == "__main__":
    main()