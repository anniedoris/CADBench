import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

FORMATS = ("step", "stl", "noisy_stl", "image", "multiview_image", "pbr")

# Maps format name -> (dataset field name, file extension)
FORMAT_MAP = {
    "step":            ("step",             "step"),
    "stl":             ("stl",              "stl"),
    "noisy_stl":       ("noisy_stl",        "stl"),
    "image":           ("singleview_image", "png"),
    "multiview_image": ("multiview_image",  "png"),
    "pbr":             ("pbr",              "png"),
}

DIR_NAMES = {
    "stl":            "mesh",
    "noisy_stl":      "noisy_mesh",
    "image":          "singleview",
    "multiview_image": "multiview",
    "pbr":            "pbr",
}

MODEL_TYPE_FORMATS = {
    "mesh":  ["stl", "noisy_stl"],
    "image": ["image", "multiview_image", "pbr"],
}

SPLITS = {
    "benchB":  "benchB",
    "benchF": "benchF",
    "benchE": "benchE",
    "benchA": "benchA",
    "benchM":  "benchM",
    "benchO":  "benchO"
}


def main():
    parser = argparse.ArgumentParser(description="Download CADBench files from HuggingFace to disk.")
    parser.add_argument("--split",      nargs="+", choices=list(SPLITS), default=list(SPLITS), help="Dataset split(s) to download (default: all)")
    parser.add_argument("--model_type", required=True, choices=list(MODEL_TYPE_FORMATS), help="Modality group to download: 'mesh' (stl, noisy_stl) or 'image' (image, multiview_image, pbr)")
    parser.add_argument("--out",        default="benchmark_data", help="Output directory (default: benchmark_data)")
    parser.add_argument("--repo",       default="DeCoDELab/CADBench", help="HuggingFace dataset repo (default: DeCoDELab/CADBench)")
    parser.add_argument("--limit",      type=int, default=None, help="Only download the first N rows per split (default: all)")
    args = parser.parse_args()

    formats = MODEL_TYPE_FORMATS[args.model_type]
    n_splits = len(args.split)

    print(f"\nDownloading model_type={args.model_type} | modalities={formats} | splits={args.split}")
    print(f"Output root: {Path(args.out)}\n")

    total_written = 0
    for split_idx, split_name in enumerate(args.split, 1):
        hf_split = SPLITS[split_name]

        print(f"[{split_idx}/{n_splits}] Split: {split_name}")
        print(f"  Loading {args.repo} split={hf_split} ...")
        ds = load_dataset(args.repo, split=hf_split, streaming=bool(args.limit))

        written = 0
        rows = ds.take(args.limit) if args.limit else ds
        for row in tqdm(rows, desc=f"  {split_name}", total=args.limit):
            label = row.get("label") or ""

            for fmt in formats:
                field, ext = FORMAT_MAP[fmt]
                val = row.get(field)
                if val is None:
                    continue

                subdir = Path(args.out) / DIR_NAMES.get(fmt, fmt) / split_name
                if label and label not in ("easy", "medium", "hard"):
                    subdir = subdir / label
                subdir.mkdir(parents=True, exist_ok=True)

                if ext == "png":
                    val.save(subdir / f"{row['file_id']}.png")
                else:
                    filename = f"{row['file_id']}.{ext}"
                    (subdir / filename).write_bytes(val)
                written += 1

        print(f"  Done. {written} file(s) written to {Path(args.out) / split_name}\n")
        total_written += written

    print(f"All splits complete. {total_written} total file(s) written to {Path(args.out)}")


if __name__ == "__main__":
    main()
