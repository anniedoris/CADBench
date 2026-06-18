import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

FORMATS = ("step", "stl", "obj", "glb", "singleview_image", "multiview_image","pbr","noisy_stl")
IMAGE_FORMATS = {"singleview_image", "multiview_image", "pbr"}

SPLITS = {
    "bench0":  "bench0",
    "bench1A": "bench1A",
    "bench1B": "bench1B",
    "bench2": "bench2",
    "bench3": "bench3",
    "bench0F":"bench0F"
}

EXT_MAP = {
    "step": "step",
    "stl": "stl",
    "noisy_stl": "stl",
    "obj": "obj",
    "glb": "glb"
}


def save_row(row: dict, formats: list[str], root_out: Path) -> int:
    label = row.get("label") or "unlabeled"
    written = 0

    for fmt in formats:
        data = row.get(fmt)
        if data is None:
            continue

        target_dir = root_out / label / fmt
        target_dir.mkdir(parents=True, exist_ok=True)

        if fmt in IMAGE_FORMATS:
            # Handle potential lists (multiview) or single images
            images = data if isinstance(data, list) else [data]
            
            for i, img in enumerate(images):
                # If singleview, we usually just want _0, if multiview, _0, _1, etc.
                if fmt == "pbr":
                    file_path = target_dir / f"{row['file_id']}.png"
                else:
                    suffix = f"_{i}" if fmt == "singleview_image" else ""
                    file_path = target_dir / f"{row['file_id']}{suffix}.png"

                if hasattr(img, "save"):
                    img.save(file_path)
                else:
                    file_path.write_bytes(img)
                written += 1
        else:
            # Use mapping for 3D formats (e.g., noisy_stl -> .stl)
            ext = EXT_MAP.get(fmt, fmt)
            file_path = target_dir / f"{row['file_id']}.{ext}"
            file_path.write_bytes(data)
            written += 1

        written += 1

    return written


def main():
    parser = argparse.ArgumentParser(description="Download CADBench files from HuggingFace to disk.")
    parser.add_argument("--split",  required=True, choices=list(SPLITS), help="Dataset split.")
    parser.add_argument("--out",    required=True, help="Output directory")
    parser.add_argument("--repo",   default="DeCoDELab/CADBench", help="HuggingFace dataset repo")
    parser.add_argument("--format", default="step", choices=(*FORMATS, "all"), help="File format(s) to save")
    parser.add_argument("--limit",  type=int, default=None, help="Only download the first N rows")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads for saving files")
    args = parser.parse_args()

    formats = list(FORMATS) if args.format == "all" else [args.format]
    hf_split = SPLITS[args.split]

    print(f"Loading {args.repo} split={hf_split} ({args.split}) ...")
    ds = load_dataset(args.repo, split=hf_split, streaming=True)

    root_out = Path(args.out)
    root_out.mkdir(parents=True, exist_ok=True)

    written = 0
    rows = ds.take(args.limit) if args.limit else ds

    max_pending = max(args.workers * 4, 1)
    pending = set()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row in tqdm(rows, desc="Queueing", total=args.limit):
            pending.add(executor.submit(save_row, row, formats, root_out))

            if len(pending) >= max_pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    written += future.result()

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                written += future.result()

    print(f"\nDone. {written} file(s) written to {args.out}")


if __name__ == "__main__":
    main()
