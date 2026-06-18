import argparse
from pathlib import Path

from PIL import Image as PILImage

from datasets import DatasetDict, Features, Image, load_dataset
from huggingface_hub import HfApi


def get_all_splits(repo_name: str) -> set[str]:
    api = HfApi()
    repo_files = list(api.list_repo_files(repo_name, repo_type="dataset"))
    splits = set()
    for f in repo_files:
        parts = f.split("/")
        if len(parts) == 2 and parts[0] == "data" and parts[1].endswith(".parquet"):
            splits.add(parts[1].split("-")[0])
    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Re-upload a HuggingFace dataset split with a new image field."
    )
    parser.add_argument("--repo_name", required=True, help="HuggingFace repo, e.g. 'username/my-cad-dataset'.")
    parser.add_argument("--split", required=True, help="Split to upload images for (e.g. 'train', 'test').")
    parser.add_argument("--image_dir", required=True, help="Directory containing images named <file_id>_0.png.")
    parser.add_argument("--field", default="singleview_image", help="Name of the image field to populate (default: singleview_image).")
    parser.add_argument("--partial_replacement", action="store_true", help="Only replace images found in --image_dir; keep existing values for all other rows.")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    field = args.field
    all_splits = get_all_splits(args.repo_name)

    # Step 1: check if field already exists; if not, backfill all splits with None first
    first_split = next(iter(all_splits))
    probe_ds = load_dataset(args.repo_name, split=first_split)
    if field not in probe_ds.column_names or not isinstance(probe_ds.features[field], Image):
        print(f"'{field}' field not found — backfilling all splits with None ...")
        dataset_dict = {}
        for split_name in all_splits:
            print(f"  Backfilling '{split_name}' ...")
            ds = load_dataset(args.repo_name, split=split_name)
            if field in ds.column_names:
                ds = ds.remove_columns(field)
            new_features = Features({**ds.features, field: Image()})
            dataset_dict[split_name] = ds.map(lambda row: {**row, field: None}, features=new_features)
        print("Pushing all splits together ...")
        DatasetDict(dataset_dict).push_to_hub(args.repo_name)
    else:
        print(f"'{field}' field already exists, skipping backfill.")

    # Step 2: attach real images for the target split and push
    print(f"Loading split '{args.split}' ...")
    ds = load_dataset(args.repo_name, split=args.split)

    # Build a lookup from file_id stem to image path, searching recursively
    image_lookup = {p.stem: p for p in image_dir.rglob("*.png")}

    def attach_image(row):
        img_path = image_lookup.get(f"{row['file_id']}")
        if img_path is not None:
            if args.partial_replacement:
                row[field] = PILImage.open(img_path)
            else:
                row[field] = str(img_path)  # Let HuggingFace handle loading the image from the path
        elif not args.partial_replacement:
            row[field] = None
        return row

    if args.partial_replacement:
        replaced = sum(1 for row in ds if image_lookup.get(f"{row['file_id']}") is not None)
        print(f"Partially replacing {replaced} image(s) out of {len(ds)} rows.")

    ds = ds.map(attach_image, writer_batch_size=100)
    print(f"Pushing split '{args.split}' with images ...")
    ds.push_to_hub(args.repo_name, split=args.split)
    print("Done.")


if __name__ == "__main__":
    main()
