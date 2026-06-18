import argparse
from pathlib import Path

from datasets import Dataset, Features, Value, Image, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError


def collect_samples(dataset_dir: str) -> list[dict]:
    """Recursively walk dataset_dir and collect 3D files and associated PNGs.
    
    - .step, .stl, .obj are collected as binary.
    - Files named 'A_0.png' are mapped to 'singleview_image' for the ID 'A'.
    - The top-level subdirectory is treated as the 'label'.
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    extensions = {".step", ".stl", ".obj", ".png"}
    by_stem: dict[str, dict] = {}

    # Sort to ensure deterministic behavior
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in extensions:
            continue

        ext = f.suffix.lower()
        raw_stem = f.stem
        
        # Handle the specific naming convention for images
        if ext == ".png" and raw_stem.endswith("_0"):
            stem = raw_stem[:-2]  # Remove '_0'
            field_name = "singleview_image"
        else:
            stem = raw_stem
            field_name = ext.lstrip(".")

        parts = f.relative_to(root).parts
        # Label is the immediate child of the root directory
        label = parts[0] if len(parts) > 1 else ""

        if stem not in by_stem:
            by_stem[stem] = {"file_id": stem, "label": label}
        
        # Read the file content
        by_stem[stem][field_name] = f.read_bytes()

    return list(by_stem.values())


def build_dataset(samples: list[dict]) -> Dataset:
    """Converts a list of dictionaries into a Hugging Face Dataset with proper schema."""
    features = Features(
        {
            "file_id": Value("string"),
            "label": Value("string"),
            "step": Value("binary"),
            "stl": Value("binary"),
            "obj": Value("binary"),
            "singleview_image": Image(), # Uses HF Image type for rendering/processing
        }
    )

    # Ensure every sample has every key to avoid schema mismatches
    for s in samples:
        for key in ("step", "stl", "obj", "singleview_image"):
            s.setdefault(key, None)

    return Dataset.from_list(samples, features=features)


def main():
    parser = argparse.ArgumentParser(
        description="Upload 3D CAD files and PNG previews to a HuggingFace dataset repo."
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--split_name",
        required=True,
        help="Name of the split (e.g. 'train', 'test').",
    )
    parser.add_argument(
        "--repo_name",
        required=True,
        help="HuggingFace repo (e.g. 'username/my-cad-dataset').",
    )
    args = parser.parse_args()

    print(f"Collecting samples from {args.dataset_dir} ...")
    samples = collect_samples(args.dataset_dir)
    print(f"Found {len(samples)} unique file IDs.")

    new_dataset = build_dataset(samples)
    print(f"Generated dataset structure:\n{new_dataset}")

    # HF API check for existing data
    api = HfApi()
    split_exists = False
    try:
        repo_files = api.list_repo_files(args.repo_name, repo_type="dataset")
        split_prefix = f"data/{args.split_name}-"
        split_exists = any(f.startswith(split_prefix) for f in repo_files)
    except RepositoryNotFoundError:
        print(f"Repository {args.repo_name} not found. It will be created on push.")
    except Exception as e:
        print(f"Note: Could not verify remote repo files: {e}")

    dataset_to_push = new_dataset
    if split_exists:
        print(f"\nSplit '{args.split_name}' already exists in '{args.repo_name}'.")
        while True:
            choice = input("Overwrite or append? [o/a]: ").strip().lower()
            if choice == "o":
                print("Overwriting existing split.")
                break
            elif choice == "a":
                print("Loading existing split to append to...")
                existing = load_dataset(args.repo_name, split=args.split_name)
                dataset_to_push = concatenate_datasets([existing, new_dataset])
                print(f"Combined dataset has {len(dataset_to_push)} samples.")
                break
            else:
                print("Please enter 'o' to overwrite or 'a' to append.")

    print(f"Pushing to {args.repo_name} (split={args.split_name}) ...")
    dataset_to_push.push_to_hub(args.repo_name, split=args.split_name)
    print("Done.")


if __name__ == "__main__":
    main()