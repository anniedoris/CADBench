import argparse
from pathlib import Path

from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError


def collect_samples(dataset_dir: str) -> list[dict]:
    """Recursively walk dataset_dir and collect all .STEP/.stl/.obj files.

    The immediate subdirectories of dataset_dir are treated as labels.
    For example, dataset_dir/easy/steps/file.step gets label "easy".
    Files at the root level get label "".
    Files are grouped by stem so each row contains all formats for one model.
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    extensions = {".step", ".stl", ".obj"}
    by_stem: dict[str, dict] = {}

    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in extensions:
            continue

        stem = f.stem
        parts = f.relative_to(root).parts
        # The label is the top-level subdirectory name; empty if file is at root
        label = parts[0] if len(parts) > 1 else ""

        if stem not in by_stem:
            by_stem[stem] = {"file_id": stem, "label": label}
        by_stem[stem][f.suffix.lower().lstrip(".")] = f.read_bytes()

    return list(by_stem.values())


def build_dataset(samples: list[dict]) -> Dataset:
    features = Features(
        {
            "file_id": Value("string"),
            "label": Value("string"),
            "step": Value("binary"),
            "stl": Value("binary"),
            "obj": Value("binary"),
        }
    )

    # Ensure every sample has all keys (None if the file wasn't present)
    for s in samples:
        for key in ("step", "stl", "obj"):
            s.setdefault(key, None)

    return Dataset.from_list(samples, features=features)


def main():
    parser = argparse.ArgumentParser(
        description="Upload 3D CAD files (.STEP/.stl/.obj) to a HuggingFace dataset repo."
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Root directory of the dataset (contains split subdirectories).",
    )
    parser.add_argument(
        "--split_name",
        required=True,
        help="Name of the split to upload (e.g. 'train', 'test').",
    )
    parser.add_argument(
        "--repo_name",
        required=True,
        help="HuggingFace repo to push to, e.g. 'username/my-cad-dataset'.",
    )
    args = parser.parse_args()

    print(f"Collecting samples from {args.dataset_dir} ...")
    samples = collect_samples(args.dataset_dir)
    print(f"Found {len(samples)} samples.")

    new_dataset = build_dataset(samples)
    print(new_dataset)

    # Check whether the split already exists in the remote repo
    api = HfApi()
    split_exists = False
    try:
        repo_files = api.list_repo_files(args.repo_name, repo_type="dataset")
        split_prefix = f"data/{args.split_name}-"
        split_exists = any(f.startswith(split_prefix) for f in repo_files)
    except RepositoryNotFoundError:
        pass  # Repo doesn't exist yet; will be created on push

    dataset_to_push = new_dataset
    if split_exists:
        print(f"Split '{args.split_name}' already exists in '{args.repo_name}'.")
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
