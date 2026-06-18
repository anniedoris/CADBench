import argparse
from pathlib import Path

from datasets import DatasetDict, Features, Value, load_dataset
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


def is_binary_feature(feature) -> bool:
    return isinstance(feature, Value) and feature.dtype == "binary"


def main():
    parser = argparse.ArgumentParser(
        description="Re-upload a HuggingFace dataset split with STL meshes."
    )
    parser.add_argument("--repo_name", required=True, help="HuggingFace repo, e.g. 'username/my-cad-dataset'.")
    parser.add_argument("--split", required=True, help="Split to upload meshes for (e.g. 'train', 'test').")
    parser.add_argument("--mesh_dir", required=True, help="Directory containing STL meshes.")
    parser.add_argument("--partial_replacement", action="store_true", help="Only replace meshes found in --mesh_dir; keep existing values for all other rows.")
    args = parser.parse_args()

    mesh_dir = Path(args.mesh_dir)
    field = "stl"
    all_splits = get_all_splits(args.repo_name)
    if not all_splits:
        raise ValueError(f"No splits found in dataset repo '{args.repo_name}'.")

    # Step 1: check if field already exists; if not, backfill all splits with None first
    first_split = sorted(all_splits)[0]
    probe_ds = load_dataset(args.repo_name, split=first_split)
    if field not in probe_ds.column_names or not is_binary_feature(probe_ds.features[field]):
        print(f"'{field}' field not found as binary — backfilling all splits with None ...")
        dataset_dict = {}
        for split_name in sorted(all_splits):
            print(f"  Backfilling '{split_name}' ...")
            ds = load_dataset(args.repo_name, split=split_name)
            if field in ds.column_names:
                ds = ds.remove_columns(field)
            new_features = Features({**ds.features, field: Value("binary")})
            dataset_dict[split_name] = ds.map(lambda row: {**row, field: None}, features=new_features)
        print("Pushing all splits together ...")
        DatasetDict(dataset_dict).push_to_hub(args.repo_name)
    else:
        print(f"'{field}' field already exists as binary, skipping backfill.")

    # Step 2: attach real meshes for the target split and push
    print(f"Loading split '{args.split}' ...")
    ds = load_dataset(args.repo_name, split=args.split)

    # Build a lookup from stem to STL path, searching recursively.
    mesh_lookup = {p.stem: p for p in mesh_dir.rglob("*.stl")}

    def attach_mesh(row):
        # Prefer exact file_id.stl, then fall back to file_id_0.stl.
        mesh_path = mesh_lookup.get(str(row["file_id"])) or mesh_lookup.get(f"{row['file_id']}_0")
        if mesh_path is not None:
            row[field] = mesh_path.read_bytes()
        elif not args.partial_replacement:
            row[field] = None
        return row

    if args.partial_replacement:
        replaced = sum(
            1
            for row in ds
            if mesh_lookup.get(str(row["file_id"])) is not None or mesh_lookup.get(f"{row['file_id']}_0") is not None
        )
        print(f"Partially replacing {replaced} mesh(es) out of {len(ds)} rows.")

    ds = ds.map(attach_mesh, writer_batch_size=100)
    print(f"Pushing split '{args.split}' with meshes ...")
    ds.push_to_hub(args.repo_name, split=args.split)
    print("Done.")


if __name__ == "__main__":
    main()
