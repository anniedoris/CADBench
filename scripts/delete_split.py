import argparse
import json
import re
import tempfile
from huggingface_hub import HfApi


def remove_split_from_readme(content: str, split: str) -> str:
    """Remove YAML config entries referencing the split from a dataset README."""
    # Remove lines like '  - split: bench0' and the following 'path:' line
    # HF dataset card YAML splits look like:
    #   - split: bench0
    #     path: data/bench0-*
    lines = content.split("\n")
    out = []
    skip_next = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if skip_next > 0:
            skip_next -= 1
            i += 1
            continue
        # Match a split entry line
        if re.match(rf"^\s*-\s+split:\s+{re.escape(split)}\s*$", line):
            # Also skip the following 'path:' line if present
            if i + 1 < len(lines) and re.match(r"^\s+path:", lines[i + 1]):
                skip_next = 1
            i += 1
            continue
        # Match a path line that references the split directly (e.g. path: data/bench0-*)
        if re.match(rf"^\s+path:\s+data/{re.escape(split)}[-.]", line):
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Delete a split from a Hugging Face dataset repo.")
    parser.add_argument("--repo_id", required=True, help="HF dataset repo ID (e.g. username/my-dataset).")
    parser.add_argument("--split", required=True, help="Split name to delete (e.g. train, test, bench0).")
    args = parser.parse_args()

    print(f"Repo:  {args.repo_id}")
    print(f"Split: {args.split}")
    answer = input(f"\nAre you sure you want to delete split '{args.split}' from '{args.repo_id}'? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    api = HfApi()
    files = list(api.list_repo_files(args.repo_id, repo_type="dataset"))

    # --- Delete parquet files ---
    to_delete = [f for f in files if f.startswith(f"data/{args.split}-") or f == f"data/{args.split}.parquet"]

    if not to_delete:
        print(f"No parquet files found for split '{args.split}'.")
    else:
        print(f"Deleting {len(to_delete)} parquet file(s):")
        for f in to_delete:
            print(f"  {f}")
        api.delete_files(
            repo_id=args.repo_id,
            repo_type="dataset",
            delete_patterns=to_delete,
        )
        print("Parquet files deleted.")

    # --- Clean up dataset_infos.json ---
    if "dataset_infos.json" in files:
        print("Checking dataset_infos.json ...")
        with tempfile.TemporaryDirectory() as tmp:
            local_path = api.hf_hub_download(
                repo_id=args.repo_id,
                filename="dataset_infos.json",
                repo_type="dataset",
                local_dir=tmp,
            )
            with open(local_path) as f:
                infos = json.load(f)
            if args.split in infos:
                del infos[args.split]
                updated = json.dumps(infos, indent=2)
                api.upload_file(
                    path_or_fileobj=updated.encode(),
                    path_in_repo="dataset_infos.json",
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    commit_message=f"Remove split '{args.split}' from dataset_infos.json",
                )
                print(f"  Removed '{args.split}' from dataset_infos.json.")
            else:
                print(f"  '{args.split}' not found in dataset_infos.json, skipping.")

    # --- Clean up README.md ---
    if "README.md" in files:
        print("Checking README.md ...")
        with tempfile.TemporaryDirectory() as tmp:
            local_path = api.hf_hub_download(
                repo_id=args.repo_id,
                filename="README.md",
                repo_type="dataset",
                local_dir=tmp,
            )
            with open(local_path) as f:
                readme = f.read()
            if args.split in readme:
                updated = remove_split_from_readme(readme, args.split)
                if updated != readme:
                    api.upload_file(
                        path_or_fileobj=updated.encode(),
                        path_in_repo="README.md",
                        repo_id=args.repo_id,
                        repo_type="dataset",
                        commit_message=f"Remove split '{args.split}' from README.md",
                    )
                    print(f"  Removed '{args.split}' references from README.md.")
                else:
                    print(f"  '{args.split}' found in README.md but no YAML entries matched — check manually.")
            else:
                print(f"  '{args.split}' not found in README.md, skipping.")

    print("Done.")


if __name__ == "__main__":
    main()
