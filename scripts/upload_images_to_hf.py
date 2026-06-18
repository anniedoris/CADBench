import argparse

from huggingface_hub import HfApi


SPLITS = ["bench0", "bench1A", "bench1B"]


def main():
    parser = argparse.ArgumentParser(description="Upload rendered images to HuggingFace dataset repo.")
    parser.add_argument("--image_dir",  required=True, help="Local folder of images to upload")
    parser.add_argument("--split",      required=True, choices=SPLITS, help="Benchmark split (bench0, bench1A, bench1B)")
    parser.add_argument("--repo",       default="DeCoDELab/CADBench", help="HuggingFace dataset repo (default: DeCoDELab/CADBench)")
    parser.add_argument("--token",      default=None, help="HuggingFace API token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    path_in_repo = f"gray_images/{args.split}"

    print(f"Uploading {args.image_dir} → {args.repo}/{path_in_repo} ...")
    api = HfApi(token=args.token)
    api.upload_folder(
        folder_path=args.image_dir,
        repo_id=args.repo,
        repo_type="dataset",
        path_in_repo=path_in_repo,
    )
    print("Done.")


if __name__ == "__main__":
    main()
