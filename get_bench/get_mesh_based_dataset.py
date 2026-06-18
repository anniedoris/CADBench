import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import glob
import pickle
import shutil
from utils import get_dino_embeddings, visualize_data_distribution, coverage_diversity, clean_directory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mesh_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_clean_dataset", action="store_true", default=False)
    parser.add_argument("--no_embeddings_compute", action="store_true", default=False)
    parser.add_argument("--visualizations_only", action="store_true", default=False)
    args = parser.parse_args()

    # Set up paths for saving things
    ORIG_MESH_DIR = args.input_mesh_dir
    OUTPUT_DIR = args.output_dir + '/meshes'
    OUTPUT_DIR_UTILS = args.output_dir + '_utils'
    IMG_DIR = args.image_dir
    CLEANED_MESH_DIR = f"{args.input_mesh_dir}_cleaned"
    EMB_PATH = f"{OUTPUT_DIR_UTILS}/all_dino_embeddings.pkl"
    SAMPLING_PATH = f"{OUTPUT_DIR_UTILS}/all_diverse_sampling.pkl"
    os.makedirs(OUTPUT_DIR_UTILS, exist_ok=True)

    # If we are only visualizing, skip everything else
    if args.visualizations_only:
        if not os.path.exists(EMB_PATH) or not os.path.exists(SAMPLING_PATH):
            print(f"Missing embeddings or sampling data, cannot run visualizations.")
            return
        with open(EMB_PATH, "rb") as f:
            emb_data = pickle.load(f)
        image_paths = emb_data["paths"]
        labels = [os.path.basename(p) for p in image_paths]
        embeddings = emb_data["embeddings"]
        with open(SAMPLING_PATH, "rb") as f:
            sampling_data = pickle.load(f)
        diverse_indices = sampling_data["diverse_indices"]
        cluster_labels = sampling_data["cluster_labels"]

        visualize_data_distribution(
            embeddings, labels=labels, perplexity=30, num_iterations=1000,
            savename=f"{OUTPUT_DIR_UTILS}/dino_all_cluster_embeddings.html",
            image_paths=image_paths, highlight_indices=diverse_indices,
            cluster_labels=cluster_labels,
            hover_text=[f"Cluster {cl}" for cl in cluster_labels]
        )
        visualize_data_distribution(
            embeddings, labels=labels, perplexity=30, num_iterations=1000,
            savename=f"{OUTPUT_DIR_UTILS}/dino_all_embeddings.html",
            image_paths=image_paths, highlight_indices=diverse_indices,
        )
        return

    # Clean the mesh files in the dataset
    if not args.no_clean_dataset:
        clean_directory(CLEANED_MESH_DIR)

        mesh_files = glob.glob(os.path.join(ORIG_MESH_DIR, "**/*.obj"), recursive=True) + \
                     glob.glob(os.path.join(ORIG_MESH_DIR, "**/*.glb"), recursive=True)
        print(f"Found {len(mesh_files)} mesh files in {ORIG_MESH_DIR}")
        missing_image = []
        for mesh_file in mesh_files:
            base = os.path.splitext(os.path.basename(mesh_file))[0]
            missing_views = [i for i in range(8) if not os.path.exists(os.path.join(IMG_DIR, f"{base}_{i}.png"))]
            if missing_views:
                missing_image.append(mesh_file)
            else:
                shutil.copy(mesh_file, os.path.join(CLEANED_MESH_DIR, os.path.basename(mesh_file)))
        if missing_image:
            print(f"Deleted {len(missing_image)} meshes with incomplete images.")
        else:
            cleaned_count = len(glob.glob(os.path.join(CLEANED_MESH_DIR, "*.obj"))) + \
                            len(glob.glob(os.path.join(CLEANED_MESH_DIR, "*.glb")))
            print(f"All {len(mesh_files)} meshes have all 8 view images, copied {cleaned_count} objs to cleaned directory.")

    # Get DINO embeddings for all samples in CLEANED_MESH_DIR
    if args.no_embeddings_compute:
        with open(EMB_PATH, "rb") as f:
            emb_data = pickle.load(f)
        image_paths = emb_data["paths"]
        embeddings = emb_data["embeddings"]
        print(f"Loaded DINO embeddings from {EMB_PATH}")
        print(f"  {len(image_paths)} images, embedding dim {embeddings.shape[1]}")
    else:
        mesh_files = sorted(glob.glob(os.path.join(CLEANED_MESH_DIR, "*.obj")) +
                            glob.glob(os.path.join(CLEANED_MESH_DIR, "*.glb")))
        print(f"\nGathering images for {len(mesh_files)} meshes in {CLEANED_MESH_DIR}...")
        image_paths = []    # one _0.png per mesh (representative)
        all_view_paths = [] # flat list: 8 views per mesh, in order
        for mesh_file in mesh_files:
            base_name = os.path.splitext(os.path.basename(mesh_file))[0]
            if not os.path.exists(os.path.join(IMG_DIR, base_name + "_0.png")):
                print(f"Warning: {base_name} not in image dir, skipping.")
                continue
            image_paths.append(os.path.join(IMG_DIR, base_name + "_0.png"))
            all_view_paths.extend([os.path.join(IMG_DIR, f"{base_name}_{i}.png") for i in range(8)])

        print(f"Found {len(image_paths)}/{len(mesh_files)} meshes with images ({len(all_view_paths)} total views)")
        all_embs = get_dino_embeddings(all_view_paths, batch_size=args.batch_size)
        embeddings = all_embs.reshape(len(image_paths), 8, -1).mean(axis=1)  # Average the 8 views per mesh

        with open(EMB_PATH, "wb") as f:
            pickle.dump({"paths": image_paths, "embeddings": embeddings}, f)
        print(f"Saved DINO embeddings to {EMB_PATH}")

    # Do diversity sampling
    print("Starting diversity sampling with k=3000 clusters...")
    diverse_indices, cluster_labels = coverage_diversity(embeddings, k=3000)
    print(f"Selected {len(diverse_indices)} samples from {len(embeddings)} meshes")

    with open(SAMPLING_PATH, "wb") as f:
        pickle.dump({"diverse_indices": diverse_indices, "cluster_labels": cluster_labels}, f)
    print(f"Saved sampling results to {SAMPLING_PATH}")

    clean_directory(OUTPUT_DIR)
    for idx in diverse_indices:
        base_name = os.path.splitext(os.path.basename(image_paths[idx]))[0].removesuffix("_0")
        for ext in (".obj", ".glb"):
            mesh_file = os.path.join(CLEANED_MESH_DIR, f"{base_name}{ext}")
            if os.path.exists(mesh_file):
                shutil.copy(mesh_file, os.path.join(OUTPUT_DIR, f"{base_name}{ext}"))
                break
    print(f"Copied {len(diverse_indices)} diverse samples to {OUTPUT_DIR}")
    print(f"Successfully saved {len(glob.glob(os.path.join(OUTPUT_DIR, '*.obj'))) + len(glob.glob(os.path.join(OUTPUT_DIR, '*.glb')))} meshes.")


if __name__ == "__main__":
    main()
