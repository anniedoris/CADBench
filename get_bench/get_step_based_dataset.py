import os

from numpy import sort
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import glob
import hashlib
import json
import pickle
import shutil
import h5py
from cadlib.visualize import vec2CADsolid
from OCC.Extend.DataExchange import write_step_file
from utils import compute_face_count, run_metric_parallel, get_complexity_easy_medium_hard, get_clip_embeddings, get_dino_embeddings, visualize_data_distribution, coverage_diversity, clean_directory
import cadquery as cq
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_step_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_clean_dataset", action="store_true", default=False)
    parser.add_argument("--no_complexity_count", action="store_true", default=False)
    parser.add_argument("--no_embeddings_compute", action="store_true", default=False)
    parser.add_argument("--visualizations_only", action="store_true", default=False)
    args = parser.parse_args()
    
    # Set up paths for saving things
    ORIG_STEP_DIR = args.input_step_dir
    OUTPUT_DIR = args.output_dir + '/steps'
    OUTPUT_DIR_UTILS = args.output_dir + '_utils'
    IMG_DIR = args.image_dir
    CLEANED_STEP_DIR = f"{args.input_step_dir}_cleaned"
    COMPLEXITY_PATH = f"{OUTPUT_DIR_UTILS}/complexity_count.txt"
    os.makedirs(OUTPUT_DIR_UTILS, exist_ok=True)
    
    # If we are only visualizing, skip everything else
    if args.visualizations_only:
        for split in ["easy", "medium", "hard"]:
            emb_path = f"{OUTPUT_DIR_UTILS}/{split}_dino_embeddings.pkl"
            sampling_path = f"{OUTPUT_DIR_UTILS}/{split}_diverse_sampling.pkl"
            if not os.path.exists(emb_path) or not os.path.exists(sampling_path):
                print(f"Missing embeddings or sampling data for {split} split, cannot run visualizations.")
                return
            with open(emb_path, "rb") as f:
                emb_data = pickle.load(f)
                image_paths = emb_data["paths"]
                labels = [os.path.basename(p) for p in image_paths]
                embeddings = emb_data["embeddings"]
            with open(sampling_path, "rb") as f:
                sampling_data = pickle.load(f)
                diverse_indices = sampling_data["diverse_indices"]
                cluster_labels = sampling_data["cluster_labels"]
            
            # Generates visuals that shows clusters and selected samples
            visualize_data_distribution(
                embeddings, labels=labels, perplexity=30, num_iterations=1000,
                savename=f"{OUTPUT_DIR_UTILS}/dino_{split}_cluster_embeddings.html",
                image_paths=image_paths, highlight_indices=diverse_indices,
                cluster_labels=cluster_labels,
                hover_text=[f"Cluster {cl}" for cl in cluster_labels]
            )
            
            # Generates visuals that shows selected vs. not selected samples
            visualize_data_distribution(
                embeddings, labels=labels, perplexity=30, num_iterations=1000,
                savename=f"{OUTPUT_DIR_UTILS}/dino_{split}_embeddings.html",
                image_paths=image_paths, highlight_indices=diverse_indices,
            )  
        return
    
    # Clean the step files in the dataset
    if not args.no_clean_dataset:
        # Clean and create directory for cleaned steps
        clean_directory(CLEANED_STEP_DIR)
        
        # Check if views are missing for any step file, and if so, don't copy to the new directory
        step_files = glob.glob(os.path.join(ORIG_STEP_DIR, "*.step"))
        missing_image = []
        for step_file in step_files:
            base = os.path.splitext(os.path.basename(step_file))[0]
            missing_views = [i for i in range(8) if not os.path.exists(os.path.join(IMG_DIR, f"{base}_{i}.png"))]
            if missing_views:
                missing_image.append(step_file)
            else:
                shutil.copy(step_file, os.path.join(CLEANED_STEP_DIR, os.path.basename(step_file)))
        if missing_image:
            print(f"Deleted {len(missing_image)} step(s) with incomplete images.")
        else:
            cleaned_count = len(glob.glob(os.path.join(CLEANED_STEP_DIR, "*.step")))
            print(f"All {len(step_files)} steps have all 8 view images, copied {cleaned_count} steps to cleaned directory.")
    
    # Get the complexity splits
    if not args.no_complexity_count:
        step_files = sorted(glob.glob(os.path.join(CLEANED_STEP_DIR, "*.step")))
        results = run_metric_parallel(compute_face_count, step_files, save_path=COMPLEXITY_PATH, num_workers=args.num_workers)
    else:
        with open(COMPLEXITY_PATH, "rb") as f:
            results = pickle.load(f)
    easy, medium, hard, t1, t2 = get_complexity_easy_medium_hard(results)
    
    print(f"Thresholds: t1={t1:.1f}, t2={t2:.1f}")
    print(f"Easy   (≤{t1:.1f} faces):          {len(easy)}")
    print(f"Medium ({t1:.1f}–{t2:.1f} faces): {len(medium)}")
    print(f"Hard   (>{t2:.1f} faces):          {len(hard)}")
    
    # Get DINO embeddings for all the steps
    # Compute embeddings for easy, medium, and hard splits
    split_data = {}
    for split_name, split_items in [("easy", easy), ("medium", medium), ("hard", hard)]:
        save_path = f"{OUTPUT_DIR_UTILS}/{split_name}_dino_embeddings.pkl"
        if args.no_embeddings_compute:
            with open(save_path, "rb") as f:
                emb_data = pickle.load(f)
            image_paths = emb_data["paths"]
            embeddings = emb_data["embeddings"]
            print(f"Loaded DINO embeddings for {split_name} split from {save_path}")
            print(f"  {len(image_paths)} images, embedding dim {embeddings.shape[1]}")
        else:
            print(f"\nGathering images for {split_name} split...")
            # All steps are guaranteed to have all 8 views (checked above)
            image_paths = []    # one _0.png per step (representative)
            all_view_paths = [] # flat list: 8 views per step, in order
            for fname, _ in split_items:
                base_name = os.path.splitext(os.path.basename(fname))[0]
                if not os.path.exists(os.path.join(IMG_DIR, base_name + "_0.png")):
                    print(f"Warning: step {base_name} not in image dir, skipping.")
                    continue
                image_paths.append(os.path.join(IMG_DIR, base_name + "_0.png"))
                all_view_paths.extend([os.path.join(IMG_DIR, f"{base_name}_{i}.png") for i in range(8)])

            print(f"Found {len(image_paths)}/{len(split_items)} steps ({len(all_view_paths)} total views)")
            all_embs = get_dino_embeddings(all_view_paths, batch_size=args.batch_size)
            embeddings = all_embs.reshape(len(image_paths), 8, -1).mean(axis=1) # Average the 8 views per step

            with open(save_path, "wb") as f:
                pickle.dump({"paths": image_paths, "embeddings": embeddings}, f)
            print(f"Saved DINO embeddings for {split_name} split to {save_path}")

        split_data[split_name] = {"paths": image_paths, "embeddings": embeddings}
        
    # Do diversity sampling
    for split in split_data:
        
        embeddings = split_data[split]["embeddings"]
        image_paths = split_data[split]["paths"]
        
        print("Starting diversity sampling with k=1000 clusters...")
        diverse_indices, cluster_labels = coverage_diversity(embeddings, k=1000)
        print(f"Selected {len(diverse_indices)} samples from {len(embeddings)} parts")
        
        samples_save_path = f"{OUTPUT_DIR_UTILS}/{split}_diverse_sampling.pkl"
        with open(samples_save_path, "wb") as f:
            pickle.dump({"diverse_indices": diverse_indices, "cluster_labels": cluster_labels}, f)
        print(f"Saved sampling results to {samples_save_path}")
        
        split_data[split]["diverse_indices"] = diverse_indices
        split_data[split]["cluster_labels"] = cluster_labels
    
    for split in split_data:
        diverse_indices = split_data[split]["diverse_indices"]
        image_paths = split_data[split]["paths"]
        
        clean_directory(f"{OUTPUT_DIR}/{split}")
        
        for idx in diverse_indices:
            base_name = os.path.splitext(os.path.basename(image_paths[idx]))[0].removesuffix("_0")
            step_file = os.path.join(CLEANED_STEP_DIR, f"{base_name}.step")
            shutil.copy(step_file, os.path.join(OUTPUT_DIR, split, f"{base_name}.step"))
        print(f"Copied {len(diverse_indices)} diverse samples for {split} split to {OUTPUT_DIR}/{split}")
        print(f"Successfully saved {len(glob.glob(os.path.join(OUTPUT_DIR, split, '*.step')))} for {split} split.")
    

if __name__ == "__main__":
    main()
