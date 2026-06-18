import os
# Set thread limits for linear algebra libraries to avoid over-subscription
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
from concurrent.futures import ProcessPoolExecutor
from cadlib.visualize import vec2CADsolid
from OCC.Extend.DataExchange import write_step_file
from utils import (
    compute_face_count, 
    run_metric_parallel, 
    get_complexity_easy_medium_hard, 
    get_clip_embeddings, 
    get_dino_embeddings, 
    visualize_data_distribution, 
    coverage_diversity
)
import cadquery as cq
from tqdm import tqdm
from PIL import Image, ImageStat

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_gen_testset", action="store_true", default=False)
    parser.add_argument("--no_complexity_count", action="store_true", default=False)
    parser.add_argument("--no_embeddings_compute", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_model", choices=["clip", "dino"], default="dino")
    parser.add_argument("--ops_cat", choices=["abc_sketch_extrude", "abc_other"])
    parser.add_argument("--out_dir_path", default = "./bench1_dir")
    args = parser.parse_args()

    ops_cat = args.ops_cat
    out_dir_path = args.out_dir_path

    os.makedirs(out_dir_path, exist_ok=True)

    OUT_DIR = f"{out_dir_path}/{ops_cat}"
    IMG_OUT_DIR = f"{out_dir_path}/{ops_cat}_images"
    
    _step_files = glob.glob(os.path.join(OUT_DIR, "*.step"))
    _missing_image = []

    for _sf in _step_files:
        _base = os.path.splitext(os.path.basename(_sf))[0]
        _missing_views = [i for i in range(8) if not os.path.exists(os.path.join(IMG_OUT_DIR, f"{_base}_{i}.png"))]

        if _missing_views:
            _missing_image.append(_sf)
            print(f"Missing views {_missing_views} for {_base}, deleting step.")
            os.remove(_sf)
        if _missing_image:
            print(f"Deleted {len(_missing_image)} step(s) with incomplete images.")
        else:
            print("All steps have all 8 view images.")

    # --- 2. Complexity Analysis ---
    COMPLEXITY_PATH = f"{out_dir_path}/{ops_cat}_complexity.txt"
    # Re-fetch the list of steps that survived the cleanup
    step_files = sorted(glob.glob(os.path.join(OUT_DIR, "*.step")))
    
    if not args.no_complexity_count:
        print(f"Computing face counts for {len(step_files)} files...")
        results = run_metric_parallel(compute_face_count, step_files, save_path=COMPLEXITY_PATH, num_workers=args.num_workers)
    else:
        with open(COMPLEXITY_PATH, "rb") as f:
            results = pickle.load(f)

    easy, medium, hard, t1, t2 = get_complexity_easy_medium_hard(results)

    print(f"Thresholds: t1={t1:.1f}, t2={t2:.1f}")
    print(f"Easy   (≤{t1:.1f} faces):          {len(easy)}")
    print(f"Medium ({t1:.1f}–{t2:.1f} faces): {len(medium)}")
    print(f"Hard   (>{t2:.1f} faces):          {len(hard)}")

    # --- 3. Embedding Computation ---
    split_data = {}
    for split_name, split_items in [("easy", easy), ("medium", medium), ("hard", hard)]:
        save_dir = f"{out_dir_path}/{ops_cat}_embeddings"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{split_name}_{args.embedding_model}_embeddings.pkl")
        
        if args.no_embeddings_compute:
            with open(save_path, "rb") as f:
                emb_data = pickle.load(f)
            image_paths = emb_data["paths"]
            embeddings = emb_data["embeddings"]
            print(f"Loaded {args.embedding_model.upper()} embeddings for {split_name}")
        else:
            print(f"\nGathering images for {split_name} split...")
            image_paths = []    
            all_view_paths = [] 
            for fname, _ in split_items:
                base_name = os.path.splitext(os.path.basename(fname))[0]
                # Double check existence (safety)
                img_zero = os.path.join(IMG_OUT_DIR, base_name + "_0.png")
                if not os.path.exists(img_zero):
                    continue
                image_paths.append(img_zero)
                all_view_paths.extend([os.path.join(IMG_OUT_DIR, f"{base_name}_{i}.png") for i in range(8)])

            print(f"Found {len(image_paths)} steps ({len(all_view_paths)} total views)")
            if args.embedding_model == "dino":
                all_embs = get_dino_embeddings(all_view_paths, batch_size=args.batch_size)
            else:
                all_embs = get_clip_embeddings(all_view_paths, batch_size=args.batch_size)

            # Average the 8 per-view embeddings into one embedding per step
            embeddings = all_embs.reshape(len(image_paths), 8, -1).mean(axis=1)

            with open(save_path, "wb") as f:
                pickle.dump({"paths": image_paths, "embeddings": embeddings}, f)

        split_data[split_name] = {"paths": image_paths, "embeddings": embeddings}

    # --- 4. Diverse Selection & Visualization ---
    vis_dir = f"{out_dir_path}/{ops_cat}_embeddings/visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    for split_name in ["easy", "medium", "hard"]:
        img_paths  = split_data[split_name]["paths"]
        embeddings = split_data[split_name]["embeddings"]
        labels     = [os.path.basename(p) for p in img_paths]
        
        # Select 1000 most diverse samples
        diverse_indices, cluster_labels = coverage_diversity(embeddings, k=1000)
        split_data[split_name]["diverse_indices"] = diverse_indices
        
        visualize_data_distribution(
            embeddings, labels=labels, perplexity=30, num_iterations=1000,
            savename=os.path.join(vis_dir, f"{args.embedding_model}_{split_name}_cluster_embeddings.html"),
            image_paths=img_paths, highlight_indices=diverse_indices,
            cluster_labels=cluster_labels,
            hover_text=[f"Cluster {cl}" for cl in cluster_labels],
        )

    # --- 5. Export Final Dataset ---
    for split_name in ["easy", "medium", "hard"]:
        img_paths       = split_data[split_name]["paths"]
        diverse_indices = split_data[split_name]["diverse_indices"]
        
        base_bench_dir = f"{out_dir_path}/data/bench1/{ops_cat}/{split_name}"
        steps_dir  = os.path.join(base_bench_dir, "steps")
        images_dir = os.path.join(base_bench_dir, "images")
        
        for d in [steps_dir, images_dir]:
            if os.path.exists(d): shutil.rmtree(d)
            os.makedirs(d)

        copied_steps = 0
        for idx in diverse_indices:
            # Clean base name (removing view suffix)
            base_name = os.path.splitext(os.path.basename(img_paths[idx]))[0].removesuffix("_0")
            src_step = os.path.join(OUT_DIR, base_name + ".step")
            src_img = img_paths[idx]
            
            if os.path.exists(src_step):
                shutil.copy2(src_step, os.path.join(steps_dir, base_name + ".step"))
                shutil.copy2(src_img, os.path.join(images_dir, os.path.basename(src_img)))
                copied_steps += 1

        print(f"Split {split_name}: Finalized {copied_steps} samples in {base_bench_dir}")

if __name__ == "__main__":
    main()

