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
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval_models.IOUProcessors import IOUProcessorBench

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--abc_steps_dir", type=str, required=True)
    parser.add_argument("--deepcad_steps_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()
    
    # Set up paths for saving things
    OUTPUT_DIR_UTILS = 'data/bench0_utils'
    COMPLEXITY_PATH_ABC = f"{OUTPUT_DIR_UTILS}/complexity_count_abc.txt"
    COMPLEXITY_PATH_DEEPCAD = f"{OUTPUT_DIR_UTILS}/complexity_count_deepcad.txt"
    CLEANED_DIR = f"data/{args.deepcad_steps_dir.split('/')[-1]}_cleaned"
    os.makedirs(OUTPUT_DIR_UTILS, exist_ok=True)
    
    # # Get complexity of abc and deepcad steps by counting faces, and compare distributions
    # step_files = sorted(glob.glob(os.path.join(args.abc_steps_dir, "*.step")))
    # results = run_metric_parallel(compute_face_count, step_files, save_path=COMPLEXITY_PATH_ABC, num_workers=args.num_workers)

    # step_files = sorted(glob.glob(os.path.join(args.deepcad_steps_dir, "*.step")))
    # results = run_metric_parallel(compute_face_count, step_files, save_path=COMPLEXITY_PATH_DEEPCAD, num_workers=args.num_workers)

    # with open(COMPLEXITY_PATH_ABC, "rb") as f:
    #     complexity_data_abc = pickle.load(f)
        
    # with open(COMPLEXITY_PATH_DEEPCAD, "rb") as f:
    #     complexity_data_deepcad = pickle.load(f)

    # deepcad_results = {}
    # for sample in complexity_data_deepcad:
    #     deepcad_id = os.path.basename(sample[0]).split(".")[0]
    #     deepcad_results[deepcad_id] = sample[1]

    # match = 0
    # abc_more = 0
    # abc_fewer = 0
    # not_found = 0
    # match_ids = []
    # fewer_ids = []
    # for sample in complexity_data_abc:
    #     part_id = os.path.basename(sample[0]).split("_")[0]
    #     abc_face_count = sample[1]
    #     deepcad_face_count = deepcad_results.get(part_id)
    #     if deepcad_face_count is None:
    #         not_found += 1
    #     elif abc_face_count == deepcad_face_count:
    #         match += 1
    #         match_ids.append(part_id)
    #     elif abc_face_count > deepcad_face_count:
    #         abc_more += 1
    #     else:
    #         abc_fewer += 1
    #         fewer_ids.append(part_id)
    # print(f"Same face count: {match}, ABC more faces: {abc_more}, ABC fewer faces: {abc_fewer}, DeepCAD not found: {not_found}")

    # # Copy files with the same face count to data/abc_deepcad_same_facecount_steps for further analysis
    # if os.path.exists(ABC_DEEPCAD_SAME_FACECOUNT_DIR):
    #     shutil.rmtree(ABC_DEEPCAD_SAME_FACECOUNT_DIR)
    # os.makedirs(ABC_DEEPCAD_SAME_FACECOUNT_DIR)
    # for sample in complexity_data_abc:
    #     part_id = os.path.basename(sample[0]).split("_")[0]
    #     if part_id in match_ids:
    #         shutil.copy2(sample[0], os.path.join(ABC_DEEPCAD_SAME_FACECOUNT_DIR, os.path.basename(sample[0])))
    # print(f"Copied {len(match_ids)} files to {ABC_DEEPCAD_SAME_FACECOUNT_DIR}")
    
    # Copy single-body steps to CLEANED_DIR
    if os.path.exists(CLEANED_DIR):
        shutil.rmtree(CLEANED_DIR)
    os.makedirs(CLEANED_DIR)
    copied = 0
    skipped = 0
    step_files = glob.glob(os.path.join(args.deepcad_steps_dir, "*.step"))
    for step_file in tqdm(step_files, desc="Filtering multi-body"):
        try:
            cq_model = cq.importers.importStep(step_file)
            if len(cq_model.solids().vals()) == 1:
                shutil.copy2(step_file, os.path.join(CLEANED_DIR, os.path.basename(step_file)))
                copied += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  Error processing {step_file}: {e}")
            skipped += 1
    print(f"Copied {copied} single-body files to {CLEANED_DIR}, skipped {skipped} multi-body/invalid files.")
    
if __name__ == "__main__":
    main()
