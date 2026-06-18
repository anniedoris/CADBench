import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import objaverse
import trimesh
from tqdm import tqdm

# --- Configuration ---

BATCH_SIZE         = 100       # Number of meshes to download and process in each batch (adjust based on memory and performance)
DOWNLOAD_PROCESSES = 4         # Parallel download processes for objaverse.load_objects (adjust based on network and CPU)
MAX_DOWNLOAD_RETRIES = 3
RETRY_DELAY_SECONDS  = 5.0
WORKER_TIMEOUT_SECONDS = 120
MIN_VOLUME = 1e-9

CATEGORIES = {
    "furniture-home":      0,
    "science-technology":  1,
    "architecture":        2,
    "cars-vehicles":       3,
    "electronics-gadgets": 4,
    "weapons-military":    5,
}

_NON_MANIFOLD_FILTERS = [
    "meshing_repair_non_manifold_edges",
    "meshing_repair_non_manifold_vertices",
    "meshing_remove_non_manifold_edges",
    "meshing_remove_non_manifold_vertices",
]

# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# --- Repair ---

def make_watertight_glb(input_path, out_repair_path=None):
    """
    Attempt to repair a mesh to watertight using PyMeshLab and export as GLB.

    Steps:
        1. Load the mesh with trimesh.
        2. Export to a temporary PLY (PyMeshLab reads PLY most reliably).
        3. Run a sequence of PyMeshLab cleanup and hole-closing filters.
        4. Reload the repaired mesh and validate: must be watertight with
           positive volume.
        5. Export the validated mesh to out_repair_path as GLB.

    Returns:
        (trimesh.Trimesh, output_path) on success.
        (None, None) if repair failed or mesh did not pass validation.
    """
    base = os.path.splitext(input_path)[0]
    if out_repair_path is None:
        out_repair_path = base + "_repaired.glb"

    if not os.path.isfile(input_path):
        return None, None

    try:
        import pymeshlab

        mesh = trimesh.load(input_path, force="mesh")

        tmp_in_ply  = base + "_tmp_in.ply"
        tmp_out_ply = base + "_tmp_out.ply"

        if not isinstance(mesh, trimesh.Trimesh) and hasattr(mesh, "geometry"):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        mesh.export(tmp_in_ply)

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(tmp_in_ply)

        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_remove_null_faces()

        try: ms.meshing_merge_close_vertices(threshold=1e-6)
        except Exception: pass

        try: ms.meshing_close_holes(maxholesize=1000)
        except Exception: pass

        for fn in _NON_MANIFOLD_FILTERS:
            if hasattr(ms, fn):
                try: getattr(ms, fn)()
                except Exception: pass

        ms.save_current_mesh(tmp_out_ply)
        mesh2 = trimesh.load(tmp_out_ply, force="mesh")

        for p in (tmp_in_ply, tmp_out_ply):
            try: os.remove(p)
            except Exception: pass

        if not mesh2.is_watertight:
            return None, None
        if mesh2.volume <= MIN_VOLUME:
            return None, None

        mesh2.export(out_repair_path)
        return mesh2, out_repair_path

    except ImportError:
        log.error("pymeshlab is not installed.")
        return None, None
    except Exception:
        return None, None

# --- Worker (runs in subprocess) ---

def _worker_target(item, conn):
    """
    Runs in a subprocess. Repairs the mesh and sends (uid, success, msg)
    back to the parent via a Pipe (no shared memory between processes).
    """
    uid, src_path, save_path = item
    try:
        if Path(save_path).exists():
            conn.send((uid, True, "already_exists"))
            return
        mesh, out_path = make_watertight_glb(src_path, out_repair_path=save_path)
        if mesh is None:
            conn.send((uid, False, "repair_failed"))
        else:
            conn.send((uid, True, "saved"))
    except Exception as exc:
        conn.send((uid, False, f"worker_exception: {exc}"))
    finally:
        conn.close()


def _process_one(item):
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    p = multiprocessing.Process(target=_worker_target, args=(item, child_conn))
    p.start()
    child_conn.close()

    finished = parent_conn.poll(WORKER_TIMEOUT_SECONDS)
    if finished:
        try:
            result = parent_conn.recv()
        except Exception as exc:
            result = (item[0], False, f"pipe_error: {exc}")
    else:
        result = (item[0], False, f"timeout: exceeded {WORKER_TIMEOUT_SECONDS}s")

    p.terminate()
    p.join()
    parent_conn.close()
    return result

# --- Download ---

def download_batch(uids):
    if not uids:
        return {}
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            log.info("Downloading %d UIDs (attempt %d/%d) …", len(uids), attempt, MAX_DOWNLOAD_RETRIES)
            result = objaverse.load_objects(uids=list(uids), download_processes=DOWNLOAD_PROCESSES)
            log.info("Downloaded %d / %d objects.", len(result), len(uids))
            return result
        except Exception as exc:
            log.warning("Download attempt %d/%d failed: %s", attempt, MAX_DOWNLOAD_RETRIES, exc)
            if attempt < MAX_DOWNLOAD_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
    return {}

# --- UID Filtering ---

def _filter_chunk(items):
    filtered, mapping = [], {}
    for uid, meta in items:
        for cat in meta.get("categories", []):
            name = cat.get("name", "").lower()
            if name in CATEGORIES:
                filtered.append(uid)
                mapping[uid] = CATEGORIES[name]
                break
    return filtered, mapping


def get_filtered_uids(out_dir, num_workers=4):
    """
    Return (filtered_uids list, uid_to_category dict).

    Results are cached in {out_dir}/filtered_uids.json and
    {out_dir}/uid_to_category.json so the annotation fetch only runs once.
    """
    uids_path = Path(out_dir) / "filtered_uids.json"
    cat_path  = Path(out_dir) / "uid_to_category.json"

    if uids_path.exists() and cat_path.exists():
        log.info("Loading cached filtered UIDs from %s", uids_path)
        with open(uids_path) as f:
            filtered_uids = json.load(f)
        with open(cat_path) as f:
            uid_to_category = json.load(f)
        log.info("Loaded %d filtered UIDs.", len(filtered_uids))
        return filtered_uids, uid_to_category

    log.info("Fetching all Objaverse UIDs …")
    all_uids = objaverse.load_uids()
    log.info("Fetched %d UIDs. Loading annotations …", len(all_uids))
    annotations = objaverse.load_annotations(all_uids)
    log.info("Loaded annotations for %d objects.", len(annotations))

    items = list(annotations.items())
    chunk_size = max(1, len(items) // num_workers)
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    filtered_uids, uid_to_category = [], {}
    with multiprocessing.Pool(processes=num_workers) as pool:
        for f, m in tqdm(pool.imap_unordered(_filter_chunk, chunks),
                         total=len(chunks), desc="Filtering UIDs"):
            filtered_uids.extend(f)
            uid_to_category.update(m)

    log.info("Filtered %d UIDs across %d categories.", len(filtered_uids), len(CATEGORIES))

    with open(uids_path, "w") as f:
        json.dump(filtered_uids, f)
    with open(cat_path, "w") as f:
        json.dump(uid_to_category, f)
    log.info("Saved filtered UIDs to %s", uids_path)

    return filtered_uids, uid_to_category


# --- Resume helpers ---

def _already_saved(save_dir):
    p = Path(save_dir)
    if not p.exists(): return set()
    return {f.stem for f in p.glob("*.glb")}

def _failed_log(save_dir):
    return Path(save_dir) / "failed_uids.txt"

def _load_failed(save_dir):
    log_path = _failed_log(save_dir)
    if not log_path.exists(): return set()
    return set(log_path.read_text().splitlines())

def _append_failed(save_dir, uids):
    if not uids: return
    with _failed_log(save_dir).open("a") as fh:
        fh.write("\n".join(uids) + "\n")

def _cleanup(paths, saved_uids):
    for uid, path in paths.items():
        if uid in saved_uids: continue
        try: Path(path).unlink(missing_ok=True)
        except Exception: pass

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Objaverse watertight mesh repair pipeline")
    parser.add_argument("--uids",           type=str, default=None, help="Path to filtered_uids.json (omit to filter automatically)")
    parser.add_argument("--out",            type=str, required=True, help="Output directory for repaired GLBs")
    parser.add_argument("--skip",           type=int, default=0,    help="Skip first N UIDs (default: 0)")
    parser.add_argument("--end",            type=int, default=None, help="Stop at index N (default: all)")
    parser.add_argument("--workers",        type=int, default=8,    help="Parallel repair workers (default: 8)")
    parser.add_argument("--filter_workers", type=int, default=4,    help="Workers for UID filtering (default: 4)")
    parser.add_argument("--batch",          type=int, default=BATCH_SIZE, help=f"Download batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.uids:
        with open(args.uids) as f:
            all_uids = json.load(f)
    else:
        all_uids, _ = get_filtered_uids(args.out, num_workers=args.filter_workers)
    all_uids = all_uids[args.skip:args.end]

    done   = _already_saved(args.out)
    failed = _load_failed(args.out)
    remaining = [u for u in all_uids if u not in done and u not in failed]

    print(f"Slice: {len(all_uids)}  |  Already saved: {len(done)}  |  Previously failed: {len(failed)}  |  Remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    batches = [remaining[i:i + args.batch] for i in range(0, len(remaining), args.batch)]
    print(f"Workers: {args.workers}  |  Batches: {len(batches)}  |  Timeout: {WORKER_TIMEOUT_SECONDS}s/mesh")

    saved_count = failed_count = 0

    with tqdm(total=len(remaining), desc="Overall", unit="mesh") as overall_bar:
        for batch_idx, batch_uids in enumerate(batches, 1):
            downloaded = download_batch(batch_uids)
            if not downloaded:
                failed_count += len(batch_uids)
                overall_bar.update(len(batch_uids))
                continue

            work_items = [
                (uid, path, str(Path(args.out) / f"{uid}.glb"))
                for uid, path in downloaded.items()
                if Path(path).exists()
            ]

            batch_saved = set()

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(_process_one, item): item for item in work_items}
                with tqdm(total=len(work_items), desc=f"Batch {batch_idx}/{len(batches)}", unit="mesh", leave=False) as batch_bar:
                    for future in as_completed(futures):
                        uid, success, msg = future.result()
                        if success and msg != "already_exists":
                            saved_count += 1
                            batch_saved.add(uid)
                        elif not success:
                            failed_count += 1
                        batch_bar.update(1)
                        overall_bar.update(1)

            batch_failed = {item[0] for item in work_items if item[0] not in batch_saved}
            _append_failed(args.out, batch_failed)
            _cleanup(downloaded, saved_uids=batch_saved)

            print(f"Batch {batch_idx}/{len(batches)} — saved: {len(batch_saved)}  failed: {len(work_items) - len(batch_saved)}")

    print("=" * 40)
    print(f"Saved: {saved_count}  |  Failed: {failed_count}")
    print("=" * 40)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
