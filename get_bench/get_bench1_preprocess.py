import argparse
import os
import glob
import re
import shutil
import multiprocessing as mp
from tqdm import tqdm

# --- Configuration ---
STRICT_BASE = {'origin', 'defaultPlane', 'newSketch', 'extrude'}
OTHER_BASE = {'origin', 'defaultPlane', 'newSketch'}
ALLOWED_EXTRAS = {'revolve', 'fillet', 'chamfer', 'loft', 'sweep', 'externalThreads', 'internalThreads'}

FORBIDDEN_SKETCH_EXTRUDE = [
    'SPHERICAL_SURFACE', 'CONICAL_SURFACE', 'TOROIDAL_SURFACE', 
    'B_SPLINE_SURFACE', 'SURFACE_OF_REVOLUTION', 'SWEPT_SURFACE', 
    'OFFSET_SURFACE', 'CURVE_BOUNDED_SURFACE'
]

FORBIDDEN_GLOBAL = [
    'POLYLINE', 'POLY_LOOP', 'VERTEX_LOOP', 'FACETED_BREP', 
    'TRIANGULATED_FACE', 'SHELL_BASED_SURFACE_MODEL', 
    'OPEN_SHELL', 'GEOMETRIC_CURVE_SET', 'SURFACE_CURVE',
    'BOUNDED_SURFACE', 'COMPOUND_SURFACE',
    'DRAUGHTING_PRE_DEFINED_CURVE_FONT', 'MAPPED_ITEM', 
    'CARTESIAN_TRANSFORMATION_OPERATOR', 'BREP_WITH_VOIDS'
]

# --- Functions ---

def parse_step_structure(content):
    content = content.replace('\n', '').replace('\r', '')
    raw_lines = content.split(';')
    entity_map = {}
    for line in raw_lines:
        if '=' in line and line.strip().startswith('#'):
            try:
                parts = line.split('=', 1)
                eid = parts[0].strip().replace('#', '')
                definition = parts[1].strip()
                type_match = re.match(r'([A-Z0-9_]+)', definition)
                etype = type_match.group(1) if type_match else "UNKNOWN"
                refs = re.findall(r'#(\d+)', definition)
                entity_map[eid] = {'type': etype, 'refs': set(refs)}
            except: continue
    return entity_map

def check_geometry_and_topology(step_path):
    try:
        with open(step_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if any(ent in content for ent in FORBIDDEN_GLOBAL):
            return False, False
        
        if content.count('MANIFOLD_SOLID_BREP') != 1: 
            return False, False

        graph = parse_step_structure(content)
        solid_roots = [eid for eid, d in graph.items() if d['type'] == 'MANIFOLD_SOLID_BREP']
        
        # Dependency Walk
        visited, stack = set(), list(solid_roots)
        while stack:
            curr = stack.pop()
            if curr in graph and curr not in visited:
                visited.add(curr)
                stack.extend(graph[curr]['refs'] - visited)
        
        # Audit all geometric surfaces in the file
        geometric_types = {
            'ADVANCED_FACE', 'FACE_SURFACE', 'B_SPLINE_SURFACE', 
            'RECTANGULAR_TRIMMED_SURFACE', 'PLANE', 'CYLINDRICAL_SURFACE'
        }
        all_geo_ids = {eid for eid, d in graph.items() if d['type'] in geometric_types}
        
        # Reject if there is any geometry not linked to the Solid
        if not all_geo_ids.issubset(visited):
            return False, False

        has_forbidden_sketch = any(surf in content for surf in FORBIDDEN_SKETCH_EXTRUDE)
        return True, has_forbidden_sketch
    except:
        return False, False

def get_yml_features(yml_path):
    features = set()
    try:
        with open(yml_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "featureType" in line and ":" in line:
                    val = line.split(":", 1)[1].strip().strip("'\"")
                    if val: features.add(val)
    except: pass
    return features

def worker_init(v_pc, v_mesh):
    global PC_CACHE, MESH_CACHE
    PC_CACHE = v_pc
    MESH_CACHE = v_mesh

def classify_part(args):
    step_path, yml_dir = args
    step_base = os.path.splitext(os.path.basename(step_path))[0]
    step_prefix = step_base.split('_')[0]

    if step_prefix not in PC_CACHE or step_prefix not in MESH_CACHE:
        return step_path, None

    is_valid, has_forbidden_sketch = check_geometry_and_topology(step_path)
    if not is_valid:
        return step_path, None

    target_dir = os.path.join(yml_dir, step_prefix)
    if not os.path.exists(target_dir): return step_path, None

    yml_path = None
    for f in os.listdir(target_dir):
        if f.startswith(step_prefix) and f.endswith('.yml'):
            yml_path = os.path.join(target_dir, f)
            break
    if not yml_path: return step_path, None

    found_ops = get_yml_features(yml_path)

    if found_ops == STRICT_BASE and not has_forbidden_sketch:
        return step_path, "abc_sketch_extrude"

    total_allowed = OTHER_BASE.union(ALLOWED_EXTRAS).union({'extrude'})
    if OTHER_BASE.issubset(found_ops) and found_ops.issubset(total_allowed):
        return step_path, "abc_other"

    return step_path, None

# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_dir', type=str, required=True)
    parser.add_argument('--yml_dir', type=str, required=True)
    parser.add_argument('--pc_dir', type=str, required=True)
    parser.add_argument('--mesh_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=128)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    def get_prefixes(directory, ext):
        if not os.path.exists(directory): return set()
        return {f.split('_')[0] for f in os.listdir(directory) if f.endswith(ext)}

    print("Building ID prefix cache...")
    v_pc = get_prefixes(args.pc_dir, '.npy')
    v_mesh = get_prefixes(args.mesh_dir, '.stl')

    out_extrude = os.path.join(args.output_dir, "abc_sketch_extrude")
    out_other = os.path.join(args.output_dir, "abc_other")
    
    if not args.dry_run:
        os.makedirs(out_extrude, exist_ok=True)
        os.makedirs(out_other, exist_ok=True)

    step_files = glob.glob(os.path.join(args.step_dir, "*.step"))
    tasks = [(f, args.yml_dir) for f in step_files]
    
    count_e, count_o = 0, 0

    print(f"Mode: {'DRY RUN' if args.dry_run else 'ACTIVE COPY'}")
    print(f"Workers: {args.num_workers} | Total files to scan: {len(step_files)}")

    with mp.Pool(processes=args.num_workers, initializer=worker_init, initargs=(v_pc, v_mesh)) as pool:
        with tqdm(total=len(step_files), desc="Categorizing") as pbar:
            for path, category in pool.imap_unordered(classify_part, tasks):
                if category == "abc_sketch_extrude":
                    count_e += 1
                    if not args.dry_run: shutil.copy2(path, out_extrude)
                elif category == "abc_other":
                    count_o += 1
                    if not args.dry_run: shutil.copy2(path, out_other)
                
                pbar.set_postfix(sk_ext=count_e, other=count_o)
                pbar.update(1)

    print(f"\n" + "="*40)
    print(f"FINAL CLASSIFICATION RESULTS")
    print(f"="*40)
    print(f"ABC Sketch-Extrude: {count_e}")
    print(f"ABC Other:          {count_o}")
    print(f"Total Categorized:  {count_e + count_o}")
    print(f"Total Scanned:      {len(step_files)}")
    print(f"="*40)

if __name__ == "__main__":
    main()