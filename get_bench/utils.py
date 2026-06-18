import sys
import csv
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue
from tqdm import tqdm
import os
import pickle
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import trimesh
import shutil

TARGETS = [
    'CLOSED_SHELL', 'ADVANCED_FACE', 'PLANE', 'CYLINDRICAL_SURFACE',
    'SPHERICAL_SURFACE', 'CONICAL_SURFACE', 'TOROIDAL_SURFACE',
    'B_SPLINE_SURFACE', 'SURFACE_OF_REVOLUTION', 'SURFACE_OF_LINEAR_EXTRUSION',
    'RECTANGULAR_COMPOSITE_SURFACE', 'OFFSET_SURFACE', 'CURVE_BOUNDED_SURFACE',
    'EDGE_CURVE', 'LINE', 'POLYLINE', 'PARABOLA', 'HYPERBOLA',
    'CIRCLE', 'ELLIPSE', 'B_SPLINE_CURVE', 'BEZIER_CURVE',
    'RATIONAL_B_SPLINE_CURVE', 'OFFSET_CURVE_3D', 'INTERSECTION_CURVE', 'SEAM_CURVE'
]

def clean_directory(dir_path):
    """Deletes the directory if it exists and creates a new empty directory."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Deleted existing directory: {dir_path}")
    os.makedirs(dir_path, exist_ok=True)

def is_single_body(file_path):
    """
    Determine whether a mesh file contains a single connected component or mulitple components.
    
    Args: file_path (str): Path to the mesh file (e.g., .glb)
    Returns: tuple: (file_path, is_single) where is_single is True if the mesh has one connected component, False otherwise.
    """
    
    if file_path.lower().endswith(".glb"):
        try:
            mesh = trimesh.load(file_path, force="mesh")
            components = mesh.split(only_watertight=False)
            if len(components) == 1:
                return file_path, True
            else:
                return file_path, False
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            return file_path, False
    else:
        raise ValueError(f"Unsupported file type for {file_path}")


def analyze_geometry_simple(input_path, queue=None):
    """Worker function: counts STEP entity occurrences via string matching."""
    try:
        stats = {t: 0 for t in TARGETS}
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                for entity in TARGETS:
                    if entity in line:
                        stats[entity] += 1
        if queue is not None:
            queue.put({"filename": os.path.basename(input_path), **stats})
    except Exception:
        if queue is not None:
            queue.put(None)


def _geometry_process_wrapper(input_path, timeout=300):
    """Runs analyze_geometry_simple in a subprocess with a timeout."""
    queue = Queue()
    process = Process(target=analyze_geometry_simple, args=(input_path, queue))
    process.start()
    try:
        result = queue.get(timeout=timeout)
        process.join()
        return result
    except Exception:
        process.terminate()
        process.join()
        return "TIMEOUT"


def compute_geometry_csv(step_files, csv_path, num_workers=128, timeout=300):
    """
    Analyze a list of STEP files in parallel and write entity counts to a CSV.

    Args:
        step_files: list of paths to .step files
        csv_path: output CSV path
        num_workers: number of parallel workers
        timeout: per-file timeout in seconds
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    headers = ["filename"] + TARGETS
    processing_fails = timeout_fails = 0

    ctx = mp.get_context('fork') if hasattr(os, 'fork') else mp.get_context('spawn')

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            future_to_file = {
                executor.submit(_geometry_process_wrapper, f, timeout): f
                for f in step_files
            }
            prog = tqdm(as_completed(future_to_file), total=len(step_files), desc="Analyzing Geometry")
            for future in prog:
                result = future.result()
                if result == "TIMEOUT":
                    timeout_fails += 1
                elif result is None:
                    processing_fails += 1
                else:
                    writer.writerow(result)
                prog.set_postfix(fails=processing_fails, timeouts=timeout_fails)

    print(f"Done. Results saved to {csv_path}")

STEP_TEMPLATE = '''import cadquery as cq
result = cq.importers.importStep('{}')
num_faces = result.faces().size()
print(num_faces)
'''

PY_TEMPLATE = '''{}
num_faces = {}.faces().size()
print(num_faces)
'''

def read_py_as_string(file_path):
    """
    Read a Python file and return its contents as a string, along with the variable name
    from the last non-empty line.

    Args:
        file_path (str): Path to the Python file

    Returns:
        tuple: (file_contents, variable_name) where variable_name is the last non-empty line
               stripped of whitespace. The variable must be either "solid" or "r".

    Raises:
        ValueError: If the variable name is not "solid" or "r"
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Find the last non-empty line
    lines = content.split('\n')
    last_var = None
    for line in reversed(lines):
        if line.strip():  # Skip empty lines and lines with only whitespace
            last_var = line.strip().split('=')[0].strip()
            break

    # Validate the variable name
    if last_var not in ["solid", "r"]:
        raise ValueError(f"Last line variable must be 'solid' or 'r', got '{last_var}'")

    return content, last_var


# Defines how to compute face count for a single CAD file
def compute_face_count(cad_file):
    if ".py" in cad_file:
        original_code, variable_name = read_py_as_string(cad_file)
        python_code = PY_TEMPLATE.format(original_code, variable_name)
    if ".step" in cad_file.lower():
        python_code = STEP_TEMPLATE.format(cad_file)
    try:
        result = subprocess.run(
            [sys.executable, "-c", python_code],
            capture_output=True,
            text=True,
            timeout=300
        )
    except subprocess.TimeoutExpired:
        print(f"Timeout while processing {cad_file}")
        return cad_file, -1

    if result.returncode == 0:
        # Parse the last line of output which should contain the face count
        output_lines = result.stdout.strip().split('\n')
        if output_lines and output_lines[-1].strip().isdigit():
            return cad_file, int(output_lines[-1].strip())
        else:
            return cad_file, -1
    else:
        return cad_file, -1
    
    
def run_metric_parallel(metric_function, data_list, save_path=None, num_workers=8):
    """
    Run a metric function in parallel across a data list using ProcessPoolExecutor.

    Args:
        metric_function: The function to run on each data item. Should accept a single
                        data item and return a tuple of (item_name, result).
        data_list: List of data items to process
        num_workers: Number of parallel workers (default: 8)

    Returns:
        list: List of tuples containing (item_name, result) for each processed item
    """
    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:

        # Submit all tasks
        future_to_item = {executor.submit(metric_function, item): item for item in data_list}

        completed = 0
        for future in tqdm(as_completed(future_to_item), total=len(data_list),
                          desc=f"Processing {metric_function.__name__}"):
            item_name, result = future.result()
            all_results.append((str(item_name), result))
            completed += 1

    # Report average result
    try:
        print(f"Average for {metric_function.__name__}: {sum(r for _, r in all_results if r >= 0) / max(1, sum(1 for _, r in all_results if r >= 0))}")
    except:
        print(f"Could not compute average for {metric_function.__name__}")
    
    if save_path is not None:
        # Check that results folder exists, if not, create it
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(all_results, f)
    return all_results


def get_complexity_easy_medium_hard(results):
    """
    Split (filename, face_count) results into easy, medium, and hard buckets
    using equal thirds along the logarithmic face count range (33%/66% log-space quantiles).

    Args:
        results: list of (filename, face_count) tuples

    Returns:
        tuple: (easy, medium, hard, t1, t2) — each bucket is a list of (filename, face_count) tuples,
               t1 and t2 are the computed thresholds
    """
    import numpy as np

    valid = sorted([(fname, c) for fname, c in results if c > 0], key=lambda x: x[0])
    face_counts = [c for _, c in valid]

    f_min, f_max = min(face_counts), max(face_counts)
    log_min, log_max = np.log10(f_min), np.log10(f_max)
    log_range = log_max - log_min

    t1 = 10 ** (log_min + log_range / 3)
    t2 = 10 ** (log_min + 2 * log_range / 3)

    while sum(1 for _, c in valid if c > t2) < 1000 and t2 > t1:
        print(f"Only {sum(1 for _, c in valid if c > t2)} samples above t2={t2:.1f}, adjusting threshold.")
        t2 = t2 - 1

    easy   = [(fname, c) for fname, c in valid if c <= t1]
    medium = [(fname, c) for fname, c in valid if t1 < c <= t2]
    hard   = [(fname, c) for fname, c in valid if c > t2]

    return easy, medium, hard, t1, t2


def get_diverse_samples(csv_path, output_path, n_clusters=1000):
    """
    Select a diverse subset of STEP files from a geometry feature CSV using KMeans
    with nearest-neighbor medoid selection (approximates K-Medoids without sklearn_extra).

    Args:
        csv_path: path to input CSV (output of compute_geometry_csv)
        output_path: path to save the diverse subset CSV
        n_clusters: number of diverse samples to select (default: 1000)

    Returns:
        list of selected filenames
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    feature_columns = [
        'ADVANCED_FACE', 'PLANE', 'CYLINDRICAL_SURFACE', 'SPHERICAL_SURFACE',
        'CONICAL_SURFACE', 'TOROIDAL_SURFACE', 'B_SPLINE_SURFACE',
        'SURFACE_OF_REVOLUTION', 'SURFACE_OF_LINEAR_EXTRUSION'
    ]

    df = pd.read_csv(csv_path)
    existing_features = [col for col in feature_columns if col in df.columns]

    df_final = df[['filename'] + existing_features].sort_values('filename')
    df_unique = df_final.drop_duplicates(subset=existing_features, keep='first')

    print(f"Original rows: {len(df_final)}, unique geometry patterns: {len(df_unique)}")

    X = df_unique.drop(columns=['filename']).values
    X_scaled = StandardScaler().fit_transform(X)

    n = min(n_clusters, len(X_scaled))
    if n < n_clusters:
        print(f"Warning: only {len(X_scaled)} unique patterns, reducing clusters to {n}")

    print(f"Clustering {len(X_scaled)} files into {n} groups...")
    kmeans = KMeans(n_clusters=n, init='k-means++', n_init=1, random_state=42)
    kmeans.fit(X_scaled)

    # For each cluster center, find the nearest actual data point (medoid approximation)
    centers = kmeans.cluster_centers_
    medoid_indices = []
    for center in centers:
        dists = np.linalg.norm(X_scaled - center, axis=1)
        medoid_indices.append(int(np.argmin(dists)))
    medoid_indices = list(dict.fromkeys(medoid_indices))  # deduplicate, preserve order

    diverse_samples = df_unique.iloc[medoid_indices]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    diverse_samples.to_csv(output_path, index=False)
    print(f"Saved {len(diverse_samples)} diverse samples to {output_path}")

    return diverse_samples['filename'].tolist()

def get_clip_embeddings(list_of_image_paths, model_name = "openai/clip-vit-large-patch14-336", batch_size = 32):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    
    all_embeddings = []
    
    with torch.inference_mode():
        for i in tqdm(range(0, len(list_of_image_paths), batch_size)):
            
            batch_paths = list_of_image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            features = model.get_image_features(**inputs)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8) # normalize embeddings and each feature has dimension 768
            
            print("Features shape:")
            print(features.shape)

            all_embeddings.append(features.cpu())

    embs = torch.cat(all_embeddings, dim=0).numpy()  # (N=2000, D=768)
    return embs


def visualize_data_distribution(data, labels = None, x_feature = "t-SNE 1", y_feature = "t-SNE 2", num_components = 2,
    perplexity = 1, num_iterations = 250, savename = "visualizations/data_distribution.html", hover_text = None, color_by_label = False, highlight_indices = None, image_paths = None, cluster_labels = None):
    """
    Visualizes the distribution of a specified feature in a DataFrame.

    Args:
        data (np.array): The dataset that you plan on using as a 2D representation.
        image_paths (list): Optional list of image file paths. If provided, hovering over a
                            point will display the corresponding image.
        cluster_labels (np.ndarray): Optional array of integer cluster ids (length N). When
            provided, points are colored by cluster using a continuous colorscale, and
            selected points (highlight_indices) are drawn as stars while all others are circles.
    """
    import base64
    import numpy as np

    tsne = TSNE(n_components = num_components, perplexity = perplexity, max_iter = num_iterations, random_state=42)
    tsne_data = tsne.fit_transform(data)

    if hover_text is None:
        hover_text = labels if labels is not None else None

    # Encode images as base64 for embedding in the HTML
    encoded_images = None
    if image_paths is not None:
        encoded_images = []
        for path in image_paths:
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            encoded_images.append(f"data:image/{mime};base64,{encoded}")

    highlighted_set = set(np.atleast_1d(highlight_indices).tolist()) if highlight_indices is not None else set()

    fig = go.Figure()

    if cluster_labels is not None:
        # Split into non-selected (circles) and selected (stars), both colored by cluster
        normal_idx = [i for i in range(len(tsne_data)) if i not in highlighted_set]
        hi_idx     = [i for i in range(len(tsne_data)) if i in highlighted_set]
        cmin, cmax = int(cluster_labels.min()), int(cluster_labels.max())

        if normal_idx:
            fig.add_trace(go.Scatter(
                x=tsne_data[normal_idx, 0],
                y=tsne_data[normal_idx, 1],
                mode="markers",
                text=[hover_text[i] for i in normal_idx] if hover_text is not None else None,
                hovertemplate="%{text}<extra></extra>" if hover_text is not None else "<extra></extra>",
                customdata=[encoded_images[i] for i in normal_idx] if encoded_images is not None else None,
                marker=dict(
                    symbol="circle",
                    color=[int(cluster_labels[i]) for i in normal_idx],
                    coloraxis="coloraxis",
                    size=6,
                    opacity=0.6,
                ),
                name="All samples",
                showlegend=False,
            ))

        if hi_idx:
            fig.add_trace(go.Scatter(
                x=tsne_data[hi_idx, 0],
                y=tsne_data[hi_idx, 1],
                mode="markers",
                text=[hover_text[i] for i in hi_idx] if hover_text is not None else None,
                hovertemplate="%{text}<extra></extra>" if hover_text is not None else "<extra></extra>",
                customdata=[encoded_images[i] for i in hi_idx] if encoded_images is not None else None,
                marker=dict(
                    symbol="star",
                    color=[int(cluster_labels[i]) for i in hi_idx],
                    coloraxis="coloraxis",
                    size=10,
                    opacity=0.95,
                    line=dict(width=0.5, color="black"),
                ),
                name="Selected samples",
                showlegend=False,
            ))

        fig.update_layout(
            title="t-SNE Visualization",
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            coloraxis=dict(
                colorscale="Turbo",
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(title="Cluster"),
            ),
        )
    else:
        # Legacy path: red for highlighted, blue for others
        if color_by_label and labels is not None:
            point_colors = labels.astype(str).tolist()
        else:
            point_colors = ["red" if i in highlighted_set else "blue" for i in range(len(tsne_data))]

        fig.add_trace(go.Scatter(
            x=tsne_data[:, 0],
            y=tsne_data[:, 1],
            mode="markers",
            text=hover_text,
            hovertemplate="%{text}<extra></extra>" if hover_text is not None else "<extra></extra>",
            customdata=encoded_images,
            marker=dict(color=point_colors),
            showlegend=False,
        ))
        fig.update_layout(title="t-SNE Visualization", xaxis_title=x_feature, yaxis_title=y_feature)

    post_script = ""
    if encoded_images is not None:
        post_script = """
(function() {
    var tooltip = document.createElement('div');
    tooltip.style.cssText = 'position:fixed;background:white;border:1px solid #ccc;padding:4px;z-index:9999;pointer-events:none;display:none;box-shadow:2px 2px 6px rgba(0,0,0,0.2);';
    document.body.appendChild(tooltip);

    document.addEventListener('mousemove', function(e) {
        if (tooltip.style.display !== 'none') {
            var x = e.clientX + 15;
            var y = e.clientY + 15;
            if (x + 170 > window.innerWidth)  x = e.clientX - 170;
            if (y + 170 > window.innerHeight) y = e.clientY - 170;
            tooltip.style.left = x + 'px';
            tooltip.style.top  = y + 'px';
        }
    });

    var plot = document.getElementsByClassName('plotly-graph-div')[0];
    plot.on('plotly_hover', function(data) {
        var pt = data.points[0];
        if (pt && pt.customdata) {
            tooltip.innerHTML = '<img src="' + pt.customdata + '" style="max-width:160px;max-height:160px;">';
            tooltip.style.display = 'block';
        }
    });
    plot.on('plotly_unhover', function() {
        tooltip.style.display = 'none';
    });
})();
"""

    fig.write_html(savename, post_script=post_script)
    return

def get_dino_embeddings(list_of_image_paths, model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", batch_size=32):
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_embeddings = []

    with torch.inference_mode():
        for i in tqdm(range(0, len(list_of_image_paths), batch_size)):
            batch_paths = list_of_image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # CLS token from the last hidden state
            features = outputs.last_hidden_state[:, 0, :]
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
            all_embeddings.append(features.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()


def coverage_diversity(embeddings, k=1000):
    """
    Select k samples that maximise coverage diversity using KMeans clustering.

    For each of the k cluster centroids, the sample whose embedding is closest
    (Euclidean distance) to that centroid is selected. Duplicate selections are
    deduplicated while preserving order.

    Args:
        embeddings (np.ndarray): Array of shape (N, D) containing image embeddings.
        k (int): Number of samples to select (default: 1000).

    Returns:
        tuple:
            - list[int]: Indices into `embeddings` of the selected samples.
            - np.ndarray: Cluster label for each embedding (shape N,), values in [0, k).
    """
    from sklearn.cluster import KMeans
    
    n = len(embeddings)
    k = min(k, n)

    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=42)
    kmeans.fit(embeddings)

    centers = kmeans.cluster_centers_
    unique_selected = []
    seen = set()

    for center in centers:
        # Calculate distances from this centroid to all embeddings
        dists = np.linalg.norm(embeddings - center, axis=1)
        
        # Sort indices by distance (closest first)
        sorted_indices = np.argsort(dists)
        
        # Find the first index that hasn't been picked yet
        for idx in sorted_indices:
            idx_val = int(idx)
            if idx_val not in seen:
                seen.add(idx_val)
                unique_selected.append(idx_val)
                break 

    return unique_selected, kmeans.labels_


def greedy_diverse_subset(embeddings, k, seed=42):
    n = len(embeddings)

    if n == 0 or k <= 0:
        return []

    k = min(k, n)
    selected_mask = np.zeros(n, dtype=bool)

    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(n))]
    selected_mask[selected[0]] = True
    min_distances = cosine_distances(embeddings, embeddings[[selected[0]]]).flatten()
    min_distances[selected_mask] = -np.inf

    for _ in range(1, k):
        idx = int(np.argmax(min_distances))
        selected.append(idx)
        selected_mask[idx] = True

        new_dist = cosine_distances(embeddings, embeddings[[idx]]).flatten()
        min_distances = np.minimum(min_distances, new_dist)
        min_distances[selected_mask] = -np.inf

    return selected
