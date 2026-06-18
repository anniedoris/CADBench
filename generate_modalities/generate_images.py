from GVis.RenderTools.blender_utils import gray_cad_image, multicolor_cad_image, multiview_cad_image
import argparse
import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import pickle
import os
import shutil

MESH_EXTS = ('*.glb', '*.GLB', '*.obj', '*.OBJ', '*.stl', '*.STL')
DEFAULT_MESH_AZIMUTH = 315.0
DEFAULT_MESH_ELEVATION = 35.26
MULTIVIEW_MESH_CAMERA_ANGLES = [
    (315.0, DEFAULT_MESH_ELEVATION),   # _0: top-left
    (135.0, DEFAULT_MESH_ELEVATION),   # _1: top-right, opposite top-hemisphere view
    (45.0, -DEFAULT_MESH_ELEVATION),   # _2: bottom-left
    (225.0, -DEFAULT_MESH_ELEVATION),  # _3: bottom-right
]

def is_badly_cropped(img, border: int, threshold: int) -> bool:
    strips = [
        img[:border, :],    # top
        img[-border:, :],   # bottom
        img[:, :border],    # left
        img[:, -border:],   # right
    ]
    return any((strip < threshold).any() for strip in strips)



def detect_mode(data_path):
    """Return 'step' or 'mesh' based on what files exist in data_path."""
    p = Path(data_path)
    for f in p.rglob('*'):
        if f.suffix.upper() == '.STEP':
            return 'step'
    for ext in MESH_EXTS:
        if any(p.rglob(ext)):
            return 'mesh'
    raise ValueError(f'No STEP or mesh files found in {data_path}')


def get_mesh_files(data_path):
    """Return sorted list of GLB/OBJ files under data_path."""
    p = Path(data_path)
    files = []
    for ext in MESH_EXTS:
        files += sorted(p.rglob(ext))
    if not files:
        raise ValueError(f'No mesh files found in {data_path}')
    return files


def mesh_output_base_path(
    mesh_file: Path,
    input_root: Path,
    save_root: Path,
    preserve_tree: bool,
) -> Path:
    if preserve_tree:
        relative_parent = mesh_file.parent.relative_to(input_root)
        out_dir = save_root / relative_parent
    else:
        out_dir = save_root
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / mesh_file.stem


def compose_multiview_grid(image_paths, compound_path, separator_width: int = 10):
    from PIL import Image, ImageDraw

    images = [Image.open(str(image_path)).convert("RGB") for image_path in image_paths]
    try:
        if not images:
            raise ValueError("No images provided for multiview grid composition.")

        img_width, img_height = images[0].size
        cols = 2
        rows = 2
        compound_width = cols * img_width + (cols - 1) * separator_width
        compound_height = rows * img_height + (rows - 1) * separator_width

        compound = Image.new("RGB", (compound_width, compound_height), (255, 255, 255))
        for idx, image in enumerate(images):
            row = idx // cols
            col = idx % cols
            x = col * (img_width + separator_width)
            y = row * (img_height + separator_width)
            compound.paste(image, (x, y))

        draw = ImageDraw.Draw(compound)
        for col in range(1, cols):
            x = col * img_width + (col - 1) * separator_width
            draw.rectangle([x, 0, x + separator_width - 1, compound_height], fill=(0, 0, 0))
        for row in range(1, rows):
            y = row * img_height + (row - 1) * separator_width
            draw.rectangle([0, y, compound_width, y + separator_width - 1], fill=(0, 0, 0))

        compound_path.parent.mkdir(parents=True, exist_ok=True)
        compound.save(str(compound_path))
    finally:
        for image in images:
            image.close()


def render_mesh_view(
    mesh_file: Path,
    out_path: Path,
    visualize_mesh_cad,
    azimuth_degrees: float = DEFAULT_MESH_AZIMUTH,
    elevation_degrees: float = DEFAULT_MESH_ELEVATION,
):
    import numpy as np
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scale = 1.2
    while True:
        visualize_mesh_cad(
            str(mesh_file),
            str(out_path),
            parallel_scale_multiplier=scale,
            azimuth_degrees=azimuth_degrees,
            elevation_degrees=elevation_degrees,
        )
        with Image.open(str(out_path)) as image:
            img = np.array(image.convert("RGB"))
        if not is_badly_cropped(img, border=1, threshold=250):
            return scale
        scale += 0.1


def _render_mesh_worker(args):
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'visualize_stl'))
    from visualize_stl_davinci import visualize_mesh_cad
    f, out_path = args
    try:
        if not Path(out_path).exists():
            scale = render_mesh_view(
                Path(f),
                Path(out_path),
                visualize_mesh_cad,
            )
            retried = scale > 1.2
            return str(f), f'ok (final scale={scale:.1f})' if retried else 'ok'
        return str(f), 'skip'
    except Exception as e:
        return str(f), f'failed: {e}'


def _render_mesh_multiview_worker(args):
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'visualize_stl'))
    from visualize_stl_davinci import visualize_mesh_cad

    mesh_file, output_base = args
    mesh_file = Path(mesh_file)
    output_base = Path(output_base)
    view_paths = [
        Path(f"{output_base}_{view_index}.png")
        for view_index in range(len(MULTIVIEW_MESH_CAMERA_ANGLES))
    ]
    compound_path = output_base.with_suffix(".png")

    try:
        if compound_path.exists() and all(view_path.exists() for view_path in view_paths):
            return str(mesh_file), "skip"

        scales = []
        for view_index, (azimuth_degrees, elevation_degrees) in enumerate(
            MULTIVIEW_MESH_CAMERA_ANGLES
        ):
            scale = render_mesh_view(
                mesh_file,
                view_paths[view_index],
                visualize_mesh_cad,
                azimuth_degrees=azimuth_degrees,
                elevation_degrees=elevation_degrees,
            )
            scales.append(scale)

        compose_multiview_grid(view_paths, compound_path)
        retried = any(scale > 1.2 for scale in scales)
        return (
            str(mesh_file),
            f"ok (combined -> {compound_path.name})" if retried else f"ok ({compound_path.name})",
        )
    except Exception as e:
        return str(mesh_file), f"failed: {e}"

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
    num_failures = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:

        # Submit all tasks
        future_to_item = {executor.submit(metric_function, item): item for item in data_list}

        completed = 0
        for future in tqdm(as_completed(future_to_item), total=len(data_list),
                          desc=f"Processing {getattr(metric_function, '__name__', None) or metric_function.func.__name__}"):
            try:
                item_name, result = future.result()
                all_results.append((str(item_name), result))
            except Exception as e:
                print(f"Error processing item {future_to_item[future]}: {e}")
                num_failures += 1
            completed += 1

    # Report number of successes vs failures
    print(f"Completed {completed} tasks with {num_failures} failures.")
    
    # Report average result
    try:
        print(f"Average for {metric_function.__name__}: {sum(r for _, r in all_results if r >= 0) / max(1, sum(1 for _, r in all_results if r >= 0))}")
    except:
        print(f"Could not compute average for {metric_function.__name__}")

    # Report failures
    if num_failures > 0:
        print(f"Number of failures: {num_failures}")

    if save_path is not None:
        # Check that results folder exists, if not, create it
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(all_results, f)
    return all_results

def get_step_files(data_path):
    """
    Get all STEP files from the given folder and its subdirectories.

    Args:
        data_path: Path to the folder containing data

    Returns:
        list: List of absolute file paths to STEP files

    Raises:
        ValueError: If path doesn't exist, is not a directory, or no STEP files found
    """
    if not os.path.exists(data_path):
        raise ValueError(f"Path does not exist: {data_path}")

    if not os.path.isdir(data_path):
        raise ValueError(f"Path is not a directory: {data_path}")

    # Collect all STEP files
    step_files = []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.upper().endswith('.STEP'):
                step_files.append(os.path.abspath(os.path.join(root, file)))

    if len(step_files) == 0:
        raise ValueError("No STEP files found in the directory")

    return step_files

def main():
    parser = argparse.ArgumentParser(description='Evaluate dataset metrics')

    parser.add_argument(
        '--singleview_gray_images',
        action='store_true',
        default=False,
        help='Generate gray CAD images'
    )
    
    parser.add_argument(
        '--multicolor_images',
        action='store_true',
        default=False,
        help='Generate multicolor CAD images'
    )
    
    parser.add_argument(
        '--multiview_gray_images',
        action='store_true',
        default=False,
        help='Generate multiview CAD images'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        required = True,
        help='Path to the folder containing data'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of worker processes to use'
    )

    parser.add_argument(
        '--no_gpu',
        action='store_true',
        default=False,
        help='Disable GPU usage'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Optional output directory for generated images. For mesh inputs, subdirectories are preserved when this is set.',
    )

    args = parser.parse_args()

    # Your evaluation logic here
    print(f"Will generate single-view gray CAD images: {args.singleview_gray_images}")
    print(f"Will generate multicolor CAD images: {args.multicolor_images}")
    print(f"Will generate multiview gray CAD images: {args.multiview_gray_images}")
    print(f"Data path: {args.data_path}")
    print(f"Number of workers: {args.num_workers}")
    
    mode = detect_mode(args.data_path)
    print(f'Detected mode: {mode}')

    input_dir_name = os.path.basename(os.path.normpath(args.data_path))

    # --- Mesh rendering path ---
    if mode == 'mesh':
        mesh_files = get_mesh_files(args.data_path)
        print(f'Found {len(mesh_files)} mesh files')

        input_root = Path(args.data_path).resolve()
        preserve_tree = args.save_dir is not None
        if args.save_dir is not None:
            save_dir = Path(args.save_dir).resolve()
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            default_dir_name = (
                f"mesh_images_multiview_{input_dir_name}"
                if args.multiview_gray_images
                else f"mesh_images_{input_dir_name}"
            )
            save_dir = Path(f'data/images/{default_dir_name}')
            if save_dir.exists():
                shutil.rmtree(save_dir)
            save_dir.mkdir(parents=True)

        if args.multiview_gray_images:
            worker = _render_mesh_multiview_worker
            tasks = [
                (f, mesh_output_base_path(f, input_root, save_dir, preserve_tree))
                for f in mesh_files
            ]
        else:
            worker = _render_mesh_worker
            tasks = [
                (
                    f,
                    Path(
                        f"{mesh_output_base_path(f, input_root, save_dir, preserve_tree)}_0.png"
                    ),
                )
                for f in mesh_files
            ]

        if args.num_workers > 1:
            with Pool(processes=args.num_workers) as pool:
                for i, (path, status) in enumerate(pool.imap_unordered(worker, tasks), 1):
                    print(f'[{i}/{len(mesh_files)}] {Path(path).name}: {status}')
        else:
            for i, task in enumerate(tasks, 1):
                mesh_path = Path(task[0])
                print(f'[{i}/{len(mesh_files)}] {mesh_path.name}')
                try:
                    path, status = worker(task)
                    print(f'  {status}')
                except KeyboardInterrupt:
                    print('\nInterrupted.')
                    return
                except Exception as e:
                    print(f'  failed: {e}')
        return

    # --- STEP rendering path ---
    # Get the data files
    data_list = get_step_files(args.data_path)

    if args.singleview_gray_images:

        # Setup folder for saving
        save_dir = f'data/images/singleview_gray_images_{input_dir_name}'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        run_metric_parallel(
            partial(gray_cad_image, save_dir=save_dir, use_gpu=not args.no_gpu),
            data_list,
            num_workers=args.num_workers
        )
        
    # if args.multicolor_images:
    #     run_metric_parallel(
    #         multicolor_cad_image,
    #         data_list,
    #         num_workers=args.num_workers
    #     )
        
    if args.multiview_gray_images:
        
        # Setup folder for saving
        save_dir = f'data/images/multiview_gray_images_{input_dir_name}'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        run_metric_parallel(
            partial(multiview_cad_image, save_dir=save_dir, use_gpu=not args.no_gpu),
            data_list,
            num_workers=args.num_workers
        )
    
if __name__ == "__main__":
    main()
