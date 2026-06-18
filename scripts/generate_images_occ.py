import os
import argparse
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os 
import sys
import glob
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
import re
import h5py
import argparse

from OCC.Extend.DataExchange import read_step_file
from OCC.Display.OCCViewer import Viewer3d
from OCC.Core.Prs3d import Prs3d_Drawer, Prs3d_LineAspect
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.Aspect import Aspect_TOL_SOLID, Aspect_TypeOfLine
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.Aspect import Aspect_TOL_SOLID, Aspect_TypeOfLine
from OCC.Extend.ShapeFactory import scale_shape, get_boundingbox


from OCC.Core.Graphic3d import (
    Graphic3d_NOM_BRASS,
    Graphic3d_NOM_BRONZE,
    Graphic3d_NOM_COPPER,
    Graphic3d_NOM_GOLD,
    Graphic3d_NOM_PEWTER,
    Graphic3d_NOM_PLASTER,
    Graphic3d_NOM_PLASTIC,
    Graphic3d_NOM_SILVER,
    Graphic3d_NOM_STEEL,
    Graphic3d_NOM_STONE,
    Graphic3d_NOM_SHINY_PLASTIC,
    Graphic3d_NOM_SATIN,
    Graphic3d_NOM_METALIZED,
    Graphic3d_NOM_NEON_GNC,
    Graphic3d_NOM_CHROME,
    Graphic3d_NOM_ALUMINIUM,
    Graphic3d_NOM_OBSIDIAN,
    Graphic3d_NOM_NEON_PHC,
    Graphic3d_NOM_JADE,
    Graphic3d_NOM_CHARCOAL,
    Graphic3d_NOM_WATER,
    Graphic3d_NOM_GLASS,
    Graphic3d_NOM_DIAMOND,
    Graphic3d_NOM_TRANSPARENT)

from PIL import Image
import re 
import json


from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRep import BRep_Tool
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.gp import gp_Vec, gp_Pnt, gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCC.Core.TopoDS import TopoDS_Shape

import trimesh

from OCC.Extend.DataExchange import read_stl_file

MAX_FACES = 5000

def read_python_file(filepath):
    """
    Reads the contents of a Python (.py) file and returns it as a string.

    Args:
        filepath (str): Path to the .py file.

    Returns:
        str: Contents of the file as a single string.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def load_obj_file(filename: str) -> TopoDS_Shape:
    """Load an OBJ file and return a shape."""
    mesh = trimesh.load(filename)

    tmp_stl = filename + f".{os.getpid()}.stl"
    try:
        mesh.export(tmp_stl)
        shape = read_stl_file(tmp_stl)
    finally:
        if os.path.exists(tmp_stl):
            os.remove(tmp_stl)

    if shape is None:
        raise Exception(f"Error: Cannot read OBJ file {filename}.")
    return shape

def load_glb_file(filename: str) -> TopoDS_Shape:
    """Load a GLB file, extracting only mesh geometry (skipping materials/textures)."""
    print("Loading GLB file:", filename)
    mesh = trimesh.load(filename, process=False, force='mesh')
    if len(mesh.faces) > MAX_FACES:
        target_reduction = 1.0 - (MAX_FACES / len(mesh.faces))
        mesh = mesh.simplify_quadric_decimation(target_reduction)

    tmp_stl = filename + f".{os.getpid()}.stl"
    try:
        mesh.export(tmp_stl)
        shape = read_stl_file(tmp_stl)
    finally:
        if os.path.exists(tmp_stl):
            os.remove(tmp_stl)

    if shape is None:
        raise Exception(f"Error: Cannot read GLB file {filename}.")
    return shape

def load_py_file(filename: str) -> TopoDS_Shape:
    """Load a Python file and return a shape."""

    code = read_python_file(filename)
    code += "\ncq.exporters.export(result, 'output.step')\n"
    
    # Load the .py file
    exec(code)

    return load_step_file('output.step')

def load_step_file(filename : str) -> TopoDS_Shape:
    """Load a STEP file and return the shape."""
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)
    if status != IFSelect_RetDone:
        raise Exception("Error: Cannot read STEP file.")
    # Transfer the roots and get the shape
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    return shape


def remove_bg(image_path):
    # Replace 'path_to_your_image.jpg' with the correct path to your image
    image = Image.open(image_path)

    # Convert image to RGBA (if not already in this mode)
    image = image.convert("RGBA")

    # Make white (and shades of white) pixels transparent
    datas = image.getdata()
    new_data = []
    for item in datas:
        if item[0] > 200 and item[1] > 200 and item[2] > 200:  # Adjust these values if necessary
            new_data.append((255, 255, 255, 0))  # Making white pixels transparent
        else:
            new_data.append(item)

    image.putdata(new_data)
    image.save(image_path, "PNG")


# 8 isometric view directions (eye position octants).
# Top hemisphere (Z > 0) first as _0–_3, then bottom hemisphere (Z < 0) as _4–_7.
VIEW_DIRECTIONS = [
    ( 1, -1,  1),  # _0: standard iso  (top,   front-right)
    (-1, -1,  1),  # _1: top,   front-left
    ( 1,  1,  1),  # _2: top,   rear-right
    (-1,  1,  1),  # _3: top,   rear-left
    ( 1, -1, -1),  # _4: bottom, front-right
    (-1, -1, -1),  # _5: bottom, front-left
    ( 1,  1, -1),  # _6: bottom, rear-right
    (-1,  1, -1),  # _7: bottom, rear-left
]


def convert_part_to_image(file_name, view_type, save_path, b_rep_name, resolution_height=224*2, resolution_width=224*2, rotation_angle=True, scale=None, remove_bg=False):
    
    file_type = None

    if ".glb" in file_name:
        shape = load_glb_file(file_name)
        file_type = "obj"
    elif ".obj" in file_name:
        shape = load_obj_file(file_name)
        file_type = "obj"
    elif ".step" in file_name:
        shape = load_step_file(file_name)
        file_type = "step"
    elif ".py" in file_name:
        shape = load_py_file(file_name)
        file_type = "py"
    else:
        raise ValueError("Unrecognized file type")

    # Initialize the offscreen renderer
    offscreen_renderer = Viewer3d()
    if view_type == "iso":
        offscreen_renderer.View_Iso()
    elif view_type == "front": 
        offscreen_renderer.View_Front()
    elif view_type == "rear": 
            offscreen_renderer.View_Rear()
    elif view_type == "left": 
            offscreen_renderer.View_Left()
    elif view_type == "right": 
            offscreen_renderer.View_Right()
    elif view_type == "top": 
            offscreen_renderer.View_Top()
    elif view_type == "bottom": 
            offscreen_renderer.View_Bottom()
    else: 
        raise Exception("please choose: top, bottom, front, rear, left, right, iso")


    # scale the shape 
    if scale is not None:
        shape = scale_shape(shape, fx=scale[0], fy=scale[1], fz=scale[2])

    # offscreen renderer
    if file_type == "obj":
        offscreen_renderer.Create(draw_face_boundaries=False)
        offscreen_renderer.SetModeShaded()
    else:
        offscreen_renderer.Create()
        offscreen_renderer.SetModeShaded()

    # Display the shape
    # Graphic3d_NOM_TRANSPARENT, Graphic3d_NOM_SILVER
    # drawer = Prs3d_Drawer()
    # drawer.SetDisplayEdges(False)
    # offscreen_renderer.Context.SetDefaultDrawer(drawer)
    offscreen_renderer.DisplayShape(shape, update=True, material=Graphic3d_NOM_SILVER, transparency=0.0)
    offscreen_renderer.View.SetBackgroundColor(0, 1, 1, 1)
    
    # Fit the entire shape in the view
    offscreen_renderer.View.FitAll(0.5)

    # Set a high resolution for the renderer
    high_resolution_width = resolution_height
    high_resolution_height = resolution_width
    offscreen_renderer.SetSize(high_resolution_width, high_resolution_height)                

    # Set the view direction
    if rotation_angle is not None:
        rotation_x, rotation_y, rotation_z = rotation_angle
        offscreen_renderer.View.SetProj(rotation_x, rotation_y, rotation_z)
        offscreen_renderer.View.FitAll(0.5)

    # Render and save the image in high resolution
    offscreen_renderer.View.Dump(save_path)
    if remove_bg:
        remove_bg(save_path)


def get_cad_paths(root_dir):
    model_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.step', '.obj', '.py', '.glb')):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root_dir)
                model_paths.append(rel_path)
    return model_paths


# CAMERA_POSITIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "camera_positions.json")
# with open(CAMERA_POSITIONS_PATH) as _f:
#     _cam = json.load(_f)
# CAMERA_POSITIONS = [(v["x"], v["y"], v["z"]) for v in _cam.values()]  # list of 20 (x,y,z) tuples


def process_file(file, input_parts, output_images):
    input_path = os.path.join(input_parts, file)
    base_name = os.path.splitext(os.path.basename(file))[0]
    errors = []
    for i, (rx, ry, rz) in enumerate(VIEW_DIRECTIONS, start=0):
        output_path = os.path.join(output_images, f"{base_name}_{i}.png")  # _0 through _7
        try:
            convert_part_to_image(input_path, "iso", output_path, "BRepName",
                                   rotation_angle=(rx, ry, rz), remove_bg=True)
        except Exception as e:
            errors.append(f"view {i}: {e}")
    if errors:
        print(f"Errors on {file}: {'; '.join(errors)}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Render 3D parts to image files.")
    parser.add_argument("--input_parts", type=str, required=True, help="Path to folder of .obj or .step files or .py files.")
    parser.add_argument("--output_images", type=str, required=True, help="Path to save output image files.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel worker processes.")
    args = parser.parse_args()

    os.makedirs(args.output_images, exist_ok=True)
    print(f"Created folder: {args.output_images}")

    print(f"Input path: {args.input_parts}")
    all_cad_files = get_cad_paths(args.input_parts)
    print(f"Found {len(all_cad_files)} files, using {args.num_workers} workers.")
    
    # Reduce all_cad_files to those for which the images don't already exist
    existing_images = set(os.listdir(args.output_images))
    # Each CAD file should produce 8 images, so we check for the existence of all 8 before skipping
    def images_exist_for_file(file):
        base_name = os.path.splitext(os.path.basename(file))[0]
        expected_images = {f"{base_name}_{i}.png" for i in range(8)}
        return expected_images.issubset(existing_images)

    all_cad_files = [f for f in all_cad_files if not images_exist_for_file(f)]
    
    print(f"{len(all_cad_files)} files need processing after checking existing images.")

    errors = 0
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp.get_context('spawn')) as executor:
        futures = {executor.submit(process_file, f, args.input_parts, args.output_images): f for f in all_cad_files}
        for future in tqdm(as_completed(futures), total=len(all_cad_files)):
            if not future.result():
                errors += 1

    elapsed = time.time() - start
    print(f"Done. Total time: {elapsed:.1f}s | Errors: {errors}/{len(all_cad_files)}")


if __name__ == "__main__":
    main()
