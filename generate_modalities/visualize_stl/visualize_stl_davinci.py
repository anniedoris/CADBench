#!/usr/bin/env python3
"""
Visualize mesh files (STL, OBJ, GLB, PLY, etc.) using DaVinciVisualizer with CAD mode.
Usage: python visualize_stl_davinci.py <mesh_file> [output_image.png]
"""

import sys
import argparse
import trimesh
import numpy as np
from utils import visualizer_util
from geometry import pyvista_util


def visualize_mesh_cad(
    input_file,
    output_image=None,
    parallel_scale_multiplier=1.2,
    azimuth_degrees=315.0,
    elevation_degrees=35.26,
):
    try:
        from utils.visualizer_util import DaVinciVisualizer
    except ImportError:
        print("Error: Could not import DaVinciVisualizer from visualizer_util.py")
        sys.exit(1)

    mesh = trimesh.load(input_file, force='mesh')
    mesh.apply_transform(trimesh.transformations.rotation_matrix(
        np.radians(180), [0, 0, 1], point=mesh.centroid))
    verts = mesh.vertices
    faces = mesh.faces

    ext = input_file.rsplit('.', 1)[-1].lower()
    print(f"Loaded {ext.upper()} mesh: {len(verts)} vertices, {len(faces)} faces")

    if len(verts) == 0:
        print(f"Error: No vertices found in {input_file}")
        return

    visualizer = DaVinciVisualizer(geometry_loader=None)
    visualizer.plot_picture(
        style="cad",
        verts=verts,
        faces=faces,
        save_path=output_image,
        center=True,
        mesh=mesh,
        draw_edges=True,
        override_color="#bfbeba",
        parallel_scale_multiplier=parallel_scale_multiplier,
        azimuth_degrees=azimuth_degrees,
        elevation_degrees=elevation_degrees,
    )

    if output_image:
        print(f"Saved -> {output_image}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render mesh file to image using DaVinciVisualizer.')
    parser.add_argument('input', help='Path to mesh file (STL, OBJ, GLB, PLY, ...)')
    parser.add_argument('output', nargs='?', default=None, help='Output image path (optional)')
    args = parser.parse_args()

    visualize_mesh_cad(args.input, args.output)
