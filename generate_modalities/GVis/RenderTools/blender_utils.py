import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from blendify import scene, Scene
from blendify.colors import UniformColors, VertexUV, FileTextureColors, VertexColors
from blendify.materials import PrincipledBSDFMaterial, MetalMaterial, \
    PlasticMaterial, PrincipledBSDFWireframeMaterial

import bpy
from PIL import Image
from typing import List, Tuple, Union
from OCC.Core.TopoDS import TopoDS_Shape
from .utils import extract_renderables, read_step_file
import seaborn as sns
from .svg_utils import export_shape_to_svg, gp_Dir
from svgutils import transform as st
from svgutils.compose import Unit, SVG, Figure, Text, Panel
from cairosvg import svg2png
import tempfile
import os

def is_badly_cropped(img, border: int, threshold: int) -> bool:
    strips = [
        img[:border, :],    # top
        img[-border:, :],   # bottom
        img[:, :border],    # left
        img[:, -border:],   # right
    ]
    return any((strip < threshold).any() for strip in strips)

def get_curve_material(edge_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
    curve_mat = bpy.data.materials.get("CurveMaterial")
    if curve_mat is None:
        curve_mat = bpy.data.materials.new(name="CurveMaterial")
        curve_mat.use_nodes = True
        nodes = curve_mat.node_tree.nodes
        # Ensure a Principled BSDF node exists
        bsdf = nodes.get("Principled BSDF")
        if bsdf is None:
            bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        # default to black-ish color; you can change per-curve below
        bsdf.inputs['Base Color'].default_value = (*edge_color, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.4
    else:
        bsdf = curve_mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (*edge_color, 1.0)
    return curve_mat

def delete_curve_material():
    curve_mat = bpy.data.materials.get("CurveMaterial")
    if curve_mat is not None:
        bpy.data.materials.remove(curve_mat)

def prepare_mesh_and_edges(verts, edges):
    
    d = verts.min(0)
    edges = [e - d for e in edges]
    verts = verts - d
    s = verts.max()
    edges = [e / s for e in edges]
    verts = verts / s
    d = verts.mean(0)
    edges = [e - d for e in edges]
    verts = verts - d
    d = np.array([0,0,verts[:,2].min()])
    edges = [e - d for e in edges]
    verts = verts - d
    
    return verts, edges

def setup_scene(scene: Scene,
                realistic_camera: bool = False,
                ambient_only: bool = False,
                view: str = 'isometric',
                relative_camera_distance: Union[float, str] = 3.0,
                isometric_setup: Tuple[int, int, int] = (1, 1, 1),
                ambient_light_strength: float = 0.01,
                light_strength: float = 200.0,
                light_size: float = 1.0,
                relative_light_distance: float = 5.0,
                light_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                resolution: Tuple[int, int] = (1920, 1080),
                random_distance_range: Tuple[float, float] = (1.5, 6.0),
                ):
    
    # get all curves and remove them
    curves = [obj for obj in bpy.data.scenes[0].objects if obj.type == 'CURVE']
    for curve in curves:
        bpy.data.objects.remove(curve)

    if view == 'isometric':
        cam_position = np.array([1, 1, np.sqrt(2) * np.tan(np.deg2rad(90-54.736))])
        cam_position /= np.linalg.norm(cam_position)
        cam_position = cam_position * np.array(isometric_setup)
    elif view == 'top':
        cam_position = np.array([0, 0, 1])
    elif view == 'front':
        cam_position = np.array([0, -1, 0])
    elif view == 'left':
        cam_position = np.array([-1, 0, 0])
    elif view == 'back':
        cam_position = np.array([0, 1, 0])
    elif view == 'right':
        cam_position = np.array([1, 0, 0])
    elif view == 'bottom':
        cam_position = np.array([0, 0, -1])
    elif view == 'random':
        cam_position = np.random.rand(3)
        cam_position /= np.linalg.norm(cam_position)
    else:
        raise ValueError(f"Unknown view: {view}")

    if isinstance(relative_camera_distance, str) and relative_camera_distance == 'random':
        relative_camera_distance = float(np.random.uniform(random_distance_range[0], random_distance_range[1]))
    elif isinstance(relative_camera_distance, str):
        raise ValueError(f"Unknown relative_camera_distance: {relative_camera_distance}")
    elif isinstance(relative_camera_distance, (int, float)):
        relative_camera_distance = float(relative_camera_distance)
    
    if realistic_camera:
        cam_position = cam_position * relative_camera_distance
    else:
        cam_position = cam_position * 10
    look_to = (0.0, 0.0, 0.0)
    
    # reset everything else
    scene.clear()
    if realistic_camera:
        cam = scene.set_perspective_camera(
            resolution, 
            fov_x=np.deg2rad(20.8),
            translation=cam_position,
            rotation_mode='look_at',
            rotation=look_to,
        )
    else:
        cam = scene.set_orthographic_camera(
            resolution,
            ortho_scale=relative_camera_distance,
            translation=cam_position,
            rotation_mode='look_at',
            rotation=look_to,
        )
        
    bpy.context.scene.camera = cam.blender_camera
        
    if view == 'random':
        rotation_axis = cam_position / np.linalg.norm(cam_position)
        rotation_angle = np.random.uniform(0, 2*np.pi)
        rotation_matrix = trimesh.transformations.rotation_matrix(
            rotation_angle,
            rotation_axis,
            point=(0,0,0)
        )[:3,:3]
        current_rotation_quaternion = np.array(scene.camera.blender_camera.rotation_quaternion)
        current_rotation_matrix = trimesh.transformations.quaternion_matrix(current_rotation_quaternion)[:3,:3]
        new_rotation = rotation_matrix @ current_rotation_matrix
        new_rotation_quaternion = trimesh.transformations.quaternion_from_matrix(
            new_rotation
        )
        scene.camera.blender_camera.rotation_quaternion = new_rotation_quaternion

    scene.lights.set_background_light(
        ambient_light_strength,
        color=light_color)
    
    if not ambient_only:
        look_vector = cam_position.astype(np.float64)
        height_scale = look_vector[2] / np.linalg.norm(look_vector)
        if np.abs(height_scale) == 1.0:
            projected_look_vector = np.array([1.0, 0.0, 0.0])
        else:
            projected_look_vector = look_vector
        projected_look_vector[2] = 0
        projected_look_vector /= np.linalg.norm(projected_look_vector)
        K_R = trimesh.transformations.rotation_matrix(np.deg2rad(45), [0,0,1])[:3,:3]
        K_F = trimesh.transformations.rotation_matrix(np.deg2rad(-45), [0,0,1])[:3,:3]
        K_B = trimesh.transformations.rotation_matrix(np.deg2rad(135), [0,0,1])[:3,:3]
        l1 = K_R @ projected_look_vector
        l2 = K_F @ projected_look_vector
        l3 = K_B @ projected_look_vector
        l1[2] = height_scale
        l2[2] = height_scale
        l3[2] = height_scale
        l1 = l1 / np.linalg.norm(l1) * relative_light_distance
        l2 = l2 / np.linalg.norm(l2) * relative_light_distance
        l3 = l3 / np.linalg.norm(l3) * relative_light_distance

        lights = [
            scene.lights.add_area(
                'square',
                light_size,
                light_strength * 2,
                rotation_mode='look_at',
                rotation= (0.0, 0.0, 0.0),
                translation=(l1[0], l1[1], l1[2]),
                color=light_color,
            ),
            scene.lights.add_area(
                'square',
                light_size,
                light_strength,
                rotation_mode='look_at',
                rotation= (0.0, 0.0, 0.0),
                translation=(l2[0], l2[1], l2[2]),
                color=light_color,
            ),
            scene.lights.add_area(
                'square',
                light_size,
                light_strength * 4,
                rotation_mode='look_at',
                rotation= (0.0, 0.0, 0.0),
                translation=(l3[0], l3[1], l3[2]),
                color=light_color,
            )
        ]
    else:
        lights = []

        
    return lights

def hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    return tuple(int(hex_color[i:i + lv // 3], 16) / 255.0 for i in range(0, lv, lv // 3)) + (1.0,)

def render_object(shape: Union[TopoDS_Shape, str],
                  draw_edges:  bool = True,
                  material: Union[str, PrincipledBSDFMaterial, PrincipledBSDFWireframeMaterial] = 'metal',
                  color: Union[str, Tuple[float, float, float]] = (1.0, 0.8862745098039215, 0.6392156862745098),
                  edge_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                  realistic_camera: bool = False,
                  ambient_only: bool = False,
                  view: str = 'isometric',
                  relative_camera_distance: Union[float, str] = "auto",
                  isometric_setup: Tuple[int, int, int] = (1, 1, 1),
                  ambient_light_strength: float = 0.1,
                  light_strength: float = 800.0,
                  light_size: float = 10.0,
                  relative_light_distance: float = 20.0,
                  light_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                  resolution: Tuple[int, int] = (1280, 720),
                  random_distance_range: Union[str, Tuple[float, float]] = "auto",
                  use_gpu: bool = True,
                  num_samples: int = 128,
                  albedo = False,
                  depth = False,
                  camera_dist = 1.5
                  ) -> Tuple[np.ndarray, np.ndarray]:
    
    if depth or albedo:
        if not realistic_camera:
            raise ValueError("Depth and albedo rendering only supported with realistic_camera=True. Isomorphic camera does not provide depth or albedo information.")
    
    # get the mesh and edges
    verts, tris, faces, edges = extract_renderables(shape)
    
    # bounding box normalization and centering
    verts, edges = prepare_mesh_and_edges(verts, edges)
    
    if isinstance(relative_camera_distance, str):
        if relative_camera_distance == "auto":
            if realistic_camera:
                # print("Inside realistic camera")
                relative_camera_distance = 6.0
            else:
                # print("Inside not realistic camera")
                relative_camera_distance = camera_dist
        elif relative_camera_distance != 'random':
            raise ValueError(f"Unknown relative_camera_distance: {relative_camera_distance}")
    
    if isinstance(random_distance_range, str):
        if random_distance_range == "auto":
            if realistic_camera:
                random_distance_range = (4.0, 12.0)
            else:
                random_distance_range = (1.2, 6.0)
        else:
            raise ValueError(f"Unknown random_distance_range: {random_distance_range}")

    # setup the scene
    lights = setup_scene(
        scene,
        realistic_camera=realistic_camera,
        ambient_only=ambient_only,
        view=view,
        relative_camera_distance=relative_camera_distance,
        isometric_setup=isometric_setup,
        ambient_light_strength=ambient_light_strength,
        light_strength=light_strength,
        light_size=light_size,
        relative_light_distance=relative_light_distance,
        light_color=light_color,
        resolution=resolution,
        random_distance_range=random_distance_range
    )

    centroid = (verts.max(0) + verts.min(0)) / 2
    scene.camera.blender_camera.location = np.array(scene.camera.blender_camera.location) + centroid
    for light in lights:
        light.blender_light.location = np.array(light.blender_light.location) + centroid

    if isinstance(material, str):
        if material == 'metal':
            mat = MetalMaterial(metallic=1.0, roughness=0.5)
        elif material == 'plastic':
            mat = PlasticMaterial()
        else:
            raise ValueError(f"Unknown material: {material}")
    else:
        mat = material
        
    if isinstance(color, str):
        if color.startswith('#'):
            color = hex_to_rgba(color)[:3]
        elif color == 'random':
            color = tuple(np.random.rand(3).tolist())
        elif color == 'per_face_tableau':
            tableau_colors = sns.hls_palette(n_colors=len(faces))
            face_colors = []
            for i in range(len(faces)-1):
                face_colors.append(tuple(tableau_colors[i]))
            color = np.array(face_colors)
            num_repeats = np.diff(faces)
            
        elif color == 'per_face_random':
            face_colors = []
            for _ in range(len(faces)-1):
                face_colors.append(tuple(np.random.rand(3).tolist()))
            color = np.array(face_colors)
            num_repeats = np.diff(faces)
    else:
        if len(color) == 4:
            color = tuple(color[:3])
            
    if isinstance(color, tuple) or isinstance(color, list):
        color = UniformColors(color)
        cad = scene.renderables.add_mesh(
            verts, tris, material=mat, colors=color
        )
        cad.set_smooth(True)
    else:
        meshes = []
        for i in range(len(faces)-1):
            c = UniformColors(color[i])
            f_tris = tris[faces[i]:faces[i+1]]
            f_verts = verts
            meshes.append(scene.renderables.add_mesh(
                f_verts, f_tris, material=mat, colors=c
            ))
            meshes[-1].set_smooth(True)
    if draw_edges and len(edges) > 0:
        curve_mat = get_curve_material(edge_color=edge_color)        
        for i, curve in enumerate(edges):
            curveData = bpy.data.curves.new(f'Curve_{i}', type='CURVE')
            curveData.dimensions = '3D'
            curveData.resolution_u = 2
            polyline = curveData.splines.new('POLY')
            polyline.points.add(len(curve)-1)
            for j, coord in enumerate(curve):
                x,y,z = coord
                polyline.points[j].co = (x, y, z, 1)

            curveData.fill_mode = 'FULL'
            curveData.bevel_depth = 0.001
            curveData.bevel_resolution = 3

            curveOB = bpy.data.objects.new(f'Curve_{i}', curveData)

            if curveOB.data.materials:
                curveOB.data.materials[0] = curve_mat
            else:
                curveOB.data.materials.append(curve_mat)

            try:
                # Blender 2.8+ object color
                curveOB.color = (*edge_color, 1.0)
            except Exception:
                pass

            bpy.data.scenes[0].collection.objects.link(curveOB)
            
    outs = scene.render(use_gpu=use_gpu, samples=num_samples, save_albedo=albedo, save_depth=depth)
    
    if albedo and depth:
        render, render_depth, render_albedo = outs
        render = Image.fromarray(render)
        return render, render_depth, render_albedo
    if albedo and not depth:
        render, render_albedo = outs
        render = Image.fromarray(render)
        return render, render_albedo
    if depth and not albedo:
        render, render_depth = outs
        render = Image.fromarray(render)
        return render, render_depth
    else:
        render = outs
        render = Image.fromarray(render)
        return render

def drawing_style_visualization(shape: Union[TopoDS_Shape, str]):
    if isinstance(shape, str):
        # 1. Read the STEP file
        shape = read_step_file(shape)
        
    iso_svg = export_shape_to_svg(shape, direction=gp_Dir(1,1,1))
    top_svg = export_shape_to_svg(shape, direction=gp_Dir(0,0,-1))
    bottom_svg = export_shape_to_svg(shape, direction=gp_Dir(0,0,1))
    front_svg = export_shape_to_svg(shape, direction=gp_Dir(0,-1,0))
    back_svg = export_shape_to_svg(shape, direction=gp_Dir(0,1,0))
    left_svg = export_shape_to_svg(shape, direction=gp_Dir(-1,0,0))
    right_svg = export_shape_to_svg(shape, direction=gp_Dir(1,0,0))

    iso_svg_ = st.fromstring(iso_svg)
    iso_h_w = (float(iso_svg_.get_size()[0]), float(iso_svg_.get_size()[1]))
    iso_svg_ = iso_svg_.getroot()
    top_svg_ = st.fromstring(top_svg)
    top_h_w = (float(top_svg_.get_size()[0]), float(top_svg_.get_size()[1]))
    top_svg_ = top_svg_.getroot()
    bottom_svg_ = st.fromstring(bottom_svg)
    bottom_h_w = (float(bottom_svg_.get_size()[0]), float(bottom_svg_.get_size()[1]))
    bottom_svg_ = bottom_svg_.getroot()
    front_svg_ = st.fromstring(front_svg)
    front_h_w = (float(front_svg_.get_size()[0]), float(front_svg_.get_size()[1]))
    front_svg_ = front_svg_.getroot()
    back_svg_ = st.fromstring(back_svg)
    back_h_w = (float(back_svg_.get_size()[0]), float(back_svg_.get_size()[1]))
    back_svg_ = back_svg_.getroot()
    left_svg_ = st.fromstring(left_svg)
    left_h_w = (float(left_svg_.get_size()[0]), float(left_svg_.get_size()[1]))
    left_svg_ = left_svg_.getroot()
    right_svg_ = st.fromstring(right_svg)
    right_h_w = (float(right_svg_.get_size()[0]), float(right_svg_.get_size()[1]))
    right_svg_ = right_svg_.getroot()

    spacing = 20

    ###################################
    ####       top             iso ####
    #### left front right back     ####
    ####      bottom               ####
    ###################################

    total_width = iso_h_w[0]+ back_h_w[0] + right_h_w[0] + front_h_w[0] + left_h_w[0] + 6 * spacing
    total_height = max(iso_h_w[1], top_h_w[1]) + bottom_h_w[1] + max(left_h_w[1], front_h_w[1], right_h_w[1], back_h_w[1]) + 4 * spacing

    bottom_position = [left_h_w[0] + spacing * 2 + front_h_w[0]/2 - bottom_h_w[0]/2, spacing]
    left_position = [spacing, bottom_h_w[1] + 2 * spacing]
    front_position = [left_h_w[0] + spacing * 2, bottom_h_w[1] + 2 * spacing]
    right_position = [front_position[0] + front_h_w[0] + spacing , bottom_h_w[1] + 2 * spacing]
    back_position = [right_position[0] + right_h_w[0] + spacing, bottom_h_w[1] + 2 * spacing]
    top_position = [left_h_w[0] + spacing * 2 + front_h_w[0]/2 - top_h_w[0]/2, front_position[1] + front_h_w[1] + spacing]
    iso_position = [back_position[0] + back_h_w[0] + spacing, max(spacing, top_h_w[1]/2 - iso_h_w[1]/2)]

    bottom_svg_.moveto(bottom_position[0], bottom_position[1])
    left_svg_.moveto(left_position[0], left_position[1])
    front_svg_.moveto(front_position[0], front_position[1])
    right_svg_.moveto(right_position[0], right_position[1])
    back_svg_.moveto(back_position[0], back_position[1])
    top_svg_.moveto(top_position[0], top_position[1])
    iso_svg_.moveto(iso_position[0], iso_position[1])

    fig = Figure(total_width, 
                total_height,
                bottom_svg_,
                left_svg_,
                front_svg_,
                right_svg_,
                back_svg_,
                top_svg_,
                iso_svg_
    )

    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=True)
    svg2png(bytestring=fig.tostr(), write_to=temp_file.name, dpi=150)
    drawing_style_image = Image.open(temp_file.name)
    temp_file.close()
    
    return drawing_style_image, fig.tostr().decode()

def comprehensive_visualization(shape: Union[TopoDS_Shape, str], 
                                material='metal', 
                                color=(1.0, 0.8862745098039215, 0.6392156862745098), 
                                show_edge: bool = True,
                                edge_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                fancy_lighting: bool = True
                                ):

    if isinstance(shape, str):
        # 1. Read the STEP file
        shape = read_step_file(shape)
        
    drawing_style_image, fig = drawing_style_visualization(shape)

    # get the mesh and edges
    verts, tris, faces, edges = extract_renderables(shape)
    
    # bounding box normalization and centering
    verts, edges = prepare_mesh_and_edges(verts, edges)
    
    if isinstance(color, str) and color == 'random':
        color = tuple(np.random.rand(3).tolist())
        
    iso_image = render_object(
        shape,
        draw_edges=show_edge,
        material=material,
        edge_color=edge_color,
        color=color,
        realistic_camera=False,
        view='isometric',
        ambient_only=not fancy_lighting,
        ambient_light_strength=0.1 if fancy_lighting else 1.0)
    
    top_image = render_object(
        shape,
        draw_edges=show_edge,
        material=material,
        edge_color=edge_color,
        color=color,
        realistic_camera=False,
        view='top',
        ambient_only=not fancy_lighting,
        ambient_light_strength=0.1 if fancy_lighting else 1.0)
    
    bottom_image = render_object(
        shape,
        draw_edges=show_edge,
        material=material,
        edge_color=edge_color,
        color=color,
        realistic_camera=False,
        view='bottom',
        ambient_only=not fancy_lighting,
        ambient_light_strength=0.1 if fancy_lighting else 1.0)
    
    front_image = render_object(
        shape,
        draw_edges=show_edge,
        material=material,
        edge_color=edge_color,
        color=color,
        realistic_camera=False,
        view='front',
        ambient_only=not fancy_lighting,
        ambient_light_strength=0.1 if fancy_lighting else 1.0)
    
    back_image = render_object(
        shape,
        draw_edges=show_edge,
        material=material,
        edge_color=edge_color,
        color=color,
        realistic_camera=False,
        view='back',
        ambient_only=not fancy_lighting,
        ambient_light_strength=0.1 if fancy_lighting else 1.0)
    
    left_image = render_object(
        shape,
        draw_edges=show_edge,
        material=material,
        edge_color=edge_color,
        color=color,
        realistic_camera=False,
        view='left',
        ambient_only=not fancy_lighting,
        ambient_light_strength=0.1 if fancy_lighting else 1.0)
    
    right_image = render_object(
        shape,
        draw_edges=show_edge,
        material=material,
        edge_color=edge_color,
        color=color,
        realistic_camera=False,
        view='right',
        ambient_only=not fancy_lighting,
        ambient_light_strength=0.1 if fancy_lighting else 1.0)
    
    random_image = render_object(
        shape,
        draw_edges=show_edge,
        material=material,
        edge_color=edge_color,
        color=color,
        realistic_camera=True,
        view='random',
        relative_camera_distance='random',
        ambient_only=not fancy_lighting,
        ambient_light_strength=0.1 if fancy_lighting else 1.0)
    
    return {
        'drawing_style': drawing_style_image,
        'drawing_style_svg': fig,
        'isometric': iso_image,
        'top': top_image,
        'bottom': bottom_image,
        'front': front_image,
        'back': back_image,
        'left': left_image,
        'right': right_image,
        'random': random_image
    }


# Specific multi-view visualizations

def gray_cad_image(cad_file, save_dir='data/images/singleview_images', use_gpu=True):
        
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(cad_file))[0]

    isometric_views = [
    (1, 1, 1),    # Front-right-top
    # (-1, 1, 1),   # Front-left-top
    # (-1, -1, 1),  # Back-left-top
    # (1, -1, 1),   # Back-right-top
    # (1, 1, -1),   # Front-right-bottom
    # (-1, 1, -1),  # Front-left-bottom
    # (-1, -1, -1), # Back-left-bottom
    # (1, -1, -1)   # Back-right-bottom
    ]

    images = []

    for i, iso_type in enumerate(isometric_views):
        
        cam_dist = 1.5
        bad_cropping = True
        
        while bad_cropping:

            # single render of a brep
            image = render_object(
                cad_file,
                draw_edges =  True,
                material = 'metal',
                color = (191/256, 190/256, 186/256), # could be hex , per_face_random, per_face_tableau
                edge_color = (0.0, 0.0, 0.0),
                realistic_camera = False, # isomorphic or perspective camera (isomorphic is better for CAD models)
                ambient_only = False, # only ambient light
                view = 'isometric', # top, bottom, left, right, front, back, isometric, random
                relative_camera_distance = "auto", # distance of camera from the model, could be "auto" or a float value, or random
                isometric_setup = iso_type, # direction of isometric view
                ambient_light_strength =  0.1, # increase if ambient only is True
                light_strength = 800.0,
                light_size = 10.0,
                relative_light_distance = 20.0,
                light_color = (1.0, 1.0, 1.0),
                resolution = (1280, 1280),
                random_distance_range = "auto",
                use_gpu = use_gpu,
                num_samples = 256,
                albedo = False, # render albedo pass if true multiple outputs
                depth = False, # render depth if true multiple outputs,
                camera_dist = cam_dist
            )

            # print(f"Image size: {image.size}")
            # Composite image onto white background
            white_bg = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                white_bg.paste(image, (0, 0), image)  # Use alpha channel as mask
            else:
                white_bg.paste(image, (0, 0))
                
            bad_cropping = is_badly_cropped(np.array(white_bg), border=5, threshold=250)
            print(f"View {i}: bad cropping = {bad_cropping} with camera distance = {cam_dist}")
            cam_dist += 0.5

        # save individual images
        white_bg.save(f'{save_dir}/{base_filename}_{i}.png')

        # store for compound image
        images.append(white_bg)
        
        

    # # Create compound image with black line separators
    # if images:
    #     separator_width = 10  # width of black lines in pixels
    #     img_width, img_height = images[0].size

    #     # Calculate grid layout (2x2 for 4 images)
    #     cols = 2
    #     rows = (len(images) + cols - 1) // cols  # ceiling division

    #     # Calculate compound image dimensions
    #     compound_width = cols * img_width + (cols - 1) * separator_width
    #     compound_height = rows * img_height + (rows - 1) * separator_width

    #     # Create compound image with white background
    #     compound = Image.new('RGB', (compound_width, compound_height), (255, 255, 255))

    #     # Paste images with separators
    #     for idx, img in enumerate(images):
    #         row = idx // cols
    #         col = idx % cols

    #         x = col * (img_width + separator_width)
    #         y = row * (img_height + separator_width)

    #         compound.paste(img, (x, y))

    #     # Draw black separators
    #     from PIL import ImageDraw
    #     draw = ImageDraw.Draw(compound)

    #     # Vertical separators
    #     for col in range(1, cols):
    #         x = col * img_width + (col - 1) * separator_width
    #         draw.rectangle([x, 0, x + separator_width - 1, compound_height], fill=(0, 0, 0))

    #     # Horizontal separators
    #     for row in range(1, rows):
    #         y = row * img_height + (row - 1) * separator_width
    #         draw.rectangle([0, y, compound_width, y + separator_width - 1], fill=(0, 0, 0))

    #     # Save compound image
    #     save_compound = f'{save_dir}/{base_filename}.png'
    #     compound.save(save_compound)
    #     # print(f"Compound image saved as {save_dir}/{base_filename}_compound.png with dimensions: {compound.size}")
        return cad_file, save_compound
    
def multiview_cad_image(cad_file, save_dir='data/images/multiview_images', use_gpu=True):
        
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(cad_file))[0]

    isometric_views = [
    (1, 1, 1),    # Front-right-top
    # (-1, 1, 1),   # Front-left-top
    (-1, -1, 1),  # Back-left-top
    # (1, -1, 1),   # Back-right-top
    # (1, 1, -1),   # Front-right-bottom
    (-1, 1, -1),  # Front-left-bottom
    # (-1, -1, -1), # Back-left-bottom
    (1, -1, -1)   # Back-right-bottom
    ]

    images = []

    for i, iso_type in enumerate(isometric_views):

        cam_dist = 1.5
        bad_cropping = True

        while bad_cropping:

            # single render of a brep
            image = render_object(
                cad_file,
                draw_edges =  True,
                material = 'metal',
                color = (191/256, 190/256, 186/256), # could be hex , per_face_random, per_face_tableau
                edge_color = (0.0, 0.0, 0.0),
                realistic_camera = False, # isomorphic or perspective camera (isomorphic is better for CAD models)
                ambient_only = False, # only ambient light
                view = 'isometric', # top, bottom, left, right, front, back, isometric, random
                relative_camera_distance = "auto", # distance of camera from the model, could be "auto" or a float value, or random
                isometric_setup = iso_type, # direction of isometric view
                ambient_light_strength =  0.1, # increase if ambient only is True
                light_strength = 800.0,
                light_size = 10.0,
                relative_light_distance = 20.0,
                light_color = (1.0, 1.0, 1.0),
                resolution = (1280, 1280),
                random_distance_range = "auto",
                use_gpu = use_gpu,
                num_samples = 256,
                albedo = False, # render albedo pass if true multiple outputs
                depth = False, # render depth if true multiple outputs
                camera_dist = cam_dist
            )

            # print(f"Image size: {image.size}")
            # Composite image onto white background
            white_bg = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                white_bg.paste(image, (0, 0), image)  # Use alpha channel as mask
            else:
                white_bg.paste(image, (0, 0))

            bad_cropping = is_badly_cropped(np.array(white_bg), border=5, threshold=250)
            cam_dist += 0.5

        # # save individual images
        # white_bg.save(f'{save_dir}/{base_filename}_{i}.png')

        # store for compound image
        images.append(white_bg)

    # Create compound image with black line separators
    if images:
        separator_width = 10  # width of black lines in pixels
        img_width, img_height = images[0].size

        # Calculate grid layout (2x2 for 4 images)
        cols = 2
        rows = (len(images) + cols - 1) // cols  # ceiling division

        # Calculate compound image dimensions
        compound_width = cols * img_width + (cols - 1) * separator_width
        compound_height = rows * img_height + (rows - 1) * separator_width

        # Create compound image with white background
        compound = Image.new('RGB', (compound_width, compound_height), (255, 255, 255))

        # Paste images with separators
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols

            x = col * (img_width + separator_width)
            y = row * (img_height + separator_width)

            compound.paste(img, (x, y))

        # Draw black separators
        from PIL import ImageDraw
        draw = ImageDraw.Draw(compound)

        # Vertical separators
        for col in range(1, cols):
            x = col * img_width + (col - 1) * separator_width
            draw.rectangle([x, 0, x + separator_width - 1, compound_height], fill=(0, 0, 0))

        # Horizontal separators
        for row in range(1, rows):
            y = row * img_height + (row - 1) * separator_width
            draw.rectangle([0, y, compound_width, y + separator_width - 1], fill=(0, 0, 0))

        # Save compound image
        save_compound = f'{save_dir}/{base_filename}.png'
        compound.save(save_compound)
        # print(f"Compound image saved as {save_dir}/{base_filename}_compound.png with dimensions: {compound.size}")
        return cad_file, save_compound

def multicolor_cad_image(cad_file, save_dir='dataset_results/images/multicolor_cad'):
    
    # If the save directory does not exist, create it
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(cad_file))[0]

    isometric_views = [
    (1, 1, 1),    # Front-right-top
    # (-1, 1, 1),   # Front-left-top
    (-1, -1, 1),  # Back-left-top
    # (1, -1, 1),   # Back-right-top
    # (1, 1, -1),   # Front-right-bottom
    (-1, 1, -1),  # Front-left-bottom
    # (-1, -1, -1), # Back-left-bottom
    (1, -1, -1)   # Back-right-bottom
    ]

    images = []

    for i, iso_type in enumerate(isometric_views):

        # single render of a brep
        image = render_object(
            cad_file,
            draw_edges =  True,
            material = 'metal',
            color = "per_face_tableau", # could be hex , per_face_random, per_face_tableau
            edge_color = (0.0, 0.0, 0.0),
            realistic_camera = False, # isomorphic or perspective camera (isomorphic is better for CAD models)
            ambient_only = False, # only ambient light
            view = 'isometric', # top, bottom, left, right, front, back, isometric, random
            relative_camera_distance = "auto", # distance of camera from the model, could be "auto" or a float value, or random
            isometric_setup = iso_type, # direction of isometric view
            ambient_light_strength =  0.1, # increase if ambient only is True
            light_strength = 800.0,
            light_size = 10.0,
            relative_light_distance = 20.0,
            light_color = (1.0, 1.0, 1.0),
            resolution = (1280, 1280),
            random_distance_range = "auto",
            use_gpu = True,
            num_samples = 256,
            albedo = False, # render albedo pass if true multiple outputs
            depth = False, # render depth if true multiple outputs
        )

        # print(f"Image size: {image.size}")
        # Composite image onto white background
        white_bg = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'RGBA':
            white_bg.paste(image, (0, 0), image)  # Use alpha channel as mask
        else:
            white_bg.paste(image, (0, 0))

        # save individual images
        white_bg.save(f'{save_dir}/{base_filename}_{i}.png')

        # store for compound image
        images.append(white_bg)

    # Create compound image with black line separators
    if images:
        separator_width = 10  # width of black lines in pixels
        img_width, img_height = images[0].size

        # Calculate grid layout (2x2 for 4 images)
        cols = 2
        rows = (len(images) + cols - 1) // cols  # ceiling division

        # Calculate compound image dimensions
        compound_width = cols * img_width + (cols - 1) * separator_width
        compound_height = rows * img_height + (rows - 1) * separator_width

        # Create compound image with white background
        compound = Image.new('RGB', (compound_width, compound_height), (255, 255, 255))

        # Paste images with separators
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols

            x = col * (img_width + separator_width)
            y = row * (img_height + separator_width)

            compound.paste(img, (x, y))

        # Draw black separators
        from PIL import ImageDraw
        draw = ImageDraw.Draw(compound)

        # Vertical separators
        for col in range(1, cols):
            x = col * img_width + (col - 1) * separator_width
            draw.rectangle([x, 0, x + separator_width - 1, compound_height], fill=(0, 0, 0))

        # Horizontal separators
        for row in range(1, rows):
            y = row * img_height + (row - 1) * separator_width
            draw.rectangle([0, y, compound_width, y + separator_width - 1], fill=(0, 0, 0))

        # Save compound image
        save_compound = f'{save_dir}/{base_filename}.png'
        compound.save(save_compound)
        # print(f"Compound image saved as {save_dir}/{base_filename}_compound.png with dimensions: {compound.size}")
        return cad_file, save_compound