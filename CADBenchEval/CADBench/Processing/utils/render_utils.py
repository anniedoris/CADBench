import os
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
from OCC.Display.OCCViewer import Viewer3d
from OCC.Core.Graphic3d import Graphic3d_NOM_SILVER
import tempfile
from multiprocessing import Process, Queue

from contextlib import contextmanager
import sys
from contextlib import redirect_stdout, redirect_stderr


@contextmanager
def suppress_output_os():
    if sys.platform == "win32":
        yield
        return
    stdout_fd, stderr_fd = 1, 2
    saved_stdout_fd, saved_stderr_fd = os.dup(stdout_fd), os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    try:
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            yield

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

def quick_render(verts, tris, edges=None, resolution=(1280, 1280)):
    if edges is None:
        edges = []

    verts, edges = prepare_mesh_and_edges(verts, edges)

    lights = setup_scene(
        scene,
        realistic_camera=False,
        ambient_only=False,
        view='isometric',
        relative_camera_distance=1.5,
        isometric_setup=(1, 1, 1),
        ambient_light_strength=0.1,
        light_strength=800.0,
        light_size=10.0,
        relative_light_distance=20.0,
        light_color=(1.0, 1.0, 1.0),
        resolution=resolution,
    )

    centroid = (verts.max(0) + verts.min(0)) / 2
    scene.camera.blender_camera.location = np.array(scene.camera.blender_camera.location) + centroid
    for light in lights:
        light.blender_light.location = np.array(light.blender_light.location) + centroid

    # mat = MetalMaterial(metallic=0.0, roughness=0.5)
    mat = PlasticMaterial()
    color = UniformColors((191/256, 190/256, 186/256))
    cad = scene.renderables.add_mesh(verts, tris, material=mat, colors=color)
    cad.set_smooth(True)

    if len(edges) > 0:
        edge_color = (0.0, 0.0, 0.0)
        curve_mat = get_curve_material(edge_color=edge_color)
        for i, curve in enumerate(edges):
            curveData = bpy.data.curves.new(f'Curve_{i}', type='CURVE')
            curveData.dimensions = '3D'
            curveData.resolution_u = 2
            polyline = curveData.splines.new('POLY')
            polyline.points.add(len(curve)-1)
            for j, coord in enumerate(curve):
                x, y, z = coord
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
                curveOB.color = (*edge_color, 1.0)
            except Exception:
                pass
            bpy.data.scenes[0].collection.objects.link(curveOB)

    render = scene.render(use_gpu=False, samples=32)
    render = Image.fromarray(render)

    white_bg = Image.new('RGB', render.size, (255, 255, 255))
    if render.mode == 'RGBA':
        white_bg.paste(render, (0, 0), render)
    else:
        white_bg.paste(render, (0, 0))

    return white_bg

def quick_render_occ(shape: TopoDS_Shape, resolution=(448, 448)) -> Image.Image:
    offscreen_renderer = Viewer3d()
    offscreen_renderer.Create()
    offscreen_renderer.View_Iso()
    offscreen_renderer.SetModeShaded()
    offscreen_renderer.DisplayShape(shape, update=True, material=Graphic3d_NOM_SILVER, transparency=0.0)
    offscreen_renderer.View.SetBackgroundColor(0, 1, 1, 1)
    offscreen_renderer.View.FitAll(0.5)
    offscreen_renderer.SetSize(resolution[0], resolution[1])

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name

    offscreen_renderer.View.Dump(tmp_path)
    img = Image.open(tmp_path).convert('RGB')
    os.remove(tmp_path)

    return img

def quick_render_silent(verts, tris, edges=None, resolution=(1280, 1280)):
    with suppress_output_os(), suppress_output():
        return quick_render(verts, tris, edges=edges, resolution=resolution)

def _quick_render_worker(verts, tris, edges, resolution, queue):
    with suppress_output_os(), suppress_output():
        try:
            result = quick_render(verts, tris, edges=edges, resolution=resolution)
            queue.put(result)
        except Exception:
            queue.put(None)


def quick_render_safe(verts, tris, edges=None, resolution=(1280, 1280), timeout=10):
    queue = Queue()
    process = Process(target=_quick_render_worker, args=(verts, tris, edges, resolution, queue))
    process.start()

    try:
        result = queue.get(timeout=timeout)
    except Exception:
        process.kill()
        return None

    process.join(timeout=5)
    if process.is_alive():
        process.kill()

    return result