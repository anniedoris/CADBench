import numpy as np
from PIL import Image
import os
import trimesh
import pyrender

# Setting EGL for headless rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# CADVis imports
from CADVis.Geom import extract_mesh_and_edges
from CADVis.PyR.Material import ColorTexture, add_uv
from CADVis.PyR.Scene import get_ground_mesh, get_camera_pose_isometric, get_three_light_poses

def render_shape(shape,
                 shadows=True,
                 resolution=(1200, 1200),
                 tol=0.1,
                 normalize=True,
                 show_edges=False,
                 camera_position='isometric', 
                 ground=True, 
                 ground_material=None, 
                 model_material=None,
                 ground_color="#B1AAAC",
                 model_color="#004e89",
                 light_color=(1., 1., 1.),
                 light_intensity=1.0,  # Reverted back to 1.0
                 mesh_uv_prjection='spherical',
                 mesh_uv_scale=2.0,
                 ground_dim=np.array([50,50]),
                 ground_uv_scale=None,
                 z_rotation_deg=90):
    
    # --- 1. Data Extraction ---
    if isinstance(shape, (tuple, list)) and len(shape) == 2:
        verts, faces = shape
        curves = []
        z_min = np.min(verts[:, 2])
    else:
        # Step Extraction
        verts, faces, curves, z_min = extract_mesh_and_edges(
            shape, tol=tol, normalize=False, apply_random_rotation=False, ReturnBottom=True
        )

    # --- 2. Apply Rotation (90 deg around Z) ---
    if z_rotation_deg != 0:
        theta = np.radians(z_rotation_deg)
        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        verts = verts @ Rz.T
        if curves:
            curves = [curve @ Rz.T for curve in curves]

    # --- 3. Normalization & Centering ---
    obj_center = verts.mean(axis=0)
    if normalize:
        verts = verts - obj_center
        if curves:
            curves = [c - obj_center for c in curves]
        z_min = z_min - obj_center[2]
        cam_target = np.array([0., 0., 0.])
    else:
        cam_target = obj_center

    # --- 4. Mesh Sanitization & UVs ---
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.process(validate=True)
    mesh = add_uv(mesh, method=mesh_uv_prjection, scale=mesh_uv_scale)
    
    # Reverted back to default (no ambient light)
    scene = pyrender.Scene()
    
    # --- 5. Materials & Ground ---
    if ground_material is None:
        ground_material = ColorTexture(color=ground_color)
    if model_material is None:
        model_material = ColorTexture(color=model_color)
    
    g_mat = ground_material.get_material() if hasattr(ground_material, "get_material") else ground_material
    m_mat = model_material.get_material() if hasattr(model_material, "get_material") else model_material
    
    radius = np.max(np.linalg.norm(verts - cam_target, axis=1))
    if radius == 0: radius = 1.0 
    
    if ground:
        g_mesh = get_ground_mesh(
            dims=np.array([radius*6, radius*6]),
            origin=np.array([cam_target[0], cam_target[1], z_min]),
            up=np.array([0, 0, 1]),
            texture_scale=ground_uv_scale
        )
        scene.add(pyrender.Mesh.from_trimesh(g_mesh, material=g_mat, smooth=False))

    if hasattr(m_mat, 'doubleSided'):
        m_mat.doubleSided = True 
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=m_mat, smooth=False))
    
    # --- 6. Edge Rendering ---
    if show_edges and curves:
        edge_verts, edge_ids = [], []
        for c in curves:
            start_idx = len(edge_verts)
            edge_verts += c.tolist()
            edge_ids += np.vstack([np.arange(start_idx, start_idx + len(c) - 1), 
                                   np.arange(start_idx + 1, start_idx + len(c))]).T.tolist()
        if edge_verts:
            all_curves = np.array(edge_verts)[np.array(edge_ids)]
            lines = [pyrender.Primitive(all_curves[i], mode=1, color_0=[0,0,0]) for i in range(len(all_curves))]
            scene.add(pyrender.Mesh(lines))

    # --- 7. Sphere-Based Camera Framing ---
    fov_y = np.deg2rad(60)
    camera_distance = (radius / np.sin(fov_y / 2)) * 1.3
    
    # Dynamic Clipping Planes
    znear = max(0.01, camera_distance - radius * 2)
    zfar = camera_distance + radius * 10
    
    camera = pyrender.PerspectiveCamera(yfov=fov_y, znear=znear, zfar=zfar)
    cam_direction = np.array([-1., -1., 1.])
    cam_direction /= np.linalg.norm(cam_direction)
    cam_pos = cam_target + (cam_direction * camera_distance)
    
    camera_pose = get_camera_pose_isometric(position=cam_pos, target=cam_target, up=[0,0,1])
    scene.add(camera, pose=camera_pose)

    # --- 8. Lighting (REVERTED TO ORIGINAL) ---
    lp = get_three_light_poses(camera_position=cam_pos, target_position=cam_target, up_vector=[0,0,1])
    scene.add(pyrender.DirectionalLight(color=light_color, intensity=4.0 * light_intensity), pose=lp[0])
    scene.add(pyrender.DirectionalLight(color=light_color, intensity=1.0 * light_intensity), pose=lp[1])
    scene.add(pyrender.DirectionalLight(color=light_color, intensity=1.0 * light_intensity), pose=lp[2])

    # --- 9. Render & Data Type Safety ---
    r = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
    color, depth = r.render(scene, flags=pyrender.RenderFlags.ALL_SOLID | pyrender.RenderFlags.SHADOWS_ALL)
    r.delete()

    color_img = Image.fromarray(color.astype(np.uint8))
    
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth_scaled = (depth - d_min) / (d_max - d_min) * 255
        depth_img = Image.fromarray(depth_scaled.astype(np.uint8))
    else:
        depth_img = Image.fromarray(np.zeros(resolution, dtype=np.uint8))

    return color_img, depth_img