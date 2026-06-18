import numpy as np
import polyscope as ps
from ..Geom import extract_mesh_and_edges
from .Texture import Texture

def render_shape(shape, draw_edges=True, shadow=True, resolution=(1024,1024), transparend_bg=False, tol=0.1, normalize=True, apply_random_rotation=False, texture : Texture = None, solid_args={}, edge_args={}):
    """Render the shape using Polyscope."""
    
    ps.set_autocenter_structures(True)
    ps.set_autoscale_structures(True)
    ps.init("openGL3_egl")
    
    verts, faces, curves = extract_mesh_and_edges(shape, tol=tol, normalize=normalize, apply_random_rotation=apply_random_rotation, ReturnBottom=False)
    
    # Add the mesh to Polyscope
    if 'color' not in solid_args and texture is None:
        solid_args['color'] = (0.5, 0.5, 0.5)
    
    mesh = ps.register_surface_mesh("shape", verts, faces, **solid_args)
    
    if texture is not None:
        texture.apply(mesh, verts)
    
    if draw_edges:
        # Add the edges to Polyscope
        edge_verts = []
        edge_ids = []
        for i, c in enumerate(curves):
            start_idx = len(edge_verts)
            end_idx = start_idx + len(c)
            f = np.vstack([np.arange(start_idx, end_idx-1), np.arange(start_idx+1, end_idx)]).T
            edge_verts += c.tolist()
            edge_ids += f.tolist()
        edge_verts = np.array(edge_verts)
        edge_ids = np.array(edge_ids)
        ps.register_curve_network('edges', edge_verts, edge_ids, radius=0.002, color=(0,0,0))
    
    ps.set_ground_plane_height_factor(0.08)
    ps.set_up_dir("z_up")
    if shadow:
        ps.set_ground_plane_mode("shadow_only")
    else:
        ps.set_ground_plane_mode("none")
    ps.look_at((-1.0,-1.0,1.0),(.0, .0, .0))
    ps.set_background_color((1,1,1))
    ps.set_window_size(*resolution)
    # ps.screenshot(save_name, transparent_bg=transparend_bg)
    img = ps.screenshot_to_buffer(transparent_bg=transparend_bg)
    return np.array(img)