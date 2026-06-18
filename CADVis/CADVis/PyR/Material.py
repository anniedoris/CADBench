import pyrender
import numpy as np
from PIL import Image
from pyrender import MetallicRoughnessMaterial, Texture, Material
import trimesh
from trimesh.visual.texture import TextureVisuals
import os


class PBRMaterial:
    def __init__(self, base_color_image = None, normal_image = None, metallic_image = None, roughness_image = None, ambient_occlusion_image = None):
        self.base_color_image = base_color_image
        self.metallic_image = metallic_image
        self.roughness_image = roughness_image
        self.ambient_occlusion_image = ambient_occlusion_image
        self.normal_image = normal_image
        
        if base_color_image is None and metallic_image is None and roughness_image is None and ambient_occlusion_image is None:
            raise ValueError("At least one texture image must be provided.")
        
        if base_color_image is None:
            raise ValueError("Base color image must be provided.")
        base_img   = np.array(Image.open(base_color_image).convert('RGBA'))
        if normal_image is not None:
            normal_img = np.array(Image.open(normal_image).convert('RGB'))
        else:
            normal_img = None
        if roughness_image is not None:
            rough_arr  = np.array(Image.open(roughness_image).convert('L'))
        else:
            rough_arr = np.ones(base_img.shape[:2], dtype=np.uint8) * 0
        if metallic_image is not None:
            metal_arr  = np.array(Image.open(metallic_image).convert('L'))
        else:
            metal_arr = np.ones(rough_arr.shape, dtype=np.uint8) * 0

        if ambient_occlusion_image is not None:
            ao_img     = np.array(Image.open(ambient_occlusion_image).convert('RGB'))
        else:
            ao_img = None
            
        h, w       = rough_arr.shape
        
        mr_arr           = np.zeros((h, w, 2), dtype=np.uint8)
        mr_arr[..., 0]   = rough_arr
        mr_arr[..., 1]   = metal_arr
        
        if metallic_image is None and roughness_image is None:
            mr_arr = None
        
        self.material = MetallicRoughnessMaterial(
            baseColorTexture         = base_img,
            normalTexture            = normal_img,
            metallicRoughnessTexture = mr_arr,
            occlusionTexture         = ao_img
        )
        
    def get_material(self):
        return self.material

class AutoPBR(PBRMaterial):
    def __init__(self, folder):
        base_color_image = None
        normal_image     = None
        metallic_image   = None
        roughness_image  = None
        ambient_occlusion_image = None
        
        for file in os.listdir(folder):
            if file.endswith('.png') or file.endswith('.jpg'):
                if 'base' in file.lower() or 'color' in file.lower() or 'diffuse' in file:
                    base_color_image = os.path.join(folder, file)
                elif 'normal' in file.lower():
                    normal_image = os.path.join(folder, file)
                elif 'metallic' in file.lower():
                    metallic_image = os.path.join(folder, file)
                elif 'roughness' in file.lower():
                    roughness_image = os.path.join(folder, file)
                elif 'ao' in file.lower() or 'ambient' in file.lower():
                    ambient_occlusion_image = os.path.join(folder, file)
        
        super().__init__(base_color_image, normal_image, metallic_image, roughness_image, ambient_occlusion_image)

class ColorTexture:
    def __init__(self, color="#7f7f7f", texture = None):
        
        if isinstance(texture, str):
            texture = np.array(Image.open(texture).convert('RGB'))
        elif isinstance(texture, Image.Image):
            texture = np.array(texture.convert('RGB'))
        elif texture is not None and not isinstance(texture, np.ndarray):
            raise ValueError("Texture must be a file path, PIL Image, or numpy array.")

        if texture is not None:
            self.texture = Texture(
                source=texture,
                source_channels='RGB'
            )
        else:
            # turn the color into a texture
            color = np.array([int(color[i:i+2], 16) for i in (1, 3, 5)]).reshape((1, 1, 3))
            self.texture = Texture(
                source=color,
                source_channels='RGB'
            )
            
        self.texture = MetallicRoughnessMaterial(baseColorTexture=self.texture)

    def get_material(self):
        return self.texture

def _add_spherical_uv(mesh: trimesh.Trimesh, scale=1.0) -> trimesh.Trimesh:
    """
    Generate spherical UVs from vertex positions and attach them so that
    Mesh.from_trimesh(...) will see valid TEXCOORD_0 data.
    """
    verts = mesh.vertices
    x, y, z = verts[:,0], verts[:,1], verts[:,2]
    r = np.linalg.norm(verts, axis=1)
    r[r == 0] = 1.0

    u = 0.5 + (np.arctan2(z, x) / (2*np.pi))
    v = 0.5 + (np.arcsin(y / r) / np.pi)
    mesh.visual = TextureVisuals(uv=np.column_stack((u, v))*scale)
    return mesh

def _add_cylindrical_uv(mesh, axis=2, scale=1.0):
    """
    Simple cylindrical UV project around the given axis (0=X,1=Y,2=Z).
    - u wraps around the circle orthogonal to `axis`
    - v is the linear coordinate along the `axis` direction
    """
    verts = mesh.vertices
    # pick which two axes to use for angle
    a1, a2 = [(0,1),(0,2),(1,2)][axis]
    # compute angle around cylinder
    theta = np.arctan2(verts[:,a2], verts[:,a1])  # [-π,π]
    u = (theta + np.pi) / (2*np.pi)
    # v = position along axis, normalized
    coord = verts[:, axis]
    v = (coord - coord.min()) / (coord.max() - coord.min())
    mesh.visual = TextureVisuals(uv=np.column_stack((u, v))*scale)
    return mesh

def _add_box_uv(mesh, scale=1.0):
    """
    Axis‐aligned box mapping: for each vertex, pick the face-normal’s
    dominant axis and project onto the other two axes.
    This is like unfolding a cube around your mesh’s bounding box.
    """
    verts  = mesh.vertices
    norms  = mesh.vertex_normals
    # dimensions for normalization
    mins, maxs = verts.min(0), verts.max(0)
    extents    = maxs - mins

    uv = np.zeros((len(verts), 2))
    for i, (v, n) in enumerate(zip(verts, norms)):
        # choose the projection plane by largest |normal component|
        major = np.argmax(np.abs(n))
        # axes to keep for UV
        axes = [0,1,2]
        axes.remove(major)
        # normalize to [0,1]
        uv[i,0] = (v[axes[0]] - mins[axes[0]]) / extents[axes[0]]
        uv[i,1] = (v[axes[1]] - mins[axes[1]]) / extents[axes[1]]

    mesh.visual = TextureVisuals(uv=uv*scale)
    return mesh

def add_uv(mesh: trimesh.Trimesh, method='box', axis=2, scale=1.0):
    """
    Add UV coordinates to a trimesh object using the specified method.
    
    Parameters:
        mesh (trimesh.Trimesh): The trimesh object to which UV coordinates will be added.
        method (str): The method to use for generating UV coordinates. Options are 'box', 'cylindrical', or 'spherical'.
        axis (int): The axis to use for cylindrical projection (0=X, 1=Y, 2=Z).
    
    Returns:
        trimesh.Trimesh: The trimesh object with added UV coordinates.
    """
    if method == 'box':
        return _add_box_uv(mesh, scale)
    elif method == 'cylindrical':
        return _add_cylindrical_uv(mesh, axis, scale)
    elif method == 'spherical':
        return _add_spherical_uv(mesh, scale)
    else:
        raise ValueError("Invalid method. Choose from 'box', 'cylindrical', or 'spherical'.")