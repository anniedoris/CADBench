import numpy as np
from PIL import Image
import polyscope as ps

class Texture:
    def __init__(self, image_path: str, filter_mode: str = 'linear', projection: str = 'spherical', planar_axis: str = 'z'):
        """
        image_path   : path to your texture image file
        filter_mode  : 'linear' for bilinear interpolation, 'nearest' for point sampling
        """
        self.image_path = image_path
        self.filter_mode = filter_mode
        # load image once
        img = Image.open(self.image_path).convert("RGB")
        self.tex = np.asarray(img, dtype=np.float32) / 255.0  # H×W×3 in [0,1]
        self.projection = projection
        self.planar_axis = planar_axis

    def apply(self,
              ps_mesh,
              verts: np.ndarray):
        """
        ps_mesh     : polyscope SurfaceMesh object from register_surface_mesh()
        verts       : (N,3) vertex array used to build that mesh
        projection  : 'spherical' or 'planar'
        planar_axis : if projection=='planar', drop this axis ('x','y','z')
        """
        # 1) compute UVs
        projection = self.projection
        planar_axis = self.planar_axis
        if projection == 'spherical':
            Vn = verts / np.linalg.norm(verts, axis=1, keepdims=True)
            x, y, z = Vn[:,0], Vn[:,1], Vn[:,2]
            u = 0.5 + (np.arctan2(z, x) / (2*np.pi))
            v = 0.5 - (np.arcsin(y)    / np.pi)
            uv = np.vstack([u, v]).T
        else:
            idx = {'x':(1,2), 'y':(0,2), 'z':(0,1)}[planar_axis]
            uv_raw = verts[:, idx]
            mn, mx = uv_raw.min(axis=0), uv_raw.max(axis=0)
            uv = (uv_raw - mn) / (mx - mn)

        # 2) register UV map
        ps_mesh.add_parameterization_quantity(
            "uv map", uv,
            defined_on='vertices',
            coords_type='unit'
        )

        # 3) register texture color
        ps_mesh.add_color_quantity(
            "texture", self.tex,
            defined_on='texture',
            param_name="uv map",
            filter_mode=self.filter_mode,
            enabled=True
        )
