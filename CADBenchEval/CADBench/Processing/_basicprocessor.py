from ._base import Processor
from datasets import Dataset
from typing import Optional, Dict, Any
import trimesh
from io import BytesIO
from PIL import Image
from .utils.pc_utils import process_mesh
import tempfile


class BasicProcessor(Processor):
    def __init__(self, 
                 dataset: Dataset,
                 image: Optional[bool] = True,
                 point_cloud: Optional[bool] = False,
                 mesh: Optional[bool] = False,
                 CAD: Optional[bool] = False,
                 num_points: int = 10000,
                 image_size: Optional[int] = 512,
                 image_field: Optional[str] = None,
                 id_field: str = "file_id",
                 render_backend: Optional[str] = "blender",
                 render_style: Optional[str] = "mesh"):
        super().__init__(dataset)
        self.image = image
        self.mesh = mesh
        self.CAD = CAD
        self.render_backend = render_backend
        self.render_style = render_style
        self.num_points = num_points
        self.image_size = image_size
        self.point_cloud = point_cloud
        self.image_field = image_field
        self.id_field = id_field

        sample = self.dataset[0]

        if id_field not in sample:
            raise ValueError(f"ID field '{id_field}' not found in dataset")

        if image_field is not None and image_field not in sample:
            raise ValueError(f"Image field '{image_field}' not found in dataset")

        needs_stl = mesh or point_cloud or (image and image_field is None)
        if needs_stl and "stl" not in sample:
            raise ValueError("STL file not found in dataset")

        if CAD is True and "step" not in sample:
            raise ValueError("STEP file not found in dataset")
        
        if CAD is True:
            raise NotImplementedError("CAD processing not implemented yet")
        
        if image and image_field is not None and not (mesh or point_cloud or CAD):
            return

        if render_backend not in ['blender', 'OCC']:
            raise NotImplementedError("Only Blender and OCC backends are supported for now")
        
        if render_backend == 'OCC' and render_style != 'CAD':
            raise ValueError("OCC backend only supports CAD rendering style")
        
        if render_style == 'CAD' and 'step' not in sample:
            raise ValueError("STEP file not found in dataset. If CAD rendering style is selected, STEP file is required.")

    @staticmethod
    def _load_dataset_image(image_value: Any) -> Image.Image:
        if isinstance(image_value, Image.Image):
            return image_value.convert("RGB")

        if isinstance(image_value, bytes):
            return Image.open(BytesIO(image_value)).convert("RGB")

        if isinstance(image_value, dict):
            if image_value.get("bytes") is not None:
                return Image.open(BytesIO(image_value["bytes"])).convert("RGB")
            if image_value.get("path") is not None:
                return Image.open(image_value["path"]).convert("RGB")

        raise TypeError(
            "Unsupported image value type from dataset field: "
            f"{type(image_value).__name__}"
        )

    @staticmethod
    def _load_step_geometry(step_bytes: bytes):
        from OCC.Extend.DataExchange import read_step_file
        from .utils.cad_utils import extract_mesh_and_edges

        temp_file = tempfile.NamedTemporaryFile(delete=True)
        temp_file.write(step_bytes)
        temp_file.flush()
        cad = read_step_file(temp_file.name)
        verts, tris, curves = extract_mesh_and_edges(cad)
        temp_file.close()
        return cad, verts, tris, curves

    @staticmethod
    def _render_cad_blender_image(verts, tris, curves, image_size: int) -> Image.Image:
        from .utils.render_utils import quick_render_silent

        return quick_render_silent(
            verts,
            tris,
            curves,
            resolution=(image_size, image_size),
        )

    @staticmethod
    def _render_occ_image(cad, image_size: int) -> Image.Image:
        from .utils.render_utils import quick_render_occ

        return quick_render_occ(
            cad,
            resolution=(image_size, image_size),
        )

    @staticmethod
    def _render_mesh_image(verts, tris, image_size: int) -> Image.Image:
        from .utils.render_utils import quick_render_silent

        return quick_render_silent(
            verts,
            tris,
            resolution=(image_size, image_size),
        )

    def __call__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]

        outputs = {}

        mesh = None
        if self.mesh or self.point_cloud or (self.image and self.image_field is None):
            mesh = trimesh.load(BytesIO(item["stl"]), file_type="stl")

        if self.mesh:
            outputs["mesh"] = (mesh.vertices, mesh.faces)

        if self.point_cloud:
            points, _, _ = process_mesh(mesh, self.num_points)
            outputs["point_cloud"] = points

        if self.image:
            if self.image_field is not None:
                image_value = item[self.image_field]
                if image_value is None:
                    raise ValueError(
                        f"Image field '{self.image_field}' is None for "
                        f"{item[self.id_field]}"
                    )
                image = self._load_dataset_image(image_value)
                if self.image_size is not None:
                    image = image.resize((self.image_size, self.image_size))
                outputs["image"] = image
            elif self.render_style == 'CAD':
                cad, verts, tris, curves = self._load_step_geometry(item["step"])

                if self.render_backend == 'blender':
                    image = self._render_cad_blender_image(
                        verts,
                        tris,
                        curves,
                        self.image_size,
                    )
                    outputs["image"] = image
                elif self.render_backend == 'OCC':
                    image = self._render_occ_image(cad, self.image_size)
                    outputs["image"] = image
            else:
                if 'step' in item:
                    _, verts, tris, _ = self._load_step_geometry(item["step"])
                else:
                    verts, tris = mesh.vertices, mesh.faces
                image = self._render_mesh_image(verts, tris, self.image_size)
                outputs["image"] = image

        return outputs, item[self.id_field]
