from typing import Union, Optional
from .meshing_utils import sharp_remesh_data
from .utils import get_brep_graph, compute_mass_properties, get_mesh
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound, TopoDS_Solid
from OCC.Extend.DataExchange import read_step_file
import numpy as np
import k3d
import fpsample

class BrepGraph:
    def __init__(self, 
                 brep: Union[TopoDS_Solid, TopoDS_Compound, TopoDS_Shape, str],
                 sample_resolution: Optional[int] = 32,
                 single_solid: bool = True,
                 mesh_tolerance: float = 0.1,
                 mesh_quality: int = 8,
                 point_cloud_size = 5000):
        
        if isinstance(brep, str):
            brep = read_step_file(brep)
        
        faces, edges, corners, edgeFace_adj, edgeCorner_adj, solid = get_brep_graph(brep,
                                                                                    sampling_density=sample_resolution,
                                                                                    single_solid=single_solid)
        self.solid = solid
        self.faces = faces
        self.edges = edges
        self.corners = corners
        self.adjacency = edgeFace_adj
        self.ec_adjacency = edgeCorner_adj
        
        mass, center_of_mass, radius_of_gyration, matrix_of_inertia, P = compute_mass_properties(solid)

        self.center_of_mass = center_of_mass
        self.radius_of_gyration = radius_of_gyration
        self.mass = mass
        self.inertia_matrix = matrix_of_inertia
        self.principal_axes = P

        self.brep_mesh = get_mesh(solid, tol=mesh_tolerance)

        self.graph_bbox = np.array([self.faces.reshape(-1, 3).min(0), self.faces.reshape(-1, 3).max(0)])
        self.solid_bbox = np.array([self.brep_mesh[0].min(0), self.brep_mesh[0].max(0)])

        self.mesh = sharp_remesh_data(self.brep_mesh[0], self.brep_mesh[1], quality=mesh_quality)
        if len(self.mesh[0]) < point_cloud_size:
            self.mesh = sharp_remesh_data(self.brep_mesh[0], self.brep_mesh[1], quality=mesh_quality+1)
        pc_ids = fpsample.bucket_fps_kdline_sampling(self.mesh[0], point_cloud_size, h=7)
        self.point_cloud = (self.mesh[0][pc_ids], self.mesh[2][pc_ids])
    
    def visualize_point_cloud(self):
        
        length_scale = max(self.solid_bbox[1] - self.solid_bbox[0])
        ps = length_scale * 0.005
        
        scene = k3d.plot()
        points, norms = self.point_cloud

        p = k3d.points(
            positions=points,
            point_size=ps,
            color=0xffa500
        )
        scene += p

        scene.grid_visible = False
        scene.display()
    
    def visualize_mesh(self, wires: bool = True):
        scene = k3d.plot()
        nodes, tris, norms = self.mesh

        mesh = k3d.mesh(
            nodes,
            tris,
            wireframe=False,
            color=0xffa500,
        )
        scene += mesh
        
        if wires:
            wires = k3d.mesh(
                nodes,
                tris,
                wireframe=True,
                color=0x000000
            )
            scene += wires
        
        scene.grid_visible = False
        scene.display()
    
    def visualize_brep_mesh(self, wires: bool = True):    
        scene = k3d.plot()
        nodes, tris, norms = self.brep_mesh

        mesh = k3d.mesh(
            nodes,
            tris,
            wireframe=False,
            color=0xffa500,
        )
        scene += mesh
        
        if wires:
            wires = k3d.mesh(
                nodes,
                tris,
                wireframe=True,
                color=0x000000
            )
            scene += wires
        
        scene.grid_visible = False
        scene.display()

    def visualize_graph_objects(self):
        length_scale = max(self.solid_bbox[1] - self.solid_bbox[0])
        ps = length_scale * 0.0025
        
        scene = k3d.plot()

        for face in self.faces:
            points = face.reshape(-1, 3)
            p = k3d.points(
                positions=points,
                point_size=ps,
                color=0xff8c00
            )
            scene += p
            
        for edge in self.edges:
            points = edge.reshape(-1, 3)
            l = k3d.line(
                points,
                shader='mesh',
                width=ps,
                color=0x003c50
            )
            scene += l

        corners = self.corners.reshape(-1, 3)
        c = k3d.points(
            positions=corners,
            point_size=10*ps,
            color=0x8b0000
        )
        scene += c

        scene.grid_visible = False
        scene.display()