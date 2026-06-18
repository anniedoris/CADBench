from OCC.Core.Tesselator import ShapeTesselator
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.gp import gp_Vec, gp_Trsf, gp_Pnt, gp_Ax1, gp_Dir
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GCPnts import GCPnts_UniformDeflection
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Extend.DataExchange import read_step_file
from typing import Union, List, Tuple
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopAbs import TopAbs_REVERSED
import numpy as np

def extract_mesh_and_edges(shape : Union[TopoDS_Shape, str], tol=0.1, return_bounding_box=True) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    
    if isinstance(shape, str):
        # 1. Read the STEP file
        shape = read_step_file(shape)
    
    if return_bounding_box:
        # get bounding box
        bbox = Bnd_Box()
        # bbox.Add(shape)
        bbox.SetGap(1e-6)
        brepbndlib.Add(shape, bbox, False)
        bounding_box = bbox.Get()
        bounding_box = np.array(bounding_box).reshape(2,-1)
    
    tess = ShapeTesselator(shape)
    tess.Compute(mesh_quality=tol)
    verts = [tess.GetVertex(i) for i in range(tess.ObjGetVertexCount())]
    faces = [tess.GetTriangleIndex(i) for i in range(tess.ObjGetTriangleCount())]

    curves = []
    for edge in TopologyExplorer(shape).edges():
        # 3. Wrap as a curve
        adaptor = BRepAdaptor_Curve(edge)
        sampler = GCPnts_UniformDeflection(
            adaptor,
            tol/20,
            adaptor.FirstParameter(),
            adaptor.LastParameter()
        )
        c = []
        for i in range(1, sampler.NbPoints()+1):
            p = adaptor.Value(sampler.Parameter(i))
            c.append(p.Coord())
        c = np.array(c)
        curves.append(c)

    if return_bounding_box:
        return np.array(verts), np.array(faces), curves, bounding_box

    return np.array(verts), np.array(faces), curves

def extract_renderables(shape : Union[TopoDS_Shape, str], tol=0.0005, angle_tol=0.1):
    
    if isinstance(shape, str):
        # 1. Read the STEP file
        shape = read_step_file(shape)
    
    mesh = BRepMesh_IncrementalMesh(
            shape,
            tol,
            True,
            angle_tol,
            True,
        )
    mesh.Perform()
    
    if not mesh.IsDone():
        raise RuntimeError("Meshing failed")
    
    explorer = TopologyExplorer(shape)
    
    verts = np.empty((0,3), dtype=np.float32)
    tris = np.empty((0,3), dtype=np.int32)
    faces = [0]
    edges = []
    
    for face in explorer.faces():
        loc = TopLoc_Location()
        bt = BRep_Tool()
        tri = bt.Triangulation(face, loc)
        if tri is not None:
            f_verts = [list(tri.Node(i).Transformed(loc.Transformation()).Coord()) for i in range(1, tri.NbNodes()+1)]
            f_tris = [list(tri.Triangle(i).Get()) for i in range(1, tri.NbTriangles()+1)]
            f_verts = np.array(f_verts, dtype=np.float32)
            f_tris = np.array(f_tris, dtype=np.int32) - 1 + len(verts)
            
            if face.Orientation() == TopAbs_REVERSED:
                f_tris = f_tris[:, [0, 2, 1]]

            faces.append(len(tris) + len(f_tris))
            verts = np.vstack((verts, f_verts))
            tris = np.vstack((tris, f_tris))
    
    # remove duplicate vertices
    # unique_verts, inverse_indices = np.unique(verts, axis=0, return_inverse=True)
    # verts = unique_verts
    # tris = inverse_indices[tris]
    
    for edge in explorer.edges():
        loc = TopLoc_Location()
        curve = BRep_Tool.Curve(edge, loc)
        if curve is not None:
            adaptor = BRepAdaptor_Curve(edge)
            u_min = adaptor.FirstParameter()
            u_max = adaptor.LastParameter()
            sampler = GCPnts_UniformDeflection(
                adaptor,
                tol,
                u_min,
                u_max
            )
            c = []
            for i in range(1, sampler.NbPoints()+1):
                p = adaptor.Value(sampler.Parameter(i))
                c.append(p.Coord())
            c = np.array(c)
            edges.append(c)
            
    return verts, tris, np.array(faces, dtype=np.int32), edges