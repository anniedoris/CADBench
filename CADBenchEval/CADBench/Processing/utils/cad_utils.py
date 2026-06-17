from OCC.Core.Tesselator import ShapeTesselator
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformDeflection
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Extend.DataExchange import read_step_file
from typing import Union, List, Tuple
import numpy as np

def extract_mesh_and_edges(shape : Union[TopoDS_Shape, str],
                           tol=0.05,
                           return_bounding_box=False) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    
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