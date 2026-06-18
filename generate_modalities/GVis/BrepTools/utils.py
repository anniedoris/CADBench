from OCC.Core.Tesselator import ShapeTesselator
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Edge, TopoDS_Face, TopoDS_Compound
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.gp import gp_Vec, gp_Trsf, gp_Pnt, gp_Ax1, gp_Dir
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GCPnts import GCPnts_UniformDeflection
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepTools import breptools
from OCC.Core.BRep import BRep_Builder
from typing import Union, List, Tuple, Dict, Any
import numpy as np

def get_brep_graph(shape: TopoDS_Shape, sampling_density: int = 32, single_solid: bool = True) -> Tuple[List[np.ndarray], List[TopoDS_Edge], np.ndarray]:
    
    explorer = TopologyExplorer(shape)
    solids = list(explorer.solids())
    if len(solids) == 0:
        raise ValueError("No solid found in the given shape.")
    elif single_solid:
        solid = solids[0]
        explorer = TopologyExplorer(solid)
    else:
        solid = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(solid)
        for shape in solids:
            builder.Add(solid, shape)
            if single_solid:
                break
        explorer = TopologyExplorer(solid)
    
    face_dict = {}
    for face in explorer.faces():
        face_idx = face.__hash__()
        face_dict[face_idx] = face
        
    edgeFace_adj = []
    edge_dict = {}
    for edge in explorer.edges():
        curve = BRepAdaptor_Curve(edge)
        if not curve.Is3DCurve():
            continue

        connected_faces = list(explorer.faces_from_edge(edge))

        if len(connected_faces) == 2 and not ShapeAnalysis_Edge().IsSeam(edge, connected_faces[0]) and not ShapeAnalysis_Edge().IsSeam(edge, connected_faces[1]):
            edge_idx = edge.__hash__()
            edge_dict[edge_idx] = edge
            face_1_index = connected_faces[0].__hash__()
            face_2_index = connected_faces[1].__hash__()
            
            edgeFace_adj.append([edge_idx, face_1_index])
            edgeFace_adj.append([edge_idx, face_2_index])
        else:
            pass # ignore seam
    edgeFace_adj = np.array(edgeFace_adj)
    edgeFace_adj = np.unique(edgeFace_adj, axis=0)
    
    edge_ids = list(edge_dict.keys())
    face_ids = list(face_dict.keys())
    
    condensed_edge_ids = np.arange(len(edge_ids))
    condensed_face_ids = np.arange(len(face_ids))
    
    edges = []
    faces = []
    corners = []
    edgeCorner_adj = []
    
    for i, edge_id in enumerate(edge_ids):
        edgeFace_adj[edgeFace_adj[:,0] == edge_id, 0] = condensed_edge_ids[i]
        _, u_min, u_max = BRep_Tool.Curve(edge_dict[edge_id])
        curve = BRepAdaptor_Curve(edge_dict[edge_id])
        u_ = np.linspace(u_min, u_max, sampling_density)
        u_val = np.zeros((sampling_density, 3), dtype=np.float64)
        for j, u in enumerate(u_):
            p = curve.Value(u)
            u_val[j,:] = [p.X(), p.Y(), p.Z()]
        if edge_dict[edge_id].Orientation() == TopAbs_REVERSED:
            u_val = u_val[::-1, :]
        edges.append(u_val)
        corners.append(u_val[0, :])
        corners.append(u_val[-1, :])
        edgeCorner_adj.append([i, 2*i])
        edgeCorner_adj.append([i, 2*i + 1])
        
    corners = np.array(corners)
    edgeCorner_adj = np.array(edgeCorner_adj)
    unique_corners, inverse_ids = np.unique(corners, axis=0, return_inverse=True)
    edgeCorner_adj[:,1] = inverse_ids[edgeCorner_adj[:,1]]
    corners = unique_corners
        
    for i, face_id in enumerate(face_ids):
        edgeFace_adj[edgeFace_adj[:,1] == face_id, 1] = condensed_face_ids[i]
        umin, umax, vmin, vmax = breptools.UVBounds(face_dict[face_id])
        u_ = np.linspace(umin, umax, sampling_density)
        v_ = np.linspace(vmin, vmax, sampling_density)
        uv_val = np.zeros((sampling_density, sampling_density, 3), dtype=np.float64)
        loc = TopLoc_Location()
        surf = BRep_Tool.Surface(face_dict[face_id], loc)
        for iu, u in enumerate(u_):
            for iv, v in enumerate(v_):
                pt = surf.Value(u, v)
                pt = pt.Transformed(loc.Transformation())
                uv_val[iu, iv, :] = [pt.X(), pt.Y(), pt.Z()]

        if face_dict[face_id].Orientation() == TopAbs_REVERSED:
            uv_val = uv_val[::-1, :, :]
        faces.append(uv_val)
        
    faces = np.array(faces, dtype=np.float32)
    edges = np.array(edges, dtype=np.float32)
    corners = np.array(corners, dtype=np.float32)
    edgeFace_adj = np.array(edgeFace_adj, dtype=np.int32)
    edgeCorner_adj = np.array(edgeCorner_adj, dtype=np.int32)
    
    return faces, edges, corners, edgeFace_adj, edgeCorner_adj, solid

def compute_mass_properties(shape : TopoDS_Shape) -> Tuple[float, np.ndarray, np.ndarray]:
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    mass = float(props.Mass())  # For solids, this gives the volume (mass = volume * density)
    center_of_mass = props.CentreOfMass()
    center_of_mass = np.array([center_of_mass.X(), center_of_mass.Y(), center_of_mass.Z()])
    matrix_of_inertia = props.MatrixOfInertia()
    matrix_of_inertia = np.array([[matrix_of_inertia.Value(1, 1), matrix_of_inertia.Value(1, 2), matrix_of_inertia.Value(1, 3)],
                                  [matrix_of_inertia.Value(2, 1), matrix_of_inertia.Value(2, 2), matrix_of_inertia.Value(2, 3)],
                                  [matrix_of_inertia.Value(3, 1), matrix_of_inertia.Value(3, 2), matrix_of_inertia.Value(3, 3)]])
    principal_moments = props.PrincipalProperties()
    p1 = principal_moments.FirstAxisOfInertia()
    p1 = np.array([p1.X(), p1.Y(), p1.Z()])
    p1 /= np.linalg.norm(p1)
    p2 = principal_moments.SecondAxisOfInertia()
    p2 = np.array([p2.X(), p2.Y(), p2.Z()])
    p2 /= np.linalg.norm(p2)
    p3 = principal_moments.ThirdAxisOfInertia()
    p3 = np.array([p3.X(), p3.Y(), p3.Z()])
    p3 /= np.linalg.norm(p3)
    P = np.vstack((p1, p2, p3)).T
    radius_of_gyration = float(np.sqrt(np.trace(matrix_of_inertia) / mass))
    return mass, center_of_mass, radius_of_gyration, matrix_of_inertia, P

def get_mesh(shape : TopoDS_Shape, tol=0.1) -> Tuple[np.ndarray, np.ndarray]:
    tess = ShapeTesselator(shape)
    tess.Compute(mesh_quality=tol, parallel=True)
    
    verts = [tess.GetVertex(i) for i in range(tess.ObjGetVertexCount())]
    normals = [tess.GetNormal(i) for i in range(tess.ObjGetNormalCount())]
    faces = [tess.GetTriangleIndex(i) for i in range(tess.ObjGetTriangleCount())]
    return np.array(verts), np.array(faces), np.array(normals)