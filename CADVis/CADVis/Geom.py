from OCC.Core.Tesselator import ShapeTesselator
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop
from OCC.Core.gp import gp_Vec, gp_Trsf, gp_Pnt, gp_Ax1, gp_Dir
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GCPnts import GCPnts_UniformDeflection
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add, brepbndlib
from OCC.Extend.DataExchange import read_step_file
from typing import Union, List, Tuple
import numpy as np

def compute_mass_properties(shape : TopoDS_Shape) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute mass properties such as volume (interpreted as mass for unit density)
    and the center of mass of the given shape."""
    props = GProp_GProps()
    #brepgprop_VolumeProperties(shape, props)
    brepgprop.VolumeProperties(shape,props)
    mass = props.Mass()  # For solids, this gives the volume (mass = volume * density)
    center_of_mass = props.CentreOfMass()
    center_of_mass = np.array([center_of_mass.X(), center_of_mass.Y(), center_of_mass.Z()])
    matrix_of_inertia = props.MatrixOfInertia()
    matrix_of_inertia = np.array([[matrix_of_inertia.Value(1, 1), matrix_of_inertia.Value(1, 2), matrix_of_inertia.Value(1, 3)],
                                  [matrix_of_inertia.Value(2, 1), matrix_of_inertia.Value(2, 2), matrix_of_inertia.Value(2, 3)],
                                  [matrix_of_inertia.Value(3, 1), matrix_of_inertia.Value(3, 2), matrix_of_inertia.Value(3, 3)]])
    return mass, center_of_mass, matrix_of_inertia


def normalize_shape(shape : TopoDS_Shape) -> TopoDS_Shape:
    
    mass, center_of_mass, matrix_of_inertia = compute_mass_properties(shape)
    
    # move center of mass to origin
    translation_vector = gp_Vec(-center_of_mass[0], -center_of_mass[1], -center_of_mass[2])
    translation = gp_Trsf()
    translation.SetTranslation(translation_vector)
    shape = BRepBuilderAPI_Transform(shape, translation, True).Shape()
    
    # scale such that root mean radius of gyration is 1/2
    if (np.trace(matrix_of_inertia)/mass/2) > 0:
        scale_factor = np.sqrt(np.trace(matrix_of_inertia)/mass/2)
    else:
        scale_factor = 1.0  # or skip normalization, or handle gracefully
    scaling = gp_Trsf()
    scaling.SetScale(gp_Pnt(0,0,0), 1/scale_factor)
    shape = BRepBuilderAPI_Transform(shape, scaling, True).Shape()
    
    return shape

def random_rotation(shape : TopoDS_Shape) -> TopoDS_Shape:
    """Apply a random rotation to the shape."""
    # Generate a random rotation
    angle = np.random.uniform(0, 2 * np.pi)
    axis = np.random.rand(3)
    axis /= np.linalg.norm(axis)  # Normalize the axis vector
    rot_matrix = np.array([
        [np.cos(angle) + axis[0]**2 * (1 - np.cos(angle)), axis[0]*axis[1]*(1 - np.cos(angle)) - axis[2]*np.sin(angle), axis[0]*axis[2]*(1 - np.cos(angle)) + axis[1]*np.sin(angle)],
        [axis[1]*axis[0]*(1 - np.cos(angle)) + axis[2]*np.sin(angle), np.cos(angle) + axis[1]**2 * (1 - np.cos(angle)), axis[1]*axis[2]*(1 - np.cos(angle)) - axis[0]*np.sin(angle)],
        [axis[2]*axis[0]*(1 - np.cos(angle)) - axis[1]*np.sin(angle), axis[2]*axis[1]*(1 - np.cos(angle)) + axis[0]*np.sin(angle), np.cos(angle) + axis[2]**2 * (1 - np.cos(angle))]
    ])
    # Create a rotation transformation
    rotation = gp_Trsf()
    rotation.SetValues(
        rot_matrix[0, 0], rot_matrix[0, 1], rot_matrix[0, 2], 0,
        rot_matrix[1, 0], rot_matrix[1, 1], rot_matrix[1, 2], 0,
        rot_matrix[2, 0], rot_matrix[2, 1], rot_matrix[2, 2], 0
    )
    # Apply the rotation to the shape
    shape = BRepBuilderAPI_Transform(shape, rotation, True).Shape()
    
    return shape

def extract_mesh_and_edges(shape : Union[TopoDS_Shape, str], tol=0.1, normalize=True, apply_random_rotation=False, ReturnBottom=True) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    
    if isinstance(shape, str):
        # 1. Read the STEP file
        shape = read_step_file(shape)
    
    if normalize:
        # 2. Normalize the shape
        shape = normalize_shape(shape)
        
    if apply_random_rotation:
        # 3. Apply random rotation
        shape = random_rotation(shape)
        
    if ReturnBottom:
        # get bounding box
        bbox = Bnd_Box()
        # bbox.Add(shape)
        bbox.SetGap(1e-6)
        #brepbndlib_Add(shape, bbox, False)
        brepbndlib.Add(shape, bbox, False)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    
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

    if ReturnBottom:
        return np.array(verts), np.array(faces), curves, zmin
    
    return np.array(verts), np.array(faces), curves