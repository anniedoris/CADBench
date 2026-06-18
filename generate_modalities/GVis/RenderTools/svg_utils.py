from OCC.Extend.DataExchange import *
from OCC.Core.gp import gp_Vec, gp_Trsf, gp_Pnt, gp_Ax1, gp_Dir
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Extend.TopologyUtils import TopologyExplorer
import numpy as np


def edge_to_svg_polyline(topods_edge: TopoDS_Edge, tol: float = 0.1, unit: str = "mm"):
    """Returns a svgwrite.Path for the edge, and the 2d bounding box"""
    check_svgwrite_installed()

    unit_factor = 1  # by default

    if unit == "mm":
        unit_factor = 1
    elif unit == "m":
        unit_factor = 1e3

    points_3d = discretize_edge(topods_edge, tol)
    points_2d = []
    box2d = Bnd_Box2d()

    for point in points_3d:
        # we tak only the first 2 coordinates (x and y, leave z)
        x_p = -point[0] * unit_factor
        y_p = point[1] * unit_factor
        box2d.Add(gp_Pnt2d(x_p, y_p))
        points_2d.append((x_p, y_p))

    return np.array(points_2d)
    # return svgwrite.shapes.Polyline(points_2d, fill="none"), box2d

def export_shape_to_svg(
    shape: TopoDS_Shape,
    filename: str = None,
    width: int = 'auto',
    height: int = 'auto',
    margin_left: int = 0,
    margin_top: int = 0,
    export_hidden_edges: bool = True,
    location: gp_Pnt = gp_Pnt(0, 0, 0),
    direction: gp_Dir = gp_Dir(1, 1, 1),
    color: str = "black",
    line_width: str = "0.1em",
    unit: str = "m",
):
    """export a single shape to an svg file and/or string.
    shape: the TopoDS_Shape to export
    filename (optional): if provided, save to an svg file
    width, height (optional): integers, specify the canvas size in pixels
    margin_left, margin_top (optional): integers, in pixel
    export_hidden_edges (optional): whether or not draw hidden edges using a dashed line
    location (optional): a gp_Pnt, the lookat
    direction (optional): to set up the projector direction
    color (optional), "default to "black".
    line_width (optional, default to 1): an integer
    """
    occ_obj = shape

    bbox = Bnd_Box()
    bbox.SetGap(1e-6)
    brepbndlib.Add(occ_obj, bbox)
    bounding_box = np.array(bbox.Get()).reshape(-1,3)
    scale = max(bounding_box[1] - bounding_box[0])
    center = (bounding_box[1] + bounding_box[0])/2

    transform = gp_Trsf()
    # transform.SetScale(gp_Pnt(0,0,0), 1.0/scale)
    transform.SetTranslation(gp_Vec(-center[0], -center[1], -center[2]))

    normalized_obj = BRepBuilderAPI_Transform(
        occ_obj,
        transform,
        True
    ).Shape()

    normalize_transform = gp_Trsf()
    normalize_transform.SetScale(gp_Pnt(0,0,0), 1.0/scale)
    normalized_obj = BRepBuilderAPI_Transform(
        normalized_obj,
        normalize_transform,
        True
    ).Shape()
    
    shape = normalized_obj
    
    check_svgwrite_installed()

    if shape.IsNull():
        raise AssertionError("shape is Null")

    # find all edges
    visible_edges, hidden_edges = get_sorted_hlr_edges(
        shape,
        position=location,
        direction=direction,
        export_hidden_edges=export_hidden_edges,
    )

    # compute polylines for all edges
    # we compute a global 2d bounding box as well, to be able to compute
    # the scale factor and translation vector to apply to all 2d edges so that
    # they fit the svg canva
    # global_2d_bounding_box = Bnd_Box2d()

    polylines = []
    for visible_edge in visible_edges:
        visible_svg_line = edge_to_svg_polyline(
            visible_edge, 0.001, unit
        )
        polylines.append(visible_svg_line)
        # global_2d_bounding_box.Add(visible_edge_box2d)
    n_polylines = len(polylines)
    if export_hidden_edges:
        for hidden_edge in hidden_edges:
            hidden_svg_line = edge_to_svg_polyline(
                hidden_edge, 0.001, unit
            )
            # hidden lines are dashed style
            # hidden_svg_line.dasharray([5, 5])
            polylines.append(hidden_svg_line)
            # global_2d_bounding_box.Add(hidden_edge_box2d)
    
    if np.allclose(np.array(direction.Coord()), np.array((1., 1., 1.))/np.sqrt(3)):
        theta = np.deg2rad(-60)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        for i in range(len(polylines)):
            polylines[i] = polylines[i] @ rotation_matrix.T
    
    if np.allclose(np.array(direction.Coord()), np.array((0, 0, 1.))):
        theta = np.deg2rad(-180)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        for i in range(len(polylines)):
            polylines[i] = polylines[i] @ rotation_matrix.T
    
    if np.allclose(np.array(direction.Coord()), np.array((0, 1., 0))) or np.allclose(np.array(direction.Coord()), np.array((0, -1., 0.))):
        theta = np.deg2rad(np.array(direction.Coord())[1] * 90)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        for i in range(len(polylines)):
            polylines[i] = polylines[i] @ rotation_matrix.T
    if np.allclose(np.array(direction.Coord()), np.array((1., 0, 0))) or np.allclose(np.array(direction.Coord()), np.array((-1., 0, 0.))):
        theta = np.deg2rad(-np.array(direction.Coord())[0] * 90 - 180)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        for i in range(len(polylines)):
            polylines[i] = polylines[i] @ rotation_matrix.T
    x_min = np.inf
    y_min = np.inf
    x_max = -np.inf
    y_max = -np.inf
    for polyline in polylines:
        x_coords = polyline[:,0]
        y_coords = polyline[:,1]
        x_min = min(x_min, np.min(x_coords))
        y_min = min(y_min, np.min(y_coords))
        x_max = max(x_max, np.max(x_coords))
        y_max = max(y_max, np.max(y_coords))

    svg_polylines = [svgwrite.shapes.Polyline((polyline - np.array([x_min, y_min])), fill="none") for polyline in polylines]
    if len(svg_polylines) > n_polylines:
        for i in range(n_polylines, len(svg_polylines)):
            svg_polylines[i].dasharray([5, 5])

    # translate and scale polylines

    # first compute shape translation and scale according to size/margins
    # x_min, y_min, x_max, y_max = global_2d_bounding_box.Get()
    bb2d_width = x_max - x_min
    bb2d_height = y_max - y_min
    if isinstance(width, str) and width == 'auto' and isinstance(height, int):
        width = int((bb2d_width / bb2d_height) * height)
    elif isinstance(width, int) and isinstance(height, str) and height == 'auto':
        height = int((bb2d_height / bb2d_width) * width)
    elif isinstance(width, str) and width == 'auto' and isinstance(height, str) and height == 'auto':
        width = bb2d_width + 2 * margin_left
        height = bb2d_height + 2 * margin_top
    else:
        pass  # user provided both width and height 
    # build the svg drawing
    dwg = svgwrite.Drawing(filename, (width, height), debug=True)
    # adjust the view box so that the lines fit then svg canvas
    dwg.viewbox(
        -margin_left,
        -margin_top,
        bb2d_width + 2 * margin_left,
        bb2d_height + 2 * margin_top,
    )

    for polyline in svg_polylines:
        # apply color and style
        polyline.stroke(color, width=line_width, linecap="round")
        # then adds the polyline to the svg canva
        dwg.add(polyline)

    # export to string or file according to the user choice
    if filename is not None:
        dwg.save()
        if not os.path.isfile(filename):
            raise AssertionError("svg export failed")
        print(f"Shape successfully exported to {filename}")
        return True
    return dwg.tostring()