import argparse
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import glob
#from GVis.pc_utils import brep_2_pc
from multiprocessing import Process, Queue
import multiprocessing as mp
import open3d as o3d
from OCC.Core.Tesselator import ShapeTesselator
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformDeflection
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Extend.DataExchange import read_step_file
from typing import Union, List, Tuple
import trimesh


args = argparse.ArgumentParser()
args.add_argument('--input_dir', type=str, required=True)
args.add_argument('--output_dir', type=str, required=True)
args.add_argument('--num_points', type=int, default=10000)
args.add_argument('--num_workers', type=int, default=128)
args.add_argument('--mesh_tolerance', type=float, default=0.25)
args.add_argument('--timeout', type=int, default=20)
args = args.parse_args()



def extract_mesh_and_edges(shape : Union[TopoDS_Shape, str], tol=args.mesh_tolerance, return_bounding_box=True) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    
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

def brep_2_mesh(brep_file, num_points=10000):
    verts, tris, _ = extract_mesh_and_edges(brep_file, return_bounding_box=False)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.compute_vertex_normals()

    return mesh


def brep_2_mesh_trimesh(brep_file, num_points=10000):
    verts, tris, _ = extract_mesh_and_edges(brep_file, return_bounding_box=False)
    mesh = trimesh.Trimesh(verts, tris)

    return mesh

def __process_file_and_save(input_path, output_path, num_points=10000, queue=None, method='o3d'):
    try:
        if method == 'o3d':
            mesh = brep_2_mesh(input_path, num_points)
            o3d.io.write_triangle_mesh(output_path, mesh, write_vertex_normals = False)
        elif method == 'trimesh':
            mesh = brep_2_mesh_trimesh(input_path, num_points)
            mesh.export(output_path)
        else:
            raise ValueError(f"Invalid method: {method}")
        #np.save(output_path, points)
        if queue is not None:
            queue.put(True)
            return
    except Exception as e:
        if queue is not None:
            queue.put(False)
            return

def _process_file_and_save(input_path, output_path, num_points=10000, timeout=30):
    # isolate the call to __process_file_and_save
    queue = Queue()
    process = Process(target=__process_file_and_save, args=(input_path, output_path, num_points, queue))
    process.start()
    
    try:
        result = queue.get(timeout=timeout)
    except Exception as e:
        process.kill()
        os.remove(input_path)
        return 2
    
    return int(result)

def main():
    all_files = glob.glob(os.path.join(args.input_dir, "*.step"))
    os.makedirs(args.output_dir, exist_ok=True)
    
    processing_fails = 0
    timeout_fails = 0
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp.get_context('fork')) as executor:
        future_to_file = {executor.submit(_process_file_and_save, file, os.path.join(args.output_dir, os.path.basename(file).replace('.step', '.stl')), num_points=args.num_points, timeout=args.timeout): file for file in all_files}
        prog = tqdm(as_completed(future_to_file), total=len(all_files))
        for future in prog:
            file = future_to_file[future]
            try:
                result = future.result()
                if result == 0:
                    #print(f"Error processing {file} (Processing Error)")
                    processing_fails += 1
                elif result == 2:
                    # print(f"Error processing {file} (Timeout)")
                    timeout_fails += 1
            except Exception as e:
                #print(f"Error processing {file}: {e}")
                processing_fails += 1
            prog.set_postfix(processing_fails=processing_fails, timeout_fails=timeout_fails)
if __name__ == "__main__":
    main()
