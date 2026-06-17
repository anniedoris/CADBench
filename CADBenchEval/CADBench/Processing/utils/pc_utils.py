import open3d as o3d
import numpy as np
import trimesh

def process_mesh(mesh: trimesh.Trimesh, num_points: int = 10000):
    
    verts = mesh.vertices
    tris = mesh.faces
    
    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    
    points = np.asarray(pcd.points)
    
    return points, verts, tris