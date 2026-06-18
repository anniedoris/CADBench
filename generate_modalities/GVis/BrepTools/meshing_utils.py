import bpy
import numpy as np

def sharp_remesh_data(verts, faces, quality=8):
    """
    Performs a SHARP Remesh on raw vertex and face data.

    This operation MUST create a temporary object and use the
    dependency graph, as the SHARP Remesher is implemented
    as a modifier. The temporary objects are cleaned up afterward.

    Args:
        verts (list of tuples): List of (x, y, z) vertex coordinates.
        faces (list of tuples): List of (v1, v2, v3, ...) vertex indices.
        quality (int): Quality level for the remesher (octree depth).

    Returns:
        tuple: A (new_verts, new_faces, new_normals) tuple.
               Faces are guaranteed to be triangles.
    """

    mesh = bpy.data.meshes.new(name="temp_remesh_mesh")
    obj = bpy.data.objects.new(name="temp_remesh_obj", object_data=mesh)

    mesh.from_pydata(verts, [], faces)
    mesh.update()
    
    try:
        bpy.context.collection.objects.link(obj)
    except Exception as e:
        bpy.data.scenes[0].collection.objects.link(obj)

    mod = obj.modifiers.new(name="Remesh", type='REMESH')
    mod.mode = 'SHARP'
    mod.octree_depth = quality
    mod.scale = 0.99

    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    
    mesh_eval = obj_eval.to_mesh()
    
    new_verts = [v.co[:] for v in mesh_eval.vertices]
    new_normals = [v.normal[:] for v in mesh_eval.vertices]

    mesh_eval.calc_loop_triangles()
    new_faces = []
    for tri in mesh_eval.loop_triangles:
        new_faces.append(tuple(tri.vertices))

    bpy.data.objects.remove(obj)
    bpy.data.meshes.remove(mesh)
    obj_eval.to_mesh_clear()

    new_verts = np.array(new_verts, dtype=np.float32)
    new_faces = np.array(new_faces, dtype=np.int32)
    new_normals = np.array(new_normals, dtype=np.float32)
    
    return new_verts, new_faces, new_normals