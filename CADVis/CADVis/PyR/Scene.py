import numpy as np
from PIL import Image
import os
import trimesh
import pyrender
from ..Geom import extract_mesh_and_edges
from .Material import add_uv

def get_camera_pose_isometric(position=np.array([-4.,-4.,4.]), target=np.array([0.,0.,0.]), up=np.array([0.,0.,1.])):
    look_at = target - position
    look_at_normalized = look_at / np.linalg.norm(look_at)
    up_normalized = up / np.linalg.norm(up)
    default_look_at = np.array([0., 0., -1.])
    default_up = np.array([0., 1., 0.])
    
    # Create a rotation matrix to align the camera with the target
    #align look_at with default_look_at
    rot_axis = np.cross(default_look_at,look_at_normalized)
    rot_angle = np.arccos(np.dot(look_at_normalized, default_look_at))
    rot_axis_normalized = rot_axis / np.linalg.norm(rot_axis)
    K = np.array([[0, -rot_axis_normalized[2], rot_axis_normalized[1]],
                  [rot_axis_normalized[2], 0, -rot_axis_normalized[0]],
                  [-rot_axis_normalized[1], rot_axis_normalized[0], 0]])
    R_1 = np.eye(3) + np.sin(rot_angle)*K + (1-np.cos(rot_angle))*(K@K)
    
    #align the up after the rotation
    rotation_axis = look_at_normalized
    new_up = R_1 @ default_up
    
    # remove the component of target up in the direction of the look_at
    up_projected = up_normalized - np.dot(up_normalized, rotation_axis) * rotation_axis
    up_projected = up_projected / np.linalg.norm(up_projected)
    rot_angle = np.arccos(np.dot(new_up, up_projected))
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                  [rotation_axis[2], 0, -rotation_axis[0]],
                  [-rotation_axis[1], rotation_axis[0], 0]])
    R_2 = np.eye(3) + np.sin(rot_angle) * K + (1 - np.cos(rot_angle)) * np.dot(K, K)
    
    # # Combine the two rotations
    R = np.dot(R_2, R_1)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = R
    camera_pose[:3, 3] = position
    return camera_pose

def get_emmisive_material(color=(1., 1., 1.), intensity=0.5):
    return pyrender.Material(
        emissiveTexture=np.array([color[0]*255, color[1]*255, color[2]*255]).reshape(1, 1, 3).astype(np.uint8),
        emissiveFactor=np.ones(3)*intensity
    )

def get_ground_mesh(dims=np.array([20,20]), up=np.array([0., 0., 1.]), origin=np.array([0., 0., 0.]), texture_scale=None):
    
    if texture_scale is None:
        texture_scale = max(dims) / 5
    
    base_verts = np.array([[-dims[0]/2, -dims[1]/2, 0],
                           [-dims[0]/2, dims[1]/2, 0],
                           [dims[0]/2, dims[1]/2, 0],
                           [dims[0]/2, -dims[1]/2, 0]])
    faces = np.array([[2, 1, 0],
                      [3, 2, 0]])
    
    uv = np.array([[0, 0],
                    [0, 1],
                    [1, 1],
                    [1, 0]]) * texture_scale
    
    # Rotate the ground plane to align with the up vector
    if not np.all(up == np.array([0., 0., 1.])):
        # Compute the rotation axis and angle
        z_axis = np.array([0., 0., 1.])
        rotation_axis = np.cross(z_axis, up)
        rotation_angle = np.arccos(np.dot(z_axis, up) / (np.linalg.norm(z_axis) * np.linalg.norm(up)))
        
        # Create the rotation matrix
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                      [rotation_axis[2], 0, -rotation_axis[0]],
                      [-rotation_axis[1], rotation_axis[0], 0]])
        R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
        
        # Apply the rotation to the vertices
        base_verts = base_verts @ R.T
    
    # Translate the ground plane to the origin
    base_verts += origin
    
    g_mesh = trimesh.Trimesh(vertices=base_verts, faces=faces)
    g_mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
    # g_mesh = add_uv(g_mesh, method='box')
    
    return g_mesh

def get_three_light_poses(camera_position : np.array, target_position : np.array, up_vector : np.array):
    
    look_at = target_position - camera_position
    up_normalized = up_vector / np.linalg.norm(up_vector)
    look_at_no_up = look_at - np.dot(look_at, up_normalized) * up_normalized
    radius = np.linalg.norm(look_at_no_up)
    height = np.linalg.norm(look_at-look_at_no_up)
    
    # get light positions
    light_positions = []
    
    # key light (45 degrees to the right of camera)
    theta = np.pi/4
    axis = up_normalized
    rel_cam_pos = camera_position - target_position
    
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    rel_light_pos = rel_cam_pos @ R.T
    light_positions.append(rel_light_pos + target_position)
    
    # Fill light (45 degrees to the left of camera)
    theta = -np.pi/4
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    rel_light_pos = rel_cam_pos @ R.T
    light_positions.append(rel_light_pos + target_position)
    
    # Back light (45 degrees to the back of camera)
    theta = np.pi/2
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    rel_light_pos = rel_cam_pos @ R.T
    light_positions.append(rel_light_pos + target_position)
    
    
    poses = []
    for lp in light_positions:
        poses.append(get_camera_pose_isometric(lp, target_position, up_vector))
        
    return poses