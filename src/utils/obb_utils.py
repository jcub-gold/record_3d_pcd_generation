import open3d as o3d
import numpy as np
import trimesh

def get_rotation_from_pca(pcd):
    points = np.asarray(pcd.points)
    xz_points = points[:, [0, 2]]  # Project to XZ plane
    xz_centered = xz_points - xz_points.mean(axis=0)
    u, s, vh = np.linalg.svd(xz_centered, full_matrices=False)
    direction_1 = vh[0]  # Principal direction in XZ
    angle = np.arctan2(direction_1[1], direction_1[0])
    return angle

def get__mesh_rotation_from_pca(mesh: trimesh.Trimesh):
    points = np.asarray(mesh.vertices)  # Extract vertices from the mesh
    xz_points = points[:, [0, 2]]       # Project to XZ plane
    xz_centered = xz_points - xz_points.mean(axis=0)
    u, s, vh = np.linalg.svd(xz_centered, full_matrices=False)
    direction_1 = vh[0]  # Principal direction in XZ
    angle = np.arctan2(direction_1[1], direction_1[0])
    return angle