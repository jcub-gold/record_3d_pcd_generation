import open3d as o3d
import numpy as np
import trimesh
from scipy.spatial import ckdtree

def get_rotation_from_pca(pcd):
    points = np.asarray(pcd.points)
    xz_points = points[:, [0, 2]]  # Project to XZ plane
    xz_centered = xz_points - xz_points.mean(axis=0)
    u, s, vh = np.linalg.svd(xz_centered, full_matrices=False)
    direction_1 = vh[0]  # Principal direction in XZ
    angle = np.arctan2(direction_1[1], direction_1[0])
    return angle

def get_mesh_rotation_from_pca(mesh: trimesh.Trimesh):
    points = np.asarray(mesh.vertices)  # Extract vertices from the mesh
    xz_points = points[:, [0, 2]]       # Project to XZ plane
    xz_centered = xz_points - xz_points.mean(axis=0)
    u, s, vh = np.linalg.svd(xz_centered, full_matrices=False)
    direction_1 = vh[0]  # Principal direction in XZ
    angle = np.arctan2(direction_1[1], direction_1[0])
    return angle


def get_mesh_rotation_from_aabb_min_xz(
    mesh,
    *,
    metric: str = "area",        # "area" (default) or "perimeter"
    coarse_samples: int = 720,   # number of angles in the coarse sweep (0..π)
    refine_steps: int = 2,       # how many times to refine around the best angle
    refine_window_deg: float = 5 # +/- window (degrees) for each refinement
) -> float:
    """
    Returns angle θ (radians) about +Y so that rotating the mesh by -θ
    minimizes the XZ-plane AABB according to `metric`.

    Notes:
    - Only the XZ projection is used.
    - Periodicity is π (180°), so the returned θ is in [0, π).
    - For degenerate cases (few points / collinear), returns 0.0.
    """
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return 0.0

    # Project to XZ and (optionally) reduce to the convex hull for speed
    xz = pts[:, (0, 2)]
    xz = xz - xz.mean(axis=0, keepdims=True)

    def _maybe_hull(p):
        try:
            from scipy.spatial import ConvexHull
            if p.shape[0] >= 4:
                hull = ConvexHull(p)
                return p[hull.vertices]
            return p
        except Exception:
            return p

    xz = _maybe_hull(xz)

    extents = np.ptp(xz, axis=0)          # [range_x, range_z]
    if np.allclose(extents, 0.0):
        return 0.0


    def _metric_for(th):
        c, s = np.cos(th), np.sin(th)
        # rotate XZ by th (2x2)
        xr =  c * xz[:, 0] - s * xz[:, 1]
        zr =  s * xz[:, 0] + c * xz[:, 1]
        w = xr.max() - xr.min()
        h = zr.max() - zr.min()
        if metric == "perimeter":
            return 2.0 * (w + h)
        return w * h  # "area"

    def _sweep(angles):
        best_val = np.inf
        best_th  = 0.0
        for th in angles:
            val = _metric_for(th)
            if val < best_val:
                best_val, best_th = val, th
        return best_val, best_th

    # --- coarse sweep over [0, π) ---
    angles = np.linspace(0.0, np.pi / 2, num=coarse_samples, endpoint=False)
    best_val, best_th = _sweep(angles)

    # --- refine around best angle ---
    window = np.deg2rad(refine_window_deg)
    for _ in range(refine_steps):
        # sample densely around current best, wrapping into [0, π)
        local = np.linspace(best_th - window, best_th + window, num=coarse_samples, endpoint=True)
        local = (local % np.pi)
        best_val, best_th = _sweep(local)
        window *= 0.3  # shrink window each iteration

    # normalize to [0, π)
    return float(best_th % np.pi)



# mesh_path = "data/basement_test_2/output copy/object_8/object_8_mesh.obj"
# mesh = trimesh.load(mesh_path)
# angle = get_mesh_rotation_from_pca(mesh)
# angle = get_mesh_rotation_from_aabb_min_xz(mesh)
# print(f"Rotation angle (deg): {angle * 180 / np.pi}")

# # Create rotation matrix around Y axis (since XZ plane)
# rotation_matrix = trimesh.transformations.rotation_matrix(-angle, [0, 1, 0], mesh.centroid)
# mesh.apply_transform(rotation_matrix)

# # Save rotated mesh
# output_path = "output.obj"
# mesh.export(output_path)