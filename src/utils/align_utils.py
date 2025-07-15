import open3d as o3d
import numpy as np

def remove_outliers_largest_cluster(pcd, eps=0.05, min_points=10):

    if len(np.asarray(pcd.points)) < min_points:
        return pcd
    
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    
    if len(np.unique(labels)) <= 1 and np.unique(labels)[0] == -1:
        return pcd
    
    labels_count = np.bincount(labels[labels >= 0])
    if len(labels_count) == 0:  # No valid clusters found
        return pcd
    
    largest_cluster_label = np.argmax(labels_count)
    
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    largest_cluster_pcd = pcd.select_by_index(largest_cluster_indices)
    
    if pcd.has_colors():
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(
            np.asarray(pcd.colors)[largest_cluster_indices])
    
    return largest_cluster_pcd


def create_pcd_from_frame(data, frame_index, samples=5000, remove_outliers=None):

    intrinsics = data["intrinsics"]
    frame = data["frames"][frame_index]
    
    mask_img = frame[0]
    depth_img = frame[1].copy()
    extrinsics = frame[2]

    if mask_img.shape[2] == 4:
        alpha_mask = mask_img[:, :, 3] > 0
    else:
        alpha_mask = np.ones(mask_img.shape[:2], dtype=bool)        

    # depth_img[~alpha_mask] = 0

    # Shrink mask by 5 pixels from each edge
    shrink = 5
    alpha_mask_shrunk = np.zeros_like(alpha_mask, dtype=bool)
    alpha_mask_shrunk[shrink:-shrink, shrink:-shrink] = alpha_mask[shrink:-shrink, shrink:-shrink]

    # Apply the shrunk mask
    depth_img[~alpha_mask_shrunk] = 0

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(mask_img[:, :, :3].astype(np.uint8)),
        o3d.geometry.Image(depth_img),
        depth_scale=1.0,
        depth_trunc=1000.0,
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsics, extrinsics
    )

    pcd = pcd.voxel_down_sample(voxel_size=0.002)
    pcd = pcd.remove_duplicated_points()
    
    if remove_outliers is not None:
        pcd = remove_outliers_largest_cluster(pcd, eps=remove_outliers['eps'], min_points=remove_outliers['min_points'])
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=remove_outliers['nb_neighbors'], std_ratio=remove_outliers['std_ratio'])
        pcd = pcd.select_by_index(ind)
    
    return pcd
