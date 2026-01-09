from src.utils.align_utils import create_pcd_from_frame, remove_outliers_largest_cluster
from src.utils.data_utils import prepare_record3d_data
import open3d as o3d
import argparse
import os
import json
from tqdm import tqdm

"""
Function: generate_pcds
-----------------------
scene_name: name of the scene folder under `data/` (e.g., "basement_test")
eps: DBSCAN epsilon parameter used by `remove_outliers_largest_cluster` (distance threshold)
min_points: minimum points per cluster for DBSCAN (used by `remove_outliers_largest_cluster`)
nb_neighbors: `nb_neighbors` parameter for Open3D statistical outlier removal
std_ratio: `std_ratio` parameter for Open3D statistical outlier removal
save_frames: optional dict to populate and save as `data/{scene_name}/cached_frames.json`
load_cached_frames: if True, load frame indices from `data/{scene_name}/cached_frames.json` instead of prompting

Scans `data/{scene_name}/record3d_input/` for `object_{n}` folders, prepares per-frame data with
`prepare_record3d_data`, and builds each object's point cloud by calling `create_pcd_from_frame`
for the selected frames. Applies outlier filtering: keeps the largest DBSCAN cluster and then
runs statistical outlier removal (parameters above). Writes final point clouds to
`data/{scene_name}/pcds/object_{n}_pcd.ply`. If `save_frames` is provided it will be written
to `data/{scene_name}/cached_frames.json`. This function performs I/O and prints progress via `tqdm`.
"""
def generate_pcds(scene_name, eps=0.05, min_points=10, nb_neighbors=10, std_ratio=3.0, save_frames=None, load_cached_frames=False):
    obj_data = {}

    pcds_output_path = f"data/{scene_name}/pcds"
    os.makedirs(pcds_output_path, exist_ok=True)

    record3d_input_path = f"data/{scene_name}/record3d_input/"

    input_depth_dir = f"data/{scene_name}/record3d_input/input_depth"
    new_metadata_path = f"data/{scene_name}/record3d_input/new_metadata.json"

    num_objects = 0
    for dirpath, dirnames, filenames in os.walk(record3d_input_path):
        for dirname in dirnames:
            if dirname.startswith("object_"):
                num_objects += 1

    if load_cached_frames:
        with open(f'data/{scene_name}/cached_frames.json') as f:
            cache = json.load(f)
    else:
        cache = None

    for obj in range(1, num_objects + 1):
        images_dir = f"data/{scene_name}/record3d_input/object_{obj}/images"
        obj_data[f'object_{obj}'] = {}
        obj_data[f'object_{obj}']['data'] = prepare_record3d_data(images_dir, input_depth_dir, new_metadata_path)
        if load_cached_frames:
            frame_indices = cache[f'object_{obj}']
        else:
            frame_indices = input(f"Enter the frames used for pcd generation for object {obj}: ").split(" ")
        obj_data[f'object_{obj}']['frames'] = frame_indices
        if save_frames is not None:
            save_frames[f'object_{obj}'] = frame_indices

    if save_frames is not None:
        save_frames_path = f'data/{scene_name}/cached_frames.json'
        json.dump(save_frames, open(save_frames_path, 'w'), indent=4)
    
    remove_outliers = {
        'eps': eps,  # Distance threshold for clustering 
        'min_points': min_points,  # Minimum number of points in a cluster
        'nb_neighbors': nb_neighbors,  # Number of neighbors to consider for statistical outlier removal
        'std_ratio': std_ratio  # Standard deviation ratio for statistical outlier removal
    }
    for obj in tqdm(range(1, num_objects + 1), desc="generating pcds"):
        obj_name = f"object_{obj}"
        pcd = o3d.geometry.PointCloud()
        for frame in obj_data[obj_name]['frames']:
            frame = int(frame)

            pcd += create_pcd_from_frame(obj_data[obj_name]['data'], frame, remove_outliers=remove_outliers) 

        if remove_outliers is not None:
            pcd = remove_outliers_largest_cluster(
                pcd, 
                eps=remove_outliers['eps'], 
                min_points=remove_outliers['min_points']
            )
            cl, ind = pcd.remove_statistical_outlier(
                nb_neighbors=remove_outliers['nb_neighbors'], 
                std_ratio=remove_outliers['std_ratio']
            )
            pcd = pcd.select_by_index(ind)

        pcd_path = os.path.join(pcds_output_path, f"{obj_name}_pcd.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)

