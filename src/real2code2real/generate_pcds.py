from src.utils.align_utils import create_pcd_from_frame, remove_outliers_largest_cluster
from src.utils.data_utils import prepare_record3d_data
import open3d as o3d
import argparse
import os
import json
from tqdm import tqdm

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


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create directory structure for a new scene")
    parser.add_argument("--scene_name", required=True, help="Name of the scene to create directories for")
    parser.add_argument("--eps", type=float, default=0.05, help="Distance threshold for clustering")
    parser.add_argument("--min_points", type=int, default=10, help="Minimum number of points in a cluster")
    parser.add_argument("--nb_neighbors", type=int, default=10, help="Number of neighbors to consider for statistical outlier removal")
    parser.add_argument("--std_ratio", type=float, default=3.0, help="Standard deviation ratio for statistical outlier removal")
    parser.add_argument("--save_frames", type=bool, default=False, help="Save a cache of the frames used for pcd generation")
    parser.add_argument("--load_cached_frames", type=bool, default=False, help="Load cached frames for pcd generation")

    args = parser.parse_args()
    if args.save_frames:
        sf = {}
    else:
        sf = None
    generate_pcds(args.scene_name, eps=args.eps, min_points=args.min_points, 
                   nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio, save_frames=sf, load_cached_frames=args.load_cached_frames)