from src.utils.align_utils import create_pcd_from_frame
from src.utils.data_utils import prepare_record3d_data
import open3d as o3d
import argparse
import os

def generate_pcds(scene_name, eps=0.05, min_points=10, nb_neighbors=10, std_ratio=3.0):
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

    for obj in range(1, num_objects + 1):
        images_dir = f"data/{scene_name}/record3d_input/object_{obj}/images"
        obj_data[f'object_{obj}'] = {}
        obj_data[f'object_{obj}']['data'] = prepare_record3d_data(images_dir, input_depth_dir, new_metadata_path)
        obj_data[f'object_{obj}']['frames'] = frame_indices = input(f"Enter the frames used for pcd generation for object {obj}: ").split(" ")

    remove_outliers = {
        'eps': eps,  # Distance threshold for clustering 
        'min_points': min_points,  # Minimum number of points in a cluster
        'nb_neighbors': nb_neighbors,  # Number of neighbors to consider for statistical outlier removal
        'std_ratio': std_ratio  # Standard deviation ratio for statistical outlier removal
    }
    for obj in range(1, num_objects + 1):
        obj_name = f"object_{obj}"
        pcd = o3d.geometry.PointCloud()
        for frame in obj_data[obj_name]['frames']:
            frame = int(frame)

            pcd += create_pcd_from_frame(obj_data[obj_name]['data'], frame, remove_outliers=remove_outliers) 
        pcd_path = os.path.join(pcds_output_path, f"{obj_name}_pcd.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create directory structure for a new scene")
    parser.add_argument("--scene_name", required=True, help="Name of the scene to create directories for")
    parser.add_argument("--eps", type=float, default=0.05, help="Distance threshold for clustering")
    parser.add_argument("--min_points", type=int, default=10, help="Minimum number of points in a cluster")
    parser.add_argument("--nb_neighbors", type=int, default=10, help="Number of neighbors to consider for statistical outlier removal")
    parser.add_argument("--std_ratio", type=float, default=3.0, help="Standard deviation ratio for statistical outlier removal")

    args = parser.parse_args()
    generate_pcds(args.scene_name, eps=args.eps, min_points=args.min_points, 
                   nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio)