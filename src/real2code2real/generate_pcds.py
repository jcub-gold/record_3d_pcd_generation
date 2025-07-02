from src.utils.align_utils import create_pcd_from_frame
from src.utils.data_utils import prepare_record3d_data
import open3d as o3d
import argparse
import os

def generate_pcds(scene_name):
    obj_data = {}
    
    pcds_output_path = f"data/{scene_name}/pcds"
    record3d_input_path = f"data/{scene_name}/record3d_input/"

    input_depth_dir = f"data/{scene_name}/record3d_input/input_depth"
    new_metadata_path = f"data/{scene_name}/record3d_input/new_metadata.json"

    num_objects = 0
    for dirpath, dirnames, filenames in os.walk(record3d_input_path):
        for dirname in dirnames:
            if dirname.startswith("object_"):
                num_objects += 1

    for obj in range(1, num_objects + 1):
        images_dir = f"data/{scene_name}/object_{obj}/images"
        obj_data[f'object_{obj}'] = prepare_record3d_data(images_dir, input_depth_dir, new_metadata_path)

    for obj in range(1, num_objects + 1):
        obj_name = f"object_{obj}"
        frame_indices = input(f"Enter the frames used for pcd generation for object {obj}: ").split(" ")
        pcd = o3d.geometry.PointCloud()
        for frame in frame_indices:
            frame = int(frame)
            pcd += create_pcd_from_frame(obj_data[obj_name], frame) 
        pcd_path = os.path.join(pcds_output_path, f"{obj_name}_pcd.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create directory structure for a new scene")
    parser.add_argument("--scene_name", required=True, help="Name of the scene to create directories for")

    args = parser.parse_args()
    generate_pcds(args.scene_name)