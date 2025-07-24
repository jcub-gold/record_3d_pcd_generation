import argparse
from src.utils.pcd_to_urdf_utils import prepare_pcd_data, pcd_to_urdf_simple_geometries
from src.utils.dataset_utils import copy_cached_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create directory structure for a new scene")
    parser.add_argument("--scene_name", required=True, help="Name of the scene to create directories for")
    parser.add_argument("--save_labels", type=bool, default=False, help="Save a cache of the labels used for urdf generation")
    parser.add_argument("--load_cached_labels", type=bool, default=False, help="Load cached labels for urdf generation")

    args = parser.parse_args()
    if args.save_labels:
        sl = {}
    else:
        sl = None

    pcds_path = f"data/{args.scene_name}/pcds"
    pcd_data, center, labels, label_keywords = prepare_pcd_data(pcds_path, save_labels=sl, load_cached_labels=args.load_cached_labels)
    pcd_to_urdf_simple_geometries(pcd_data, center, labels, output_path=f"simple_urdf_scenes/{args.scene_name}/{args.scene_name}.urdf")
    copy_cached_frames(args.scene_name)