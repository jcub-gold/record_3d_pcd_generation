from src.utils.data_utils import prepare_record3d_data, get_number
from src.utils.align_utils import create_pcd_from_frame
from src.utils.pcd_to_urdf_utils import align_pcd_scene_via_object_aabb_minimization, prepare_pcd_data, pcd_to_urdf_simple_geometries
from src.utils.pcd_to_urdf_utils import pcd_to_urdf_simple_geometries
import os
import open3d as o3d
import pickle
from tqdm import tqdm
import numpy as np

"""
    Function: visualize_pcd_from_frames
    -----------------------------------
    data: dictionary containing the prepared data from prepare_record3d_data
    frame_indices: list of frame indices to visualize
    samples: number of points to sample from the point cloud
    remove_outliers: whether to remove outliers from the point cloud

    Visualizes the point cloud from a specific frame in the data, testing the
    functionality of create_pcd_from_frame.
"""
def visualize_pcd_from_frames(data, frame_indices, samples=5000, remove_outliers=True):
    pcd = None
    for frame_index in tqdm(frame_indices, desc="Processing frames"):
        if pcd is None:
            pcd = create_pcd_from_frame(data, frame_index, samples, remove_outliers)
        else:
            pcd += create_pcd_from_frame(data, frame_index, samples, remove_outliers)
    o3d.visualization.draw_geometries([pcd])

"""
    Function: test_prepare_record3d_data
    ------------------------------------

    Tests the prepare_record3d_data function and return the output data.
    Remember to specify the correct paths for images_dir, depth_dir, and metadata_path.
"""
def test_prepare_record3d_data(dir, output_metadata_path=None):
    print('Preparing Record3D data...')

    images_dir = dir + "/object_1/images"
    input_depth_dir = dir + "/input_depth"
    new_metadata_path = dir + "/new_metadata.json"

    data = prepare_record3d_data(images_dir, input_depth_dir, new_metadata_path)
    
    assert len(data["frames"]) > 0, "No frames found in the data."
    assert "intrinsics" in data, "Intrinsics not found in the data."

    print(f"Data prepared with {len(data['frames'])} frames")

    if output_metadata_path is not None:
        with open(output_metadata_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Metadata saved to {output_metadata_path}")
    
    return data

"""
    Function: visualize_pcd_aabb_and_center
    ---------------------------------------
    ply_paths: list of paths to .ply files to visualize 

    Visualizes the point clouds from the given .ply files, along with their oriented
    bounding boxes (AABBs) and centers.
"""
def visualize_pcd_aabb_and_center(ply_paths=None):
    pcds = []
    for pcd_path in ply_paths:
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcds.append(pcd)

    R, combined_pcd_center = align_pcd_scene_via_object_aabb_minimization(pcds)

    geometries = []
    for ply_path in ply_paths:
        pcd = o3d.io.read_point_cloud(ply_path)
        pcd.rotate(R, center=combined_pcd_center)
        aabb = pcd.get_axis_aligned_bounding_box()
        # aabb = pcd.get_oriented_bounding_box()
        aabb.color = (1, 0, 0)
        center = aabb.get_center()

        # Create a small sphere to mark the center
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        center_sphere.translate(center)
        center_sphere.paint_uniform_color([0, 1, 0])  # green

        geometries.extend([pcd, aabb, center_sphere])
    
    o3d.visualization.draw_geometries(geometries)
    
def test_prepare_pcd_data():
    print("Preparing PCD data...")

    pcds_path = "pcds"
    pcd_data = prepare_pcd_data(pcds_path)

    assert len(pcd_data) > 0, "No PCD data found."
    return pcd_data

def test_pcd_to_urdf_simple_geometries():
    print("Testing pcd_to_urdf_simple_geometries...")

    pcds_path = "data/basement_base_cabinet/pcds"
    pcd_data = prepare_pcd_data(pcds_path, save_labels=None, load_cached_labels=True)

    assert len(pcd_data) > 0, "No PCD data found."
    
    pcd_to_urdf_simple_geometries(pcd_data)
    print("PCD to URDF conversion completed successfully.")

if __name__ == "__main__":
    # ## testign data_utils and create_pcd_from_frame
    # print("Testing data_utils and create_pcd_from_frame...")

    # images_dir = "data/basement_base_cabinet/record3d_input"
    # # test_data = test_prepare_record3d_data(output_metadata_path="data.pkl")
    # test_data = test_prepare_record3d_data(dir=images_dir)
    # # frame_indices = [get_number(os.path.splitext(p)[0]) for p in os.listdir(images_dir)]
    # # frame_indices.sort()
    # frame_indices = [24, 124]  # Example frame indices to visualize
    # samples = 5000
    # visualize_pcd_from_frames(test_data, frame_indices, samples, remove_outliers=True)

    # testing visualize_pcd_aabb_and_center
    print("Testing visualize_pcd_aabb_and_center...")
    ply_paths = []
    root_path = "data/base_cabinet_test/aa_pcds" # "data/basement_test/pcds" # 'test_pcd' #  "data/basement_base_cabinet/pcds" # "data/basement_base_cabinet_extension/aa_pcds" # pcds

    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(".ply"):
                ply_path = os.path.join(dirpath, filename)
                ply_paths.append(ply_path)

    # visualize_pcd_obb_and_center(ply_paths=ply_paths)
    visualize_pcd_aabb_and_center(ply_paths=ply_paths)

    # ## testing test_prepare_pcd_data
    # print("Testing test_prepare_pcd_data...")
    # pcd_data = test_prepare_pcd_data()

    # ## testing pcd_to_urdf_simple_geometries
    # print("Testing pcd_to_urdf_simple_geometries...")
    # pcds_path = "pcds"
    # pcd_data = prepare_pcd_data(pcds_path)
    # pcd_to_urdf_simple_geometries(pcd_data)

    # testing pcd_to_urdf_simple_geometries with specific data
    # test_pcd_to_urdf_simple_geometries()


    
