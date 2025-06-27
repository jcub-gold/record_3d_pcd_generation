from src.utils.data_utils import prepare_record3d_data, get_number
from src.utils.align_utils import create_pcd_from_frame
import os
import open3d as o3d
import pickle
from tqdm import tqdm

"""
    Function: visualize_pcd_from_frame
    ---------------------------------
    data: dictionary containing the prepared data from prepare_record3d_data
    frame_index: index of the frame to visualize
    samples: number of points to sample from the point cloud
    remove_outliers: whether to remove outliers from the point cloud

    Visualizes the point cloud from a specific frame in the data, testing the
    functionality of create_pcd_from_frame.
"""
def visualize_pcd_from_frame(data, frame_indices, samples=5000, remove_outliers=True):
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
def test_prepare_record3d_data(output_metadata_path=None):
    print('Preparing Record3D data...')

    images_dir = "scans/basement_kitchen_test_3/object_1/images"
    input_depth_dir = "scans/basement_kitchen_test_3/input_depth"
    new_metadata_path = "scans/basement_kitchen_test_3/new_metadata.json"

    data = prepare_record3d_data(images_dir, input_depth_dir, new_metadata_path)
    
    assert len(data["frames"]) > 0, "No frames found in the data."
    assert "intrinsics" in data, "Intrinsics not found in the data."

    print(f"Data prepared with {len(data['frames'])} frames")

    if output_metadata_path is not None:
        with open(output_metadata_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Metadata saved to {output_metadata_path}")
    
    return data

if __name__ == "__main__":
    print("Testing data_utils...")

    images_dir = "scans/basement_kitchen_test_3/object_1/images"
    # test_data = test_prepare_record3d_data(output_metadata_path="data.pkl")
    test_data = test_prepare_record3d_data()
    frame_indices = [get_number(os.path.splitext(p)[0]) for p in os.listdir(images_dir)]
    frame_indices.sort()
    samples = 10000
    visualize_pcd_from_frame(test_data, frame_indices, samples, remove_outliers=True)


    
