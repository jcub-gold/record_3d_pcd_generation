import numpy as np
import open3d as o3d
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa
from tqdm import tqdm
import copy
import os


"""
    Function: prepare_pcd_data
    --------------------------
    pcd_path: path to the .pcd file to be processed

    Returns a list dictionaries of the pcd data, each entry containing the pcd path,
    the point cloud object, the axis aligned bounding box, and the center 
    coordinates of each object.
"""
def prepare_pcd_data(pcds_path):
    print(pcds_path)
    pcd_data = []
    pcds = []
    for dirpath, dirnames, filenames in os.walk(pcds_path):
        for filename in filenames:
            pcd_path = os.path.join(dirpath, filename)
            pcd = o3d.io.read_point_cloud(pcd_path)
            pcds.append(pcd)

    R, combined_pcd_center = align_pcd_scene_via_object_aabb_minimization(pcds)
    
    input_path = os.path.dirname(pcds_path)
    aa_pcds_path = os.path.join(input_path, "aa_pcds")
    os.makedirs(aa_pcds_path, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(pcds_path):
        for filename in filenames:
            pcd_path = os.path.join(dirpath, filename)
            pcd = o3d.io.read_point_cloud(pcd_path)
            pcd.rotate(R, center=combined_pcd_center)
            # save the aligned pcd
            aa_pcd_path = os.path.join(aa_pcds_path, filename)
            o3d.io.write_point_cloud(aa_pcd_path, pcd)
            dict_data = {
                "pcd_path": aa_pcd_path,
                "pcd": pcd,
                "aabb": pcd.get_axis_aligned_bounding_box(),
                "center": pcd.get_center(),
                "label": input(f"Enter label for {filename}: ")
            }
            pcd_data.append(dict_data)

    return pcd_data

"""
    Function: align_pcd_scene_via_object_aabb_minimization
    ------------------------------------------------------
    pcds: list of open3d PointCloud objects to be aligned

    Aligns a list of point clouds by minimizing the total volume of their axis-aligned bounding boxes (AABBs)
    along the Y-axis amongst a -45 degree to 45 degree sweep of rotations, along a 0.5 degree increment.
"""
def align_pcd_scene_via_object_aabb_minimization(pcds):
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined_pcd += pcd

    best_angle = 0
    min_total_volume = float("inf")

    for deg in tqdm(np.arange(-45, 45.01, 0.5)):
        rad = np.deg2rad(deg)
        total_volume = 0

        for pcd in pcds:
            pcd_copy = copy.deepcopy(pcd)
            R = pcd_copy.get_rotation_matrix_from_axis_angle([0, rad, 0])
            pcd_copy.rotate(R, center=pcd_copy.get_center())

            aabb = pcd_copy.get_axis_aligned_bounding_box()
            extent = aabb.get_extent()
            volume = extent[0] * extent[1] * extent[2]
            total_volume += volume

        if total_volume < min_total_volume:
            min_total_volume = total_volume
            best_angle = rad

    best_R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, best_angle, 0])
    combined_pcd_center = combined_pcd.get_center()

    return best_R, combined_pcd_center



# pcds = []
# combined_pcd = o3d.geometry.PointCloud()
# for pcd_path in pcd_paths:
#     pcd = o3d.io.read_point_cloud(pcd_path)
#     combined_pcd += pcd
#     pcds.append(pcd)