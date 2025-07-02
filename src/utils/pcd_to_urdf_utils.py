import numpy as np
import open3d as o3d
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa
from tqdm import tqdm
import copy
import os
import re


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
            match = re.search(r'object_(\d+)_state', aa_pcd_path)
            assert match is not None, f"Filename {filename} does not match expected pattern."
            object_number = int(match.group(1))



            dict_data = {
                "pcd_path": aa_pcd_path,
                "pcd": pcd,
                "aabb": pcd.get_axis_aligned_bounding_box(),
                "center": pcd.get_center(),
                "label": input(f"Enter label for {filename}: "),
                "object_number": object_number
            }

            extents = dict_data['aabb'].get_extent()
            width, depth, height = extents[0], extents[1], extents[2]
            func_name = f"get_{dict_data['label']}_asset"
            asset_func = globals()[func_name]
            asset_name = f"{dict_data['label']}_{width}_{height}_{depth}_object_{dict_data["object_number"]}"

            dict_data["asset_func"] = asset_func
            dict_data["asset_name"] = asset_name

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

    for deg in tqdm(np.arange(-45, 45.01, 0.5), desc="Rotating degrees"):
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

# translation plus recurrent nature of placing assets
def pcd_to_urdf_simple_geometries(pcd_data, output_dir=None):
    s = synth.Scene()
    assert len(pcd_data) > 0, "No PCD data provided."

    unplaced_assets = pcd_data.copy()
    placed_assets = []

    parent_asset = unplaced_assets.pop()
    extents = parent_asset['aabb'].get_extent()
    width, depth, height = extents[0], extents[1], extents[2]

    s.add_object(parent_asset["asset_func"](width, height, depth), parent_asset['asset_name'])
    print(parent_asset["object_number"])

    placed_assets.append(parent_asset)


    while len(unplaced_assets) > 0:
        placed = False

        for potential_parent in placed_assets:
            pp_half_box = get_half_aabb_box(potential_parent)
            for potential_child in unplaced_assets:
                potential_child_half_box = get_half_aabb_box(potential_child)
                # if the width and depth of the child are centered around the parent and it is directly next to the parent (there is height overlap in the pcd)
                if aabbs_overlap_xy(pp_half_box, potential_child_half_box) and check_overlap(axis=2, child_center=potential_child['center'], 
                                                                                                        parent_center=potential_parent['center'], 
                                                                                                        child_extents=potential_child['aabb'].get_extent(), 
                                                                                                        parent_extents=potential_parent['aabb'].get_extent()):
                    depth, width = potential_parent['aabb'].get_extent()[1], potential_parent['aabb'].get_extent()[0]
                    height = potential_child['aabb'].get_extent()[2]
                
                    place_top_or_bottom_asset(s=s,
                                              parent_asset=potential_parent,
                                              child_asset=potential_child,
                                              child_width=width,
                                              child_height=height,
                                              child_depth=depth)

                    placed_assets.append(potential_child)
                    unplaced_assets.remove(potential_child)
                    placed = True
                    break  # break inner for loop

                if aabbs_overlap_yz(pp_half_box, potential_child_half_box):
                    depth, height = potential_parent['aabb'].get_extent()[1], potential_parent['aabb'].get_extent()[2]
                    width = potential_child['aabb'].get_extent()[0]

                    place_left_or_right_asset(s=s,
                                              parent_asset=potential_parent,
                                              child_asset=potential_child,
                                              child_width=width,
                                              child_height=height,
                                              child_depth=depth)

                    placed_assets.append(potential_child)
                    unplaced_assets.remove(potential_child)
                    placed = True
                    break  # break inner for loop
                
            if placed:
                break  # break outer for loop

    s.show()















def place_top_or_bottom_asset(s, parent_asset, child_asset, child_width, child_height, child_depth):
    # place on top of the parent asset
    if (child_asset['center'][2] > parent_asset['center'][2]):
        s.add_object(child_asset["asset_func"](child_width, child_height, child_depth), 
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=('center', 'back', 'top'), 
                    connect_obj_anchor=('center', 'back', 'bottom'))
        
    # place on bottom of the parent asset
    else:
        s.add_object(child_asset["asset_func"](child_width, child_height, child_depth), 
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=('left', 'back', 'bottom'), 
                    connect_obj_anchor=('left', 'back', 'top'))
        
def place_left_or_right_asset(s, parent_asset, child_asset, child_width, child_height, child_depth):
    # place on top of the parent asset
    if (child_asset['center'][0] > parent_asset['center'][0]):
        s.add_object(child_asset["asset_func"](child_width, child_height, child_depth), 
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=('left', 'back', 'top'), 
                    connect_obj_anchor=('right', 'back', 'top'))
        
    # place on bottom of the parent asset
    else:
        s.add_object(child_asset["asset_func"](child_width, child_height, child_depth), 
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=('right', 'back', 'top'), 
                    connect_obj_anchor=('left', 'back', 'top'))

# axis: axis along which to check overlap (0 for x, 1 for y, 2 for z)
def check_overlap(axis, child_center, parent_center, child_extents, parent_extents):
    child_min = child_center[axis] - child_extents[axis] / 2.0
    child_max = child_center[axis] + child_extents[axis] / 2.0
    parent_min = parent_center[axis] - parent_extents[axis] / 2.0
    parent_max = parent_center[axis] + parent_extents[axis] / 2.0

    return not (child_max < parent_min or child_min > parent_max)

    
def get_half_aabb_box(asset):
    extents = asset['aabb'].get_extent()
    width, height, depth = extents[0], extents[1], extents[2]
    center = asset['center']
    quarter_extents = np.array([width, height, depth]) / 4.0
    min_bound = center - quarter_extents
    max_bound = center + quarter_extents
    small_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return small_box

def aabbs_overlap_xy(aabb1, aabb2):
    return _aabbs_overlap_2d(aabb1, aabb2, axes=(0, 1))

def aabbs_overlap_xz(aabb1, aabb2):
    return _aabbs_overlap_2d(aabb1, aabb2, axes=(0, 2))

def aabbs_overlap_yz(aabb1, aabb2):
    return _aabbs_overlap_2d(aabb1, aabb2, axes=(1, 2))

def _aabbs_overlap_2d(aabb1, aabb2, axes):
    min1, max1 = aabb1.get_min_bound(), aabb1.get_max_bound()
    min2, max2 = aabb2.get_min_bound(), aabb2.get_max_bound()

    for axis in axes:
        if max1[axis] < min2[axis] or max2[axis] < min1[axis]:
            return False
    return True







### asset generation functions
def get_drawer_asset(width, height, depth):
    return pa.BaseCabinetAsset(
        width=width, 
        height=height, 
        depth=depth, 
        num_drawers_horizontal=1,
        include_cabinet_doors=False,
        include_foot_panel=False)

def get_lower_left_cabinet_asset(width, height, depth):
    return pa.BaseCabinetAsset(width=width, 
        height=height, 
        depth=depth, 
        num_drawers_vertical=0,
        include_cabinet_doors=True,
        include_foot_panel=False,
        lower_compartment_types=("door_right",),
        handle_offset=(height * 0.35, width * 0.05))

def get_lower_right_cabinet_asset(width, height, depth):
    return pa.BaseCabinetAsset(width=width, 
        height=height, 
        depth=depth, 
        num_drawers_vertical=0,
        include_cabinet_doors=True,
        include_foot_panel=False,
        upper_compartment_types=("door_left",),
        handle_offset=(height * 0.35, width * 0.05))


