import numpy as np
import open3d as o3d
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa
from tqdm import tqdm
import copy
import os
import re
import json


"""
    Function: prepare_pcd_data
    --------------------------
    pcd_path: path to the .pcd file to be processed
    save_labels: pass an empty dict to save labels as a json file
    use_cached_labels: boolean indicating whether to use cached labels from a previous run

    Returns a list dictionaries of the pcd data, each entry containing the pcd path,
    the point cloud object, the axis aligned bounding box, and the center 
    coordinates of each object.
"""
def prepare_pcd_data(pcds_path, save_labels=None, load_cached_labels=False):
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

    if load_cached_labels:
        with open(os.path.join(input_path, "cached_labels.json"), 'r') as f:
            cache = json.load(f)
    else:
        cache = None

    for dirpath, dirnames, filenames in os.walk(pcds_path):
        for filename in filenames:
            pcd_path = os.path.join(dirpath, filename)
            pcd = o3d.io.read_point_cloud(pcd_path)
            pcd.rotate(R, center=combined_pcd_center)
            # save the aligned pcd
            aa_pcd_path = os.path.join(aa_pcds_path, filename)
            o3d.io.write_point_cloud(aa_pcd_path, pcd)
            match = re.search(r'object_(\d+)', aa_pcd_path)
            assert match is not None, f"Filename {filename} does not match expected pattern."
            object_number = int(match.group(1))

            
            if cache is not None:
                label = cache[f"object_{object_number}"]
            else:
                label = input(f"Enter label for {filename}: ")
                if save_labels is not None:
                    save_labels[f"object_{object_number}"] = label

            dict_data = {
                "pcd_path": aa_pcd_path,
                "pcd": pcd,
                "aabb": pcd.get_axis_aligned_bounding_box(),
                "center": pcd.get_center(),
                "label": label,
                "object_number": object_number,
                "width": pcd.get_axis_aligned_bounding_box().get_extent()[0],
                "height": pcd.get_axis_aligned_bounding_box().get_extent()[1],
                "depth": pcd.get_axis_aligned_bounding_box().get_extent()[2]
            }
            # print(dict_data['aabb'].get_extent())

            width, height, depth = dict_data['width'], dict_data['height'], dict_data['depth']
            func_name = f"get_{dict_data['label']}_asset"
            asset_func = globals()[func_name]
            asset_name = f"{dict_data['label']}_{width:.3f}_{height:.3f}_{depth:.3f}_object_{dict_data["object_number"]}"

            dict_data["asset_func"] = asset_func
            dict_data["asset_name"] = asset_name

            pcd_data.append(dict_data)

    if save_labels is not None:
        json.dump(save_labels, open(os.path.join(input_path, "cached_labels.json"), 'w'), indent=4)
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
    width, height, depth = parent_asset['width'], parent_asset['height'], parent_asset['depth']

    s.add_object(parent_asset["asset_func"](width, height, depth), parent_asset['asset_name'])

    placed_assets.append(parent_asset)

    pbar = tqdm(total=len(unplaced_assets))

    while len(unplaced_assets) > 0:
        placed = False

        for potential_parent in placed_assets:
            pp_half_box = get_half_aabb_box(potential_parent)
            print(f"looking to find a child for object {potential_parent['object_number']}")
            for potential_child in unplaced_assets:
                potential_child_half_box = get_half_aabb_box(potential_child)
                # if the width and depth of the child are centered around the parent and it is directly next to the parent (there is height overlap in the pcd)
                if aabbs_overlap_xz(pp_half_box, potential_child_half_box) and check_overlap(axis=1, child_center=potential_child['center'], 
                                                                                                        parent_center=potential_parent['center'], 
                                                                                                        child_extents=potential_child['aabb'].get_extent(), 
                                                                                                        parent_extents=potential_parent['aabb'].get_extent()):
                    print(f"Placing object {potential_child['object_number']} on object {potential_parent['object_number']}")
                    potential_child['depth'], potential_child['width'] = potential_parent['depth'], potential_parent['width']

                    print(f"object {potential_child['object_number']} width {potential_child['width']} height {potential_child['height']} depth {potential_child['depth']}")
                    print(f"object {potential_parent['object_number']} width {potential_parent['width']} height {potential_parent['height']} depth {potential_parent['depth']}")
                
                    print("placing vertically")
                    place_top_or_bottom_asset(s=s,
                                              parent_asset=potential_parent,
                                              child_asset=potential_child,
                                              child_width=potential_child['width'],
                                              child_height=potential_child['height'],
                                              child_depth=potential_child['depth'])
                    print("placed")

                    placed_assets.append(potential_child)
                    unplaced_assets.remove(potential_child)
                    placed = True
                    break  # break inner for loop

                if aabbs_overlap_yz(pp_half_box, potential_child_half_box) and check_overlap(axis=0, child_center=potential_child['center'], 
                                                                                                        parent_center=potential_parent['center'], 
                                                                                                        child_extents=potential_child['aabb'].get_extent(), 
                                                                                                        parent_extents=potential_parent['aabb'].get_extent()):
                    print(f"Placing object {potential_child['object_number']} on object {potential_parent['object_number']}")
                    potential_child['depth'], potential_child['height'] = potential_parent['depth'], potential_parent['height']

                    print(f"object {potential_child['object_number']} width {potential_child['width']} height {potential_child['height']} depth {potential_child['depth']}")
                    print(f"object {potential_parent['object_number']} width {potential_parent['width']} height {potential_parent['height']} depth {potential_parent['depth']}")
                
                    print("placing laterally")
                    place_left_or_right_asset(s=s,
                                              parent_asset=potential_parent,
                                              child_asset=potential_child,
                                              child_width=potential_child['width'],
                                              child_height=potential_child['height'],
                                              child_depth=potential_child['depth'])
                    print("placed\n")
                    placed_assets.append(potential_child)
                    unplaced_assets.remove(potential_child)
                    placed = True
                    break  # break inner for loop
                
            if placed:
                break  # break outer for loop
        else:
            for potential_parent in placed_assets:
                pp_half_box = get_half_aabb_box(potential_parent)
                print(f"looking to find a child for object {potential_parent['object_number']}")
                for potential_child in unplaced_assets:
                    potential_child_half_box = get_half_aabb_box(potential_child)
                    # if the width and depth of the child are centered around the parent and it is directly next to the parent (there is height overlap in the pcd)
                    if aabbs_overlap_xz(pp_half_box, potential_child_half_box):
                        print(f"Placing object {potential_child['object_number']} on object {potential_parent['object_number']}")
                        potential_child['depth'], potential_child['width'] = potential_parent['depth'], potential_parent['width']

                        print(f"object {potential_child['object_number']} width {potential_child['width']} height {potential_child['height']} depth {potential_child['depth']}")
                        print(f"object {potential_parent['object_number']} width {potential_parent['width']} height {potential_parent['height']} depth {potential_parent['depth']}")
                    
                        print("placing vertically")
                        translation = abs(potential_child['center'][1] - potential_parent['center'][1]) - (potential_child['aabb'].get_extent()[1] + potential_parent['aabb'].get_extent()[1]) / 2.0
                        # translation = 0.2
                        transform = np.eye(4)
                        transform[2, 3] = translation
                        place_top_or_bottom_asset(s=s,
                                                parent_asset=potential_parent,
                                                child_asset=potential_child,
                                                child_width=potential_child['width'],
                                                child_height=potential_child['height'],
                                                child_depth=potential_child['depth'],
                                                transform=transform)
                        print("placed")

                        placed_assets.append(potential_child)
                        unplaced_assets.remove(potential_child)
                        placed = True
                        break  # break inner for loop

                    if aabbs_overlap_yz(pp_half_box, potential_child_half_box):
                        print(f"Placing object {potential_child['object_number']} on object {potential_parent['object_number']}")
                        potential_child['depth'], potential_child['height'] = potential_parent['depth'], potential_parent['height']

                        print(f"object {potential_child['object_number']} width {potential_child['width']} height {potential_child['height']} depth {potential_child['depth']}")
                        print(f"object {potential_parent['object_number']} width {potential_parent['width']} height {potential_parent['height']} depth {potential_parent['depth']}")
                    
                        print("placing laterally")
                        translation = abs(potential_child['center'][0] - potential_parent['center'][0]) - (potential_child['aabb'].get_extent()[0] + potential_parent['aabb'].get_extent()[0]) / 2.0
                        # translation = 0.2
                        transform = np.eye(4)
                        transform[0, 3] = translation
                        place_left_or_right_asset(s=s,
                                                parent_asset=potential_parent,
                                                child_asset=potential_child,
                                                child_width=potential_child['width'],
                                                child_height=potential_child['height'],
                                                child_depth=potential_child['depth'],
                                                transform=transform)
                        print("placed\n")
                        placed_assets.append(potential_child)
                        unplaced_assets.remove(potential_child)
                        placed = True
                        break  # break inner for loop
                    
                if placed:
                    break  # break outer for loop
            else:
                # If we reach here, it means no placement was made
                print("No suitable placement found for remaining assets.")
                break
        pbar.update(1)
    pbar.close()
    s.show()







def place_top_or_bottom_asset(s, parent_asset, child_asset, child_width, child_height, child_depth, transform=np.eye(4)):
    # place on top of the parent asset
    if (child_asset['center'][2] > parent_asset['center'][2]):
        s.add_object(child_asset["asset_func"](child_width, child_height, child_depth), 
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=('center', 'back', 'top'), 
                    connect_obj_anchor=('center', 'back', 'bottom'),
                    transform=transform)
        
    # place on bottom of the parent asset
    else:
        transform[2,3] = transform[2,3] * -1
        s.add_object(child_asset["asset_func"](child_width, child_height, child_depth), 
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=('left', 'back', 'bottom'), 
                    connect_obj_anchor=('left', 'back', 'top'),
                    transform=transform)
        
def place_left_or_right_asset(s, parent_asset, child_asset, child_width, child_height, child_depth, transform=np.eye(4)):
    # place on left of the parent asset
    if (child_asset['center'][0] < parent_asset['center'][0]):
        transform[0,3] = transform[0,3] * -1
        s.add_object(child_asset["asset_func"](child_width, child_height, child_depth), 
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=('left', 'back', 'top'), 
                    connect_obj_anchor=('right', 'back', 'top'),
                    transform=transform)
        
    # place on right of the parent asset
    else:
        s.add_object(child_asset["asset_func"](child_width, child_height, child_depth), 
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=('right', 'back', 'top'), 
                    connect_obj_anchor=('left', 'back', 'top'),
                    transform=transform)

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
        lower_compartment_types=("door_left",),
        handle_offset=(height * 0.35, width * 0.05))


# TODO: Fix three dimension URDF syncronization
# Either
# 1. Get a better PCD for dimensions
# 2. Try and look for neighbors to match the dimensions
# 3.       Would require some sort of threshold because I want to be able to include multiple heigh objects (i.e. drawers stacked on top of one another)