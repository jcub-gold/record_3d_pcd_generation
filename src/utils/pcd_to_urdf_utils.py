import numpy as np
import open3d as o3d
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa
from tqdm import tqdm
import copy
import os
import re
import json
from scene_synthesizer.assets import BoxAsset
from sklearn.cluster import KMeans
from collections import defaultdict


labels = ['drawer', 'lower_left_cabinet', 'lower_right_cabinet', 'upper_left_cabinet', 'upper_right_cabinet', 'box']
label_clusters = [3, 1, 1, 1]
extent_labels = ['width', 'height', 'depth']

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

    # Define additional 90-degree rotation around Y-axis
    extra_rotation = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.pi / 2, 0])

    # Apply the extra rotation *after* the original one
    R = extra_rotation @ R
    
    input_path = os.path.dirname(pcds_path)
    aa_pcds_path = os.path.join(input_path, "aa_pcds")
    os.makedirs(aa_pcds_path, exist_ok=True)

    if load_cached_labels:
        with open(os.path.join(input_path, "cached_labels.json"), 'r') as f:
            cache = json.load(f)
    else:
        print("Either type full label or else a number cooresponding to the following label:\n0: drawer\n1: lower_left_cabinet\n2: lower_right_cabinet\n3: upper_left_cabinet\n4: upper_right_cabinet\n5: box")
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
                label_input = input(f"Enter label for {filename}: ")
                try:
                    index = int(label_input)
                    if 0 <= index < len(labels):
                        label = labels[index]
                    else:
                        raise ValueError("Index out of range")
                except ValueError:
                    if label_input in labels:
                        label = label_input
                    else:
                        raise ValueError("Invalid label or index")
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
                "depth": pcd.get_axis_aligned_bounding_box().get_extent()[2],
                "relative_alignment": [0, 1, 2]
            }
            if 'cabinet' in dict_data['label']:
                dict_data['width'] = np.sqrt(dict_data['width']**2 + dict_data['depth']**2)
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
    
    # use function here
    label_keywords = ['drawer', 'lower', 'upper', 'box']
    extent_indices = [(0, 1, 2), (1,), (0, 1), (0, 1)]
    for i in range(len(label_keywords)):
        label_keyword = label_keywords[i]
        clusters = label_clusters[i]
        extent_indice = extent_indices[i]
        assign_extent_clusters(pcd_data, label_keyword, clusters, extent_indice)
    
    for i in range(len(label_keywords)):
        label_keyword = label_keywords[i]
        extent_indice = extent_indices[i]
        set_cluster_fixed_extents(pcd_data, label_keyword, extent_indice)

    # print(pcd_data)
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

    # for deg in tqdm(np.arange(-45, 45.01, 0.5), desc="Rotating degrees"):
    #     rad = np.deg2rad(deg)
    #     total_volume = 0

    #     for pcd in pcds:
    #         pcd_copy = copy.deepcopy(pcd)
    #         R = pcd_copy.get_rotation_matrix_from_axis_angle([0, rad, 0])
    #         pcd_copy.rotate(R, center=pcd_copy.get_center())

    #         aabb = pcd_copy.get_axis_aligned_bounding_box()
    #         extent = aabb.get_extent()
    #         volume = extent[0] * extent[1] * extent[2]
    #         total_volume += volume

    #     if total_volume < min_total_volume:
    #         min_total_volume = total_volume
    #         best_angle = rad

    for deg in tqdm(np.arange(-45, 45.01, 0.5), desc="Rotating degrees"):
        rad = np.deg2rad(deg)
        total_volume = 0

        pcd_copy = copy.deepcopy(combined_pcd)
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


def pcd_to_urdf_simple_geometries(pcd_data, output_dir=None):
    unplaced_assets = {}
    for asset in pcd_data:
        if unplaced_assets.get(asset['label']) is None:
            unplaced_assets[asset['label']] = []
        unplaced_assets[asset['label']].append(asset)

    s = synth.Scene()
    placed_assets = []
    parent_asset = unplaced_assets['drawer'].pop()
    # print(parent_asset['object_number'])
    set_dimension_parent_default(parent_asset, aligned_axis=parent_asset['relative_alignment'], target_axis=0, data=pcd_data)
    set_dimension_parent_default(parent_asset, aligned_axis=parent_asset['relative_alignment'], target_axis=1, data=pcd_data)
    set_dimension_parent_default(parent_asset, aligned_axis=parent_asset['relative_alignment'], target_axis=2, data=pcd_data)

    width, height, depth = parent_asset['width'], parent_asset['height'], parent_asset['depth']
    s.add_object(parent_asset["asset_func"](width, height, depth), parent_asset['asset_name'])
    placed_assets.append(parent_asset)

    try_to_place_strict(unplaced_assets, 'drawer', placed_assets, pcd_data, s, 0)
    try_to_place_medium(unplaced_assets, 'drawer', placed_assets, pcd_data, s, 0)
    try_to_place_strict(unplaced_assets, 'drawer', placed_assets, pcd_data, s, 0)
    try_to_place_medium(unplaced_assets, 'drawer', placed_assets, pcd_data, s, 0)
    try_to_place_strict(unplaced_assets, 'drawer', placed_assets, pcd_data, s, 0)

    try_to_place_strict(unplaced_assets, 'lower_left_cabinet', placed_assets, pcd_data, s, 1)
    try_to_place_strict(unplaced_assets, 'lower_right_cabinet', placed_assets, pcd_data, s, 1)
    try_to_place_strict(unplaced_assets, 'drawer', placed_assets, pcd_data, s, 0)
    try_to_place_strict(unplaced_assets, 'lower_left_cabinet', placed_assets, pcd_data, s, 0)
    try_to_place_strict(unplaced_assets, 'lower_right_cabinet', placed_assets, pcd_data, s, 0)
    try_to_place_medium(unplaced_assets, 'lower_left_cabinet', placed_assets, pcd_data, s, 0)

    try_to_place_strict(unplaced_assets, 'box', placed_assets, pcd_data, s, 0)
    # try_to_place_medium(unplaced_assets, 'lower_right_cabinet', placed_assets, pcd_data, s, 0)
    for asset in placed_assets:
        print(asset['object_number'])

    min_corner, max_corner = s.get_bounds()      # scene_synthesizer.scene.Scene 
    max_x = max_corner[0]                        # the right-most world-x
    scene_width = max_corner[0] - min_corner[0]
    for asset in placed_assets:
        if asset['label'] == 'drawer':
            height = 0.2 * asset['fixed_height']
            depth = 1.02 * asset['fixed_depth']
    
    corner_asset = 6
    for asset in placed_assets:
        if asset['object_number'] == corner_asset:
            corner_asset = asset
            break

    counter = get_box_asset(scene_width, depth, height)
    s.add_object(counter, 
                    'counter',
                    connect_parent_id=corner_asset['asset_name'],
                    connect_parent_anchor=('left', 'back', 'top'), 
                    connect_obj_anchor=('left', 'back', 'bottom'))
    
    print(scene_width)
    s.export('broken.urdf')
    s.show()
    return

# translation plus recurrent nature of placing assets
# # def _pcd_to_urdf_simple_geometries(pcd_data, output_dir=None):
#     for asset in pcd_data:
#         if 'cabinet' in asset['label']:
#             asset['width'] = (np.sqrt(asset['width']**2 + asset['depth']**2))

#     s = synth.Scene()
#     assert len(pcd_data) > 0, "No PCD data provided."

#     unplaced_assets = pcd_data.copy()
#     placed_assets = []

#     parent_asset = unplaced_assets.pop()
#     width, height, depth = parent_asset['width'], parent_asset['height'], parent_asset['depth']

#     s.add_object(parent_asset["asset_func"](width, height, depth), parent_asset['asset_name'])

#     placed_assets.append(parent_asset)

#     pbar = tqdm(total=len(unplaced_assets))

#     while len(unplaced_assets) > 0:
#         placed = False

#         for potential_parent in placed_assets:
#             pp_half_box = get_half_aabb_box(potential_parent)
#             print(f"looking to find a child for object {potential_parent['object_number']}")
#             for potential_child in unplaced_assets:

#                 potential_child_half_box = get_half_aabb_box(potential_child)
#                 # if the width and depth of the child are centered around the parent and it is directly next to the parent (there is height overlap in the pcd)
#                 if aabbs_overlap_xz(pp_half_box, potential_child_half_box) and check_overlap(axis=1, potential_child=potential_child, potential_parent=potential_parent):

#                     print(f"Placing object {potential_child['object_number']} on object {potential_parent['object_number']}")
#                     potential_child['depth'], potential_child['width'] = potential_parent['depth'], potential_parent['width']

#                     print(f"object {potential_child['object_number']} width {potential_child['width']} height {potential_child['height']} depth {potential_child['depth']}")
#                     print(f"object {potential_parent['object_number']} width {potential_parent['width']} height {potential_parent['height']} depth {potential_parent['depth']}")
                
#                     print("placing vertically")
#                     place_top_or_bottom_asset(s=s,
#                                               parent_asset=potential_parent,
#                                               child_asset=potential_child,
#                                               child_width=potential_child['width'],
#                                               child_height=potential_child['height'],
#                                               child_depth=potential_child['depth'])
#                     print("placed")

#                     placed_assets.append(potential_child)
#                     unplaced_assets.remove(potential_child)
#                     placed = True
#                     break  # break inner for loop

#                 if aabbs_overlap_yz(pp_half_box, potential_child_half_box) and check_overlap(axis=0, potential_child=potential_child, potential_parent=potential_parent):
#                     print(f"Placing object {potential_child['object_number']} on object {potential_parent['object_number']}")
#                     potential_child['depth'], potential_child['height'] = potential_parent['depth'], potential_parent['height']

#                     print(f"object {potential_child['object_number']} width {potential_child['width']} height {potential_child['height']} depth {potential_child['depth']}")
#                     print(f"object {potential_parent['object_number']} width {potential_parent['width']} height {potential_parent['height']} depth {potential_parent['depth']}")
                
#                     print("placing laterally")
#                     place_left_or_right_asset(s=s,
#                                               parent_asset=potential_parent,
#                                               child_asset=potential_child,
#                                               child_width=potential_child['width'],
#                                               child_height=potential_child['height'],
#                                               child_depth=potential_child['depth'])
#                     print("placed\n")
#                     placed_assets.append(potential_child)
#                     unplaced_assets.remove(potential_child)
#                     placed = True
#                     break  # break inner for loop
                
#             if placed:
#                 break  # break outer for loop
#         else:
#             for potential_parent in placed_assets:
#                 pp_half_box = get_half_aabb_box(potential_parent)
#                 print(f"looking to find a child for object {potential_parent['object_number']}")
#                 for potential_child in unplaced_assets:
#                     potential_child_half_box = get_half_aabb_box(potential_child)
#                     # if the width and depth of the child are centered around the parent and it is directly next to the parent (there is height overlap in the pcd)
#                     if aabbs_overlap_xz(pp_half_box, potential_child_half_box):
#                         print(f"Placing object {potential_child['object_number']} on object {potential_parent['object_number']}")
#                         potential_child['depth'], potential_child['width'] = potential_parent['depth'], potential_parent['width']

#                         print(f"object {potential_child['object_number']} width {potential_child['width']} height {potential_child['height']} depth {potential_child['depth']}")
#                         print(f"object {potential_parent['object_number']} width {potential_parent['width']} height {potential_parent['height']} depth {potential_parent['depth']}")
                    
#                         print("placing vertically")
#                         translation = abs(potential_child['center'][1] - potential_parent['center'][1]) - (potential_child['aabb'].get_extent()[1] + potential_parent['aabb'].get_extent()[1]) / 2.0
#                         # translation = 0.2
#                         transform = np.eye(4)
#                         transform[2, 3] = translation
#                         place_top_or_bottom_asset(s=s,
#                                                 parent_asset=potential_parent,
#                                                 child_asset=potential_child,
#                                                 child_width=potential_child['width'],
#                                                 child_height=potential_child['height'],
#                                                 child_depth=potential_child['depth'],
#                                                 transform=transform)
#                         print("placed")

#                         placed_assets.append(potential_child)
#                         unplaced_assets.remove(potential_child)
#                         placed = True
#                         break  # break inner for loop

#                     if aabbs_overlap_yz(pp_half_box, potential_child_half_box):
#                         print(f"Placing object {potential_child['object_number']} on object {potential_parent['object_number']}")
#                         potential_child['depth'], potential_child['height'] = potential_parent['depth'], potential_parent['height']

#                         print(f"object {potential_child['object_number']} width {potential_child['width']} height {potential_child['height']} depth {potential_child['depth']}")
#                         print(f"object {potential_parent['object_number']} width {potential_parent['width']} height {potential_parent['height']} depth {potential_parent['depth']}")
                    
#                         print("placing laterally")
#                         if (potential_child['object_number'] == 10):
#                             s.show()
#                         translation = abs(potential_child['center'][0] - potential_parent['center'][0]) - (potential_child['aabb'].get_extent()[0] + potential_parent['aabb'].get_extent()[0]) / 2.0
#                         # translation = 0.2
#                         transform = np.eye(4)
#                         transform[0, 3] = translation
#                         place_left_or_right_asset(s=s,
#                                                 parent_asset=potential_parent,
#                                                 child_asset=potential_child,
#                                                 child_width=potential_child['width'],
#                                                 child_height=potential_child['height'],
#                                                 child_depth=potential_child['depth'],
#                                                 transform=transform)
#                         print("placed\n")
#                         placed_assets.append(potential_child)
#                         unplaced_assets.remove(potential_child)
#                         placed = True
#                         break  # break inner for loop
                    
#                 if placed:
#                     break  # break outer for loop
#             else:
#                 # If we reach here, it means no placement was made
#                 print("No suitable placement found for remaining assets.")
#                 # s.show()
#                 s.export('broken.urdf')
#                 break
#         pbar.update(1)
#     pbar.close()
#     # s.show()
#     s.export('simple.urdf')







def place_vertical_asset(s, parent_asset, child_asset, child_width, child_height, child_depth, transform=np.eye(4)):
    # place on top of the parent asset
    if (child_asset['center'][2] > parent_asset['center'][2]):
        s.add_object(child_asset["asset_func"](child_width, child_height, child_depth), 
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=('left', 'back', 'top'), 
                    connect_obj_anchor=('left', 'back', 'bottom'),
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
        
def place_lateral_asset(s, parent_asset, child_asset, child_width, child_height, child_depth, transform=np.eye(4)):
    # place on left of the parent asset
    if (child_asset['center'][0] < parent_asset['center'][0]):
        transform[0,3] = transform[0,3] * -1
        s.add_object(child_asset["asset_func"](child_width, child_height, child_depth), 
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=('left', 'back', 'bottom'), 
                    connect_obj_anchor=('right', 'back', 'bottom'),
                    transform=transform)
        
    # place on right of the parent asset
    else:
        s.add_object(child_asset["asset_func"](child_width, child_height, child_depth), 
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=('right', 'back', 'bottom'), 
                    connect_obj_anchor=('left', 'back', 'bottom'),
                    transform=transform)

# axis: axis along which to check overlap (0 for x, 1 for y, 2 for z)
def check_overlap(axis, potential_parent, potential_child):

    child_center=potential_child['center']
    parent_center=potential_parent['center']
    child_extents=potential_child['aabb'].get_extent()
    parent_extents=potential_parent['aabb'].get_extent()

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

def half_extent_overlap_2d(parent: dict,
                           child: dict,
                           threshold=1.0,
                           axes: tuple[int, int] = (0, 2)) -> bool:
    def is_half_inside(shrink: dict, fixed: dict) -> bool:
        shrink_center = shrink["center"]
        fixed_center  = fixed["center"]
        shrink_half   = shrink["aabb"].get_extent() * 0.25  # half of half = quarter
        fixed_half    = fixed["aabb"].get_extent() * 0.5

        for ax in axes:
            shrink_min = shrink_center[ax] - shrink_half[ax]
            shrink_max = shrink_center[ax] + shrink_half[ax]

            # Expand fixed boundary by (threshold - 1) * half_extent on each side
            expansion = fixed_half[ax] * (threshold - 1)
            fixed_min = fixed_center[ax] - fixed_half[ax] - expansion
            fixed_max = fixed_center[ax] + fixed_half[ax] + expansion

            if shrink_min < fixed_min or shrink_max > fixed_max:
                return False
        return True

    return is_half_inside(child, parent) or is_half_inside(parent, child)

def same_depth(parent, child, depth_axis, weight=0.05):
    p_depth = parent["aabb"].get_extent()[depth_axis]
    c_depth = child["aabb"].get_extent()[depth_axis]
    p_center = parent["center"][depth_axis]
    c_center = child["center"][depth_axis]
    max_depth_threshold = max(p_depth, c_depth) * weight
    p_min = p_center - (p_depth / 2)
    c_min = c_center - (c_depth / 2)
    if abs(p_min - c_min) <= max_depth_threshold:
        return True
    return False

def _half_extent_overlap_2d(parent: dict,
                           child: dict,
                           axes: tuple[int, int] = (0, 2)) -> bool:
    # Gather centres and half-extents (width/2) on the two axes
    p_c  = parent["center"]
    c_c  = child ["center"]
    p_he = parent["aabb"].get_extent() * 0.5   # half-extents vector
    c_he = child ["aabb"].get_extent() * 0.5

    p = [None, None, None]
    c = [None, None, None]
    for ax in axes:
        p_min = p_c[ax] - p_he[ax]
        p_max = p_c[ax] + p_he[ax]
        c_min = c_c[ax] - c_he[ax]
        c_max = c_c[ax] + c_he[ax]

        p[ax] = [p_min, p_max]
        c[ax] = [c_min, c_max]
    
    for i in range(2):
        if i == 1:
            temp = p[axes[0]]
            p[axes[0]] = c[axes[0]] 
            c[axes[0]] = temp
        if compare_axes(p, c, axes) or compare_axes(c, p, axes):
            return True
    return False

def compare_axes(fixed, shrink, axes):
    for ax in axes:
        if shrink[ax][0] / 2 < fixed[ax][0] or shrink[ax][1] / 2 > fixed[ax][1]:
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

def get_upper_left_cabinet_asset(width, height, depth):
    return pa.WallCabinetAsset(width=width, 
                                height=height, 
                                depth=depth, 
                                compartment_types=("door_right",),
                                handle_offset=(height * -0.4, width * 0.05))

def get_upper_right_cabinet_asset(width, height, depth):
    return pa.WallCabinetAsset(width=width, 
                                height=height, 
                                depth=depth, 
                                compartment_types=("door_left",),
                                handle_offset=(height * -0.4, width * 0.05))

def get_box_asset(width, height, depth):
    return BoxAsset(extents=[width, height, depth])


# TODO: Fix three dimension URDF syncronization
# Either
# 1. Get a better PCD for dimensions
# 2. Try and look for neighbors to match the dimensions
# 3.       Would require some sort of threshold because I want to be able to include multiple heigh objects (i.e. drawers stacked on top of one another)




def assign_extent_clusters(data,
                           label_keyword: str,
                           n_clusters: int = 3,
                           extent_indices=(0, 1, 2)):
    matches = [d for d in data if label_keyword in d["label"]]

    if len(matches) == 0:
        print(f"[assign_extent_clusters] No assets matched '{label_keyword}'.")
        return

    feats = np.array([[ [d["width"], d["height"], d["depth"]][i]
                        for i in extent_indices ]
                       for d in matches ])

    k = min(n_clusters, len(matches))
    kmeans = KMeans(n_clusters=k, random_state=42).fit(feats)

    key = f"{label_keyword}_cluster_id"
    for d, cid in zip(matches, kmeans.labels_):
        d[key] = int(cid)



def set_cluster_fixed_extents(data,
                              label_keyword: str,
                              extent_indices=(0, 1, 2)):
    if isinstance(extent_indices, int):
        extent_indices = (extent_indices,)
    idx2name = {0: "width", 1: "height", 2: "depth"}
    names    = [idx2name[i] for i in extent_indices]

    key_cluster = f"{label_keyword}_cluster_id"
    matches = [d for d in data
               if label_keyword in d["label"] and key_cluster in d]
    if not matches:
        print(f"[set_cluster_fixed_extents] Nothing to do for '{label_keyword}'.")
        return
    clusters = defaultdict(list)
    for d in matches:
        clusters[d[key_cluster]].append(d)
    for cid, objs in clusters.items():
        means = {name: float(np.mean([o[name] for o in objs])) for name in names}
        # print("-------")
        for o in objs:
            # if o['label'] == 'drawer':
                # print(o['object_number'])
            for name in names:
                o[f"fixed_{name}"] = means[name]


'''
k means cluster for each object
set fixed heights/widths for each object based off of average in the cluster
- drawer: fix height, width, depth (depth across all clusters)
- lower cabinet: fix height
- upper cabinet: fix height, caculated width, calulated depth abs(drawer_center_depth - drawer_depth * 1.5 - (cabinet_center_depth - cabinet_Depth * 0.5))
- box fix height and width
place objects and override fixed heights if needed.
track reference frame for each object, only look in that reference frame

if overlap turn 90 degrees so there is no collision

when placing objects:
-try to place all drawers first, allowing translations but not overlap
-try to place all lower cabinets, allowing translations but not overlap
-try to place all boxes allowing translation and not overlap
-allow overlap
-unlock upper cabinet placement
 -if can't place, laterally and could depth wise, rotate


*** -if found go back to top


'''
def try_to_place_strict(unplaced_assets, label, placed_assets, pcd_data, s, axis, allow_overlap=False):
    if axis == 1:
        func_name = "place_vertical_asset"
        axes=[0,2,1]
    elif axis == 0:
        func_name = 'place_lateral_asset'
        axes = [1,2,0]
    placement_func = globals()[func_name]

    placed = False
    for potential_parent in placed_assets:
        aligned_axis = potential_parent['relative_alignment']
        pp_half_box = get_half_aabb_box(potential_parent)
        for potential_child in unplaced_assets[label]:
            pc_half_box = get_half_aabb_box(potential_child)
            if not allow_overlap and overlaps_any_placed(potential_child, placed_assets):
                continue
            # if (potential_child['object_number'] == 4):
            #     # s.show()
            #     continue
            # if (potential_child['object_number'] == 5):
            #     # s.show()
            #     continue
            if axis == 0 and potential_child['object_number'] == 10 and potential_parent['object_number'] == 9:
                print(_aabbs_overlap_2d(pp_half_box, pc_half_box, axes=(aligned_axis[axes[0]], aligned_axis[axes[1]])))
                print(check_overlap(axis=aligned_axis[axes[2]],
                                                                                                                      potential_child=potential_child,
                                                                                                                         potential_parent=potential_parent))
                print(
                    f"Parent  – center: {potential_parent['center']}, "
                    f"extents: {potential_parent['aabb'].get_extent()}"
                )
                print(
                    f"Child   – center: {potential_child['center']}, "
                    f"extents: {potential_child['aabb'].get_extent()}"
                )

                print(get_translation(potential_parent, potential_child,
                                          aligned_axis, axis))
            
            c_o = check_overlap(axis=aligned_axis[axes[2]], potential_child=potential_child, potential_parent=potential_parent)
            aabbs = _aabbs_overlap_2d(pp_half_box, pc_half_box, axes=(aligned_axis[axes[0]], aligned_axis[axes[1]]))
            # two_d = half_extent_overlap_2d(potential_parent, potential_child, threshold=1, axes=(aligned_axis[axes[0]], aligned_axis[axes[1]]))
            one_d = half_extent_overlap_2d(potential_parent, potential_child, threshold=1, axes=(aligned_axis[axes[0]],))
            c_depth = same_depth(potential_parent, potential_child, depth_axis=aligned_axis[2], weight=0.2)
            if axis == 0 and potential_child['object_number'] == 12 and potential_parent['object_number'] == 13:
                print(_aabbs_overlap_2d(pp_half_box, pc_half_box, axes=(aligned_axis[axes[0]], aligned_axis[axes[1]])))
                print(check_overlap(axis=aligned_axis[axes[2]],
                                                                                                                      potential_child=potential_child,
                                                                                                                         potential_parent=potential_parent))
                print(
                    f"Parent  – center: {potential_parent['center']}, "
                    f"extents: {potential_parent['aabb'].get_extent()}"
                )
                print(
                    f"Child   – center: {potential_child['center']}, "
                    f"extents: {potential_child['aabb'].get_extent()}"
                )

                print(get_translation(potential_parent, potential_child,
                                          aligned_axis, axis))
                print(c_o)
                print(c_depth and one_d)
            if (potential_child['object_number'] == 4):
                # s.show()
                print(c_depth and one_d)
            if  c_depth and one_d and c_o:
                
                
                set_dimension_parent_default(potential_child, aligned_axis, axes[0], pcd_data, potential_parent=potential_parent)
                set_dimension_parent_default(potential_child, aligned_axis, axes[1], pcd_data, potential_parent=potential_parent)
                set_dimension_parent_default(potential_child, aligned_axis, axes[2], pcd_data)

                placement_func(s=s,
                                              parent_asset=potential_parent,
                                              child_asset=potential_child,
                                              child_width=potential_child[extent_labels[aligned_axis[0]]],
                                              child_height=potential_child[extent_labels[aligned_axis[1]]],
                                              child_depth=potential_child[extent_labels[aligned_axis[2]]])
                placed_assets.append(potential_child)
                unplaced_assets[label].remove(potential_child)
                placed = True
                break
    return placed

def try_to_place_medium(unplaced_assets, label, placed_assets,
                        pcd_data, s, axis, allow_overlap=False):
    # ----------------------------------------------------
    # 0.  Determine helper function and axis mapping
    # ----------------------------------------------------
    if axis == 1:
        func_name = "place_vertical_asset"
        axes = [0, 2, 1]      # x–z–y order
    elif axis == 0:
        func_name = "place_lateral_asset"
        axes = [1, 2, 0]      # y–z–x order
    else:
        raise ValueError("axis must be 0 (lateral) or 1 (vertical)")

    placement_func = globals()[func_name]

    # ----------------------------------------------------
    # 1.  Search for the parent/child pair with *minimum*
    #     |translation| along axes[2]
    # ----------------------------------------------------
    best = {
        "mag": float("inf"),
        "parent": None,
        "child": None,
        "aligned_axis": None,
        "translation": None,
    }

    for parent in placed_assets:
        aligned_axis = parent["relative_alignment"]
        pp_half_box = get_half_aabb_box(parent)

        # iterate over a *copy* so we don't mutate during search
        for child in list(unplaced_assets[label]):
            pc_half_box = get_half_aabb_box(child)

            # Skip if overlap is prohibited and ≥50 % overlap detected
            if not allow_overlap and overlaps_any_placed(child, placed_assets):
                continue

            # Must overlap on the two non-placement axes
            aabbs = _aabbs_overlap_2d(pp_half_box, pc_half_box, axes=(aligned_axis[axes[0]], aligned_axis[axes[1]]))
            two_d = half_extent_overlap_2d(parent, child, axes=(aligned_axis[axes[0]], aligned_axis[axes[1]]))
            if not two_d:
                continue

            # Your change ①: use axes[2] here
            translation = get_translation(parent, child,
                                          aligned_axis, axes[2])

            if abs(translation) < best["mag"]:
                best.update(dict(
                    mag=abs(translation),
                    parent=parent,
                    child=child,
                    aligned_axis=aligned_axis,
                    translation=translation,
                ))

    # ----------------------------------------------------
    # 2.  Bail early if nothing legal was found
    # ----------------------------------------------------
    if best["parent"] is None:
        return False

    parent = best["parent"]
    child = best["child"]
    aligned_axis = best["aligned_axis"]
    translation = best["translation"]

    # ----------------------------------------------------
    # 3.  Dimension propagation
    # ----------------------------------------------------
    set_dimension_parent_default(child, aligned_axis, axes[0], pcd_data,
                                 potential_parent=parent)
    set_dimension_parent_default(child, aligned_axis, axes[1], pcd_data,
                                 potential_parent=parent)
    set_dimension_parent_default(child, aligned_axis, axes[2], pcd_data)

    # ----------------------------------------------------
    # 4.  Build 4×4 transform
    #     Your change ②: index derived from aligned_axis[axes[2]]
    # ----------------------------------------------------
    indices = [0, 2, 1]        # map x/y/z → homogeneous-matrix col
    index = indices[aligned_axis[axes[2]]]
    transform = np.eye(4)
    transform[index, 3] = translation

    # ----------------------------------------------------
    # 5.  Call placement helper once
    # ----------------------------------------------------
    placement_func(
        s=s,
        parent_asset=parent,
        child_asset=child,
        child_width=child[extent_labels[aligned_axis[0]]],
        child_height=child[extent_labels[aligned_axis[1]]],
        child_depth=child[extent_labels[aligned_axis[2]]],
        transform=transform,
    )

    # ----------------------------------------------------
    # 6.  Book-keeping
    # ----------------------------------------------------
    placed_assets.append(child)
    unplaced_assets[label].remove(child)

    return True



# # def try_to_place_vertical_strict(unplaced_assets, label, placed_assets, pcd_data, s, allow_overlap=False):

# #     for potential_parent in placed_assets:
# #         aligned_axis = potential_parent['relative_alignment']
# #         pp_half_box = get_half_aabb_box(potential_parent)
# #         for potential_child in unplaced_assets[label]:
# #             pc_half_box = get_half_aabb_box(potential_child)
# #             if not allow_overlap and aabb_overlap_fraction_with_centers(potential_parent['center'], potential_parent['aabb'].get_extent(), potential_child['center'], potential_child['aabb'].get_extent()):
# #                 continue
# #             if _aabbs_overlap_2d(pp_half_box, pc_half_box, axes=(aligned_axis[0], aligned_axis[2])) and check_overlap(axis=aligned_axis[1],
# #                                                                                                                       potential_child=potential_child,
# #                                                                                                                          potential_parent=potential_parent):
# #                 print(potential_child.get(f'fixed_{extent_labels[0]}'))
# #                 print(f"potential p {potential_parent['object_number']}")
# #                 set_dimension_parent_default(potential_child, aligned_axis, 0, pcd_data, potential_parent=potential_parent)
# #                 set_dimension_parent_default(potential_child, aligned_axis, 2, pcd_data, potential_parent=potential_parent)
# #                 set_dimension_parent_default(potential_child, aligned_axis, 1, pcd_data)

# #                 place_top_or_bottom_asset(s=s,
# #                                               parent_asset=potential_parent,
# #                                               child_asset=potential_child,
# #                                               child_width=potential_child[extent_labels[aligned_axis[0]]],
# #                                               child_height=potential_child[extent_labels[aligned_axis[1]]],
# #                                               child_depth=potential_child[extent_labels[aligned_axis[2]]])
# #                 placed_assets.append(potential_child)
# #                 unplaced_assets[label].remove(potential_child)
# #                 break

# # def try_to_place_lateral_strict(unplaced_assets, label, placed_assets, pcd_data, s, allow_overlap=False):

#     for potential_parent in placed_assets:
#         aligned_axis = potential_parent['relative_alignment']
#         pp_half_box = get_half_aabb_box(potential_parent)
#         for potential_child in unplaced_assets[label]:
#             pc_half_box = get_half_aabb_box(potential_child)
#             if not allow_overlap and aabb_overlap_fraction_with_centers(potential_parent['center'], potential_parent['aabb'].get_extent(), potential_child['center'], potential_child['aabb'].get_extent()):
#                 continue
            
#             if _aabbs_overlap_2d(pp_half_box, pc_half_box, axes=(aligned_axis[1], aligned_axis[2])) and check_overlap(axis=aligned_axis[0],
#                                                                                                                       potential_child=potential_child,
#                                                                                                                       potential_parent=potential_parent):
#                 if (potential_child['object_number'] == 4):
#                     # s.show()
#                     continue
#                 if (potential_child['object_number'] == 5):
#                     # s.show()
#                     continue
#                 set_dimension_parent_default(potential_child, aligned_axis, 1, pcd_data, potential_parent=potential_parent)
#                 set_dimension_parent_default(potential_child, aligned_axis, 2, pcd_data, potential_parent=potential_parent)
#                 set_dimension_parent_default(potential_child, aligned_axis, 0, pcd_data)

#                 place_left_or_right_asset(s=s,
#                                               parent_asset=potential_parent,
#                                               child_asset=potential_child,
#                                               child_width=potential_child[extent_labels[aligned_axis[0]]],
#                                               child_height=potential_child[extent_labels[aligned_axis[1]]],
#                                               child_depth=potential_child[extent_labels[aligned_axis[2]]])
#                 placed_assets.append(potential_child)
#                 unplaced_assets[label].remove(potential_child)
#                 break
                

# # def try_to_place_vertical_medium(unplaced_assets, label, placed_assets, pcd_data, s, allow_overlap=False):
# #     placed = False
# #     for potential_parent in placed_assets:
# #         aligned_axis = potential_parent['relative_alignment']
# #         pp_half_box = get_half_aabb_box(potential_parent)
# #         for potential_child in unplaced_assets[label]:
# #             pc_half_box = get_half_aabb_box(potential_child)
# #             if not allow_overlap and aabb_overlap_fraction_with_centers(potential_parent['center'], potential_parent['aabb'].get_extent(), potential_child['center'], potential_child['aabb'].get_extent()):
# #                 continue
# #             if _aabbs_overlap_2d(pp_half_box, pc_half_box, axes=(aligned_axis[0], aligned_axis[2])):
# #                 set_dimension_parent_default(potential_child, aligned_axis, 0, pcd_data, potential_parent=potential_parent)
# #                 set_dimension_parent_default(potential_child, aligned_axis, 2, pcd_data, potential_parent=potential_parent)
# #                 set_dimension_parent_default(potential_child, aligned_axis, 1, pcd_data)

# #                 translation = get_translation(potential_parent, potential_child, aligned_axis, 1)
# #                 indices = [0, 2, 1]
# #                 index = indices[aligned_axis[1]]
# #                 transform = np.eye(4)
# #                 transform[index, 3] = translation

# #                 place_top_or_bottom_asset(s=s,
# #                                               parent_asset=potential_parent,
# #                                               child_asset=potential_child,
# #                                               child_width=potential_child[extent_labels[aligned_axis[0]]],
# #                                               child_height=potential_child[extent_labels[aligned_axis[1]]],
# #                                               child_depth=potential_child[extent_labels[aligned_axis[2]]],
# #                                               transform=transform)
                
# #                 placed_assets.append(potential_child)
# #                 unplaced_assets[label].remove(potential_child)
# #                 placed = True
# #                 break
# #         if placed:
# #             break

# # def try_to_place_lateral_medium(unplaced_assets, label, placed_assets, pcd_data, s, allow_overlap=False):
# #     placed = False
# #     for potential_parent in placed_assets:
# #         aligned_axis = potential_parent['relative_alignment']
# #         pp_half_box = get_half_aabb_box(potential_parent)
# #         for potential_child in unplaced_assets[label]:
# #             pc_half_box = get_half_aabb_box(potential_child)
# #             if not allow_overlap and aabb_overlap_fraction_with_centers(potential_parent['center'], potential_parent['aabb'].get_extent(), potential_child['center'], potential_child['aabb'].get_extent()):
# #                 continue
# #             print(_aabbs_overlap_2d(pp_half_box, pc_half_box, axes=(aligned_axis[1], aligned_axis[2])))
# #             if _aabbs_overlap_2d(pp_half_box, pc_half_box, axes=(aligned_axis[1], aligned_axis[2])):
# #                 print('here') 
# #                 print(potential_child['object_number'])
# #                 print('waht')
# #                 print(potential_parent['object_number'])
# #                 print('waht')
                
                
# #                 set_dimension_parent_default(potential_child, aligned_axis, 1, pcd_data, potential_parent=potential_parent)
# #                 set_dimension_parent_default(potential_child, aligned_axis, 2, pcd_data, potential_parent=potential_parent)
# #                 set_dimension_parent_default(potential_child, aligned_axis, 0, pcd_data)

# #                 translation = get_translation(potential_parent, potential_child, aligned_axis, 0)
# #                 print(translation)
# #                 indices = [0, 2, 1]
# #                 index = indices[aligned_axis[0]]
# #                 transform = np.eye(4)
# #                 transform[index, 3] = translation
# #                 print(transform)

# #                 place_left_or_right_asset(s=s,
# #                                               parent_asset=potential_parent,
# #                                               child_asset=potential_child,
# #                                               child_width=potential_child[extent_labels[aligned_axis[0]]],
# #                                               child_height=potential_child[extent_labels[aligned_axis[1]]],
# #                                               child_depth=potential_child[extent_labels[aligned_axis[2]]],
# #                                               transform=transform)
# #                 placed_assets.append(potential_child)
# #                 unplaced_assets[label].remove(potential_child)
# #                 placed = True
# #                 break
# #         if placed:
# #             break


def get_translation(potential_parent, potential_child, aligned_axis, target_axis):
    axis = aligned_axis[target_axis]
    return abs(potential_child['center'][axis] - potential_parent['center'][axis]) - (abs(potential_child['aabb'].get_extent()[axis] + potential_parent['aabb'].get_extent()[axis])) + abs(_effective_extent(potential_child, axis) + _effective_extent(potential_parent, axis)) / 2.0
    

def set_dimension_parent_default(potential_child, aligned_axis, target_axis, data, potential_parent=None):
    label = potential_child['label']
    if potential_child.get(f'fixed_{extent_labels[target_axis]}') is not None:
        if potential_child.get(f'fixed_{extent_labels[aligned_axis[target_axis]]}') is None:
            potential_child[extent_labels[aligned_axis[target_axis]]] = get_cluster_fixed_extents_one_asset(data, potential_child, label_keyword=label.split('_')[0], extent_index=aligned_axis[target_axis])
        else:
            potential_child[extent_labels[aligned_axis[target_axis]]] = potential_child.get(f'fixed_{extent_labels[aligned_axis[target_axis]]}')
    else:
        if potential_parent is not None:
            potential_child[extent_labels[aligned_axis[target_axis]]] = potential_parent[extent_labels[aligned_axis[target_axis]]]

def get_cluster_fixed_extents_one_asset(data,
                                        asset,
                              label_keyword: str,
                              extent_index: int):
    idx2name = {0: "width", 1: "height", 2: "depth"}
    name = idx2name[extent_index]

    key_cluster = f"{label_keyword}_cluster_id"
    matches = [d for d in data
               if label_keyword in d["label"] and key_cluster in d]
    if not matches:
        print(f"[set_cluster_fixed_extents] Nothing to do for '{label_keyword}'.")
        return
    clusters = defaultdict(list)
    for d in matches:
        clusters[d[key_cluster]].append(d)
    for cid, objs in clusters.items():
        means = {name: float(np.mean([o[name] for o in objs]))}
        for o in objs:
            if o['object_number'] == asset['object_number']:
                return means[name]
            
def aabb_overlap_fraction_with_centers(center1, extent1, center2, extent2):
    def get_bounds(center, extent):
        center = np.array(center)
        extent = np.array(extent)
        return center - extent, center + extent

    def get_overlap_volume(min1, max1, min2, max2):
        overlap_min = np.maximum(min1, min2)
        overlap_max = np.minimum(max1, max2)
        overlap_dims = np.maximum(overlap_max - overlap_min, 0)
        return np.prod(overlap_dims)

    min1, max1 = get_bounds(center1, extent1)
    min2, max2 = get_bounds(center2, extent2)

    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)
    overlap_vol = get_overlap_volume(min1, max1, min2, max2)

    return (overlap_vol >= 0.7 * vol1) or (overlap_vol >= 0.7 * vol2)

def _effective_extent(asset, axis):
    label = extent_labels[axis]                # e.g. "width"
    fixed_key = f"fixed_{label}"               # e.g. "fixed_width"

    if asset.get(fixed_key) is not None:       # explicit fixed dimension
        return asset[fixed_key]
    # Fallback: geometry extent from AABB
    return asset["aabb"].get_extent()[axis]

def overlaps_any_placed(child: dict, placed_assets: list) -> bool:
    """
    Return True if `child` overlaps (≥50 % of either volume, using
    aabb_overlap_fraction_with_centers) with **any** asset in `placed_assets`.
    """
    c_center  = child["center"]
    c_extent  = child["aabb"].get_extent()

    for asset in placed_assets:
        if aabb_overlap_fraction_with_centers(asset["center"],
                                              asset["aabb"].get_extent(),
                                              c_center, c_extent):
            return True
    return False

# check if one half exentents fill another full extents