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
from src.utils.asset_utils import get_asset

"""
    Function: get_depth_axis_from_pcd_extents
    -----------------------------------------
    extent: 3D vector representing the dimensions of an object (dx, dy, dz)
    ratio_threshold: float, threshold to determine if the smallest dimension is significantly smaller than the second smallest

    Returns the index of the depth axis (0 for dx, 1 for dy, 2 for dz) assuming that this depth dimension is significatly skinnier than its other dimensions,
    whening scanning closed assets. If this depth axis cannot be determined (i.e. the smallest dimension is not significantly smaller than the second smallest), returns None.
"""
def get_depth_axis_from_pcd_extents(extent, ratio_threshold: float = 0.2):
    ext = np.asarray(extent, dtype=float)
    if ext.shape != (3,):
        raise ValueError("extent must be length-3 (dx, dy, dz)")

    smallest_idx = int(np.argmin(ext))
    second_smallest = np.partition(ext, 1)[1]   # next-smallest value
    return smallest_idx if ext[smallest_idx] < ratio_threshold * second_smallest else None


"""
    Function: place_vertical_asset
    --------------------------------
    s: SceneSynthesizer instance
    parent_asset: dictionary containing the parent asset information
    child_asset: dictionary containing the child asset information
    child_width: width of the child asset
    child_height: height of the child asset
    child_depth: depth of the child asset
    transform: 4x4 transformation matrix to apply to the child asset
    depth_clamp: string indicating the depth clamping direction ("back" or "front")
    clamp: integer indicating the clamping direction (-1 for left, 1 for right)

    Places the child asset vertically relative to the parent asset, either on top or bottom. The placement is determined based on 
    the center of the child asset relative to the parent asset's center.
"""
def place_vertical_asset(s, parent_asset, child_asset, child_width, child_height, child_depth, transform=np.eye(4), depth_clamp='back', clamp=-1, name_add=""):
    # we do not need to worry about axis alignment for the vertical axis
    lateral_anchor = 'left' if clamp == -1 else 'right'
    # place on top of the parent asset
    if (child_asset['center'][1] > parent_asset['center'][1]):
        s.add_object(get_asset(child_asset['label'], child_width, child_height, child_depth), 
                    f"{child_asset['asset_name']}{name_add}",
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=(lateral_anchor, depth_clamp, 'top'), 
                    connect_obj_anchor=(lateral_anchor, depth_clamp, 'bottom'),
                    transform=transform)
        
    # place on bottom of the parent asset
    else:
        transform[2,3] = transform[2,3] * -1
        s.add_object(get_asset(child_asset['label'], child_width, child_height, child_depth), 
                    f"{child_asset['asset_name']}{name_add}",
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=(lateral_anchor, depth_clamp, 'bottom'), 
                    connect_obj_anchor=(lateral_anchor, depth_clamp, 'top'),
                    transform=transform)

"""
    Function: place_lateral_asset
    -----------------------------
    s: SceneSynthesizer instance
    parent_asset: dictionary containing the parent asset information
    child_asset: dictionary containing the child asset information
    child_width: width of the child asset
    child_height: height of the child asset
    child_depth: depth of the child asset
    transform: 4x4 transformation matrix to apply to the child asset
    depth_clamp: string indicating the depth clamping direction ("back" or "front")
    clamp: integer indicating the clamping direction (-1 for bottom, 1 for top)

    This function places the child asset laterally relative to the parent asset, either on the left or right side.
    The placement is determined based on the center of the child asset relative to the parent asset's center.
    The function makes sure to use axis alignment based on the parent asset's relative alignment.
"""
def place_lateral_asset(s, parent_asset, child_asset, child_width, child_height, child_depth, transform=np.eye(4), depth_clamp='back', clamp=1, default_countertop_thickness = 0.04):
    aa = parent_asset['relative_alignment']
    vertical_anchor = 'top' if clamp == 1 else 'bottom'
    parent_vertical_axis = parent_asset['center'][1] + clamp * parent_asset['aabb'].get_extent()[1] / 2
    child_vertical_axis = child_asset['center'][1] + clamp * child_asset['aabb'].get_extent()[1] / 2
    vertical_delta = child_vertical_axis - parent_vertical_axis
    if abs(vertical_delta) > (parent_asset['aabb'].get_extent()[1] + child_asset['aabb'].get_extent()[1]) / 2 * 0.05:
        extra_translation = vertical_delta
    else:
        extra_translation = 0


    # unique handling for sink assets
    if child_asset['label'] == 'sink' or parent_asset['label'] == 'sink':
        vertical_anchor = 'top'
        if child_asset['label'] == 'sink':
            tranlation = default_countertop_thickness
        else:
            tranlation = -1 * default_countertop_thickness
        transform = np.eye(4)
        transform[2, 3] = tranlation
    transform[2, 3] += extra_translation
    # print(extra_translation, vertical_anchor, clamp)
    # print(parent_asset['center'][1])

    # place on left of the parent asset
    if (child_asset['center'][aa[0]] * parent_asset['weight'] < parent_asset['center'][aa[0]] * parent_asset['weight']):
        transform[0,3] = transform[0,3] * -1
        s.add_object(get_asset(child_asset['label'], child_width, child_height, child_depth),
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=('left', depth_clamp, vertical_anchor), 
                    connect_obj_anchor=('right', depth_clamp, vertical_anchor),
                    transform=transform)
        
    # place on right of the parent asset
    else:
        s.add_object(get_asset(child_asset['label'], child_width, child_height, child_depth), 
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=('right', depth_clamp, vertical_anchor), 
                    connect_obj_anchor=('left', depth_clamp, vertical_anchor),
                    transform=transform)

"""
    Function: check_BB_overlap
    -----------------------
    Checks if two objects overlap along a specified axis, with a buffer based on the larger object's extent.
"""
def check_BB_overlap(axis, potential_parent, potential_child, threshold = 0.1):

    child_center=potential_child['center']
    parent_center=potential_parent['center']
    child_extents=potential_child['aabb'].get_extent()
    parent_extents=potential_parent['aabb'].get_extent()

    buffer = max(parent_extents[axis], child_extents[axis]) * threshold

    child_min = child_center[axis] - child_extents[axis] / 2.0
    child_max = child_center[axis] + child_extents[axis] / 2.0
    parent_min = parent_center[axis] - parent_extents[axis] / 2.0
    parent_max = parent_center[axis] + parent_extents[axis] / 2.0

    return not (child_max < parent_min - buffer or child_min > parent_max + buffer)

"""
    Function: check_extent_overlap
    ------------------------------
    Checks if two objects overlap in extent along specified axes, with a threshold for overlap.
"""
def check_extent_overlap(parent: dict,
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

            # if shrink['object_number'] == 12 and fixed['label'] == 'counter':
            #     # Visualize shrink, fixed, and expanded fixed AABBs
            #     shrink_aabb = o3d.geometry.AxisAlignedBoundingBox(
            #         min_bound=np.array(shrink_center) - shrink_half,
            #         max_bound=np.array(shrink_center) + shrink_half
            #     )
                
            #     shrink_aabb.color = (1, 0, 0)  # Red for shrink

            #     fixed_expanded_half = fixed_half + fixed_half * (threshold - 1)
            #     fixed_aabb = o3d.geometry.AxisAlignedBoundingBox(
            #         min_bound=np.array(fixed_center) - fixed_expanded_half,
            #         max_bound=np.array(fixed_center) + fixed_expanded_half
            #     )
            #     fixed_aabb.color = (0, 1, 0)  # Green for fixed

            #     # Also visualize the point clouds if you want
            #     o3d.visualization.draw_geometries([
            #         shrink["pcd"],
            #         fixed["pcd"],
            #         shrink_aabb,
            #         fixed_aabb
            #     ])

            if shrink_min < fixed_min or shrink_max > fixed_max:
                return False
        return True

    return is_half_inside(child, parent) or is_half_inside(parent, child)

def get_depth_delta(parent: dict, child: dict) -> float:
    aa = parent['relative_alignment']
    weight = parent['weight']
    if aa == [2,1,0]:
        scale = -1
    else:
        scale = 1
    parent_front = parent['center'][aa[2]] * weight - parent['aabb'].get_extent()[aa[2]] / 2.0
    child_front = child['center'][aa[2]] * weight - child['aabb'].get_extent()[aa[2]] / 2.0
    return (child_front - parent_front) * scale

## assume upper is smaller depth than lower
def get_aligned_depth(parent, child, depth_axis = 2):
    extent_labels = ['width', 'height', 'depth']
    aa = parent['relative_alignment']
    p_s = parent['weight']
    p_depth = parent["aabb"].get_extent()[aa[2]]
    c_depth = child["aabb"].get_extent()[aa[2]]
    p_center = parent["center"][aa[2]]
    c_center = child["center"][aa[2]]

    p_fixed_depth = parent[extent_labels[depth_axis]]

    if (parent['relative_alignment'] == [0,1,2]):
        p_s = p_s * -1
    p_min = p_center + p_s * (p_depth / 2)
    c_min = c_center + p_s * (c_depth / 2)

    # if child['object_number'] == 1:
    #     print()
    #     print(parent['object_number'])
    #     print(p_s)
    #     print(depth_axis)
    #     print(p_depth)
    #     print(p_center, c_center)
    #     print(p_min, c_min)
    #     print()
    # if (child['object_number'] == 29 and parent['object_number'] == 25):
    #     print('----')
    #     print(p_s)
    #     print(depth_axis)
    #     print(p_center, c_center)
    #     print(p_min, c_min)
    #     print(p_depth)
    #     print(p_fixed_depth - abs(p_min - c_min))
    #     print('----')

    if (p_s == -1 and (p_min > c_min)) or (p_s == 1 and c_min > p_min):
        return p_fixed_depth - abs(p_min - c_min)
    else:
        return None
    
# input child depth axis
def get_not_aligned_depth(parent, child, depth_axis):
    aa = child['relative_alignment']
    c_s = child['weight']
    p_depth = parent["aabb"].get_extent()[aa[2]]
    c_depth = child["aabb"].get_extent()[aa[2]]
    p_center = parent["center"][aa[2]]
    c_center = child["center"][aa[2]]

    p_min = p_center + c_s * (p_depth / 2)
    c_min = c_center + c_s * (c_depth / 2)

    if child['object_number'] == 34:
        print(parent['object_number'])
        print(c_s)
        print(depth_axis)
        print(p_center, c_center)
        print(p_min, c_min)

    # print((c_s == 1 and c_min < p_min))
    if (c_s == -1 and (p_min < c_min)) or (c_s == 1 and c_min < p_min):
        return abs(p_min - c_min)
    else:
        return None
    
def get_depth_from_counter(counters, asset):
    aa = asset['relative_alignment']
    counter_asset = None
    for counter in counters:
        if check_extent_overlap(counter, asset, threshold=1, axes=(aa[0],)):
            counter_asset = counter
    counter_pcd = counter_asset['pcd']
    axis = aa[0]
    min_val = asset['center'][axis] - asset['aabb'].get_extent()[axis] * 0.4
    max_val = asset['center'][axis] + asset['aabb'].get_extent()[axis] * 0.4
    cropped_counter_pcd = crop_point_cloud_along_axis(counter_pcd, axis, min_val, max_val)
    cropped_counter_pcd = crop_point_cloud_along_axis(cropped_counter_pcd, aa[2], asset['center'][aa[2]] - 1, asset['center'][aa[2]] + 1) # hardcoded bound of 2m (can't have depth be more than two meters)
    # print(cropped_counter_pcd.get_axis_aligned_bounding_box().get_extent()[aa[2]])
    # o3d.visualization.draw_geometries([cropped_counter_pcd + asset['pcd']])
    
    return cropped_counter_pcd.get_axis_aligned_bounding_box().get_extent()[aa[2]]
    
def crop_point_cloud_along_axis(pcd, axis, min_val, max_val):
    """
    Crop a point cloud along a single axis (x, y, or z).

    Parameters:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        axis (str): Axis to crop along: 'x', 'y', or 'z'.
        min_val (float): Minimum value along the axis.
        max_val (float): Maximum value along the axis.

    Returns:
        o3d.geometry.PointCloud: Cropped point cloud.
    """
    points = np.asarray(pcd.points)
    mask = (points[:, axis] >= min_val) & (points[:, axis] <= max_val)
    filtered_points = points[mask]

    # Create new point cloud
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # Copy colors/normals if they exist
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        cropped_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        cropped_pcd.normals = o3d.utility.Vector3dVector(normals[mask])

    return cropped_pcd

def place_counter_top(s, counter, pcd_data, default_counter_thickness=0.04):
    considered_assets = [[], []]
    for asset in pcd_data:
        aa = asset['relative_alignment']
        if asset['object_number'] == 12:
            print("help", aa)
        if 'upper' not in asset['label'] and asset['label'] != 'counter' and check_extent_overlap(counter, asset, threshold=1, axes=(aa[0],)):
            # print(asset['object_number'])
            if not check_in_between_assets(pcd_data, counter, asset):
                considered_assets[int(aa[0] / 2)].append(asset)
    depth = 0
    for i in range(2):
        if len(considered_assets[i]) == 0:
            continue
        combined_pcd = o3d.geometry.PointCloud()
        width = 0
        for asset in considered_assets[i]:
            combined_pcd += asset['pcd']
            depth = max(depth, asset['depth'])
            width += asset['width']
        # o3d.visualization.draw_geometries([combined_pcd + counter['pcd'], counter['aabb']])
        width_axis = i * 2
        points = np.asarray(combined_pcd.points)
        min_val = np.min(points[:, width_axis])
        max_val = np.max(points[:, width_axis])
        print(width_axis, width)

        best_asset = None
        clamp = 0
        best_score = float('inf')
        height_axis = 1

        for asset in considered_assets[i]:
            asset_min = np.min(np.asarray(asset['pcd'].points)[:, width_axis])
            asset_max = np.max(np.asarray(asset['pcd'].points)[:, width_axis])
            
            # Distance to either edge
            dist_to_min = abs(asset_min - min_val)
            dist_to_max = abs(asset_max - max_val)
            edge_dist = min(dist_to_min, dist_to_max)

            # Height difference
            counter_height = counter['pcd'].get_center()[height_axis]
            asset_height = asset['pcd'].get_center()[height_axis]
            height_diff = abs(counter_height - asset_height)

            score = 10 * edge_dist + height_diff

            if score < best_score:
                best_score = score
                best_asset = asset
                if edge_dist == dist_to_min:
                    clamp = -1 * asset['weight']
                else:
                    clamp = 1 * asset['weight']
        print(f"_{width_axis}")
        place_vertical_asset(s, best_asset, counter, width, default_counter_thickness, depth, clamp=clamp, name_add=f"_{width_axis}")
        print('made it here', best_asset['object_number'])

def check_in_between_assets(pcd_data, upper, lower):
    for asset in pcd_data:
        if asset['object_number'] != upper['object_number'] and asset['object_number'] != upper['object_number']:
            if asset['center'][1] < upper['center'][1] and asset['center'][1] > lower['center'][1] and check_extent_overlap(asset, lower, threshold=1, axes=(lower['relative_alignment'][0],)) and lower['relative_alignment'] == asset['relative_alignment']:
                if lower['object_number'] == 20:
                    print(asset['object_number'])
                return True
    return False

def is_point_in_aabb(point, aabb):
    min_bound = aabb.get_min_bound()  # np.array([x_min, y_min, z_min])
    max_bound = aabb.get_max_bound()  # np.array([x_max, y_max, z_max])
    return np.all(point >= min_bound) and np.all(point <= max_bound)