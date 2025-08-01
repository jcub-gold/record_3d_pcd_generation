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


GREEN = "\033[32m"
RESET = "\033[0m"

possible_labels = ['drawer', 'lower_left_cabinet', 'lower_right_cabinet', 'upper_left_cabinet', 'upper_right_cabinet', 'box', 'sink']

extent_labels = ['width', 'height', 'depth']

default_countertop_thickness = 0.04
default_drawer_depth = 0.024

"""
second_floor_test
label_clusters = [3, 1, 1, 1, 1]
extent_indices = [(0, 1, 2), (1,), (1,), (0, 1), (0,)]
"""
# # basement_test
# label_clusters = [5, 1, 1, 1]
# extent_indices = [(0, 1, 2), (0, 1, 2), (1,), (1,), ]
# # second_floor_test
# label_clusters = [1, 1, 3, 1, 1]
# extent_indices = [(1,), (0,), (0, 1, 2), (1,), (0, 1)]
# base_cabinet_test
label_clusters = [1, 1]
extent_indices = [(0, 1, 2), (1,)]

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
    def iter_pcd_files(folder):
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(".ply"):
                yield fn, o3d.io.read_point_cloud(os.path.join(folder, fn))

    input_path   = os.path.dirname(pcds_path)
    aa_pcds_path = os.path.join(input_path, "aa_pcds")
    cache_exists = (
        os.path.isdir(aa_pcds_path)
        and any(f.lower().endswith(".ply") for f in os.listdir(aa_pcds_path))
    )

    if cache_exists:
        print(f"⚡  Using cached aligned PCDs in: {aa_pcds_path}")
        files = list(iter_pcd_files(aa_pcds_path))
        pcds = [pcd for _, pcd in files]
        combined_pcd = o3d.geometry.PointCloud()
        for pcd in pcds:
            combined_pcd += pcd
        center = combined_pcd.get_center()

    else:
        print(f"🌀  No cache – aligning PCDs from: {pcds_path}")
        files = list(iter_pcd_files(pcds_path))

        raw_pcds = [pcd for _, pcd in files]
        R, center = align_pcd_scene_via_object_aabb_minimization(raw_pcds)
        # extra_R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.pi/2, 0])
        # R = extra_R @ R

        os.makedirs(aa_pcds_path, exist_ok=True)
        for fname, pcd in files:
            pcd.rotate(R, center=center)
            o3d.io.write_point_cloud(os.path.join(aa_pcds_path, fname), pcd)
        print(f"✅  Wrote aligned copies to: {aa_pcds_path}")
        files = list(iter_pcd_files(aa_pcds_path))

    if load_cached_labels:
        with open(os.path.join(input_path, "cached_labels.json"), "r") as f:
            cache = json.load(f)
    else:
        string = "Either type full label or a number for the label list below:\n"
        for i in range(len(possible_labels)):
            string += f"{i}: {possible_labels[i]}:\n"
        print(string)
        cache = None

    pcd_data = []
    labels = []
    asset_info = {}
    for fname, pcd in files:
        match = re.search(r'object_(\d+)', fname)
        assert match, f"Filename {fname} does not match expected pattern."
        obj_num = int(match.group(1))

        if cache is not None:
            label = cache[f"object_{obj_num}"]
        else:
            label_in = input(f"Enter label for {fname}: ")
            try:
                idx = int(label_in)
                if 0 <= idx < len(possible_labels):
                    label = possible_labels[idx]
                else:
                    raise ValueError
            except ValueError:
                if label_in in possible_labels:
                    label = label_in
                else:
                    raise ValueError("Invalid label or index")
            if save_labels is not None:
                save_labels[f"object_{obj_num}"] = label
        if label not in labels:
            labels.append(label)

        aabb = pcd.get_axis_aligned_bounding_box()
        width, height, depth = aabb.get_extent()

        # # second_floor_test
        # # not neccesary assuming closed scan and need to set relative axes based on dimensions
        # if 'cabinet' in label:
        #     if 'upper' in label:
        #         print(f"Old object {obj_num} width: {width}")
        #     width = np.sqrt(width**2 + depth**2)
        #     depth = width
        #     if 'upper' in label:
        #         print(f"New object {obj_num} width: {width}")

        # if 'box' in label:
        #     print(aabb)

        dict_data = {
            "pcd_path": os.path.join(aa_pcds_path, fname),
            "pcd": pcd,
            "aabb": aabb,
            "center": pcd.get_center(),
            "label": label,
            "object_number": obj_num,
            "width": width,
            "height": height,
            "depth": depth,
            "relative_alignment": [0, 1, 2],
            "weight": 1
        }
        if obj_num == 18:
            prnt = True
        else:
            prnt=False

        # # second_floor_test
        # if 'upper' in label:
        #     dict_data["relative_alignment"] = [2, 1, 0]

        # basement_test
        thin = thin_axis(aabb.get_extent(), ratio_threshold=0.15, p=prnt)
        if 'cabinet' in label:
            if thin == 0:
                dict_data["relative_alignment"] = [2, 1, 0]
                # print(f'90 degree rotation {obj_num}')
            else:
                dict_data["relative_alignment"] = [0, 1, 2]
                if thin == None:
                    # print(f'45 degree degree rotation {obj_num}')
                    transform = np.eye(4)
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                            [0, 0, -1 * np.pi / 4])
                    transform[:3, :3] = R
                    dict_data['rotation'] = transform
                # else:
                #     print(f'No rotation {obj_num}')

        # if obj_num == 35 and obj_num == 36:
        #     dict_data["relative_alignment"] = [2, 1, 0]

        func_name = f"get_{label}_asset"
        dict_data["asset_func"] = globals()[func_name]
        dict_data["asset_name"] = (
            f"{label}_{width:.3f}_{height:.3f}_{depth:.3f}_object_{obj_num}"
        )
        asset_info[f"object_{obj_num}"] = dict_data["asset_name"]

        pcd_data.append(dict_data)

    if save_labels is not None:
        json.dump(save_labels, open(os.path.join(input_path, "cached_labels.json"), "w"), indent=4)

    asset_info_path = os.path.join(input_path, "cached_asset_info.json")
    with open(asset_info_path, "w") as f:
        json.dump(asset_info, f, indent=4)
    print(f"💾 Wrote asset info for {len(asset_info)} objects to: {asset_info_path}")

    label_keywords = []
    for label in labels:
        if label.split("_")[0] not in label_keywords:
            label_keywords.append(label.split("_")[0])
    
    print(label_keywords, label_clusters, extent_indices)
    for kw, clusters, idx in zip(label_keywords, label_clusters, extent_indices):
        assign_extent_clusters(pcd_data, kw, clusters, idx)
    for kw, idx in zip(label_keywords, extent_indices):
        set_cluster_fixed_extents(pcd_data, kw, idx)
    return pcd_data, center, labels, label_keywords


# if allow_rotation = false skip asset,
# otherwise place rotated, similar to other way, make sure to commit after this step

# instead of candidates just including potential children, loop through parents too and find best pair by closet overlap extent
# may have to use this closest overlap condition to check vertical and lateral anchors

# branch test

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

    # for deg in tqdm(np.arange(-45, 45.01, 0.5), desc="Rotating degrees"):
    #     rad = np.deg2rad(deg)
    #     total_volume = 0

    #     pcd_copy = copy.deepcopy(combined_pcd)
    #     R = pcd_copy.get_rotation_matrix_from_axis_angle([0, rad, 0])
    #     pcd_copy.rotate(R, center=pcd_copy.get_center())

    #     aabb = pcd_copy.get_axis_aligned_bounding_box()
    #     extent = aabb.get_extent()
    #     volume = extent[0] * extent[1] * extent[2]
    #     total_volume += volume

    #     if total_volume < min_total_volume:
    #         min_total_volume = total_volume
    #         best_angle = rad

    best_R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, best_angle, 0])
    combined_pcd_center = combined_pcd.get_center()

    return best_R, combined_pcd_center


def thin_axis(extent, ratio_threshold: float = 0.2, p=False):
    ext = np.asarray(extent, dtype=float)
    if ext.shape != (3,):
        raise ValueError("extent must be length-3 (dx, dy, dz)")

    smallest_idx = int(np.argmin(ext))
    second_smallest = np.partition(ext, 1)[1]   # next-smallest value
    # if p:
    #     print(ext[smallest_idx], second_smallest)

    return smallest_idx if ext[smallest_idx] < ratio_threshold * second_smallest else None


def pcd_to_urdf_simple_geometries(pcd_data, combined_center, labels, output_path=None):
    unplaced_assets = {}
    for asset in pcd_data:
        if unplaced_assets.get(asset['label']) is None:
            unplaced_assets[asset['label']] = []
        unplaced_assets[asset['label']].append(asset)

    s = synth.Scene()
    placed_assets = []

    # place the first asset, prioritize a drawer asset for depth calculation
    parent_asset = unplaced_assets['drawer'].pop()
    set_dimension_parent_default(parent_asset, aligned_axis=parent_asset['relative_alignment'], target_axis=0, data=pcd_data)
    set_dimension_parent_default(parent_asset, aligned_axis=parent_asset['relative_alignment'], target_axis=1, data=pcd_data)
    set_dimension_parent_default(parent_asset, aligned_axis=parent_asset['relative_alignment'], target_axis=2, data=pcd_data)
    width, height, depth = parent_asset['width'], parent_asset['height'], parent_asset['depth']
    s.add_object(parent_asset["asset_func"](width, height, depth), parent_asset['asset_name'])
    placed_assets.append(parent_asset)

    upper_labels = []
    lower_labels = []
    for label in labels:
        if 'upper' not in label:
            lower_labels.append(label)
        else:
            upper_labels.append(label)

    while get_unplaced_assets_length(unplaced_assets, labels) > 0:
        expand_bottom = True
        while expand_bottom:
            expand_direct_neighbors(lower_labels, unplaced_assets, placed_assets, pcd_data, s, combined_center)
            expand_bottom = False
            for label in lower_labels:
                if try_to_place_strict(unplaced_assets, label, placed_assets, pcd_data, s, 0, combined_center, allow_overlap=True):
                    expand_bottom = True
                    break
        
        expand_top = True
        while expand_top:
            expand_direct_neighbors(upper_labels, unplaced_assets, placed_assets, pcd_data, s, combined_center)
            place_aligned_top = False
            for label in upper_labels:
                if try_to_place_upper_aligned(unplaced_assets, label, placed_assets, pcd_data, s):
                    place_aligned_top = True
                    break
            if place_aligned_top:
                continue
            place_not_aligned_top = False
            for label in upper_labels:
                if try_to_place_upper_not_aligned(unplaced_assets, label, placed_assets, pcd_data, s, combined_center):
                    place_not_aligned_top = True
                    break
            expand_top = place_aligned_top or place_not_aligned_top

        place_rotated = True
        while place_rotated:
            place_rotated = False
            for label in labels:
                # print('trying to place rotated!!!!!!!!!')
                while(try_to_place_strict(unplaced_assets, label, placed_assets, pcd_data, s, 1, combined_center, allow_rotation=True)):
                    place_rotated = True
                    continue
                while(try_to_place_strict(unplaced_assets, label, placed_assets, pcd_data, s, 0, combined_center, allow_rotation=True)):
                    place_rotated = True
                    continue

    s.export(output_path)
    print(f"{GREEN} Successfully generated URDF at {output_path}!{RESET}")
    s.show()
    return


def expand_direct_neighbors(labels, unplaced_assets, placed_assets, pcd_data, s, combined_center):
    place_direct_neighbor = True
    while place_direct_neighbor:
        place_direct_neighbor = False
        for label in labels:
            while(try_to_place_strict(unplaced_assets, label, placed_assets, pcd_data, s, 1, combined_center)):
                place_direct_neighbor = True
                continue
            while(try_to_place_strict(unplaced_assets, label, placed_assets, pcd_data, s, 0, combined_center)):
                place_direct_neighbor = True
                continue

def get_unplaced_assets_length(unplaced_assets, labels):
    length = 0
    for label in labels:
        length += len(unplaced_assets[label])
    return length



def place_vertical_asset(s, parent_asset, child_asset, child_width, child_height, child_depth, transform=np.eye(4), depth_clamp='back', clamp=-1):
    lateral_anchor = 'left' if clamp == -1 else 'right'
    # lateral_anchor = 'left'
    # place on top of the parent asset
    if (child_asset['center'][1] > parent_asset['center'][1]):
        s.add_object(child_asset["asset_func"](child_width, child_height, child_depth), 
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=(lateral_anchor, depth_clamp, 'top'), 
                    connect_obj_anchor=(lateral_anchor, depth_clamp, 'bottom'),
                    transform=transform)
        
    # place on bottom of the parent asset
    else:
        # print('bottom')
        transform[2,3] = transform[2,3] * -1
        s.add_object(child_asset["asset_func"](child_width, child_height, child_depth), 
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=(lateral_anchor, depth_clamp, 'bottom'), 
                    connect_obj_anchor=(lateral_anchor, depth_clamp, 'top'),
                    transform=transform)
        
def place_lateral_asset(s, parent_asset, child_asset, child_width, child_height, child_depth, transform=np.eye(4), depth_clamp='back', clamp=1):
    aa = parent_asset['relative_alignment']
    vertical_anchor = 'top' if clamp == 1 else 'bottom'
    if child_asset['label'] == 'sink' or parent_asset['label'] == 'sink':
        vertical_anchor = 'top'
        if child_asset['label'] == 'sink':
            tranlation = default_countertop_thickness
        else:
            tranlation = -1 * default_countertop_thickness
        transform = np.eye(4)
        transform[2, 3] = tranlation
        # child_height += default_countertop_thickness


    # place on left of the parent asset
    if (child_asset['center'][aa[0]] * parent_asset['weight'] < parent_asset['center'][aa[0]] * parent_asset['weight']):
        transform[0,3] = transform[0,3] * -1
        s.add_object(child_asset["asset_func"](child_width, child_height, child_depth), 
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=('left', depth_clamp, vertical_anchor), 
                    connect_obj_anchor=('right', depth_clamp, vertical_anchor),
                    transform=transform)
        
    # place on right of the parent asset
    else:
        if parent_asset['object_number'] == 5 and child_asset['object_number'] == 4:
            print("whahtttt")
            print()
        s.add_object(child_asset["asset_func"](child_width, child_height, child_depth), 
                    child_asset['asset_name'],
                    connect_parent_id=parent_asset['asset_name'],
                    connect_parent_anchor=('right', depth_clamp, vertical_anchor), 
                    connect_obj_anchor=('left', depth_clamp, vertical_anchor),
                    transform=transform)

# axis: axis along which to check overlap (0 for x, 1 for y, 2 for z)
def check_overlap(axis, potential_parent, potential_child, threshold = 0.1):

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
    if parent['relative_alignment'] == [2, 1, 0] and parent['weight'] == 1:
        p_min = p_center + (p_depth / 2)
        c_min = c_center + (c_depth / 2)
    else:
        p_min = p_center - (p_depth / 2)
        c_min = c_center - (c_depth / 2)
    # if parent['object_number'] == 4 and child['object_number'] == 36:
    #     print(p_depth, depth_axis, p_min, c_min)
    #     print('look here')
    if abs(p_min - c_min) <= max_depth_threshold:
        return True
    return False


def compare_axes(fixed, shrink, axes):
    for ax in axes:
        if shrink[ax][0] / 2 < fixed[ax][0] or shrink[ax][1] / 2 > fixed[ax][1]:
            return False
    return True






### asset generation functions
def get_sink_asset(width, height, depth):
    adjusted_height = height+default_countertop_thickness
    return pa.SinkCabinetAsset(width=width,
                               depth=depth,
                               height=adjusted_height,
                               sink_height=adjusted_height,
                               countertop_thickness=default_countertop_thickness,
                               sink_width=width/2,
                               sink_depth=depth/2,
                               include_bottom_compartment=False)

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
    return BoxAsset(extents=[width, depth, height])



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
        # if label_keyword =='drawer':
        #     print("-------")     
        for o in objs:
            # if 'drawer' in o['label']:
            #     print(o['object_number'])
            for name in names:
                o[f"fixed_{name}"] = means[name]
                    # print(name, means[name])


def _edge_alignment_score(parent, child, axis_idx, *, eps=1e-6):
    """Return  √(1/closest_edge_difference)  along axis_idx."""
    pc, pe = parent["center"][axis_idx], parent["aabb"].get_extent()[axis_idx]
    cc, ce = child ["center"][axis_idx], child ["aabb"].get_extent()[axis_idx]

    p_lo, p_hi = pc - pe/2, pc + pe/2
    c_lo, c_hi = cc - ce/2, cc + ce/2
    closest = min(abs(p_lo - c_lo), abs(p_hi - c_hi))
    if closest == abs(p_lo - c_lo):
        clamp = -1
    else:
        clamp = 1 
    return np.sqrt(1.0 / (closest + eps)), clamp

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

## upper aligneed + anchoring

def try_to_place_strict(unplaced_assets, label, placed_assets, pcd_data, s, axis, center, allow_overlap=False, allow_rotation=False):
    # allow_overlap = False
    if axis == 1:
        func_name, axes = "place_vertical_asset", [0, 2, 1]
    else:
        func_name, axes = "place_lateral_asset", [1, 2, 0]
    placement_func = globals()[func_name]

    best = None
    for potential_parent in placed_assets:
        aligned_axis = potential_parent['relative_alignment']
        if 'upper' in label and 'upper' not in potential_parent['label']:
            continue
        for potential_child in list(unplaced_assets[label]):
            # if (potential_child['object_number'] == 36 and potential_parent['object_number'] == 4 and axis == 1):
            #         print('----')
            #         print(potential_parent['center'], potential_parent['aabb'].get_extent(),
            #                                                         potential_child['center'], potential_child['aabb'].get_extent())
            #         print('----')
            # if allow_rotation and potential_child['object_number'] == 18:
                # print('well well well')
            # if allow_rotation:
            #     print(f'{potential_child['object_number']}well well well')
            if allow_rotation == False and potential_child.get("rotation") is not None:
                # print(f'skipping {potential_child['object_number']}')
                continue
            if not allow_overlap and overlaps_any_placed(potential_child, placed_assets):
                # print(potential_child['object_number'])
                # if (potential_child['object_number'] == 36 and potential_parent['object_number'] == 4 and axis == 1):
                #     print('----')
                #     print(potential_parent['center'], potential_parent['aabb'].get_extent(),
                #                                                     potential_child['center'], potential_child['aabb'].get_extent())
                #     print('----')
                potential_child['relative_alignment'] = [2, 1, 0]
                continue
            # if allow_overlap and overlaps_any_placed(potential_child, placed_assets):
            if allow_overlap and aabb_overlap_fraction_with_centers(potential_parent['center'], potential_parent['aabb'].get_extent(),
                                                                    potential_child['center'], potential_child['aabb'].get_extent()):
                # print(potential_child['object_number'])
                if place_overlapped_child(s, potential_parent, potential_child, center, pcd_data):
                    placed_assets.append(potential_child)
                    unplaced_assets[potential_child['label']].remove(potential_child)
                    return True
            elif allow_overlap and overlaps_any_placed(potential_child, placed_assets):
                continue
            # elif (for every potential child, parent pair find the one with different 
            # relative axes and find the one with the least center distance and move those must have a height overlap too to be a candidate)

            c_o     = check_overlap(axis=aligned_axis[axes[2]],
                                    potential_child=potential_child,
                                    potential_parent=potential_parent)
            one_d   = half_extent_overlap_2d(potential_parent, potential_child,
                                             threshold=1,
                                             axes=(aligned_axis[axes[0]],))
            c_depth = same_depth(potential_parent, potential_child,
                                 depth_axis=aligned_axis[axes[1]], weight=0.45)
            
            # # fix since scans were with door open:
            # # second_floor_test
            # if 'upper' in label and abs(potential_child['object_number'] - potential_parent['object_number']) < 2:
            #     c_o = True
            # if (potential_child['object_number'] == 15 and potential_parent['object_number'] == 14 and axis == 0):
            #     print('----')
            #     print(c_o, one_d, c_depth)
            #     print('----')
            # if (potential_child['object_number'] == 36):
            #     print('----')
            #     print(c_o, one_d, c_depth)
            #     print(potential_parent['center'], potential_parent['aabb'].get_extent(),
            #                                                         potential_child['center'], potential_child['aabb'].get_extent())
            #     print('----')
            # print(c_o, one_d, c_depth)

            # if (potential_child['object_number'] == 18 and potential_parent['object_number'] == 17 and axis == 0):
            #     print('----')
            #     print('testing here')
            #     print(c_o, one_d, c_depth)
            #     print('----')
            if c_depth and one_d and c_o and \
               potential_child['aabb'].get_extent()[aligned_axis[0]] > \
               potential_parent ['aabb'].get_extent()[aligned_axis[0]] / 8:

                # --- scoring ------------------------------------------------
                axis_idx   = aligned_axis[axes[0]]           # edge-comparison axis
                edge_score, clamp = _edge_alignment_score(potential_parent,
                                                   potential_child,
                                                   axis_idx)

                label_match   = int(potential_child['label'] == potential_parent['label'])
                cluster_match = int(
                    potential_child.get(f"{potential_parent['label'].split('_')[0]}_cluster_id")
                    ==
                    potential_parent.get(f"{potential_parent['label'].split('_')[0]}_cluster_id")
                )
                similarity = 2*label_match + cluster_match
                # print(similarity)
                # print(edge_score)
                score = edge_score * 0.01 + (1 + similarity)**2        # tweak weighting if desired

                if best is None or score > best[0]:
                    best = (score, potential_parent, potential_child, aligned_axis, axes, clamp)

    if best is None:
        return False

    _, best_parent, best_child, aligned_axis, axes, clamp = best



    set_dimension_parent_default(best_child, aligned_axis, axes[0], pcd_data,
                                    potential_parent=best_parent)
    set_dimension_parent_default(best_child, aligned_axis, axes[1], pcd_data,
                                    potential_parent=best_parent)
    set_dimension_parent_default(best_child, aligned_axis, axes[2], pcd_data)
    if best_child.get("rotation") is not None:
        transform = best_child['rotation']
        child_pos  = best_child["center"][aligned_axis[0]] # might need weight here (2 lines below)
        scene_pos  = center[aligned_axis[0]]
        tight_translation_weight = -1
        if scene_pos * best_parent['weight'] > child_pos * best_parent['weight']:
            transform = np.linalg.inv(transform)
            tight_translation_weight = 1

        # added this not thouroughly tested
        if aligned_axis == [2, 1, 0]:
            tight_translation_weight *= -1
        transform[0, 3] = tight_translation_weight * default_drawer_depth
        # print(f"parent {best_parent['object_number']}")
        # print(aligned_axis)
        # print(transform, tight_translation_weight)
        depth_clamp = 'front'
    else:
        depth_clamp = 'back'
        transform = np.eye(4)

    placement_func(
        s=s,
        parent_asset=best_parent,
        child_asset=best_child,
        child_width=best_child[extent_labels[aligned_axis[0]]],
        child_height=best_child[extent_labels[aligned_axis[1]]],
        child_depth=best_child[extent_labels[aligned_axis[2]]],
        transform=transform,
        depth_clamp=depth_clamp,
        clamp=clamp

    )

    placed_assets.append(best_child)
    unplaced_assets[label].remove(best_child)
    print(f"placing object {best_child['object_number']} on object {best_parent['object_number']}, {func_name}")
    # if (best_child['object_number'] == 4):
    #     print(aligned_axis)
    #     print(potential_parent['object_number'])
    best_child['relative_alignment'] = aligned_axis
    best_child['weight'] = best_parent['weight']
    
    return True

def try_to_place_upper_aligned(unplaced_assets, label, placed_assets, pcd_data, s):
    func_name, axes = "place_vertical_asset", [0, 2, 1]
    placement_func = globals()[func_name]

    best = None  # (edge_score, depth, parent, child, aligned_axis)

    for potential_parent in placed_assets:
        aligned_axis = potential_parent['relative_alignment']

        for potential_child in list(unplaced_assets[label]):
            if potential_child.get("rotation") is not None:
                # print(f'skipping {potential_child['object_number']}')
                continue
            one_d = half_extent_overlap_2d(
                potential_parent, potential_child,
                threshold=1,
                axes=(aligned_axis[axes[0]],)
            )
            if not one_d:
                continue

            depth = get_aligned_depth(
                potential_parent, potential_child,
                aligned_axis[axes[1]]
            )
            if depth is None:
                continue

            if potential_child['relative_alignment'] != aligned_axis:
                continue

            axis_idx   = aligned_axis[axes[0]]
            edge_score, clamp = _edge_alignment_score(
                potential_parent, potential_child, axis_idx
            )

            if (best is None) or (edge_score > best[0]):
                best = (edge_score, depth, potential_parent, potential_child, aligned_axis, clamp)

    if best is None:
        return False

    edge_score, depth, parent_asset, child_asset, aligned_axis, clamp = best

    # print(f'\n{clamp}\n')

    # print(f"parent {parent_asset['object_number']} found child {child_asset['object_number']}")

    # (Optional) recompute to be robust if anything changed:
    depth = get_aligned_depth(parent_asset, child_asset, aligned_axis[axes[1]])
    if depth is None:
        return False
    
    # print(depth)

    # Dimension propagation
    set_dimension_parent_default(child_asset, aligned_axis, axes[2], pcd_data)
    set_dimension_parent_default(child_asset, aligned_axis, axes[0], pcd_data)

    # account for pcd noise
    child_asset[extent_labels[aligned_axis[axes[1]]]] = depth * 1.2

    # Build transform
    translation = get_translation(parent_asset, child_asset, aligned_axis, axes[2])
    indices = [0, 2, 1]  # map x/y/z → homogeneous-matrix column
    index = indices[aligned_axis[axes[2]]]
    transform = np.eye(4)
    transform[index, 3] = translation

    # Place
    placement_func(
        s=s,
        parent_asset=parent_asset,
        child_asset=child_asset,
        child_width = child_asset[extent_labels[aligned_axis[0]]],
        child_height= child_asset[extent_labels[aligned_axis[1]]],
        child_depth = child_asset[extent_labels[aligned_axis[2]]],
        transform=transform,
        clamp=clamp
    )

    placed_assets.append(child_asset)
    unplaced_assets[label].remove(child_asset)
    print(f"placing object {child_asset['object_number']} on object {parent_asset['object_number']}, {func_name}")
    child_asset['relative_alignment'] = aligned_axis  # fixed typo
    child_asset['weight'] = parent_asset['weight']
    return True


def try_to_place_upper_not_aligned(unplaced_assets, label, placed_assets, pcd_data, s, scene_center):
    placed = False
    candidate_pairs = []
    for potential_parent in placed_assets:
        aligned_axis = potential_parent["relative_alignment"]

        for potential_child in list(unplaced_assets[label]):
            child_axes = potential_child["relative_alignment"]
            if child_axes == aligned_axis:
                continue
            if potential_child.get("rotation") is not None:
                # print(f'skipping {potential_child['object_number']}')
                continue
            one_d = half_extent_overlap_2d(
                potential_parent,
                potential_child,
                threshold=1,
                axes=(child_axes[0],)  # child's width axis
            )
            # if potential_parent['object_number'] == 34:
            #     print(potential_child['object_number'], one_d)

            if one_d:
                # Compute center distance for ranking
                dist = np.linalg.norm(np.array(potential_child["center"]) - np.array(potential_parent["center"]))
                candidate_pairs.append((dist, potential_parent, potential_child))

    if not candidate_pairs:
        return False

    candidate_pairs.sort(key=lambda x: x[0])
    _, parent_asset, child_asset = candidate_pairs[0]
    aligned_axis = parent_asset["relative_alignment"]
    child_axes = child_asset["relative_alignment"]

    # print()
    # print(parent_asset['object_number'], child_asset['object_number'], get_not_aligned_depth(parent_asset, child_asset, child_axes[2]))
    # print()
    # ------------------------------------------------------------
    # 3. Propagate dimensions
    # ------------------------------------------------------------

    lateral_axis = aligned_axis[0]
    sign = -1 if scene_center[lateral_axis] > child_asset["center"][lateral_axis] else 1
    parent_side = 'left'  if sign == -1 else 'right'
    child_side  = 'right' if sign == -1 else 'left'
    # print(parent_side, child_side)
    child_asset['weight'] = sign


    set_dimension_parent_default(child_asset, child_axes, 0, pcd_data)
    set_dimension_parent_default(child_asset, child_axes, 1, pcd_data)
    child_asset[extent_labels[child_axes[2]]] = get_not_aligned_depth(parent_asset, child_asset, child_axes[2])

    # ------------------------------------------------------------
    # 4. Build transform: rotate ±90° if misaligned
    # ------------------------------------------------------------
    translation = get_translation(parent_asset, child_asset, aligned_axis, 1)
    indices = [0, 2, 1]       
    index = indices[aligned_axis[1]]
    transform = np.eye(4)
    transform[index, 3] = translation


    # Rotate around parent's vertical axis
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                [0, 0, -1 * sign * np.pi / 2])
    transform[:3, :3] = R

    child_asset["width"], child_asset["depth"] = child_asset["depth"], child_asset["width"]

    # ------------------------------------------------------------
    # 5. Place ABOVE the parent (anchors = top/bottom)
    # ------------------------------------------------------------
    s.add_object(
        child_asset["asset_func"](
            child_asset["width"],
            child_asset["height"],
            child_asset["depth"],
        ),
        child_asset["asset_name"],
        connect_parent_id=parent_asset["asset_name"],
        connect_parent_anchor=(parent_side, "back", "top"),
        connect_obj_anchor=(child_side, "back", "bottom"),
        transform=transform
    )

    # ------------------------------------------------------------
    # 6. Bookkeeping
    # ------------------------------------------------------------
    placed_assets.append(child_asset)
    unplaced_assets[label].remove(child_asset)
    print(f"placing object {child_asset['object_number']} on object {child_asset['object_number']}, 90 degree rotation")
    child_asset["width"], child_asset["depth"] = \
            child_asset["depth"], child_asset["width"]
    return True




## assume upper is smaller depth than lower
def get_aligned_depth(parent, child, depth_axis,):
    p_s = parent['weight']
    p_depth = parent["aabb"].get_extent()[depth_axis]
    c_depth = child["aabb"].get_extent()[depth_axis]
    p_center = parent["center"][depth_axis]
    c_center = child["center"][depth_axis]

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
    # if (child['object_number'] == 14 and parent['object_number'] == 22):
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
    c_s = child['weight']
    p_depth = parent["aabb"].get_extent()[depth_axis]
    c_depth = child["aabb"].get_extent()[depth_axis]
    p_center = parent["center"][depth_axis]
    c_center = child["center"][depth_axis]

    p_min = p_center + c_s * (p_depth / 2)
    c_min = c_center + c_s * (c_depth / 2)

    # if child['object_number'] == 26:
    #     print(c_s)
    #     print(depth_axis)
    #     print(p_center, c_center)
    #     print(p_min, c_min)

    # print((c_s == 1 and c_min < p_min))
    if (c_s == -1 and (p_min < c_min)) or (c_s == 1 and c_min < p_min):
        return abs(p_min - c_min)
    else:
        return None


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
            

            # FLAG FOR TOMORROW
def aabb_overlap_fraction_with_centers(center1, extent1, center2, extent2,
                                       thresh=0.5):

    c1, e1 = np.asarray(center1), np.asarray(extent1) / 2.0
    c2, e2 = np.asarray(center2), np.asarray(extent2) / 2.0

    min1, max1 = c1 - e1, c1 + e1
    min2, max2 = c2 - e2, c2 + e2

    # Volume of each box
    vol1 = np.prod(2 * e1)
    vol2 = np.prod(2 * e2)

    # Overlap bounds
    overlap_min = np.maximum(min1, min2)
    overlap_max = np.minimum(max1, max2)
    overlap_dims = np.clip(overlap_max - overlap_min, 0, None)
    overlap_vol = np.prod(overlap_dims)

    # Fractions of each box that are overlapped
    frac1 = overlap_vol / vol1
    frac2 = overlap_vol / vol2

    return (frac1 >= thresh) or (frac2 >= thresh)

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
            # if (child['object_number'] == 35) or child['object_number'] == 36:
            #     print(asset['object_number'])
            return True
    return False

# check if one half exentents fill another full extents




# takes in a potential child and parent

# if aligned_ax_c != aligned_ax_p:
#     if center_of_scene[aligned_ax_p[0]] > aabb_c[aligned_ax_p[0]]:
        # place child to the left of the 
        # rotate the child 90 degrees and transform it left by aabb_c[aligned_ax_p[0]]


        # place child to the right of the 
        # rotate the child 90 degrees and transform it right by aabb_c[aligned_ax_p[0]]


def place_overlapped_child(s,
                             parent_asset: dict,
                             child_asset: dict,
                             scene_center: np.ndarray,
                             pcd_data: list):
    aligned_p = parent_asset["relative_alignment"]
    aligned_c = child_asset["relative_alignment"]
    lateral_axis = aligned_p[0]
    for ax in range(3):
        if ax == 0:
            set_dimension_parent_default(child_asset,
                                     aligned_p, ax,
                                     pcd_data)
        else:
            set_dimension_parent_default(child_asset,
                                     aligned_p, ax,
                                     pcd_data,
                                     potential_parent=parent_asset)
    child_pos  = child_asset["center"][lateral_axis] # might need weight here (2 lines below)
    scene_pos  = scene_center[lateral_axis]
    side_sign  = -1 if scene_pos * parent_asset['weight'] > child_pos * parent_asset['weight'] else 1 
    parent_side = 'left'  if side_sign == -1 else 'right'
    child_side  = 'right' if side_sign == -1 else 'left'
    child_asset['weight'] = side_sign
    transform = np.eye(4)

    # print(side_sign, parent_side, child_side)

    if aligned_c != aligned_p:
        # (i) ± 90° yaw so the *new* front faces the scene centre
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                [0, 0, -1 * side_sign * np.pi / 2])
        transform[:3, :3] = R
        child_asset["width"], child_asset["depth"] = \
            child_asset["depth"], child_asset["width"]
    else:
        return False

    # --------------------------------------------------
    # 4.  Pick vertical anchor (same rule you used before)
    # --------------------------------------------------
    vertical_anchor = (
        'top' if not same_depth(parent_asset, child_asset,
                                depth_axis=aligned_p[1], weight=0.2)
        else 'bottom'
    )

    # Special-case countertop offset for sinks
    if child_asset['label'] == 'sink' or parent_asset['label'] == 'sink':
        vertical_anchor = 'top'
        dz = default_countertop_thickness
        transform[2, 3] += dz if child_asset['label'] == 'sink' else -dz
    
    print(f"placing object {child_asset['object_number']} on object {parent_asset['object_number']}, 90 degree rotation")

    s.add_object(
        child_asset["asset_func"](
            child_asset["width"],
            child_asset["height"],
            child_asset["depth"],
        ),
        child_asset["asset_name"],
        connect_parent_id=parent_asset["asset_name"],
        connect_parent_anchor=(parent_side, 'front', vertical_anchor),
        connect_obj_anchor=(child_side,  'front', vertical_anchor),
        transform=transform
    )
    s.add_object(
        get_box_asset(child_asset["depth"] + default_drawer_depth, parent_asset['height'], parent_asset['depth']),
        f"box_{parent_asset['object_number']}_to_{child_asset['object_number']}",
        connect_parent_id=parent_asset["asset_name"],
        connect_parent_anchor=(parent_side, 'back', vertical_anchor),
        connect_obj_anchor=(child_side,  'back', vertical_anchor)

    )
    s.add_object(
        get_box_asset(parent_asset["depth"] + default_drawer_depth, child_asset['height'], child_asset['depth']),
        f"box_{child_asset['object_number']}_to_{parent_asset['object_number']}",
        connect_parent_id=child_asset["asset_name"],
        connect_parent_anchor=(child_side, 'back', vertical_anchor),
        connect_obj_anchor=(parent_side,  'back', vertical_anchor)

    )
    child_asset["width"], child_asset["depth"] = \
            child_asset["depth"], child_asset["width"]
    return True








# def try_to_place_medium(unplaced_assets, label, placed_assets,
#                         pcd_data, s, axis, allow_overlap=False):
#     # ----------------------------------------------------
#     # 0.  Determine helper function and axis mapping
#     # ----------------------------------------------------
#     if axis == 1:
#         func_name = "place_vertical_asset"
#         axes = [0, 2, 1]      # x–z–y order
#     elif axis == 0:
#         func_name = "place_lateral_asset"
#         axes = [1, 2, 0]      # y–z–x order
#     else:
#         raise ValueError("axis must be 0 (lateral) or 1 (vertical)")

#     placement_func = globals()[func_name]

#     # ----------------------------------------------------
#     # 1.  Search for the parent/child pair with *minimum*
#     #     |translation| along axes[2]
#     # ----------------------------------------------------
#     best = {
#         "mag": float("inf"),
#         "parent": None,
#         "child": None,
#         "aligned_axis": None,
#         "translation": None,
#     }

#     for parent in placed_assets:
#         aligned_axis = parent["relative_alignment"]
#         pp_half_box = get_half_aabb_box(parent)

#         # iterate over a *copy* so we don't mutate during search
#         for child in list(unplaced_assets[label]):
#             pc_half_box = get_half_aabb_box(child)

#             # Skip if overlap is prohibited and ≥50 % overlap detected
#             if not allow_overlap and overlaps_any_placed(child, placed_assets):
#                 continue

#             # Must overlap on the two non-placement axes
#             aabbs = _aabbs_overlap_2d(pp_half_box, pc_half_box, axes=(aligned_axis[axes[0]], aligned_axis[axes[1]]))
#             two_d = half_extent_overlap_2d(parent, child, axes=(aligned_axis[axes[0]], aligned_axis[axes[1]]))
#             if not two_d:
#                 continue

#             # Your change ①: use axes[2] here
#             translation = get_translation(parent, child,
#                                           aligned_axis, axes[2])

#             if abs(translation) < best["mag"]:
#                 best.update(dict(
#                     mag=abs(translation),
#                     parent=parent,
#                     child=child,
#                     aligned_axis=aligned_axis,
#                     translation=translation,
#                 ))

#     # ----------------------------------------------------
#     # 2.  Bail early if nothing legal was found
#     # ----------------------------------------------------
#     if best["parent"] is None:
#         return False

#     parent = best["parent"]
#     child = best["child"]
#     aligned_axis = best["aligned_axis"]
#     translation = best["translation"]

#     # ----------------------------------------------------
#     # 3.  Dimension propagation
#     # ----------------------------------------------------
#     set_dimension_parent_default(child, aligned_axis, axes[0], pcd_data,
#                                  potential_parent=parent)
#     set_dimension_parent_default(child, aligned_axis, axes[1], pcd_data,
#                                  potential_parent=parent)
#     set_dimension_parent_default(child, aligned_axis, axes[2], pcd_data)

#     # ----------------------------------------------------
#     # 4.  Build 4×4 transform
#     #     Your change ②: index derived from aligned_axis[axes[2]]
#     # ----------------------------------------------------
#     indices = [0, 2, 1]        # map x/y/z → homogeneous-matrix col
#     index = indices[aligned_axis[axes[2]]]
#     transform = np.eye(4)
#     transform[index, 3] = translation

#     # ----------------------------------------------------
#     # 5.  Call placement helper once
#     # ----------------------------------------------------
#     placement_func(
#         s=s,
#         parent_asset=parent,
#         child_asset=child,
#         child_width=child[extent_labels[aligned_axis[0]]],
#         child_height=child[extent_labels[aligned_axis[1]]],
#         child_depth=child[extent_labels[aligned_axis[2]]],
#         transform=transform,
#     )

#     # ----------------------------------------------------
#     # 6.  Book-keeping
#     # ----------------------------------------------------
#     placed_assets.append(child)
#     unplaced_assets[label].remove(child)

#     return True
