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
from src.utils.pcd_to_urdf_utils_utils import get_depth_axis_from_pcd_extents, place_lateral_asset, place_vertical_asset, check_BB_overlap, check_extent_overlap, get_depth_delta, get_aligned_depth, get_not_aligned_depth, get_depth_from_counter, place_counter_top
from src.utils.asset_utils import get_asset
from src.utils.obb_utils import get_rotation_from_pca


GREEN = "\033[32m"
RESET = "\033[0m"

possible_labels = ['drawer', 'lower_left_cabinet', 'lower_right_cabinet', 'upper_left_cabinet', 'upper_right_cabinet', 'box', 'sink', "counter"]
extent_labels = ['width', 'height', 'depth']

default_countertop_thickness = 0.04
default_drawer_depth = 0.024

"""
second_floor_test
label_clusters = [3, 1, 1, 1, 1]
extent_indices = [(0, 1, 2), (1,), (1,), (0, 1), (0,)]
"""
# # basement_test
# label_clusters = [5, 1, 1, 2]
# extent_indices = [(0, 1, 2), (0, 1, 2), (1,), (1,), ]
# second_floor_test
label_clusters = [1, 1, 3, 1, 1]
extent_indices = [(1,), (0,), (0, 1, 2), (1,), (0, 1)]
# # base_cabinet_test
# label_clusters = [1, 1]
# extent_indices = [(0, 1, 2), (1,)]

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
        dict_data = {}
        weight = 1
        confident = True
        transform = None
        depth_axis = get_depth_axis_from_pcd_extents(aabb.get_extent(), ratio_threshold=0.15)
        if depth_axis == 0:
            relative_alignment = [2, 1, 0]
            print(f'90 degree rotation {obj_num}')
        else:
            relative_alignment = [0, 1, 2]
            if depth_axis == None:
                angle = get_rotation_from_pca(pcd) * 180 / np.pi
                angle = 5 * round(angle / 5)
                angle = abs(angle % 90)
                print(f'{angle} degree degree rotation {obj_num}')
                if angle == 0:
                    confident = False
                else:
                    angle = angle / 180 * np.pi
                    transform = np.eye(4)
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                            [0, 0, -1 * angle])
                    transform[:3, :3] = R
                    
            else:
                print(f'No rotation {obj_num}')


        extent = np.array(aabb.get_extent())
        width, height, depth = extent[relative_alignment]
        if transform is not None:
            width = np.sqrt(width**2 + depth**2)
        dict_data = {
            "pcd_path": os.path.join(aa_pcds_path, fname),
            "pcd": pcd,
            "aabb": aabb,
            "center": aabb.get_center(),
            "label": label,
            "object_number": obj_num,
            "width": width,
            "height": height,
            "depth": depth,
            "relative_alignment": relative_alignment,
            "weight": weight,
            "alignment_confidence": confident
        }
        if transform is not None:
            dict_data['rotation'] = transform

       

        ### NOTE: will probobly want to define the asset name later after assignment
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
    return pcd_data, center, labels, label_keywords

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


def pcd_to_urdf_simple_geometries(pcd_data, combined_center, labels, scene_name=None):
    # if "counter" in labels:
    #     labels.remove("counter")
    #     counter = True
    # else:
    #     counter = False
    # counters = []
    # if counter:
    #     for asset in pcd_data:
    #         if asset['label'] == "counter":
    #             counters.append(asset)
    
    output_path = f"simple_urdf_scenes/{scene_name}/{scene_name}.urdf"
    cache = {}
    cache['vertical_placement_pairs'] = []
    unplaced_assets = {}

    for asset in pcd_data:
        if unplaced_assets.get(asset['label']) is None:
            unplaced_assets[asset['label']] = []
        unplaced_assets[asset['label']].append(asset)
    if "counter" in labels:
        labels.remove("counter")

    s = synth.Scene()
    placed_assets = []

    if unplaced_assets.get('drawer') is None:
        print('No drawer asset in scene :(')
        return 
    # place the first asset, prioritize a drawer asset for depth calculation
    parent_asset = unplaced_assets['drawer'].pop()
    # print(parent_asset)

    parent_asset['depth'] = .7 # hardcoded right now
    if unplaced_assets.get("counter") is not None:
        parent_asset['depth'] = get_depth_from_counter(unplaced_assets["counter"], parent_asset)
    print(f" DEPTH!!!!! {parent_asset['depth']}")


    width, height, depth = parent_asset['width'], parent_asset['height'], parent_asset['depth']
    s.add_object(get_asset(parent_asset['label'], width, height, depth), parent_asset['asset_name'])
    placed_assets.append(parent_asset)

    upper_labels = []
    lower_labels = []
    for label in labels:
        if 'upper' not in label:
            lower_labels.append(label)
        else:
            upper_labels.append(label)

    def get_unplaced_assets_length(unplaced_assets, labels):
        length = 0
        for label in labels:
            length += len(unplaced_assets[label])
        return length

    while get_unplaced_assets_length(unplaced_assets, labels) > 0:
        expand_bottom = True
        while expand_bottom:
            expand_direct_neighbors(lower_labels, unplaced_assets, placed_assets, pcd_data, s, combined_center, cache)
            expand_bottom = False
            for label in lower_labels:
                while try_to_place_strict(unplaced_assets, label, placed_assets, pcd_data, s, 0, combined_center, allow_90_degree_rotation=True):
                    # s.show()
                    expand_bottom = True
        # s.show()

        expand_top = True
        while expand_top:
            expand_direct_neighbors(upper_labels, unplaced_assets, placed_assets, pcd_data, s, combined_center, cache)
            place_aligned_top = False
            for label in upper_labels:
                if try_to_place_upper_aligned(unplaced_assets, label, placed_assets, pcd_data, s):
                    place_aligned_top = True
                    break
            if place_aligned_top:
                continue
            # s.show()
            place_not_aligned_top = False
            for label in upper_labels:
                if try_to_place_upper_not_aligned(unplaced_assets, label, placed_assets, pcd_data, s, combined_center):
                    place_not_aligned_top = True
                    break
            expand_top = place_aligned_top or place_not_aligned_top

        print("here")
        place_rotated = True
        while place_rotated:
            place_rotated = False
            for label in labels:
                # print('trying to place rotated!!!!!!!!!')
                while(try_to_place_strict(unplaced_assets, label, placed_assets, pcd_data, s, 1, combined_center, allow_other_rotation=True)):
                    place_rotated = True
                    continue
                while(try_to_place_strict(unplaced_assets, label, placed_assets, pcd_data, s, 0, combined_center, allow_other_rotation=True)):
                    place_rotated = True
                    continue
        # s.show()

    # place counters
    if len(unplaced_assets['counter']) > 0:
        for counter in unplaced_assets['counter']:
            place_counter_top(s, counter, pcd_data)

    s.export(output_path)
    print(f"{GREEN} Successfully generated URDF at {output_path}!{RESET}")
    with open(os.path.join("data", scene_name, "cached_vertical_pairs.json"), "w") as f:
        json.dump(cache, f, indent=4)
    s.show()
    return

def expand_direct_neighbors(labels, unplaced_assets, placed_assets, pcd_data, s, combined_center, cache):
    place_direct_neighbor = True
    while place_direct_neighbor:
        place_direct_neighbor = False
        for label in labels:
            while(try_to_place_strict(unplaced_assets, label, placed_assets, pcd_data, s, 1, combined_center, cache=cache)):
                place_direct_neighbor = True
                break
            while(try_to_place_strict(unplaced_assets, label, placed_assets, pcd_data, s, 0, combined_center)):
                place_direct_neighbor = True
                break



def _edge_alignment_score(parent, child, axis_idx, *, eps=1e-6):
    """Return  √(1/closest_edge_difference)  along axis_idx."""
    pc, pe = parent["center"][axis_idx], parent["aabb"].get_extent()[axis_idx]
    cc, ce = child ["center"][axis_idx], child ["aabb"].get_extent()[axis_idx]

    p_lo, p_hi = pc - pe/2, pc + pe/2
    c_lo, c_hi = cc - ce/2, cc + ce/2
    closest = min(abs(p_lo - c_lo), abs(p_hi - c_hi))
    if closest == abs(p_lo - c_lo):
        clamp = -1 * parent['weight']
    else:
        clamp = 1 * parent['weight']
    return np.sqrt(1.0 / (closest + eps)), clamp

def try_to_place_strict(unplaced_assets, label, placed_assets, pcd_data, s, axis, center, allow_90_degree_rotation=False, allow_other_rotation=False, cache=None):
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
            # TODO: need to add extra logic if unsure about the relative alignment
            if allow_90_degree_rotation == False and potential_child['relative_alignment'] != potential_parent['relative_alignment'] and potential_child['alignment_confidence'] == True:
                continue
            if allow_90_degree_rotation and potential_child['relative_alignment'] != potential_parent['relative_alignment'] and potential_child['alignment_confidence'] == True:
                if place_90_degree_rotated_child(s, potential_parent, potential_child, center, unplaced_assets):
                    placed_assets.append(potential_child)
                    unplaced_assets[label].remove(potential_child)
                    return True
                else:
                    continue
            if allow_other_rotation == False and potential_child.get("rotation") is not None:
                continue
            c_o     = check_BB_overlap(axis=aligned_axis[axes[2]],
                                    potential_child=potential_child,
                                    potential_parent=potential_parent)
            one_d   = check_extent_overlap(potential_parent, potential_child,
                                             threshold=1,
                                             axes=(aligned_axis[axes[0]],))
            if one_d and c_o and abs(potential_parent['center'][aligned_axis[2]] - potential_child['center'][aligned_axis[2]]) < potential_parent['depth'] / 2:
                # if potential_child['object_number'] == 27 and potential_parent['object_number'] == 28:
                #     print(potential_parent['width'], potential_parent['height'], potential_parent['depth'])
                #     print(aligned_axis)
                #     print(potential_child['width'], potential_child['height'], potential_child['depth'])
                #     print(potential_child['relative_alignment'])


                # --- scoring ------------------------------------------------
                axis_idx   = axes[0]           # edge-comparison axis
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
                score = edge_score * 0.01 + (1 + similarity)**2        # tweak weighting if desired

                if best is None or score > best[0]:
                    best = (score, potential_parent, potential_child, aligned_axis, axes, clamp)

    if best is None:
        return False

    _, best_parent, best_child, aligned_axis, axes, clamp = best

    # if they were not aligned, we need to realign the child
    if best_child['relative_alignment'] != aligned_axis:    
        extent = np.array(best_child['aabb'].get_extent())
        best_child['width'], best_child['height'], best_child['depth'] = extent[aligned_axis]
    if (best_child['object_number'] == 11):
        print(get_depth_delta(best_parent, best_child))
    best_child['depth'] = best_parent['depth']  + get_depth_delta(best_parent, best_child)
    

    for extent in ['width', 'height', 'depth']:
        if abs(best_child[extent] - best_parent[extent]) < 0.05 * best_parent[extent]:
            best_child[extent] = best_parent[extent]

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
        # print(transform, tight_translation_weight
        best_child['depth'] = best_parent['depth']
        depth_clamp = 'front'
    else:
        depth_clamp = 'back'
        transform = np.eye(4)

    placement_func(
        s=s,
        parent_asset=best_parent,
        child_asset=best_child,
        child_width=best_child['width'],
        child_height=best_child['height'],
        child_depth=best_child['depth'],
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

    if cache is not None:
        cache['vertical_placement_pairs'].append(
            (f"object_{best_parent['object_number']}", f"object_{best_child['object_number']}")
        )
    
    return True

def try_to_place_upper_aligned(unplaced_assets, label, placed_assets, pcd_data, s):
    func_name, axes = "place_vertical_asset", [0, 2, 1]
    placement_func = globals()[func_name]

    best = None  # (edge_score, depth, parent, child, aligned_axis)

    for potential_parent in placed_assets:
        aligned_axis = potential_parent['relative_alignment']

        for potential_child in list(unplaced_assets[label]):
            # if potential_child['object_number'] == 29 and potential_parent['object_number'] == 25:
            #         print(potential_parent['width'], potential_parent['height'], potential_parent['depth'])
            #         print(aligned_axis)
            #         print(potential_child['width'], potential_child['height'], potential_child['depth'])
            #         print(potential_child['relative_alignment'])
            if potential_child.get("rotation") is not None:
                # print(f'skipping {potential_child['object_number']}')
                continue
            one_d = check_extent_overlap(
                potential_parent, potential_child,
                threshold=1,
                axes=(aligned_axis[axes[0]],)
            )
            if not one_d:
                continue

            depth = get_aligned_depth(
                potential_parent, potential_child,
                2
            )
            # if potential_child['object_number'] == 29 and potential_parent['object_number'] == 25:
            #         print(depth)
            if depth is None:
                continue

            if potential_child['relative_alignment'] != aligned_axis:
                continue

            axis_idx   = axes[0]
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

    if depth is None:
        return False
    
    # print(depth)

    # Dimension propagation

    # account for pcd noise
    child_asset['depth'] = depth

    # Build transform
    translation = _get_translation(parent_asset, child_asset, aligned_axis, axes[2])
    indices = [0, 2, 1]  # map x/y/z → homogeneous-matrix column
    index = indices[aligned_axis[axes[2]]]
    transform = np.eye(4)
    transform[index, 3] = translation

    # if child_asset['object_number'] == 29 and parent_asset['object_number'] == 25:
    #     print(parent_asset['width'], parent_asset['height'], parent_asset['depth'])
    #     print(aligned_axis)
    #     print(child_asset['width'], child_asset['height'], child_asset['depth'])
    #     print(child_asset['relative_alignment'])

    # Place
    placement_func(
        s=s,
        parent_asset=parent_asset,
        child_asset=child_asset,
        child_width = child_asset['width'],
        child_height= child_asset['height'],
        child_depth = child_asset['depth'],
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
            one_d = check_extent_overlap(
                potential_parent,
                potential_child,
                threshold=1,
                axes=(child_axes[0],)  # child's width axis
            )
            # if potential_parent['object_number'] == 34:
            #     print(potential_child['object_number'], one_d)

            if one_d:
                # Compute center distance for ranking
                
                dist = np.linalg.norm(np.array(potential_child["center"])[potential_child['relative_alignment'][2]] - np.array(potential_parent["center"])[potential_child['relative_alignment'][2]])
                # if (potential_parent['object_number'] == 1 or potential_parent['object_number'] == 3):
                #     print(potential_parent['object_number'], dist, potential_child['object_number'])
                #     print(np.array(potential_child["center"]) - np.array(potential_parent["center"]))
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


    
    child_asset['depth'] = get_not_aligned_depth(parent_asset, child_asset, 2)

    # ------------------------------------------------------------
    # 4. Build transform: rotate ±90° if misaligned
    # ------------------------------------------------------------
    translation = _get_translation(parent_asset, child_asset, aligned_axis, 1)
    indices = [0, 2, 1]       
    index = indices[aligned_axis[1]]
    transform = np.eye(4)
    transform[index, 3] = translation


    # Rotate around parent's vertical axis
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                [0, 0, -1 * sign * np.pi / 2])
    transform[:3, :3] = R

    # ------------------------------------------------------------
    # 5. Place ABOVE the parent (anchors = top/bottom)
    # ------------------------------------------------------------
    s.add_object(
        get_asset(child_asset['label'], child_asset['width'], child_asset['height'], child_asset['depth']),
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
    return True






def _get_translation(potential_parent, potential_child, aligned_axis, target_axis):
    axis = aligned_axis[target_axis]
    return abs(potential_child['center'][axis] - potential_parent['center'][axis]) - (abs(potential_child['aabb'].get_extent()[axis] + potential_parent['aabb'].get_extent()[axis])) + abs(_effective_extent(potential_child, axis) + _effective_extent(potential_parent, axis)) / 2.0
    
            

def _effective_extent(asset, axis):
    label = extent_labels[axis]                # e.g. "width"
    fixed_key = f"fixed_{label}"               # e.g. "fixed_width"

    if asset.get(fixed_key) is not None:       # explicit fixed dimension
        return asset[fixed_key]
    # Fallback: geometry extent from AABB
    return asset["aabb"].get_extent()[axis]

# check if one half exentents fill another full extents


def place_90_degree_rotated_child(s,
                             parent_asset: dict,
                             child_asset: dict,
                             scene_center: np.ndarray,
                             unplaced_assets: list):
    aligned_p = parent_asset["relative_alignment"]
    if not (check_BB_overlap(aligned_p[0], parent_asset, child_asset) and check_extent_overlap(parent_asset, child_asset, threshold=1, axes=(1,)) and check_BB_overlap(aligned_p[2], parent_asset, child_asset, threshold=0.5)):
        return False
    # print(parent_asset['object_number'], child_asset['object_number'])
    aligned_c = child_asset["relative_alignment"]
    lateral_axis = aligned_p[0]
    child_asset['depth'] = parent_asset['depth'] # hardcoded to depth rn get_depth_from_counter(unplaced_assets['counter'], child_asset)
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
    else:
        return False

    # --------------------------------------------------
    # 4.  Pick vertical anchor (same rule you used before)
    # --------------------------------------------------
    # vertical_anchor = (
    #     'top' if not same_depth(parent_asset, child_asset,
    #                             depth_axis=aligned_p[1], weight=0.2)
    #     else 'bottom'
    # )
    vertical_anchor = "bottom"

    # Special-case countertop offset for sinks
    if child_asset['label'] == 'sink' or parent_asset['label'] == 'sink':
        vertical_anchor = 'top'
        dz = default_countertop_thickness
        transform[2, 3] += dz if child_asset['label'] == 'sink' else -dz
    
    print(f"placing object {child_asset['object_number']} on object {parent_asset['object_number']}, 90 degree rotation")

    if 'upper' not in child_asset['label']:
        vertical_addition = default_countertop_thickness
    s.add_object(
        get_asset(child_asset['label'], child_asset['width'], child_asset['height'], child_asset['depth']),
        child_asset["asset_name"],
        connect_parent_id=parent_asset["asset_name"],
        connect_parent_anchor=(parent_side, 'front', vertical_anchor),
        connect_obj_anchor=(child_side,  'front', vertical_anchor),
        transform=transform
    )
    s.add_object(
        get_asset('box', child_asset["depth"] + default_drawer_depth, parent_asset['height'] + vertical_addition, parent_asset['depth']),
        f"box_{parent_asset['object_number']}_to_{child_asset['object_number']}",
        connect_parent_id=parent_asset["asset_name"],
        connect_parent_anchor=(parent_side, 'back', vertical_anchor),
        connect_obj_anchor=(child_side,  'back', vertical_anchor)

    )
    s.add_object(
        get_asset('box', parent_asset["depth"] + default_drawer_depth, child_asset['height'] + vertical_addition, child_asset['depth']),
        f"box_{child_asset['object_number']}_to_{parent_asset['object_number']}",
        connect_parent_id=child_asset["asset_name"],
        connect_parent_anchor=(child_side, 'back', vertical_anchor),
        connect_obj_anchor=(parent_side,  'back', vertical_anchor)

    )
    return True




def post_process_placed_assets(scene_name):
    aa_pcds_dir = f"data/{scene_name}/aa_pcds"
    vertical_pairs_cache = json.load(open(f"data/{scene_name}/cached_vertical_pairs.json", "r"))

    for pair in vertical_pairs_cache['vertical_placement_pairs']:
        for _, _, filenames in os.walk(aa_pcds_dir):
            pcds = [dict(), dict()]
            for file in filenames:
                for i in range(2):
                    if pair[i] in file:
                        pcds[i]['pcd'] = o3d.io.read_point_cloud(os.path.join(aa_pcds_dir, file))
                        pcds[i]['path'] = os.path.join(aa_pcds_dir, file)
                        print(pair[i])
            resize_pcd(pcds[0]['pcd'], pcds[1]['pcd'], pcds[0]['path'], pcds[1]['path'])
            
                



from time import sleep

def resize_pcd(pcd_1, pcd_2, pcd_1_path, pcd_2_path):

    extent_1 = pcd_1.get_max_bound()[1] - pcd_1.get_min_bound()[1]
    extent_1 /= 2
    extent_2 = pcd_2.get_max_bound()[1] - pcd_2.get_min_bound()[1]
    extent_2 /= 2
    center_1 = pcd_1.get_center()[1]
    center_2 = pcd_2.get_center()[1]

    clip = (extent_1 + extent_2) - abs(center_1 - center_2)
    print(f"overlap: {clip}")
    if clip > 0:
        scale_1 = np.eye(4)
        scale_1[1,1] = (extent_1 - clip / 4) /extent_1
        scale_2 = np.eye(4)
        scale_2[1,1] = (extent_2 - clip / 4) /extent_2
        if center_1 > center_2:
            sign = 1
        else:
            sign = -1

        center = pcd_1.get_center()
        pcd_1.translate(-center)
        pcd_1.transform(scale_1)
        center[1] += sign * clip / 4
        pcd_1.translate(center)

        center = pcd_2.get_center()
        pcd_2.translate(-center)
        pcd_2.transform(scale_2)
        center[1] -= sign * clip / 4
        pcd_2.translate(center)
    
    extent_1 = pcd_1.get_max_bound()[1] - pcd_1.get_min_bound()[1]
    extent_1 /= 2
    extent_2 = pcd_2.get_max_bound()[1] - pcd_2.get_min_bound()[1]
    extent_2 /= 2
    center_1 = pcd_1.get_center()[1]
    center_2 = pcd_2.get_center()[1]

    clip = (extent_1 + extent_2) - abs(center_1 - center_2)
    print(f"overlap: {clip}")
    
    o3d.io.write_point_cloud(pcd_2_path, pcd_2)
    o3d.io.write_point_cloud(pcd_1_path, pcd_1)
    sleep(0.1)
            


# Assume we are only scanning the faces
# Start with a counter top

# if extents overlap (80% of one fits in the other), and there is overlap or very close to overlap in the searching dimension
# the relative alignments need to be the same if allow rotation is False ::::: don't need to check the depth at this step


# calculate depth by front of parent minus front of child
# if extents are within 5-10% of each other, set to parent extent



# issues, how do I solve for unknow relative alignment? 
#   - calculate 2.5 obb and then if there is no rotation needed mark as unconfident_relative_alignment, if so, look in the parent.
# issues, how do i caculate first depth?
#   - calculate depth countertop caclulation (only look at countertop pcd overlapping with the child width)
# issues, how do I place the countertop?
#  - bit hacky, but proposed solution is to add all the directly below candidates with the same relative alignment, per countertop, then you can add their width and place on the "best asset", will need to add countertop when rotating assets, will require a set countertop height


# todo countertops (extract depth and add)
