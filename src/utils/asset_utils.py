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


def get_asset(asset_name, width, height, depth, **kwargs):
    func_name = f"get_{asset_name}_asset"
    func = globals().get(func_name)
    if not func:
        raise ValueError(f"No asset function found for asset: '{asset_name}'")
    return func(width, height, depth, **kwargs)

"""
    Asset Generation Utilities
"""
def get_sink_asset(width, height, depth, default_countertop_thickness=0.04):
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
        handle_offset=(height * 0.3, width * 0.05))

def get_lower_right_cabinet_asset(width, height, depth):
    return pa.BaseCabinetAsset(width=width, 
        height=height, 
        depth=depth, 
        num_drawers_vertical=0,
        include_cabinet_doors=True,
        include_foot_panel=False,
        lower_compartment_types=("door_left",),
        handle_offset=(height * 0.3, width * 0.05))

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
