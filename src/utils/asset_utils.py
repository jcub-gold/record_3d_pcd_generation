from scene_synthesizer import procedural_assets as pa
from scene_synthesizer.assets import BoxAsset

# NOTE: for any new asset, you must define a function get_{asset_name}_asset

"""
Function: get_asset
-------------------
asset_name: string name of the asset type (e.g., "sink", "drawer")
width: width of the asset
height: height of the asset
depth: depth of the asset
**kwargs: additional keyword arguments for the asset constructor

Dynamically dispatches to the appropriate asset creation function based on asset_name.
Returns the generated asset object.
Raises ValueError if no matching asset function is found.
"""
def get_asset(asset_name, width, height, depth, **kwargs):
    func_name = f"get_{asset_name}_asset"
    func = globals().get(func_name)
    if not func:
        raise ValueError(f"No asset function found for asset: '{asset_name}'")
    return func(width, height, depth, **kwargs)


# Asset Generation Utilities
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

def get_counter_asset(width, height, depth):
    return get_box_asset(width, height, depth)