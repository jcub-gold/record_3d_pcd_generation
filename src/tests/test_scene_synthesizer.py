import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa

def get_lower_left_cabinet_asset(width, height, depth):
    return pa.BaseCabinetAsset(width=width, 
        height=height, 
        depth=depth, 
        num_drawers_vertical=0,
        include_cabinet_doors=True,
        include_foot_panel=False,
        lower_compartment_types=("door_right",),
        handle_offset=(height * 0.35, width * 0.05))

if __name__ == "__main__":
    lower_large_double_cabinet = pa.BaseCabinetAsset(
        width=0.958, 
        height=0.333, 
        depth=0.998, 
        num_drawers_horizontal=1,
        include_cabinet_doors=False,
        include_foot_panel=False # Explicitly specify door types
    )
    s = synth.Scene()

    s.add_object(lower_large_double_cabinet, 'cabinet')
    s.add_object(
        get_lower_left_cabinet_asset(0.958, 0.333, 0.998),
        'lower_left_cabinet',
        connect_parent_id='cabinet',
        connect_parent_anchor=('left', 'back', 'bottom'),  # bottom of the lower cabinet
        connect_obj_anchor=('right', 'back', 'bottom'),  # top of the lower left cabinet
        translation=(0, 0, 0.0)
    )

    s.show()

    # s.export('extracted_drawer/drawer_extracted.urdf')