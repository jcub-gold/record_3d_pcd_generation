import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa

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

s.show()

# s.export('extracted_drawer/drawer_extracted.urdf')