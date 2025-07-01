# record_3d_pcd_generation
Recreate Erics PCD generation and add pcd to simple urdf primitives heuristic

TODO: add comments on erics portion of the code, update readme with full directions of the pipeline, create pcd to urdf simple geometries heuristic, get mesh generation portion working


## Pipeline Directions:

1. Setup the data structure for the scene
```bash
python3 -m real2code2real/setup_structure --scene_name={the name of your scene} --num_objects={the number of objects in your scene} --num_states={number of states per object}
```
Scan video with Record3d
Export as EXR + JPG
Go to Files app and transfer to computer, the output folder should contain format:
scene_name
depth
0.exr
1.exr
…
rgb
0.jpg
1.jpg
…
 metadata.json


python get_masks.py -s /path/to/scene_directory --dataset_size 1200
dataset_size specifies how many input frames to down-sample and use
Run get_masks.py with the new dataset size and how many objects to segment. Then for each object, enter in:
start_frame end_frame
frame1 frame_state pixel1X pixel1Y pixel2X pixel2Y …
frame2 frame_state pixel1X pixel1Y …
