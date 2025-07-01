# record_3d_pcd_generation
Recreate Erics PCD generation and add pcd to simple urdf primitives heuristic

TODO: add comments on erics portion of the code, update readme with full directions of the pipeline, create pcd to urdf simple geometries heuristic, get mesh generation portion working

## Setup Directions
1. Clone repo
```bash
git clone https://github.com/jcub-gold/record_3d_pcd_generation
```
2. Install requirements
```bash
pip install -r requirements.txt
```
3. Download sam2 checkpoints
```bash
cd submodules/sam2/checkpoints
bash download_ckpts.sh
```

## Pipeline Directions:

1. Setup the data structure for the scene
```bash
python3 -m src.real2code2real.setup_structure --scene_name=example_scene --num_objects=1 --num_states=1
```
- scene_name specifies the name of your scene
- num_objects specifies the number of objects in your scene
- num_states specifies the number of states per object

2.  Scan input data
- Scan video with Record3d
- Export as EXR + JPG
- Go to Files app and transfer to computer, the output folder should contain format:
```
data/scene_name/record3d_input/
├── depth/
│   ├── 0.exr
│   ├── 1.exr
│   └── ...
├── rgb/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── metadata.json
```

3. Segment frames
```bash
python3 -m src.real2code2real.get_masks -s /path/to/scene_directory --dataset_size 1200
```
- dataset_size specifies how many input frames to down-sample and use
- Run real2code2real/get_masks.py with the new dataset size and how many objects to segment. Then for each object, enter in:
    - start_frame end_frame
    - frame1 frame_state pixel1X pixel1Y pixel2X pixel2Y …
    - frame2 frame_state pixel1X pixel1Y …

![Alt text](relative/or/absolute/path/to/image.png)
