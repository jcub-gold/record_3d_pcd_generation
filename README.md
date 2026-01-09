# record_3d_pcd_generation
Real2Code2Real URDF generation pipeline.

## Setup Directions
### 1. Clone repo
```bash
git clone https://github.com/jcub-gold/record_3d_pcd_generation
git submodule init
git submodule update
```
### 2. Install requirements
```bash
conda env create -f r2c2r.yml
conda env create -f trellis.yml
```
### 3. Download sam2 checkpoints
```bash
cd submodules/sam2/checkpoints
bash download_ckpts.sh
cd ../../..
```

## Pipeline Directions:

### 1. Setup the data structure for the scene
```bash
conda activate r2c2r
```
```bash
python3 -m src.real2code2real.setup_structure --scene_name=example_scene
```
- scene_name specifies the name of your scene

### 2.  Scan input data
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

### 3. Generate the template URDF
```bash
python3 -m src.real2code2real.generate_template_urdf --scene_name=example_scene --dataset_size=1200
```
- dataset_size specifies how many input frames to down-sample and use
- There will be two GUIs displayed for object labeling and segmentation
    - In the first GUI, you will be prompted to enter the start frame number (first frame of appearance without occlussion) and the end frame number (last frame of appearance without occlusion) as well as the label for each object
    - In the second GUI, you will be prompted to select points for segmentation for each object


- In this step, you can also specify parameters for point cloud generation
    - "--eps=0.03 --min_points=15 --nb_neighbors=15 --std_ratio=2"

NOTE: You need one counter object in the scene to get the depth dimension for each object

### 4. Scan indivual assets for mesh generation
- Take rgb (non HDR) videos of opened assets desired for mesh generation
- Save videos in the multiview directory under the associated object directory
```
data/scene_name/multiview/
├── object_1/
│   └── movie_1.mov
├── object_8/
│   └── movie_2.mov
```

### 5. Prepare input for TRELLIS
```bash
python3 -m src.real2code2real.prepare_mesh_input --scene_name=example_scene
```

- Here you will be prompted to select points for segmentation again in each video

### 6. Generate meshes
```bash
conda activate trellis
```
```bash
python3 -m src.real2code2real.generate_realistic_mesh --scene_name=example_scene
```

### 7. Generate final urdf
```bash
conda activate r2c2r
```
```bash
python3 -m src.real2code2real.generate_realistic_urdf --scene_name=example_scene
```