import os
import shutil
from argparse import ArgumentParser
from src.utils.mask_utils import get_labeled_images, get_object_masks
from src.utils.dataset_utils import copy_dataset, rewrite_json
import time
import sys
from src.guis.frame_selection_gui import run_frame_range_gui
from src.guis.point_prompting_gui import run_prompt_gui
import json

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

parser = ArgumentParser("Get masks and mesh extracts of objects within a scene")

parser.add_argument("--scene_name", "-s", required=True, type = str)
parser.add_argument("--model_path", "-m", type=str, default=None)
parser.add_argument("--dataset_size", default = 1100, type = int)


args = parser.parse_args(sys.argv[1:])

if args.model_path:
    os.makedirs(args.model_path, exist_ok=True)
    base_output_dir = os.path.join(args.model_path, time.strftime("%Y%m%d-%H%M%S"))

base_dir = f"data/{args.scene_name}/record3d_input"
raw_dir = os.path.join(base_dir, "rgb")
raw_depth_dir = os.path.join(base_dir, "depth")
input_dir = os.path.join(base_dir, "input")
depth_dir = os.path.join(base_dir, "input_depth")
labeled_dir = os.path.join(base_dir, "labeled")
background_masks_dir = os.path.join(base_dir, "images")

# Reduce the dataset size
if not os.path.isdir(input_dir):
    print(bcolors.OKCYAN + f"Making reduced dataset directory at {input_dir}" + bcolors.ENDC)
    frame_correspondance = copy_dataset(raw_dir, input_dir, min(args.dataset_size, len(os.listdir(raw_dir))))
    copy_dataset(raw_depth_dir, depth_dir, args.dataset_size)

    categories = ["frameTimestamps", "poses", "perFrameIntrinsicCoeffs"]
    rewrite_json(os.path.join(base_dir, "metadata.json"), os.path.join(base_dir, "new_metadata.json"), frame_correspondance, categories)

print(bcolors.OKGREEN + f"Reduced dataset successfully created at {input_dir}" + bcolors.ENDC)

# Get labeled images for precision prompting
if not os.path.isdir(labeled_dir):
    print(bcolors.OKCYAN + f"Now getting labeled images and saving at {labeled_dir}" + bcolors.ENDC)
    get_labeled_images(labeled_dir, input_dir)
print(bcolors.OKGREEN + f"Labeled dataset successfully created at {labeled_dir}" + bcolors.ENDC)

### if cache load cache
cache_path = os.path.join("data", args.scene_name, "cache.json")
if os.path.isfile(cache_path):
    print(bcolors.OKGREEN + f"Cache file found at {cache_path}, loading cache" + bcolors.ENDC)
    with open(cache_path, 'r') as f:
        cache = json.load(f)
else:
    cache = {}

# loop through asking users if they are done selecting frames
if len(cache) == 0:
    frame = 0
    user_input = "\n"
    obj_num = 1
    while(user_input.lower() != "q"):
        obj_key = f"object_{obj_num}"
        output_tuple = run_frame_range_gui(input_dir, frame, obj_key)
        cache[obj_key] = {}
        cache[obj_key]["frame_interval"] = output_tuple[0]
        cache[obj_key]["label"] = output_tuple[1]
        frame = output_tuple[0][1]
        obj_num += 1
        user_input = input("Enter 'q' to quit. Enter 'return' to continue selecting objects: ")

num_objects = obj_num - 1

# Create new directory for each object
for obj_key in cache:

    # Split images into separate directories
    object_dir = os.path.join(base_dir, obj_key)
    os.makedirs(object_dir, exist_ok=True)

    object_input_dir = os.path.join(object_dir, "input")
    if not os.path.isdir(object_input_dir):
        print(bcolors.OKCYAN + f"Making copies of input in object directory for {obj_key}"+ bcolors.ENDC)
        os.makedirs(object_input_dir, exist_ok=True)

        # Copy existing files to a separate directory
        for file_index in range(cache[obj_key]['frame_interval'][0], cache[obj_key]['frame_interval'][1]):
            file_path = os.path.join(input_dir, f"{file_index}.jpg")
            if os.path.isfile(file_path):
                shutil.copy2(file_path, object_input_dir)
    else:
        print(bcolors.OKGREEN + f"Object input directory already found for object {obj_key}" + bcolors.ENDC)

# select points for each object if not in cache

for obj_key in cache:
    if "object_prompts" not in cache[obj_key]:
        object_input_dir = os.path.join(base_dir, obj_key, "input")

        prompts = run_prompt_gui(object_input_dir)
        cache[obj_key]["object_prompts"] = prompts

# Generate masks for each object
for obj_key in cache:

    object_dir = os.path.join(base_dir, obj_key)
    # Get masks for the object directory and its background
    if not os.path.isdir(os.path.join(object_dir, "images")):
        print(bcolors.OKCYAN + f"Getting masks for object {obj_key}" + bcolors.ENDC)
        get_object_masks(object_dir, cache[obj_key]["object_prompts"], background_masks_dir)

        print(bcolors.OKGREEN + f"Masks successfully created for object {obj_key}" + bcolors.ENDC)
    else:
        print(bcolors.OKGREEN + f"Masks directory already found for object {obj_key}" + bcolors.ENDC)

with open(cache_path, 'w') as f:
    json.dump(cache, f, indent=4)

