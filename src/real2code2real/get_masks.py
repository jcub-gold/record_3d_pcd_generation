import os
import shutil
from argparse import ArgumentParser
from src.utils.mask_utils import get_labeled_images, get_object_masks
from src.utils.dataset_utils import copy_dataset, rewrite_json
import time
import sys

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

parser.add_argument("--source_path", "-s", required=True, type = str)
parser.add_argument("--model_path", "-m", type=str, default=None)
parser.add_argument("--dataset_size", default = 1100, type = int)


args = parser.parse_args(sys.argv[1:])

if args.model_path:
    os.makedirs(args.model_path, exist_ok=True)
    base_output_dir = os.path.join(args.model_path, time.strftime("%Y%m%d-%H%M%S"))

base_dir = args.source_path
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

all_object_prompts = {}
frame_intervals = []

# Get information for each object in order  
num_objects = int(input("Enter desired number of objects: "))
for obj in range(num_objects):
    frames = input(f"For the reconstruction interval of object {obj + 1}, enter its first and last frame of appearance: ").split(" ")
    start_frame, end_frame = int(frames[0]), int(frames[1])

    user_input = input(f"For part of object {obj + 1}, enter the its first frame appearance, the type of prompt, and coordinates: ").split(" ")
    object_prompts = []
    frame_intervals.append([start_frame, end_frame])
    while len(user_input) > 1:
        starting_index = int(user_input[0])
        prompts = [int(user_input[1])]

        for i in range(1, len(user_input)//2):
            prompts.append([int(user_input[2 * i]), int(user_input[2 * i + 1])])
        
        object_prompts.append((starting_index - frame_intervals[obj][0], prompts))

        user_input = input(f"For part of object {obj + 1}, enter the its first frame appearance, the type of prompt, and coordinates. Press return once done: ").split(" ")

    all_object_prompts[obj] = object_prompts


# Create new directory and calculate masks for each object
for obj in range(num_objects):

    # Split images into separate directories
    object_dir = os.path.join(base_dir, f"object_{obj + 1}")
    os.makedirs(object_dir, exist_ok=True)

    object_input_dir = os.path.join(object_dir, "input")
    if not os.path.isdir(object_input_dir):
        print(bcolors.OKCYAN + f"Making copies of input in object directory for object {obj + 1}"+ bcolors.ENDC)
        os.makedirs(object_input_dir, exist_ok=True)

        # Copy existing files to a separate directory
        for file_index in range(frame_intervals[obj][0], frame_intervals[obj][1]):
            file_path = os.path.join(input_dir, f"{file_index}.jpg")
            if os.path.isfile(file_path):
                shutil.copy2(file_path, object_input_dir)
    else:
        print(bcolors.OKGREEN + f"Object input directory already found for object {obj + 1}" + bcolors.ENDC)

    # Get masks for the object directory and its background
    if not os.path.isdir(os.path.join(object_dir, "images")):
        print(bcolors.OKCYAN + f"Getting masks for object {obj + 1}" + bcolors.ENDC)
        get_object_masks(object_dir, all_object_prompts[obj], background_masks_dir)

        print(bcolors.OKGREEN + f"Masks successfully created for object {obj + 1}" + bcolors.ENDC)
    else:
        print(bcolors.OKGREEN + f"Masks directory already found for object {obj + 1}" + bcolors.ENDC)
        

