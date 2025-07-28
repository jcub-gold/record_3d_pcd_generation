import os
import logging
from argparse import ArgumentParser
import shutil
from PIL import Image
import json

def copy_dataset(input_directory, output_directory, dataset_size):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get a sorted list of all image files in the input directory
    # Get all image files in the folder
    frame_names = [
        p for p in os.listdir(input_directory)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG", ".exr"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    frame_correspondance = {}

    # Copy and rename every nth image to the output directory
    save_frequency = len(os.listdir(input_directory)) * 1.0/dataset_size
    for index in range(dataset_size):
        src_path = os.path.join(input_directory, frame_names[int(index * save_frequency)])
        dst_path = os.path.join(output_directory, f"{index}.{frame_names[int(index * save_frequency)].split('.')[-1]}")
        frame_correspondance[index] = int(index * save_frequency)

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

    print(f"Made a copy of {dataset_size} files to '{output_directory}' with lossless quality.")

    return frame_correspondance

def rewrite_json(json_dir, output_dir, frame_correspondance, rewrite_categories):
    with open(json_dir, 'r') as file:
        data = json.load(file)
    
    new_json = {}

    for key in data:
        if key not in rewrite_categories:
            new_json[key] = data[key]
            continue

        new_values = []
        for new_frame in frame_correspondance:
            new_values.append(data[key][frame_correspondance[new_frame]])
        
        new_json[key] = new_values
    
    with open(output_dir, 'w') as file:
        json.dump(new_json, file)

def copy_cached_frames(scene_name):
    record3d_input_path = f"data/{scene_name}/record3d_input/"
    multiview_output_path = f"data/{scene_name}/multiview/"
    # dests = ['generation_state', 'state_1']
    dests = ['generation_state']

    num_objects = 0
    for dirpath, dirnames, filenames in os.walk(record3d_input_path):
        for dirname in dirnames:
            if dirname.startswith("object_"):
                num_objects += 1

    with open(f'data/{scene_name}/cached_frames.json') as f:
        cache = json.load(f)

    for obj in range(1, num_objects + 1):
        images_dir = f"data/{scene_name}/record3d_input/object_{obj}/images"
        frame_indices = cache[f'object_{obj}']

        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.split('.')[0] in frame_indices:
                    source_path = os.path.join(root, file)
                    for dest in dests:
                        dest_path = os.path.join(multiview_output_path, f"object_{obj}",dest, file)
                        shutil.copy2(source_path, dest_path)

if __name__ == "__main__":
    copy_cached_frames("basement_test")
    copy_cached_frames("second_floor_test")