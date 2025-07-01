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