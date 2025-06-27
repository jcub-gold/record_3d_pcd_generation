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

def restructure_files(input_dir):
    # Create new subdirectories
    os.makedirs(os.path.join(input_dir, 'calibrated_metadata'), exist_ok=True)
    os.makedirs(os.path.join(input_dir, 'images_depth'), exist_ok=True)
    os.makedirs(os.path.join(input_dir, 'images_original'), exist_ok=True)

    # Move and rename metadata files
    for file_name in os.listdir(input_dir):
        if file_name.startswith('calib_') and file_name.endswith('.json'):
            new_name = file_name.replace('calib_', '')
            shutil.move(
                os.path.join(input_dir, file_name),
                os.path.join(input_dir, 'calibrated_metadata', new_name)
            )

    # Move and rename depth images
    for file_name in os.listdir(input_dir):
        if file_name.startswith('depth_') and file_name.endswith('.exr'):
            new_name = file_name.replace('depth_', '')
            shutil.move(
                os.path.join(input_dir, file_name),
                os.path.join(input_dir, 'images_depth', new_name)
            )

    # Move and rename jpeg images
    for file_name in os.listdir(input_dir):
        if file_name.startswith('frame_') and file_name.endswith('.jpg'):
            new_name = file_name.replace('frame_', '')
            shutil.move(
                os.path.join(input_dir, file_name),
                os.path.join(input_dir, 'images_original', new_name)
            )

    # Rename files in images_resized subdirectory
    resized_dir = os.path.join(input_dir, 'images_resized')
    if os.path.exists(resized_dir):
        for file_name in os.listdir(resized_dir):
            if file_name.startswith('frame_') and file_name.endswith('.jpg'):
                new_name = file_name.replace('frame_', '')
                os.rename(
                    os.path.join(resized_dir, file_name),
                    os.path.join(resized_dir, new_name)
                )

    os.rename(resized_dir, os.path.join(input_dir, 'rgb'))

    # Rename files in optimized_poses subdirectory
    poses_dir = os.path.join(input_dir, 'optimized_poses')
    if os.path.exists(poses_dir):
        for file_name in os.listdir(poses_dir):
            if file_name.startswith('frame_') and file_name.endswith('.json'):
                new_name = file_name.replace('frame_', '')
                os.rename(
                    os.path.join(poses_dir, file_name),
                    os.path.join(poses_dir, new_name)
                )