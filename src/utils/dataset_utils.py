import os
import shutil
import json

"""
Function: copy_dataset
----------------------
input_directory: path to the input directory containing images
output_directory: path to the output directory for copied images
dataset_size: number of images to copy

Copies and renames every nth image from the input directory to the output directory,
ensuring lossless quality. Returns a mapping from output indices to input indices.
"""
def copy_dataset(input_directory, output_directory, dataset_size):
    os.makedirs(output_directory, exist_ok=True)

    # Get a sorted list of all image files in the input directory
    frame_names = [
        p for p in os.listdir(input_directory)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG", ".exr"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    frame_correspondance = {}

    # Copy and rename every nth image to the output directory, n is determined by dataset_size parameter
    save_frequency = len(os.listdir(input_directory)) * 1.0/dataset_size
    for index in range(dataset_size):
        src_path = os.path.join(input_directory, frame_names[int(index * save_frequency)])
        dst_path = os.path.join(output_directory, f"{index}.{frame_names[int(index * save_frequency)].split('.')[-1]}")
        frame_correspondance[index] = int(index * save_frequency)

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

    print(f"Made a copy of {dataset_size} files to '{output_directory}' with lossless quality.")

    return frame_correspondance

"""
Function: rewrite_json
----------------------
json_dir: path to the input JSON file
output_dir: path to the output JSON file
frame_correspondance: mapping from output indices to input indices
rewrite_categories: list of categories to rewrite

Rewrites specified categories in the JSON file to match the new frame correspondences.
"""
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


if __name__ == "__main__":
    print()