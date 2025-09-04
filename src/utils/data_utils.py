import os
import open3d as o3d
import numpy as np
import json
import cv2
from scipy.spatial.transform import Rotation
import OpenEXR
import Imath
import re

"""
    Function: get_number
    --------------------
    word: string containing a number
    
    Returns the integer value of the number found in the string.
"""
def get_number(word):
    numbers = ""
    for char in word:
        if char.isnumeric():
            numbers += char
    
    return int(numbers)

"""
    Function: read_exr_depth
    ------------------------
    exr_path: path to the .exr file containing depth data

    Reads the depth data from an OpenEXR file and returns it as a numpy array.
"""
def read_exr_depth(exr_path):
    exr_file = OpenEXR.InputFile(exr_path)
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.HALF)
    depth_str = exr_file.channel("R", pt)
    depth = np.frombuffer(depth_str, dtype=np.float16).astype(np.float32)
    depth = depth.reshape((height, width))
    return depth

"""
    Function: prepare_record3d_data
    -------------------------------
    images_dir: path to object_{n}/images
    depth_dir: path to input_depth
    metadata_path: path to new_metadata.json

    Sorts the frames based on their number, reads the metadata,
    and prepares the data for each frame including the image, depth map, and extrinsics.
"""
def prepare_record3d_data(images_dir, depth_dir, metadata_path):

    frames = [get_number(os.path.splitext(p)[0]) for p in os.listdir(images_dir)]
    frames.sort()

    with open(metadata_path, 'r') as file:
        metadata_dict = json.load(file)

    poses_data = np.array(metadata_dict["poses"])

    W, H = metadata_dict["w"], metadata_dict["h"]
    K = np.array(metadata_dict["K"]).reshape((3, 3)).T
    focal_length = K[0, 0]

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=W, 
        height=H,
        fx=focal_length,
        fy=focal_length, 
        cx=W/2, 
        cy=H/2 
    )
    
    output = {
        "h": H,
        "w": W,
        "intrinsics": intrinsics,
        "frames": {}
    }

    for frame in frames:
        img_file = os.path.join(images_dir, f"{frame}.png")

        if not os.path.isfile(img_file):
            img_file = os.path.join(images_dir, f"{frame}.jpg")

        exr_file = os.path.join(depth_dir, f"{frame}.exr")

        mask_img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGRA2RGBA)

        depth_img = read_exr_depth(exr_file)
        depth_img = cv2.resize(depth_img, dsize=(mask_img.shape[1], mask_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        extrinsics = np.eye(4)
        rotation = Rotation.from_quat(poses_data[frame][:4]).as_matrix()
        translation = poses_data[frame][4:]
        extrinsics[:3, :3] = rotation
        extrinsics[:3, 3] = translation

        flip_mat = np.eye(4)
        flip_mat[1, 1] = -1
        flip_mat[2, 2] = -1
        extrinsics = flip_mat @ np.linalg.inv(extrinsics)
        
        output["frames"][frame] = [mask_img, depth_img, extrinsics]

    return output



# select n evenly spaced frames from a list, ensuring first and last are included
def select_evenly_spaced(lst, n=5):
    if n < 2 or n > len(lst):
        n=2
    if n < 2 or n > len(lst):
        raise ValueError("n must be at least 2 and at most the length of the list")
    indices = [round(i * (len(lst) - 1) / (n - 1)) for i in range(n)]
    return [lst[i] for i in indices]

# select frames every 10 frames, ensuring first and last are included
def select_custom_frames(lst):
    if len(lst) == 0:
        return []

    lst = sorted([int(f) for f in lst])
    selected = lst[::10]

    # Ensure first and last are included
    if lst[0] not in selected:
        selected.insert(0, lst[0])
    if lst[-1] not in selected:
        selected.append(lst[-1])

    # Remove duplicates and sort
    selected = sorted(set(selected))

    return [str(f) for f in selected]

def extract_frame_numbers(base_dir, custom=False):
    output_path = base_dir + '/cached_frames.json'
    base_dir = base_dir + '/record3d_input'
    result = {}

    # Match directories like object_1, object_2, etc.
    pattern = re.compile(r"^object_\d+$")

    for entry in os.listdir(base_dir):
        obj_dir = os.path.join(base_dir, entry)
        images_dir = os.path.join(obj_dir, "images")

        # Check that it's a directory named object_n with an images/ subdirectory
        if pattern.match(entry) and os.path.isdir(images_dir):
            frame_numbers = []
            for file_name in os.listdir(images_dir):
                if file_name.endswith(".png"):
                    try:
                        frame_num = os.path.splitext(file_name)[0]  # Remove .png
                        int_frame = int(frame_num)  # Validate it's numeric
                        frame_numbers.append(str(int_frame))
                    except ValueError:
                        continue  # Skip if filename is not numeric

            # Replace this line in the original:
            frame_numbers.sort(key=lambda x: int(x))

            # Add this:
            good_frames = frame_numbers[1: -1]
            result[entry] = select_evenly_spaced(good_frames)
            if custom:
                result[entry] = select_custom_frames(frame_numbers)
            
            # if len(frame_numbers) >= 2:
            #     result[entry] = [frame_numbers[1], frame_numbers[-2]]
            # elif frame_numbers:
            #     result[entry] = frame_numbers  # fallback if fewer than 5 frames


    # Save the result to JSON
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)