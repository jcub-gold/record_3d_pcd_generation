import os
import open3d as o3d
import numpy as np
import json
import cv2
from scipy.spatial.transform import Rotation
import OpenEXR
import Imath
from submodules.TRELLIS.trellis.utils import render_utils, postprocessing_utils
from submodules.TRELLIS.trellis.renderers import MeshRenderer, GaussianRenderer
import imageio

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

def save_object(object_output, output_path, object_name="", is_glb=False):
    if object_name:
        object_name += "_"
        
    video = render_utils.render_video(object_output['mesh'][0])['normal']
    imageio.mimsave(os.path.join(output_path, f"{object_name}sample_mesh.mp4"), video, fps=30)
    video = render_utils.render_video(object_output['gaussian'][0])['color']
    imageio.mimsave(os.path.join(output_path, f"{object_name}sample_gs.mp4"), video, fps=30)

    obj = postprocessing_utils.to_glb(
        object_output['gaussian'][0],
        object_output['mesh'][0],
        # Optional parameters
        simplify=0.85,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
        verbose=False
    )

    if not is_glb:
        mesh_path = os.path.join(output_path, f"{object_name}mesh.obj")
    else:
        mesh_path = os.path.join(output_path, f"{object_name}mesh.glb")

    obj.export(mesh_path)