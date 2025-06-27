import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from submodules.sam2.sam2.build_sam import build_sam2_video_predictor
import torch
import subprocess
sam2_checkpoint = "submodules/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def get_labeled_images(labeled_dir, images_dir):

    # Create a directory of all the labeled images to prompt SAM to create masks
    frame_names = [
        p for p in os.listdir(images_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    os.makedirs(labeled_dir, exist_ok=True)

    for index, frame in enumerate(frame_names):
        if ((index % 2) == 0):
            plt.figure(figsize=(9, 6))
            plt.title(f"frame {index}")

            img = Image.open(f"{images_dir}/{frame}")
            width, height = img.size
            plt.imshow(img)
            plt.xticks(range(0, width, 100))
            plt.yticks(range(0, height, 100))
            plt.grid()
            plt.savefig(f"{labeled_dir}/{frame}")
            plt.close()

def get_masks(images_dir, object_prompts):
    video_segments = {}  # video_segments contains the per-frame segmentation results
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    inference_state = predictor.init_state(video_path=images_dir)

    for index in range(len(os.listdir(images_dir))):
        video_segments[index] = {}
        video_segments[index][1] = torch.zeros(inference_state["video_height"], inference_state["video_width"], dtype=torch.bool).cpu().numpy()
        video_segments[index][-1] = torch.ones(inference_state["video_height"], inference_state["video_width"], dtype=torch.bool).cpu().numpy()

    for frame, prompts in object_prompts:
        
        predictor.reset_state(inference_state)
        ann_obj_id = 1
    
        prompt = np.array(prompts[1::], dtype=np.float32)
        labels = np.array([1] * len(prompt), np.int32)

        for i, (x, y) in enumerate(prompt):
            if x < 0 and y < 0:
                labels[i] = -1

        if prompts[0] == 0:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame,
                obj_id=ann_obj_id,
                box=prompt,
            )
        else:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame,
                obj_id=ann_obj_id,
                points=prompt,
                labels = labels,
            )         
    
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):       
            if (prompts[0] == 1):    
                video_segments[out_frame_idx][1] =  video_segments[out_frame_idx][1] | (out_mask_logits > 0.0).cpu().numpy()
                video_segments[out_frame_idx][-1] = video_segments[out_frame_idx][-1] & (out_mask_logits <= 0.0).cpu().numpy()
            else:
                video_segments[out_frame_idx][1] =  video_segments[out_frame_idx][1] & (out_mask_logits <= 0.0).cpu().numpy()

    return video_segments, inference_state

def load_masks(images_dir):
    filepath = os.path.join(images_dir, "sam_masks.npz")
    loaded_data = np.load(filepath, allow_pickle=True)
    
    nested_dict = {}
    for compound_key, array in loaded_data.items():
        # Split the compound key back into outer and inner keys
        outer_key, inner_key = compound_key.split('/')
        if outer_key not in nested_dict:
            nested_dict[int(outer_key)] = {}
        nested_dict[int(outer_key)][int(inner_key)] = array
    
    print(f"Dictionary loaded from {filepath}.")
    return nested_dict

def standardize_names(images_dir, frame_names):

    # Rename each file
    for i, file in enumerate(frame_names):
        old_path = os.path.join(images_dir, file)
        new_path = os.path.join(images_dir, f"{i}.jpg")
        os.rename(old_path, new_path)

def unstandardize_names(output_dir, frame_names, extension):

    for i, file in reversed(list(enumerate(frame_names))):
        old_path = os.path.join(output_dir, f"{i}{extension}")
        new_path = os.path.join(output_dir, file.split(".")[0] + extension)
        os.rename(old_path, new_path)

# Create video from a series of frames in {output_dir}/images
def create_video(base_dir, fps=30):
    images_dir = os.path.join(base_dir, "images")
    output_dir = os.path.join(base_dir, "output")

    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

    output_video = os.path.join(output_dir, "masked_video.mp4")
    # Create a video using ffmpeg
    try:
        # Run ffmpeg command quietly
        with open(os.devnull, 'w') as devnull:
            subprocess.run([
                'ffmpeg', '-y',  # Overwrite output file if it exists
                '-framerate', str(fps),  # Set frame rate
                '-i', os.path.join(images_dir, '%d.png'),  # Input images pattern
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                output_video
            ], stdout=devnull, stderr=devnull, check=True)
        print(f"Video successfully created at: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during video creation: {e}")
        

def save_masked_image_as_rgba(image, mask, save_path):

    mask = mask.astype(bool)
    
    # Create an RGBA image
    rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    
    # Set RGB channels
    rgba_image[..., :3] = image  # Copy the RGB channels
    
    # Set alpha channel (255 for True, 0 for False)
    rgba_image[..., 3] = mask.astype(np.uint8) * 255
    
    # Set background to black where mask is False
    rgba_image[~mask, :3] = 0
    
    # Save the image using Pillow
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    Image.fromarray(rgba_image).save(save_path)

def get_object_masks(object_dir, prompts, background_dir = ""):

    object_input_dir = os.path.join(object_dir, "input")
    object_output_dir = os.path.join(object_dir, "images")
    os.makedirs(object_output_dir, exist_ok=True)

    #  scan all names to convert back to the original name
    frame_names = [
        p for p in os.listdir(object_input_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    standardize_names(object_input_dir, frame_names)

    if "sam_masks.npz" in os.listdir(object_input_dir):
        video_segments = load_masks(object_input_dir)
    else:
        video_segments, inference_state = get_masks(object_input_dir, prompts)

        for index in video_segments:
            mask = video_segments[index][1][0][0]
            image = np.array(Image.open(f"{object_input_dir}/{index}.jpg"))
            masked_image = np.where(mask[..., None], image, 0)  # Add channel dimension for broadcasting

            save_masked_image_as_rgba(masked_image, mask, f"{object_output_dir}/{index}.png")            

    create_video(object_dir)

    print(f"Masks saved to {object_output_dir}")

    unstandardize_names(object_output_dir, frame_names, ".png")
    unstandardize_names(object_input_dir, frame_names, ".jpg")
