import os
import shutil
from argparse import ArgumentParser
from src.utils.mask_utils import get_object_masks
import subprocess
import sys
import glob
import re
from src.utils.simulate_frame_caching_utils import select_evenly_spaced

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-i", video_path,
        os.path.join(output_dir, "%04d.jpg")
    ])

def _frame_num_from_name(path):
    # works for frame_00012.jpg / 00012.png / etc.
    m = re.search(r'(\d+)(?=\.(?:jpe?g|png)$)', os.path.basename(path), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse frame index from: {path}")
    return int(m.group(1))

def _list_frames(input_dir):
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(input_dir, pat)))
    frames = sorted(files, key=_frame_num_from_name)
    if not frames:
        raise FileNotFoundError(f"No image frames (.jpg/.jpeg/.png) found in {input_dir}")
    return frames

def _parse_prompt_line(line, first_frame_idx):
    toks = line.strip().split()

    # Must have at least one point: start, type, x1, y1  ->  >= 4 tokens
    if len(toks) < 4 or ((len(toks) - 2) % 2 != 0):
        raise ValueError(
            "Expected: <starting_frame> <prompt_type> <x1> <y1> [<x2> <y2> ...]"
        )

    starting_frame = int(toks[0])
    prompt_type    = int(toks[1])
    coords = list(map(int, toks[2:]))
    points = [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]

    starting_offset = starting_frame - first_frame_idx
    if starting_offset < 0:
        raise ValueError(
            f"Starting frame {starting_frame} is before first available frame {first_frame_idx}"
        )

    prompts = [prompt_type] + points
    return (starting_offset, prompts)

def _collect_prompts(first_frame_idx, dirname):
    object_prompts = []
    while True:
        line = input(f"For part of {dirname}, enter the its first frame appearance, the type of prompt, and coordinates. Press return once done: ").strip()
        if not line:
            break
        tup = _parse_prompt_line(line, first_frame_idx)
        object_prompts.append(tup)

    if not object_prompts:
        raise ValueError("No prompts provided.")
    return object_prompts

def main():
    parser = ArgumentParser("Get masks for thorough object scans")
    parser.add_argument("--scene_name", "-s", required=True, type=str)
    args = parser.parse_args(sys.argv[1:])

    base_dir = f"data/{args.scene_name}/multiview"

    for _, dirnames, _ in os.walk(base_dir):
        for dirname in dirnames:
            if not dirname.startswith("object_"):
                continue

            obj_dir = os.path.join(base_dir, dirname)

            video_path = None
            for root, _, files in os.walk(obj_dir):
                for fn in files:
                    if fn.lower().endswith((".mov", ".mp4")):
                        video_path = os.path.join(root, fn)
                        break
                if video_path:
                    break

            if not video_path:
                print(f"[SKIP] No video found in {obj_dir}")
                continue

            input_dir = os.path.join(obj_dir, "input")
            if os.path.isdir(input_dir):
                shutil.rmtree(input_dir)
            extract_frames(video_path, input_dir)

            frame_paths = _list_frames(input_dir)
            first_idx = _frame_num_from_name(frame_paths[0])

            object_prompts = _collect_prompts(first_idx, dirname)
            out_dir = os.path.join(obj_dir, "images")
            os.makedirs(out_dir, exist_ok=True)
            get_object_masks(obj_dir, object_prompts, out_dir)

            generation_state_path = os.path.join(obj_dir, "generation_state")
            frames = select_evenly_spaced(_list_frames(out_dir)[1: -1])

            if os.path.isdir(generation_state_path):
                shutil.rmtree(generation_state_path)
            os.makedirs(generation_state_path, exist_ok=True)

            for src in frames:
                dst = os.path.join(generation_state_path, os.path.basename(src))
                shutil.copy2(src, dst)

            print(f"[GEN] Copied {len(frames)} frames to: {generation_state_path}")


if __name__ == "__main__":
    main()