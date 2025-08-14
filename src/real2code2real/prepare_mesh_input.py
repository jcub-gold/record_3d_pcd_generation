import os
import shutil
from argparse import ArgumentParser
from src.utils.mask_utils import get_object_masks
import subprocess
import sys
import glob
import re
from src.utils.simulate_frame_caching_utils import select_evenly_spaced
from src.guis.point_prompting_gui import run_prompt_gui
import json

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

def main():
    parser = ArgumentParser("Get masks for thorough object scans")
    parser.add_argument("--scene_name", "-s", required=True, type=str)
    args = parser.parse_args(sys.argv[1:])

    base_dir = f"data/{args.scene_name}/multiview"
    prompts_cache_path = os.path.join(base_dir, "prompts_by_name.json")

    prepared = []             # list of (dirname, obj_dir) that have extracted frames
    prompts_by_name = {}      # dirname -> object_prompts

    # Try loading existing cache
    if os.path.isfile(prompts_cache_path):
        with open(prompts_cache_path, "r") as f:
            prompts_by_name = json.load(f)
        print(f"[CACHE] Loaded prompts from {prompts_cache_path}")

    # -------------------------
    # PASS 1: extract frames for ALL objects first
    # -------------------------
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
            print(f"[EXTRACT] Frames -> {input_dir}")
            prepared.append((dirname, obj_dir))

    # -------------------------
    # PASS 2: collect prompts for ALL objects
    # -------------------------
    changed = False
    for dirname, obj_dir in prepared:
        input_dir = os.path.join(obj_dir, "input")
        if not os.path.isdir(input_dir):
            print(f"[SKIP] No input frames in {obj_dir}; did extraction fail?")
            continue

        if dirname in prompts_by_name:
            print(f"[CACHE] Using cached prompts for {dirname}")
            continue

        # Need to collect prompts
        object_prompts = run_prompt_gui(input_dir)
        prompts_by_name[dirname] = object_prompts
        changed = True
        print(f"[SAVE] Collected prompts for {dirname}")

    # Save updated cache if needed
    if changed:
        with open(prompts_cache_path, "w") as f:
            json.dump(prompts_by_name, f, indent=2)
        print(f"[CACHE] Updated {prompts_cache_path}")

    # -------------------------
    # PASS 3: run segmentations + evenly-spaced selection
    # -------------------------
    for dirname, obj_dir in prepared:
        if dirname not in prompts_by_name:
            print(f"[SKIP] {dirname}: no prompts collected")
            continue

        object_prompts = prompts_by_name[dirname]

        out_dir = os.path.join(obj_dir, "images")
        os.makedirs(out_dir, exist_ok=True)
        get_object_masks(obj_dir, object_prompts, out_dir)

        generation_state_path = os.path.join(obj_dir, "generation_state")
        masked = _list_frames(out_dir)
        frames = select_evenly_spaced(masked[1:-1]) if len(masked) > 2 else masked

        if os.path.isdir(generation_state_path):
            shutil.rmtree(generation_state_path)
        os.makedirs(generation_state_path, exist_ok=True)

        for src in frames:
            dst = os.path.join(generation_state_path, os.path.basename(src))
            shutil.copy2(src, dst)

        print(f"[GEN] Copied {len(frames)} frames to: {generation_state_path}")



if __name__ == "__main__":
    main()