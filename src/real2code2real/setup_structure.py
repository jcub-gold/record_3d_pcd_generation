import argparse
import os

def create_scene_dirs(scene_name):
    base_path = os.path.join("data", scene_name)
    subdirs = ["record3d_input", "multiview", "output"]

    for subdir in subdirs:
        path = os.path.join(base_path, subdir)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create directory structure for a new scene")
    parser.add_argument("--scene_name", required=True, help="Name of the scene to create directories for")

    args = parser.parse_args()
    create_scene_dirs(args.scene_name)