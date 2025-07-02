import argparse
import os

def create_scene_dirs(scene_name):
    base_path = os.path.join("data", scene_name)
    subdirs = ["record3d_input", "multiview", "output"]

    for subdir in subdirs:
        path = os.path.join(base_path, subdir)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")
        
def create_multiview_dirs(scene_name, num_objects, num_states):
    input_dir = os.path.join("data", scene_name, "multiview")
    os.makedirs(input_dir, exist_ok=True)

    for obj in range(1, num_objects + 1):
        object_dir = os.path.join(input_dir, f"object_{obj}")
        os.makedirs(object_dir, exist_ok=True)

        generation_dir = os.path.join(object_dir, "generation_state")
        os.makedirs(generation_dir, exist_ok=True)

        for state in range(1, num_states + 1):
            state_dir = os.path.join(object_dir, f"state_{state}")
            os.makedirs(state_dir, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create directory structure for a new scene")
    parser.add_argument("--scene_name", required=True, help="Name of the scene to create directories for")
    parser.add_argument("--num_objects", default=1, help="Number of objects in the scene")
    parser.add_argument("--num_states", default=1, help="Number of states for each object in the scene")

    args = parser.parse_args()
    create_scene_dirs(args.scene_name)
    create_multiview_dirs(args.scene_name, int(args.num_objects), int(args.num_states))