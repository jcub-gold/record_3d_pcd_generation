from src.utils.AddRealisticMesh import AddRealisticMesh as ARM
import argparse
import os
import json
import numpy as np
import shutil
from src.utils.clustering_utils import assign_missing_meshes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace simple primitives with realistic mesh")
    parser.add_argument("--scene_name", required=True, help="Name of the scene")

    args = parser.parse_args()
    scene_name = args.scene_name

    input_mesh_paths = f"data/{scene_name}/output/"
    urdf_path = f"scenes/{scene_name}/{scene_name}.urdf"
    asset_info_path = f'data/{scene_name}/cached_asset_info.json'

    with open(asset_info_path) as f:
        asset_info = json.load(f)
    assert isinstance(asset_info, dict), f"Expected dict, got {type(asset_info)}"
    assign_missing_meshes(asset_info, input_mesh_paths)

    mesh_paths = []
    for _, dirnames, _ in os.walk(input_mesh_paths):
        for dirname in dirnames:
            if dirname.startswith("object_"):
                for root, _, filenames in os.walk(os.path.join(input_mesh_paths, dirname)):
                    for filename in filenames:
                        if filename.endswith(".obj"):
                            mesh_paths.append(os.path.join(root, filename))
    


    with open(asset_info_path) as f:
        asset_names = json.load(f)

    for obj in asset_names.keys():
        if obj != "object_10":
            continue
        print(f"Replacing simple geometry for {obj}...")
        # if obj != "object_8":
        #     continue
        if 'drawer' in asset_names[obj]:
            link_name = asset_names[obj] + "_drawer_0_0"
        elif 'cabinet' in asset_names[obj]:
            link_name = asset_names[obj] + "_door_0_0"
        elif 'box' in asset_names[obj]:
            link_name = asset_names[obj]
        else:
            continue

        mesh_path = ""
        for path in mesh_paths:
            if f"{obj}/" in path:
                mesh_path = path
                break
        if mesh_path == "":
            continue

        asset = ARM(urdf_path, mesh_path, link_name)
        asset.set_urdf()
        asset.set_mesh()
        asset.align_mesh()

        output_mesh = asset.get_mesh()
        output_mesh_base_dir = f"scenes/{scene_name}/{obj}"
        os.makedirs(output_mesh_base_dir, exist_ok=True)
        output_mesh_path = f"scenes/{scene_name}/{obj}/{obj}_mesh.obj"
        output_mesh.export(output_mesh_path)

        output_mesh_path_ref = f"{obj}/{obj}_mesh.obj"
        asset.replace_geometry(input_urdf=urdf_path, output_urdf=urdf_path, mesh_path=output_mesh_path_ref)
    
    print("[DONE]")

