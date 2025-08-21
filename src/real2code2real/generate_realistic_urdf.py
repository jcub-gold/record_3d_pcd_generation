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
        asset.extract_corners(sample_count=200000, weight_y_axis=0.05)
        asset.warp()
        # asset.debug_visualize(show_obj=True, show_urdf=True, show_warped=True, show_points=True, show_aabb=False)
        
        warped = asset.get_warped_mesh().copy()

        warped = asset.get_mesh().copy()
        inv_T = np.linalg.inv(asset.get_urdf_transformations())
        warped.apply_transform(inv_T)

        warped_mesh_path = f"scenes/{scene_name}/{obj}_mesh.obj"
        warped.export(warped_mesh_path)

        asset.replace_geometry(input_urdf=urdf_path, output_urdf=urdf_path, mesh_path=warped_mesh_path)
