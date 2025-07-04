import open3d as o3d
import os
# from src.utils.align_utils import remove_outliers_largest_cluster
import re
import copy

# def remove_outliers(pcd):
#     pcd = remove_outliers_largest_cluster(pcd)
#     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=3)
#     pcd = pcd.select_by_index(ind)
#     return pcd

def extend_pcd(aa_pcds_path):
    pcds = []
    paths = []
    combined_pcd = o3d.geometry.PointCloud()
    for dirpath, dirnames, filenames in os.walk(aa_pcds_path):
        for filename in filenames:
            pcd_path = os.path.join(dirpath, filename)
            combined_pcd += o3d.io.read_point_cloud(pcd_path)
            pcd = o3d.io.read_point_cloud(pcd_path)
            pcds.append(pcd)
            paths.append(pcd_path)

            combined_pcd += pcd
    
    extension_distance = (combined_pcd.get_max_bound()[0] - combined_pcd.get_min_bound()[0]) * 0.95


    for i in range(1, 3):
        for j in range(len(pcds)):
            pcd = copy.deepcopy(pcds[j])
            pcd_path = paths[j]
            pcd.translate([extension_distance * i, 0, 0])

            match = re.search(r'object_(\d+)', pcd_path)
            assert match is not None, f"Filename {filename} does not match expected pattern."
            object_number = int(match.group(1))

            new_path = pcd_path.replace(f"{object_number}", f"{object_number + len(pcds) * i}")
            o3d.io.write_point_cloud(new_path, pcd)
    
if __name__ == "__main__":
    aa_pcds_path = "data/basement_base_cabinet_extension/aa_pcds"
    extend_pcd(aa_pcds_path)


    

    