import open3d as o3d

path = "data/basement_test_2/output/object_9/object_9_mesh.obj"

mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)  # loads .mtl/texture if present
if mesh.is_empty():
    raise RuntimeError("Failed to load mesh")

mesh.compute_vertex_normals()  # nice lighting
o3d.visualization.draw_geometries(
    [mesh],
    window_name="Open3D Viewer",
    width=1280, height=960,
    mesh_show_back_face=True,   # helpful for thin surfaces
)
