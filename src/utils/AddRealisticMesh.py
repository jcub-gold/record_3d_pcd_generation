import trimesh
import numpy as np
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from trimesh.boolean import intersection
from itertools import permutations, product
from scipy.optimize import minimize
import copy
from src.utils.obb_utils import get_mesh_rotation_from_aabb_min_xz
from trimesh import transformations as tf
from scipy.spatial import cKDTree
import open3d as o3d

def to_homogeneous(mat3x3):
    mat4x4 = np.eye(4)
    mat4x4[:3, :3] = mat3x3
    return mat4x4

def _match_axis_aligned_extents(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Rotates and scales mesh_a so its extents align with mesh_b's, assuming both are axis-aligned.
    This permutes mesh_a's axes (no OBB or PCA needed).
    """

    # Get unsorted extents (lengths along x/y/z)
    extents_a = mesh_a.extents
    extents_b = mesh_b.extents
    # Get axis sort orders: indices that would sort from largest to smallest
    order_a = np.argsort(extents_a)[::-1]  # e.g. [2, 0, 1]
    order_b = np.argsort(extents_b)[::-1]  # e.g. [1, 0, 2]
    # Compute permutation matrix: which axes in A map to which in B
    rot = np.zeros((3, 3))
    for i in range(3):
        src = order_a[i]
        dst = order_b[i]
        rot[dst, src] = 1
    rot4 = to_homogeneous(rot)

    return rot4

class AddRealisticMesh:
    def __init__(self, urdf_path, mesh_path, urdf_link_name):
        self.urdf_path = urdf_path
        self.mesh_path = mesh_path
        self.urdf_link_name = urdf_link_name

        self.urdf = None
        self.mesh = None
        self.urdf_corners = None
        self.mesh_corners = None
        self.warped_mesh = None
        self.warped_mesh_corners = None
        self.urdf_transformations = np.eye(4)

    def set_urdf(self):
        assert (self.urdf_path != None and self.urdf_link_name != None), "initialize urdf path or link name"

        # Parse URDF
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        link = root.find(f"./link[@name='{self.urdf_link_name}']")

        # Collect all box-visuals
        boxes = []
        for visual in link.findall("visual"):
            geo = visual.find("geometry/box")
            if geo is None: 
                continue
            size = np.array(list(map(float, geo.attrib["size"].split())))
            box = trimesh.creation.box(extents=size)
            
            # Get transform from URDF
            origin = visual.find("origin")
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            if origin is not None:
                if "xyz" in origin.attrib:
                    xyz = np.array(list(map(float, origin.attrib["xyz"].split())))
                if "rpy" in origin.attrib:
                    rpy = np.array(list(map(float, origin.attrib["rpy"].split())))
            T = trimesh.transformations.euler_matrix(rpy[0], rpy[1], rpy[2], axes="sxyz")
            T[:3,3] = xyz
            box.apply_transform(T)
            boxes.append(box)

        self.urdf = trimesh.util.concatenate(boxes)

    def set_mesh(self):
        mesh = trimesh.load(self.mesh_path)
        self.mesh = mesh

    def _center_meshes(self):
        """
        1) AABB‐minimize self.mesh via align_mesh(…).
        2) Center that aligned mesh on the world origin by shifting its bbox center to (0,0,0).
        3) Center the URDF box‐mesh on the world origin in the same way.
        """
        if self.mesh is None:
            raise RuntimeError("Call set_mesh() first.")
        if self.urdf is None:
            raise RuntimeError("Call set_urdf() first.")

        # --- 1) AABB‐minimize the mesh ---
        # mesh_aligned, _ = align_mesh(self.mesh)
        # mesh_aligned = self.mesh.copy()
        mesh_aligned = self.mesh.copy()

        # --- 2) Center the aligned mesh on origin ---
        mesh_centered, T = _center_on_origin(mesh_aligned)
        self.mesh = mesh_centered.copy()

        # --- 3) Center the URDF mesh on origin the same way ---
        self.urdf, T = _center_on_origin(self.urdf.copy())
        self.urdf_transformations = T

    def align_mesh(self, deg_threshold=5.0, sample_count=100000):
        mesh = self.mesh
        urdf = self.urdf


        theta = get_mesh_rotation_from_aabb_min_xz(mesh)
        phi = _reduce_mod_90(theta)
        if abs(np.degrees(phi)) < deg_threshold:
            phi = 0.0
        print(phi)

        R = trimesh.transformations.rotation_matrix(-phi, [0, 1, 0])
        mesh.apply_transform(R)
        
        rot4 = _match_axis_aligned_extents(mesh, urdf)

        mesh.apply_transform(rot4)
        scale = urdf.extents / mesh.extents
        mesh.apply_scale(scale)

        self._center_meshes()

        rot4 = _refine_rotation_all_90_axes(self.mesh, self.urdf, sample_count)
        mesh.apply_transform(rot4)

        inv_T = np.linalg.inv(self.get_urdf_transformations())
        mesh.apply_transform(inv_T)

        self.mesh = mesh

    def replace_geometry(self, input_urdf: str, output_urdf: str, mesh_path: str):
        """
        Replace the geometry of the drawer link named by self.urdf_link_name
        with a single realistic mesh, and rebuild a simplified collision box
        whose dimensions are computed from the existing collision boxes under that link.
        """
        # Parse the URDF
        tree = ET.parse(input_urdf)
        root = tree.getroot()

        # Find the drawer link dynamically
        drawer_link = root.find(f".//link[@name='{self.urdf_link_name}']")
        if drawer_link is None:
            raise RuntimeError(f"Could not find link '{self.urdf_link_name}' in URDF.")

        # Gather all <collision><geometry><box size="..."> elements under this link
        collisions = drawer_link.findall("collision")
        if not collisions:
            raise RuntimeError(f"No <collision> elements found under link '{self.urdf_link_name}'.")

        # Compute the axis-aligned bounding box (AABB) that encloses all existing boxes
        all_mins = []
        all_maxs = []

        for col in collisions:
            geom = col.find("geometry")
            if geom is None:
                continue
            box = geom.find("box")
            if box is None or "size" not in box.attrib:
                continue

            # Parse the size "sx sy sz"
            sx, sy, sz = map(float, box.get("size").split())

            # Read the <origin xyz="ox oy oz"/> if present
            origin_tag = col.find("origin")
            if origin_tag is not None and "xyz" in origin_tag.attrib:
                ox, oy, oz = map(float, origin_tag.get("xyz").split())
            else:
                ox, oy, oz = 0.0, 0.0, 0.0

            # Compute that box’s min & max corners in the link’s frame
            half = np.array([sx / 2.0, sy / 2.0, sz / 2.0])
            origin = np.array([ox, oy, oz])
            min_corner = origin - half
            max_corner = origin + half

            all_mins.append(min_corner)
            all_maxs.append(max_corner)

        if not all_mins:
            raise RuntimeError(f"Could not parse any <box size='...'> under collisions of '{self.urdf_link_name}'.")

        # Find global min/max across x,y,z
        all_mins = np.vstack(all_mins)
        all_maxs = np.vstack(all_maxs)
        global_min = all_mins.min(axis=0)
        global_max = all_maxs.max(axis=0)

        # Compute total size of the combined AABB
        total_size = global_max - global_min  # [dx, dy, dz]
        box_size_str = f"{total_size[0]:.6f} {total_size[1]:.6f} {total_size[2]:.6f}"

        # Remove all existing <visual> and <collision> nodes under this link
        for child in list(drawer_link.findall("visual")):
            drawer_link.remove(child)
        for child in list(drawer_link.findall("collision")):
            drawer_link.remove(child)

        # 1) Add new <visual> for the realistic mesh
        visual = ET.SubElement(drawer_link, "visual")
        geometry = ET.SubElement(visual, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        mesh.set("filename", mesh_path)

        # (Optional) attach a simple gray material
        material = ET.SubElement(visual, "material")
        material.set("name", f"{self.urdf_link_name}_material")
        color = ET.SubElement(material, "color")
        color.set("rgba", "0.8 0.8 0.8 1.0")

        # 2) Add a single <collision> with a <box> of the computed size
        collision = ET.SubElement(drawer_link, "collision")
        geometry = ET.SubElement(collision, "geometry")
        box = ET.SubElement(geometry, "box")
        box.set("size", box_size_str)

        # Center the new collision box at the AABB center
        center = (global_min + global_max) / 2.0  # [cx, cy, cz]
        origin = ET.SubElement(collision, "origin")
        origin.set("xyz", f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}")
        origin.set("rpy", "0 0 0")

        # Write updated URDF
        tree.write(output_urdf, encoding="ASCII", xml_declaration=True)

 
    # Get Methods
    def get_urdf(self):
        return self.urdf
    def get_mesh(self):
        return self.mesh
    def get_urdf_corners(self):
        return self.urdf_corners
    def get_mesh_corners(self): 
        return self.mesh_corners
    def get_warped_mesh(self):  
        return self.warped_mesh
    def get_warped_mesh_corners(self):
        return self.warped_mesh_corners
    def get_urdf_transformations(self):
        return self.urdf_transformations

# ——————————————————————————————
# Helper functions (inside the same file)
# ——————————————————————————————
def _center_on_origin(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Translate `mesh` so that its AABB center sits at (0,0,0).
    We compute the axis‐aligned bounding box, take (min + max) / 2 for each axis,
    and subtract that vector from every vertex. Returns a new Trimesh.
    """
    # 1) Compute axis‐aligned bounding box corners
    aabb = mesh.bounds   # shape (2,3): [ [min_x, min_y, min_z], [max_x, max_y, max_z] ]
    min_corner = aabb[0]
    max_corner = aabb[1]

    # 2) Compute center = (min + max) / 2
    center = (min_corner + max_corner) * 0.5

    # 3) Translate every vertex by –center
    T = np.eye(4)
    T[:3, 3] = -center

    recentred = mesh.copy()
    recentred.apply_transform(T)
    return recentred, T

def cumulative_distance_to_mesh(pts: np.ndarray,
                                urdf_mesh: trimesh.Trimesh) -> float:
    closest_pts, distances, face_id = urdf_mesh.nearest.on_surface(pts)
    return float(np.sum(distances))

def _match_axis_aligned_extents(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh):
    """
    Rotates and scales mesh_a so its extents align with mesh_b's, assuming both are axis-aligned.
    This permutes mesh_a's axes (no OBB or PCA needed).
    """

    # Get unsorted extents (lengths along x/y/z)
    extents_a = mesh_a.extents
    extents_b = mesh_b.extents
    # Get axis sort orders: indices that would sort from largest to smallest
    order_a = np.argsort(extents_a)[::-1]  # e.g. [2, 0, 1]
    order_b = np.argsort(extents_b)[::-1]  # e.g. [1, 0, 2]
    # Compute permutation matrix: which axes in A map to which in B
    rot = np.zeros((3, 3))
    for i in range(3):
        src = order_a[i]
        dst = order_b[i]
        rot[dst, src] = 1
    rot4 = to_homogeneous(rot)

    return rot4

def _reduce_mod_90(theta_rad: float) -> float:
    half_pi = 0.5 * np.pi
    return ((theta_rad + np.pi/4) % half_pi) - np.pi/4

def _points_on_mesh(mesh, n_samples=20000, use_vertices=False):
    V = np.asarray(mesh.vertices, dtype=np.float64)
    if use_vertices or getattr(mesh, "faces", None) is None or len(mesh.faces) == 0:
        if V.size == 0:
            raise ValueError("Mesh has no vertices.")
        return V
    try:
        return mesh.sample(int(n_samples))  # trimesh area-weighted sampling
    except Exception:
        if V.size == 0:
            raise
        return V

def _query_nn(tree, X):
    # Use multi-threading if available; gracefully fall back if not
    try:
        return tree.query(X, k=1, workers=-1)[0]
    except TypeError:
        return tree.query(X, k=1)[0]

def _chamfer_distance(mesh_a, mesh_b, n_samples=20000, squared=True, use_vertices=False):
    P = np.ascontiguousarray(_points_on_mesh(mesh_a, n_samples, use_vertices), dtype=np.float64)
    Q = np.ascontiguousarray(_points_on_mesh(mesh_b, n_samples, use_vertices), dtype=np.float64)

    if P.shape[0] == 0 or Q.shape[0] == 0:
        raise ValueError("One of the meshes produced zero points.")

    tree_Q = cKDTree(Q)
    d_ab = _query_nn(tree_Q, P)  # distances P -> Q

    tree_P = cKDTree(P)
    d_ba = _query_nn(tree_P, Q)  # distances Q -> P

    if squared:
        d_ab = d_ab**2
        d_ba = d_ba**2

    return float(d_ab.mean() + d_ba.mean())
    
def _refine_rotation_all_90_axes(mesh, urdf, sample_count: int = 100000):
        """
        Instead of brute‐forcing all 24 axis swaps, we:
        1) Loop over the 4 possible sign‐flip matrices S = diag(±1,±1,±1) with det(S·P)=+1.
        2) For each R = S·P, apply to the OBJ, compute champfer distance.
        """
        if mesh is None or urdf is None:
            raise RuntimeError("Call set_mesh(), set_urdf() first.")

        # --- 3) Build the 8 sign‐flip rotations R = S·P with det(R)=+1 ---
        candidate_Rs = []
        for signs in product([1.0, -1.0], repeat=3):
            R = np.diag(signs)
            candidate_Rs.append(R)

        # At this point, candidate_Rs has exactly 4 matrices (each 3×3 with ±1 entries).

        best_total = np.inf
        best_R = None

        base_mesh = mesh.copy()  # should already be AABB‐aligned + centered

        # --- 4) Test each of the 4 R candidates by sampling and measuring distance to URDF ---
        for idx, R3 in enumerate(candidate_Rs):
            R_hom = np.eye(4, dtype=np.float64)
            R_hom[:3, :3] = R3

            candidate = base_mesh.copy()
            candidate.apply_transform(R_hom)

            total_dist = _chamfer_distance(candidate, urdf, n_samples=sample_count)

            # print(f"  [flip {idx+1}/8] total_dist = {total_dist:.6f}")

            if total_dist < best_total:
                best_total = total_dist
                best_R = R3.copy()

        if best_R is None:
            raise RuntimeError("No valid rotation found among the 4 candidates.")

        # --- 5) Store and report the result ---
        # print(f"refine_rotation_all_90_axes: best total_dist = {best_total:.6f}")
        
        best_R_hom = np.eye(4, dtype=np.float64)
        best_R_hom[:3, :3] = best_R
        return best_R_hom

def debug_visualize_trimesh(meshes, center=True):
    centered_meshes = []
    
    for mesh in meshes:
        mesh_copy = mesh.copy()
        if center:
            # Compute bounding box center
            bbox_center = mesh_copy.bounding_box.centroid
            mesh_copy.apply_translation(-bbox_center)
        centered_meshes.append(mesh_copy)
    
    scene = trimesh.Scene(centered_meshes)
    scene.show()
