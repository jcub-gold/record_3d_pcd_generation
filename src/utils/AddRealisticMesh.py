import trimesh
import numpy as np
import xml.etree.ElementTree as ET
from itertools import product
from src.utils.obb_utils import get_mesh_rotation_from_aabb_min_xz
from scipy.spatial import cKDTree

"""
Class: AddRealisticMesh
----------------------
urdf_path: path to the URDF file
mesh_path: path to the mesh file
urdf_link_name: name of the URDF link to modify

Provides methods to load URDF and mesh, aligns the realistics mesh to the URDF mesh
vias rotating and scaliung.
"""
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

    """"
    Function: set_urdf
    ------------------
    Loads the URDF file and extracts the box geometry for the specified link.
    Applies any origin transforms from the URDF.
    """
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

    """
    Function: set_mesh
    ------------------
    Loads the mesh from the specified mesh_path.
    """
    def set_mesh(self):
        mesh = trimesh.load(self.mesh_path)
        self.mesh = mesh

    """
    Function: _center_meshes
    ------------------------
    Centers both the mesh and URDF box mesh on the world origin using their AABB centers,
    saving the urdf transformation to later apply it's inverse the the realistic mesh.
    """
    def _center_meshes(self):
        if self.mesh is None:
            raise RuntimeError("Call set_mesh() first.")
        if self.urdf is None:
            raise RuntimeError("Call set_urdf() first.")
        mesh_aligned = self.mesh.copy()
        mesh_centered, T = _center_on_origin(mesh_aligned)
        self.mesh = mesh_centered.copy()
        self.urdf, T = _center_on_origin(self.urdf.copy())
        self.urdf_transformations = T

    """
    Function: align_mesh
    --------------------
    deg_threshold: threshold in degrees for rotation snapping, rotations below this are set to zero
    sample_count: number of samples for chamfer distance refinement

    Aligns the mesh to the URDF box using axis alignment, scaling, and chamfer refinement.
    """
    def align_mesh(self, deg_threshold=5.0, sample_count=100000):
        mesh = self.mesh
        urdf = self.urdf


        theta = get_mesh_rotation_from_aabb_min_xz(mesh)
        phi = _reduce_mod_90(theta)
        if abs(np.degrees(phi)) < deg_threshold:
            phi = 0.0
        # print(phi)

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

    """
    Function: replace_geometry
    -------------------------
    input_urdf: path to input URDF file
    output_urdf: path to output URDF file
    mesh_path: path to the realistic mesh file

    Replaces the geometry of the specified URDF link with a realistic mesh and a simplified collision box.
    """
    def replace_geometry(self, input_urdf: str, output_urdf: str, mesh_path: str):
        tree = ET.parse(input_urdf)
        root = tree.getroot()
        link = root.find(f".//link[@name='{self.urdf_link_name}']") # Find the link by name
        if link is None:
            raise RuntimeError(f"Could not find link '{self.urdf_link_name}' in URDF.")

        # Gather all <collision><geometry><box size="..."> elements under this link
        collisions = link.findall("collision")
        if not collisions:
            raise RuntimeError(f"No <collision> elements found under link '{self.urdf_link_name}'.")

        all_mins = []
        all_maxs = []

        for col in collisions:
            geom = col.find("geometry")
            if geom is None:
                continue
            box = geom.find("box")
            if box is None or "size" not in box.attrib:
                continue
            sx, sy, sz = map(float, box.get("size").split())
            origin_tag = col.find("origin")
            if origin_tag is not None and "xyz" in origin_tag.attrib:
                ox, oy, oz = map(float, origin_tag.get("xyz").split())
            else:
                ox, oy, oz = 0.0, 0.0, 0.0
            half = np.array([sx / 2.0, sy / 2.0, sz / 2.0])
            origin = np.array([ox, oy, oz])
            min_corner = origin - half
            max_corner = origin + half

            all_mins.append(min_corner)
            all_maxs.append(max_corner)

        if not all_mins:
            raise RuntimeError(f"Could not parse any <box size='...'> under collisions of '{self.urdf_link_name}'.")

        all_mins = np.vstack(all_mins)
        all_maxs = np.vstack(all_maxs)
        global_min = all_mins.min(axis=0)
        global_max = all_maxs.max(axis=0)
        total_size = global_max - global_min  # [dx, dy, dz]
        box_size_str = f"{total_size[0]:.6f} {total_size[1]:.6f} {total_size[2]:.6f}"

        # Remove all existing <visual> and <collision> nodes under this link
        for child in list(link.findall("visual")):
            link.remove(child)
        for child in list(link.findall("collision")):
            link.remove(child)

        # 1) Add new <visual> for the realistic mesh
        visual = ET.SubElement(link, "visual")
        geometry = ET.SubElement(visual, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        mesh.set("filename", mesh_path)

        material = ET.SubElement(visual, "material") 
        material.set("name", f"{self.urdf_link_name}_material")
        color = ET.SubElement(material, "color")
        color.set("rgba", "0.8 0.8 0.8 1.0") # TODO: color can be input parameter, with the average color from segmented input image (might need an entirely different function to do this for the whole object)

        # 2) Add a single <collision> with a <box> of the computed size
        collision = ET.SubElement(link, "collision")
        geometry = ET.SubElement(collision, "geometry")
        box = ET.SubElement(geometry, "box")
        box.set("size", box_size_str)

        center = (global_min + global_max) / 2.0  # [cx, cy, cz]
        origin = ET.SubElement(collision, "origin")
        origin.set("xyz", f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}")
        origin.set("rpy", "0 0 0")
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

# ————————————————
# Helper functions
# ————————————————
"""
Function: to_homogeneous
-----------------------
mat3x3: 3x3 numpy array representing a rotation matrix

Returns a 4x4 homogeneous transformation matrix with the input rotation.
"""
def to_homogeneous(mat3x3):
    mat4x4 = np.eye(4)
    mat4x4[:3, :3] = mat3x3
    return mat4x4

"""
Function: _match_axis_aligned_extents
-------------------------------------
mesh_a: trimesh.Trimesh object to be rotated/scaled
mesh_b: trimesh.Trimesh object to match extents to

Returns a 4x4 permutation matrix to align mesh_a's extents with mesh_b's by aligning mesh a's
largest to smallest dimensions with mesh b's.
"""
def _match_axis_aligned_extents(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> trimesh.Trimesh:
    extents_a = mesh_a.extents
    extents_b = mesh_b.extents
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

"""
Function: _center_on_origin
---------------------------
mesh: trimesh.Trimesh object to be centered

Returns a copy of the mesh translated so its AABB center is at the origin,
and the transformation matrix used.
"""
def _center_on_origin(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    aabb = mesh.bounds   # shape (2,3): [ [min_x, min_y, min_z], [max_x, max_y, max_z] ]
    min_corner = aabb[0]
    max_corner = aabb[1]
    center = (min_corner + max_corner) * 0.5
    T = np.eye(4)
    T[:3, 3] = -center
    recentred = mesh.copy()
    recentred.apply_transform(T)
    return recentred, T

"""
Function: _reduce_mod_90
------------------------
theta_rad: angle in radians

Returns the angle reduced modulo 90 degrees, centered around zero.
"""
def _reduce_mod_90(theta_rad: float) -> float:
    half_pi = 0.5 * np.pi
    return ((theta_rad + np.pi/4) % half_pi) - np.pi/4

"""
Function: _points_on_mesh
-------------------------
mesh: trimesh.Trimesh object
n_samples: number of points to sample
use_vertices: if True (default false), use mesh vertices instead of sampling

Returns a numpy array of sampled points on the mesh surface.
"""
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

"""
Function: _query_nn
-------------------
tree: cKDTree object
X: Nx3 numpy array of query points

Returns the nearest neighbor distances from X to the tree points.
"""
def _query_nn(tree, X):
    # Use multi-threading if available; gracefully fall back if not
    try:
        return tree.query(X, k=1, workers=-1)[0]
    except TypeError:
        return tree.query(X, k=1)[0]

"""
Function: _chamfer_distance
---------------------------
mesh_a: trimesh.Trimesh object
mesh_b: trimesh.Trimesh object
n_samples: number of points to sample
squared: if True (defualt True), use squared distances

Returns the mean bidirectional chamfer distance between two meshes.
"""
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
    
"""
Function: _refine_rotation_all_90_axes
--------------------------------------
mesh: trimesh.Trimesh object (already centered and aligned)
urdf: trimesh.Trimesh object (target)
sample_count: number of samples for chamfer distance

Returns the best sign-flip rotation matrix (4x4) to minimize chamfer distance. This includes reflections.
"""
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

"""
Function: debug_visualize_trimesh
---------------------------------
meshes: list of trimesh.Trimesh objects
center: if True, center each mesh at the origin

Visualizes the provided meshes in a trimesh.Scene.
"""
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
