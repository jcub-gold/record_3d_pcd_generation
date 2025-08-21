import trimesh
import numpy as np
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from trimesh.boolean import intersection
from itertools import permutations, product
from scipy.optimize import minimize
import copy
from src.utils.obb_utils import get_mesh_rotation_from_pca
from trimesh import transformations as tf

def to_homogeneous(mat3x3):
    mat4x4 = np.eye(4)
    mat4x4[:3, :3] = mat3x3
    return mat4x4

def match_axis_aligned_extents(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> trimesh.Trimesh:
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
    # print(order_a, order_b, rot)

    # Apply rotation
    mesh_out = mesh_a.copy()
    mesh_out.apply_transform(rot4)

    # Get sorted extents again (now A is aligned in axis order with B)
    scale = extents_b / mesh_out.extents

    mesh_out.apply_scale(scale)

    return mesh_out

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
        urdf = self.urdf
        # if mesh == None:
        #     return
        mesh, _ = align_mesh(mesh)
        # mesh.export("test.obj")
        # angle = get_mesh_rotation_from_pca(mesh)   # radians
        # R = tf.rotation_matrix(-angle, [0, 1, 0])       # -angle -> align principal dir to +X
        # mesh.apply_transform(R)
        # print(angle)
        # mesh.export("test.obj")
        
        mesh_aligned = match_axis_aligned_extents(mesh, urdf)

        # print(mesh.extents)
        # print(urdf.extents)


        # s = trimesh.Scene()
        # s.add_geometry(mesh)
        # s.add_geometry(mesh_aligned)
        # s.add_geometry(urdf)
        # s.show()
        # self.mesh = obj = trimesh.load(self.mesh_path)
        self.mesh = mesh_aligned


    def extract_corners(self, sample_count: int = 100000, weight_y_axis: float = 5.0, manual_reflection=None):
        self._simple_align_mesh()
        self._refine_rotation_all_90_axes(sample_count=sample_count, manual_reflection=manual_reflection)
        # MESH corners extraction
        if self.mesh is None:
            raise RuntimeError("Call set_mesh() first.")
        try:
            # Project vertices onto XZ plane
            vertices = self.get_mesh().vertices
            xz = vertices[:, [0, 2]]  # columns 0 (x) and 2 (z)
            y = vertices[:, 1]       # Get Y coordinates for scoring
            print(f"Projected {len(xz)} vertices to XZ plane")

            # Split points into quadrants and find furthest point in each
            corners = []

            # Calculate distances from origin for each point in XZ plane
            xz_distances = np.linalg.norm(xz, axis=1)
            # Normalize Y coordinates to [0,1] range for scoring
            y_min, y_max = np.min(y), np.max(y)
            y_normalized = (y_max - y) / (y_max - y_min)
            scores = xz_distances * (1 + weight_y_axis * y_normalized)

            # For each quadrant in XZ plane, find point with highest score
            quadrants = [
                ((xz[:, 0] >= 0) & (xz[:, 1] >= 0)),  # Q1 (x>=0, z>=0)
                ((xz[:, 0] <  0) & (xz[:, 1] >= 0)),  # Q2 (x<0,  z>=0)
                ((xz[:, 0] <  0) & (xz[:, 1] <  0)),  # Q3 (x<0,  z<0)
                ((xz[:, 0] >= 0) & (xz[:, 1] <  0))   # Q4 (x>=0, z<0)
            ]

            for quadrant_mask in quadrants:
                if np.any(quadrant_mask):
                    # Find corner with highest score in this quadrant
                    quadrant_scores = scores * quadrant_mask
                    corner_idx = np.argmax(quadrant_scores)
                    corners.append(vertices[corner_idx])

            print(f"Found {len(corners)} corners by quadrant distance")
            self.mesh_corners = np.array(corners)

        except Exception as e:
            print(f"Error in extract_corners: {e}")
            self.mesh_corners = None


        # URDF corners extraction
        if self.urdf is None:
            raise RuntimeError("Call set_urdf() first.")
        vertices = self.get_urdf().vertices
        
        # Get extremal points
        min_x = np.min(vertices[:, 0])
        max_x = np.max(vertices[:, 0])
        max_y = np.min(vertices[:, 1])  # Get the maximum y-value for the front face
        min_z = np.min(vertices[:, 2])
        max_z = np.max(vertices[:, 2])
        
        # Find actual vertices closest to the extremal combinations
        target_points = np.array([
            [max_x, max_y, max_z],  # Top right
            [min_x, max_y, max_z],  # Top left
            [min_x, max_y, min_z],  # Bottom left
            [max_x, max_y, min_z],  # Bottom right
        ])
        
        corners = []
        for target in target_points:
            # Calculate distances to all vertices
            distances = np.linalg.norm(vertices - target[None, :], axis=1)
            # Get the closest vertex
            closest_idx = np.argmin(distances)
            corners.append(vertices[closest_idx])
        
        self.urdf_corners = np.array(corners)

    def _refine_rotation_all_90_axes(self, sample_count: int = 100000, manual_reflection = None):
        """
        Instead of brute‐forcing all 24 axis swaps, we:
        1) Look at each mesh’s AABB.extents (dx, dy, dz) after AABB‐alignment & centering.
        2) Find the permutation of those three dims that best matches two of the three URDF dims—
            i.e. we compute abs(permuted_mesh_extents - urdf_extents), sort those three differences,
            and take the sum of the two smallest differences as our “score.” This discards the single
            largest‐mismatch dimension as the “outlier.”
        3) Build the corresponding permutation matrix P.
        4) Loop over the 4 possible sign‐flip matrices S = diag(±1,±1,±1) with det(S·P)=+1.
        5) For each R = S·P, apply to the OBJ, sample sample_count points, compute
            sum of nearest‐point distances to URDF, and pick the best among the 4.
        """
        if self.mesh is None or self.urdf is None:
            raise RuntimeError("Call set_mesh(), set_urdf() and simple_align_mesh() first.")

        # --- 1) Get each mesh’s AABB extents (already AABB‐aligned + centered) ---
        mesh_dims = self.mesh.bounding_box.extents   # array([dx, dy, dz])
        urdf_dims = self.urdf.bounding_box.extents   # array([Dx, Dy, Dz])

        # --- 2) Find the permutation of (dx,dy,dz) that best matches (Dx,Dy,Dz) on two dims ---
        best_perm = None
        best_score = np.inf

        # We will measure score(permutation) = sum of the two smallest entries of abs(permuted - urdf_dims).
        for perm in permutations([0, 1, 2], 3):
            permuted = np.array([
                mesh_dims[perm[0]],
                mesh_dims[perm[1]],
                mesh_dims[perm[2]]
            ])
            diffs = np.abs(permuted - urdf_dims)         # length‐3 array of absolute differences
            diffs_sorted = np.sort(diffs)                # sort ascending
            score = diffs_sorted[0] + diffs_sorted[1]    # sum of the two smallest diffs
            if score < best_score:
                best_score = score
                best_perm = perm

        # Build the 3×3 permutation matrix P so that [x';y';z'] = P @ [x;y;z]
        P = np.zeros((3, 3), dtype=np.float64)
        for row in range(3):
            col = best_perm[row]
            P[row, col] = 1.0

        print(f"Chosen permutation (mesh→URDF dims): {best_perm}")
        print(f"Mesh dims permuted: {[mesh_dims[i] for i in best_perm]}, URDF dims: {urdf_dims}")

        # --- 3) Build the 4 sign‐flip rotations R = S·P with det(R)=+1 ---
        candidate_Rs = []
        for signs in product([1.0, -1.0], repeat=3):
            S = np.diag(signs)
            R = S.dot(P)
            candidate_Rs.append(R)

        # At this point, candidate_Rs has exactly 4 matrices (each 3×3 with ±1 entries).

        best_total = np.inf
        best_mesh_aligned = None
        best_R = None

        base_mesh = self.mesh.copy()  # should already be AABB‐aligned + centered

        # --- 4) Test each of the 4 R candidates by sampling and measuring distance to URDF ---
        for idx, R3 in enumerate(candidate_Rs):
            R_hom = np.eye(4, dtype=np.float64)
            R_hom[:3, :3] = R3

            candidate = base_mesh.copy()
            candidate.apply_transform(R_hom)

            pts = candidate.sample(sample_count)            # sample `sample_count` points
            total_dist = cumulative_distance_to_mesh(pts, self.urdf)

            print(f"  [flip {idx+1}/4] total_dist = {total_dist:.6f}")

            if total_dist < best_total:
                best_total = total_dist
                best_mesh_aligned = candidate.copy()
                best_R = R3.copy()

        if best_mesh_aligned is None:
            raise RuntimeError("No valid rotation found among the 4 candidates.")

        # --- 5) Store and report the result ---
        print(f"refine_rotation_all_90_axes: best total_dist = {best_total:.6f}")
        print(f"Best 3×3 rotation matrix:\n{best_R}")

        if manual_reflection is not None:
            # Apply the manual reflection if provided
            best_mesh_aligned.apply_transform(manual_reflection)

        self.mesh = best_mesh_aligned.copy()

    def _simple_align_mesh(self):
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
        mesh_centered, T = center_on_origin(mesh_aligned)
        self.mesh = mesh_centered.copy()

        # --- 3) Center the URDF mesh on origin the same way ---
        self.urdf, T = center_on_origin(self.urdf.copy())
        self.urdf_transformations = T

    def warp(self,
             visualize_corners: bool = False,
             visualize_uv: bool = False,
             visualize_frames: bool = False,
             visualize_warped_corners: bool = False):
        """
        Extended warp() that can draw intermediate steps.
        
        Flags:
          - visualize_corners:   shows 3D scene with red/green spheres + thick yellow tubes
          - visualize_uv:        shows a 2D UV‐scatter (OBJ corners in red, URDF corners in blue)
          - visualize_frames:    shows the local (u,v,n) axes at O_src and O_tgt
          - visualize_warped_corners: plots where the four warped corners land, in 3D, versus the URDF corners 
        """
        assert (
            self.urdf is not None and
            self.mesh is not None and
            self.urdf_corners is not None and
            self.mesh_corners is not None
        ), "Must call set_urdf(), set_mesh(), extract_corners() first."

        def make_frame(corners: np.ndarray):
            """
            Given 4 points (4×3), compute:
              O = centroid,
              u = normalized (corner1 – corner0),
              n = normalized cross((corner2 – corner1), (corner0 – corner1)),
              v = cross(n, u).
            Return O, u, v, n (each shape (3,)).
            """
            O = corners.mean(axis=0)
            u = corners[1] - corners[0]
            u = u / np.linalg.norm(u)
            n = np.cross(corners[2] - corners[1], corners[0] - corners[1])
            n = n / np.linalg.norm(n)
            v = np.cross(n, u)
            return O, u, v, n

        # 1) We assume mesh_corners & urdf_corners are already exactly [BL,BR,TL,TR]
        src_pts = np.array(self.mesh_corners)   # (4×3) array of OBJ’s front‐face corners
        tgt_pts = np.array(self.urdf_corners)   # (4×3) array of URDF’s front‐face corners

        # 2) Visualize the raw corner‐pairs in 3D, if desired
        if visualize_corners:
            scene = trimesh.Scene()

            # a) Draw the URDF box in translucent blue
            urdf_copy = self.urdf.copy()
            urdf_copy.visual.face_colors = [0, 0, 255, 50]
            scene.add_geometry(urdf_copy)

            # b) Draw the OBJ mesh in solid gray
            mesh_copy = self.mesh.copy()
            mesh_copy.visual.face_colors = [200, 200, 200, 255]
            scene.add_geometry(mesh_copy)

            # c) Red spheres at URDF corners
            for c in tgt_pts:
                sph = trimesh.creation.icosphere(radius=0.01)
                sph.apply_translation(c)
                sph.visual.vertex_colors = [255, 0, 0, 255]
                scene.add_geometry(sph)

            # d) Green spheres at OBJ corners
            for c in src_pts:
                sph = trimesh.creation.icosphere(radius=0.01)
                sph.apply_translation(c)
                sph.visual.vertex_colors = [0, 255, 0, 255]
                scene.add_geometry(sph)

            # e) Thick yellow tubes for each matched pair
            for i in range(4):
                p0 = src_pts[i]
                p1 = tgt_pts[i]
                vec = p1 - p0
                length = np.linalg.norm(vec)
                if length < 1e-8:
                    continue

                # Create a cylinder of height=length, radius=0.005
                cyl = trimesh.creation.cylinder(
                    radius=0.005, 
                    height=length, 
                    sections=16
                )

                # Rotate its local +Z to align with the direction (vec/length)
                direction = vec / length
                R = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)
                cyl.apply_transform(R)

                # Translate the cylinder so its midpoint is (p0 + p1)/2
                midpoint = (p0 + p1) * 0.5
                cyl.apply_translation(midpoint)

                # Color it bright yellow
                cyl.visual.vertex_colors = [255, 255, 0, 255]
                scene.add_geometry(cyl)

            # f) Add coordinate axes at origin for reference
            axes = trimesh.creation.axis(origin_size=0.01, axis_length=0.2)
            scene.add_geometry(axes)

            # g) Show the scene and return from visualization (pause here)
            scene.show()
            # If you want the code to pause until the window is closed, you can:
            # input("Press Enter to continue...")

        # 3) Build local frames for the four corners
        O_src, u_src, v_src, n_src = make_frame(src_pts)
        O_tgt, u_tgt, v_tgt, n_tgt = make_frame(tgt_pts)

        # 4) Visualize the local frames if requested
        if visualize_frames:
            scene = trimesh.Scene()

            # Show the URDF box lightly
            urdf_copy = self.urdf.copy()
            urdf_copy.visual.face_colors = [0, 0, 255, 30]
            scene.add_geometry(urdf_copy)

            # Show the OBJ mesh lightly
            mesh_copy = self.mesh.copy()
            mesh_copy.visual.face_colors = [200, 200, 200, 30]
            scene.add_geometry(mesh_copy)

            # (a) Draw a small 3D coordinate triad at O_src
            tri_src = trimesh.creation.axis(origin_size=0.005, axis_length=0.1)
            # But we need to rotate that triad so its x-axis= u_src, y-axis= v_src, z-axis= n_src.
            M_src = np.eye(4)
            M_src[:3,:3] = np.stack([u_src, v_src, n_src], axis=1)  # columns = [u,v,n]
            M_src[:3,3] = O_src
            tri_src.apply_transform(M_src)
            scene.add_geometry(tri_src)

            # (b) Draw a small 3D coordinate triad at O_tgt
            tri_tgt = trimesh.creation.axis(origin_size=0.005, axis_length=0.1)
            M_tgt = np.eye(4)
            M_tgt[:3,:3] = np.stack([u_tgt, v_tgt, n_tgt], axis=1)
            M_tgt[:3,3] = O_tgt
            tri_tgt.apply_transform(M_tgt)
            scene.add_geometry(tri_tgt)

            # (c) Also draw the four corner spheres again so you can see O_src/O_tgt in context
            for c in src_pts:
                s = trimesh.creation.icosphere(radius=0.008)
                s.apply_translation(c)
                s.visual.vertex_colors = [0, 255, 0, 255]
                scene.add_geometry(s)

            for c in tgt_pts:
                s = trimesh.creation.icosphere(radius=0.008)
                s.apply_translation(c)
                s.visual.vertex_colors = [255, 0, 0, 255]
                scene.add_geometry(s)

            # (d) Finally show
            scene.show()
            # input("Press Enter to continue…")

        # 5) Project the four corners into their respective 2D UV‐spaces
        def to_uv(pts, O, u, v):
            rel = pts - O
            return np.stack([rel.dot(u), rel.dot(v)], axis=1)

        uv_src = to_uv(src_pts, O_src, u_src, v_src)  # shape (4×2)
        uv_tgt = to_uv(tgt_pts, O_tgt, u_tgt, v_tgt)  # shape (4×2)

        # 6) Visualize the 2D UV correspondences if requested
        if visualize_uv:
            plt.figure(figsize=(5,5))
            plt.scatter(uv_src[:,0], uv_src[:,1], c='r', s=50, label='OBJ corners')
            plt.scatter(uv_tgt[:,0], uv_tgt[:,1], c='b', s=50, label='URDF corners')

            for i in range(4):
                x0, y0 = uv_src[i]
                x1, y1 = uv_tgt[i]
                plt.arrow(x0, y0, x1 - x0, y1 - y0,
                          head_width=0.005, head_length=0.01,
                          fc='orange', ec='orange', length_includes_head=True)

            plt.legend()
            plt.title("2D UV Corner Correspondence")
            plt.xlabel("u")
            plt.ylabel("v")
            plt.axis('equal')
            plt.grid(True)
            plt.show()
            # input("Press Enter to continue…")

        # 7) Solve the 2D‐affine A, t so that: UV_tgt = A ⋅ UV_src + t
        N = 4
        G = np.zeros((2*N, 6))
        b = np.zeros(2*N)
        for i in range(N):
            xs, ys = uv_src[i]
            xt, yt = uv_tgt[i]
            G[2*i]   = [xs, ys, 1,  0,   0,  0]
            G[2*i+1] = [0,   0,  0, xs, ys, 1]
            b[2*i]   = xt
            b[2*i+1] = yt

        params, *_ = np.linalg.lstsq(G, b, rcond=None)
        A = np.array([[params[0], params[1]],
                      [params[3], params[4]]])
        t2 = np.array([params[2], params[5]])

        # 8) Now compute the warped positions of the four source corners (just the corners)
        rel_corners = src_pts - O_src                      # (4×3)
        coords_c = np.stack([rel_corners.dot(u_src),
                              rel_corners.dot(v_src),
                              rel_corners.dot(n_src)], axis=1)  # (4×3): [u,v,w]
        uv_corners = coords_c[:, :2]                       # (4×2)
        w_depth   = coords_c[:, 2:]                         # (4×1)
        
        uv_corners_warped = uv_corners.dot(A.T) + t2       # (4×2)
        warped_corners_3d = (O_tgt
                             + uv_corners_warped[:,0:1]*u_tgt
                             + uv_corners_warped[:,1:2]*v_tgt
                             + w_depth * n_tgt)           # (4×3)

        # 9) If requested, visualize those warped corners versus the URDF corners
        if visualize_warped_corners:
            scene = trimesh.Scene()

            # (a) Show URDF lightly so we see it
            urdf_copy = self.urdf.copy()
            urdf_copy.visual.face_colors = [0, 0, 255, 30]
            scene.add_geometry(urdf_copy)

            # (b) Show OBJ mesh lightly
            mesh_copy = self.mesh.copy()
            mesh_copy.visual.face_colors = [200, 200, 200, 30]
            scene.add_geometry(mesh_copy)

            # (c) Draw red spheres at the *target* corners (URDF)
            for c in tgt_pts:
                s = trimesh.creation.icosphere(radius=0.008)
                s.apply_translation(c)
                s.visual.vertex_colors = [255, 0, 0, 255]
                scene.add_geometry(s)

            # (d) Draw bright green spheres at the *warped* corner positions
            for c in warped_corners_3d:
                s = trimesh.creation.icosphere(radius=0.008)
                s.apply_translation(c)
                s.visual.vertex_colors = [0, 255, 0, 255]
                scene.add_geometry(s)

            # (e) Draw small lines connecting each warped corner → its URDF target
            for i in range(4):
                seg = np.vstack([warped_corners_3d[i], tgt_pts[i]])
                path = trimesh.load_path(seg)
                path.colors = np.array([[255, 255, 0, 255]])
                scene.add_geometry(path)

            # (f) Show
            scene.show()
            # input("Press Enter to continue…")

        # 10) Finally, apply the full warp to every vertex of the mesh
        all_verts = self.mesh.vertices.copy()               # (M×3)
        rel_all = all_verts - O_src                         # (M×3)
        coords_all = np.stack([rel_all.dot(u_src),
                               rel_all.dot(v_src),
                               rel_all.dot(n_src)], axis=1)  # (M×3)
        uv_all = coords_all[:, :2]                          # (M×2)
        depth_all = coords_all[:, 2:]                       # (M×1)

        uv_all_warped = uv_all.dot(A.T) + t2                # (M×2)
        warped_coords = np.concatenate([uv_all_warped, depth_all], axis=1)  # (M×3)

        new_verts = (O_tgt
                     + warped_coords[:, 0:1]*u_tgt
                     + warped_coords[:, 1:2]*v_tgt
                     + warped_coords[:, 2:3]*n_tgt)       # (M×3)

        # 11) Undo any prior “center‐on‐origin” translation
        inv_T = np.linalg.inv(self.urdf_transformations)
        warped = self.mesh.copy()
        warped.vertices = new_verts
        warped.apply_transform(inv_T)
        self.warped_mesh = warped

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


    def debug_visualize(self,
                        show_urdf: bool,
                        show_obj: bool,
                        show_warped: bool,
                        show_points: bool,
                        show_aabb: bool = False):
        """
        Pop up a trimesh.Scene showing, depending on the flags:
        - If show_urdf=True and self.urdf exists: add self.urdf (semi-transparent blue).
        - If show_obj=True  and self.mesh exists: add self.mesh (solid gray).
        - If show_warped=True and self.warped_mesh exists: add self.warped_mesh (solid green).
        - If show_points=True and self.urdf_corners/self.mesh_corners/self.warped_mesh_corners exist:
            add spheres at each of those corner points 
            (red for URDF corners, green for mesh corners, yellow for warped_mesh_corners).
        - If show_aabb=True, draw the axis-aligned bounding box around any displayed
            geometry (URDF, OBJ, and/or WARPED) by fetching `geom.bounding_box`.
        """
        scene = trimesh.Scene()

        # Keep track of which geometries to include in AABB
        aabb_targets = []

        # 1) URDF geometry
        if show_urdf:
            if self.urdf is None:
                raise RuntimeError("URDF mesh not set. Call set_urdf() first.")
            urdf_copy = self.urdf.copy()
            urdf_copy.visual.face_colors = [0, 0, 255, 50]  # translucent blue
            scene.add_geometry(urdf_copy)
            if show_aabb:
                aabb_targets.append(urdf_copy)

        # 2) OBJ geometry
        if show_obj:
            if self.mesh is None:
                raise RuntimeError("OBJ mesh not set. Call set_mesh() first.")
            mesh_copy = self.mesh.copy()
            mesh_copy.visual.face_colors = [200, 200, 200, 255]  # solid gray
            scene.add_geometry(mesh_copy)
            if show_aabb:
                aabb_targets.append(mesh_copy)

        # 3) WARPED geometry
        if show_warped:
            if getattr(self, "warped_mesh", None) is None:
                raise RuntimeError("Warped mesh not set. Run warp() first.")
            warped_copy = self.warped_mesh.copy()
            warped_copy.visual.face_colors = [0, 200, 0, 255]  # solid green
            scene.add_geometry(warped_copy)
            if show_aabb:
                aabb_targets.append(warped_copy)

        # 4) Corner points overlay
        if show_points:
            # URDF corners (red)
            if getattr(self, "urdf_corners", None) is not None:
                for corner in self.urdf_corners:
                    sphere = trimesh.creation.icosphere(radius=0.01)
                    sphere.apply_translation(corner)
                    sphere.visual.vertex_colors = [255, 0, 0, 255]
                    scene.add_geometry(sphere)
            # Mesh corners (green)
            if getattr(self, "mesh_corners", None) is not None:
                for corner in self.mesh_corners:
                    sphere = trimesh.creation.icosphere(radius=0.01)
                    sphere.apply_translation(corner)
                    sphere.visual.vertex_colors = [0, 255, 0, 255]
                    scene.add_geometry(sphere)
            # Warped mesh corners (yellow)
            if getattr(self, "warped_mesh_corners", None) is not None:
                for corner in self.warped_mesh_corners:
                    sphere = trimesh.creation.icosphere(radius=0.01)
                    sphere.apply_translation(corner)
                    sphere.visual.vertex_colors = [255, 255, 0, 255]
                    scene.add_geometry(sphere)

        # 5) Draw AABB wireframes
        if show_aabb and aabb_targets:
            for geom in aabb_targets:
                bb = geom.bounding_box
                # Make faces transparent, edges white
                bb.visual.face_colors = [255, 255, 255, 0]  # faces invisible
                bb.visual.vertex_colors = [255, 255, 255, 255]  # edges white
                scene.add_geometry(bb)

        # 6) Draw coordinate axes if any geometry is shown
        if ((show_urdf   and self.urdf is not None) or
            (show_obj    and self.mesh is not None) or
            (show_warped and getattr(self, "warped_mesh", None) is not None)):
            largest_extent = 0.0
            if show_urdf and self.urdf is not None:
                largest_extent = max(largest_extent, max(self.urdf.extents))
            if show_obj and self.mesh is not None:
                largest_extent = max(largest_extent, max(self.mesh.extents))
            if show_warped and getattr(self, "warped_mesh", None) is not None:
                largest_extent = max(largest_extent, max(self.warped_mesh.extents))
            if largest_extent > 0:
                axes = trimesh.creation.axis(
                    origin_size=0.01,
                    axis_length=largest_extent * 1.2
                )
                scene.add_geometry(axes)

        scene.show()
    
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
def center_on_origin(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
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

def align_mesh(mesh):
    # Try multiple starting points to avoid local minima
    starting_angles = [
        [0, 0, 0],        # No rotation
        [np.pi/2, 0, 0],  # 90° around x
        [0, np.pi/2, 0],  # 90° around y
        [0, 0, np.pi/2],  # 90° around z
        [np.pi/4, np.pi/4, np.pi/4]  # 45° all axes
    ]

    # for _ in range(5):  # three extra random guesses
    #     starting_angles.append(list(np.random.uniform(0, 2*np.pi, size=3)))
    
    best_volume = float('inf')
    best_angles = None
    best_mesh = None
    
    for start in starting_angles:
        # Optimize rotation to minimize AABB volume
        result = minimize(
            lambda x: compute_aabb_volume(x, mesh),
            x0=start,
            method='Powell',
            options={'maxiter': 1000}
        )
        
        volume = compute_aabb_volume(result.x, mesh)
        if volume < best_volume:
            best_volume = volume
            best_angles = result.x
            
            # Create the best rotated mesh
            R = trimesh.transformations.euler_matrix(best_angles[0], best_angles[1], best_angles[2])
            best_mesh = copy.deepcopy(mesh)
            best_mesh.apply_transform(R)
    
    return best_mesh, best_angles

def compute_aabb_volume(angles, mesh):
    # Create rotation matrix for all three angles (x, y, z)
    R = trimesh.transformations.euler_matrix(angles[0], angles[1], angles[2])
    rotated_mesh = copy.deepcopy(mesh)
    rotated_mesh.apply_transform(R)
    aabb = rotated_mesh.bounding_box
    return aabb.volume
