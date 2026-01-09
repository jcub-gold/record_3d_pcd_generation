from submodules.TRELLIS.trellis.utils import render_utils, postprocessing_utils
from submodules.TRELLIS.trellis.renderers import MeshRenderer, GaussianRenderer
import imageio
import os

"""
Function: save_object
---------------------
object_output: dictionary containing 'mesh' and 'gaussian' model outputs
output_path: path to the output directory
object_name: optional name prefix for output files (default "")
is_glb: if True, exports as GLB format; if False, exports as OBJ format (default False)

Renders and saves mesh and Gaussian sample videos, and exports the mesh in the specified format.
"""
def save_object(object_output, output_path, object_name="", is_glb=False):
    if object_name:
        object_name += "_"
        
    video = render_utils.render_video(object_output['mesh'][0])['normal']
    imageio.mimsave(os.path.join(output_path, f"{object_name}sample_mesh.mp4"), video, fps=30)
    video = render_utils.render_video(object_output['gaussian'][0])['color']
    imageio.mimsave(os.path.join(output_path, f"{object_name}sample_gs.mp4"), video, fps=30)

    obj = postprocessing_utils.to_glb(
        object_output['gaussian'][0],
        object_output['mesh'][0],
        # Optional parameters
        simplify=0.85,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
        verbose=False
    )

    if not is_glb:
        mesh_path = os.path.join(output_path, f"{object_name}mesh.obj")
    else:
        mesh_path = os.path.join(output_path, f"{object_name}mesh.glb")

    obj.export(mesh_path)