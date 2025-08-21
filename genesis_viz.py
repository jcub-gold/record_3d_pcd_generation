import genesis as gs
from time import sleep

gs.init()

scene = gs.Scene(
    show_viewer = True,
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True,
        world_frame_size = 1.0,
        show_link_frame  = False,
        show_cameras     = False,
        plane_reflection = True,
        ambient_light    = (0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.Rasterizer(),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
kitchen = scene.add_entity(
    gs.morphs.URDF(file='scenes/basement_test_2/basement_test_2.urdf'),
)

cam = scene.add_camera(
    res    = (640, 480),
    pos    = (-10, -12, 3),
    lookat = (5, 0, 0.5),
    fov    = 30,
    GUI    = True,
)

scene.build(compile_kernels=False)  # visualize-only mode

cam.start_recording()
for _ in range(120):
    cam.render()                     # no scene.step() => no physics
cam.stop_recording('video.mp4', fps=60)