# prompt_gui.py
import os
import re
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
from matplotlib.patches import Rectangle

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_VIDEO_EXTS = {".mp4", ".mov"}
# digits right before the extension: e.g. 00012.jpg or frame_00012.png -> "00012"
_DIGITS_BEFORE_EXT = re.compile(r'(\d+)(?=\.(?:jpe?g|png)$)', re.IGNORECASE)


def _is_video(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in _VIDEO_EXTS


def _find_first_frame_in_dir(img_dir: str):
    """
    Return (image_rgb, first_frame_number).
    first_frame_number is the lowest integer found immediately before the extension.
    """
    candidates = []
    for fn in os.listdir(img_dir):
        ext = os.path.splitext(fn)[1].lower()
        if ext in _IMAGE_EXTS:
            m = _DIGITS_BEFORE_EXT.search(fn)
            if m:
                frame_num = int(m.group(1))
                candidates.append((frame_num, os.path.join(img_dir, fn)))
    if not candidates:
        raise FileNotFoundError(
            f"No numeric-named images found in {img_dir} (e.g., 0001.jpg or frame_0012.png)."
        )
    candidates.sort(key=lambda t: t[0])
    first_num, first_path = candidates[0]
    img_bgr = cv2.imread(first_path)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {first_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, first_num


def _first_frame_from_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame 0 from: {video_path}")
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return img_rgb, 0


def load_first_frame(input_path: str):
    """Return (image_rgb, first_frame_number)."""
    if _is_video(input_path):
        return _first_frame_from_video(input_path)
    if os.path.isdir(input_path):
        return _find_first_frame_in_dir(input_path)
    raise ValueError("Input must be either a directory of images or an .mp4/.mov file.")


def run_prompt_gui(input_path: str):
    """
    Launch the GUI and return a list of tuples:
      [
        (first_frame_number, [prompt_type, [x1,y1], [x2,y2], ...]),
        ...
      ]
    Only prompt types that have points are included.
    prompt_type ∈ { 1 (add), -1 (remove), 0 (box) }.
    For 'box', points are stored as pairs (two clicks per box).
    """
    img_rgb, first_frame_number = load_first_frame(input_path)

    # Store points per type
    points_by_type = {1: [], -1: [], 0: []}  # lists of [x,y]
    # Each action pushed here so Undo can delete drawings + data
    # action = {"ptype": int, "n_points": 1|2, "artists": [matplotlib Artist, ...]}
    actions = []

    # For an in-progress box (first corner clicked, waiting for second)
    pending_box_anchor = {"pt": None, "artists": []}

    mode_to_type = {"Add (+1)": 1, "Remove (-1)": -1, "Box (0)": 0}
    selected_mode_label = ["Add (+1)"]

    # ---- Figure / Axes ----
    fig = plt.figure(figsize=(10, 7))
    ax_img = fig.add_axes([0.05, 0.15, 0.70, 0.8])
    ax_radio = fig.add_axes([0.78, 0.55, 0.18, 0.25])
    ax_undo = fig.add_axes([0.78, 0.18, 0.09, 0.07])
    ax_done = fig.add_axes([0.88, 0.18, 0.09, 0.07])

    ax_img.imshow(img_rgb)
    ax_img.set_title(f"First frame: {first_frame_number}")
    ax_img.axis("off")

    fig.text(
        0.05, 0.05,
        "Click to add points.\n"
        "Modes: Add/Remove = single point; Box = two clicks (corners).\n"
        "Undo removes the last point/box. Press Enter to finish. Z = Undo.",
        fontsize=9,
    )

    radio = RadioButtons(ax_radio, ("Add (+1)", "Remove (-1)", "Box (0)"), active=0)

    def _force_redraw():
        # Explicit redraw to ensure artists actually disappear immediately
        fig.canvas.draw()

    def _clear_pending_anchor():
        if pending_box_anchor["pt"] is not None:
            for a in pending_box_anchor["artists"]:
                try:
                    a.remove()
                except Exception:
                    pass
            pending_box_anchor["pt"] = None
            pending_box_anchor["artists"] = []
            _force_redraw()

    def on_radio(label):
        # If we leave Box while a box is half-done, drop its anchor artists
        if selected_mode_label[0] == "Box (0)":
            _clear_pending_anchor()
        selected_mode_label[0] = label

    radio.on_clicked(on_radio)

    btn_undo = Button(ax_undo, "Undo")
    btn_done = Button(ax_done, "Done")

    # draw a small "+" cross at (x, y)
    def draw_cross(x, y, color, size=8, lw=1.25):
        h1, = ax_img.plot([x - size, x + size], [y, y], color=color, linewidth=lw)
        h2, = ax_img.plot([x, x], [y - size, y + size], color=color, linewidth=lw)
        return [h1, h2]

    def on_click(event):
        if event.inaxes != ax_img or event.xdata is None or event.ydata is None:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        ptype = mode_to_type[selected_mode_label[0]]

        if ptype in (1, -1):
            color = "lime" if ptype == 1 else "red"
            arts = draw_cross(x, y, color=color)
            points_by_type[ptype].append([x, y])
            actions.append({"ptype": ptype, "n_points": 1, "artists": arts})
            _force_redraw()
        else:
            # Box mode
            if pending_box_anchor["pt"] is None:
                # first click
                anchor_arts = draw_cross(x, y, color="yellow")
                pending_box_anchor["pt"] = (x, y)
                pending_box_anchor["artists"] = anchor_arts
                _force_redraw()
            else:
                # second click
                x0, y0 = pending_box_anchor["pt"]
                # remove anchor cross
                for a in pending_box_anchor["artists"]:
                    try:
                        a.remove()
                    except Exception:
                        pass
                pending_box_anchor["pt"] = None
                pending_box_anchor["artists"] = []

                # store both corners
                points_by_type[0].extend([[x0, y0], [x, y]])

                # draw rectangle + two corner crosses
                x_min, y_min = min(x0, x), min(y0, y)
                w, h = abs(x - x0), abs(y - y0)
                rect = Rectangle((x_min, y_min), w, h, fill=False, linewidth=1.4)
                ax_img.add_patch(rect)
                c1 = draw_cross(x0, y0, color="yellow")
                c2 = draw_cross(x, y, color="yellow")
                arts = [rect] + c1 + c2
                actions.append({"ptype": 0, "n_points": 2, "artists": arts})
                _force_redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)

    def on_undo(_event=None):
        # If a box anchor is half-placed, cancel it first
        if pending_box_anchor["pt"] is not None:
            _clear_pending_anchor()
            return

        if not actions:
            return
        last = actions.pop()
        ptype = last["ptype"]
        n_pts = last["n_points"]

        # remove last n_pts from the correct bucket
        for _ in range(n_pts):
            if points_by_type[ptype]:
                points_by_type[ptype].pop()

        # remove drawn artists from canvas
        for a in last.get("artists", []):
            try:
                a.remove()
            except Exception:
                pass

        _force_redraw()

    btn_undo.on_clicked(on_undo)

    result = {"done": False, "tuples": None}

    def _build_return_list():
        # Build [(first_frame_number, [ptype, [x,y], ...]), ...] in the order 1, -1, 0 if present
        out = []
        for ptype in (1, -1, 0):
            pts = points_by_type[ptype]
            if pts:
                prompts = [ptype] + [p for p in pts]
                out.append((0, prompts)) ##### NOTE: returns the first frame offest, which is 0 since we are always loading the first frame
        return out

    def on_done(_event=None):
        # Drop a dangling box anchor visually (no points were added yet)
        _clear_pending_anchor()
        result["tuples"] = _build_return_list()
        result["done"] = True
        try:
            plt.close(fig)
        except Exception:
            pass

    btn_done.on_clicked(on_done)

    def on_key(event):
        if event.key in ("z", "Z"):
            on_undo()
        elif event.key == "enter":
            on_done()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Blocking GUI
    plt.show()

    # If window was closed without pressing Done, still return current state
    if not result["done"]:
        return _build_return_list()
    return result["tuples"]

if __name__ == "__main__":
    path = "/Users/jacobgoldberg/Documents/Stanford/REALab/record_3d_pcd_generation/data/basement_test_2/multiview/object_15/input"
    out = run_prompt_gui(path)
    print("\nResult list:")
    for item in out:
        print(item)
