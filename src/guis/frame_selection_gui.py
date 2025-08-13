# frame_range_gui.py
import os
import re
import cv2
import bisect
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# digits immediately before extension: 00012.jpg or frame_00012.png -> "00012"
_DIGITS_BEFORE_EXT = re.compile(r"(\d+)(?=\.(?:jpe?g|png)$)", re.IGNORECASE)


def _collect_frames(img_dir):
    """
    Scan a directory for images, extract the integer frame number from digits
    immediately before the extension, and return:
      - frames: sorted list of frame numbers (ints)
      - paths:  dict {frame_num: full_path}
    """
    paths = {}
    for fn in os.listdir(img_dir):
        ext = os.path.splitext(fn)[1].lower()
        if ext in _IMAGE_EXTS:
            m = _DIGITS_BEFORE_EXT.search(fn)
            if m:
                n = int(m.group(1))
                paths[n] = os.path.join(img_dir, fn)

    if not paths:
        raise FileNotFoundError(
            f"No numeric-named images found in {img_dir} "
            "(e.g., 0001.jpg or frame_0012.png)."
        )

    frames = sorted(paths.keys())
    return frames, paths


def _nearest_existing(frames, n):
    """
    Return the nearest frame number from 'frames' to integer n.
    If tie, prefers the lower one.
    """
    i = bisect.bisect_left(frames, n)
    if i == 0:
        return frames[0]
    if i == len(frames):
        return frames[-1]
    before = frames[i - 1]
    after = frames[i]
    return before if (n - before) <= (after - n) else after


def run_frame_range_gui(img_dir: str, start_frame_num: int):
    """
    Display an image viewer for selecting a [start_frame, end_frame] range.

    Args:
      img_dir: path to directory of images
      start_frame_num: desired starting frame number to display

    Returns:
      [start_frame_num, end_frame_num] (both guaranteed to exist in the folder).
      If start > end, they will be swapped on finish.
    """
    frames, paths = _collect_frames(img_dir)

    # Pick initial displayed frame: nearest to requested number (exact if present)
    current_frame = _nearest_existing(frames, int(start_frame_num))
    idx = frames.index(current_frame)

    # Matplotlib setup
    fig = plt.figure(figsize=(10, 7))
    ax_img = fig.add_axes([0.05, 0.15, 0.70, 0.8])

    # Controls area
    ax_left10 = fig.add_axes([0.80, 0.75, 0.07, 0.06])
    ax_left1  = fig.add_axes([0.88, 0.75, 0.07, 0.06])
    ax_right1 = fig.add_axes([0.80, 0.66, 0.07, 0.06])
    ax_right10= fig.add_axes([0.88, 0.66, 0.07, 0.06])

    ax_start_label = fig.add_axes([0.80, 0.52, 0.15, 0.04]); ax_start_label.axis("off")
    ax_start_box   = fig.add_axes([0.80, 0.48, 0.15, 0.05])
    ax_end_label   = fig.add_axes([0.80, 0.40, 0.15, 0.04]); ax_end_label.axis("off")
    ax_end_box     = fig.add_axes([0.80, 0.36, 0.15, 0.05])

    ax_done = fig.add_axes([0.80, 0.22, 0.31, 0.07])

    # Load and show image
    img_artist = None

    def _show(idx_):
        nonlocal img_artist
        idx_ = max(0, min(idx_, len(frames) - 1))
        frame_num = frames[idx_]
        img_bgr = cv2.imread(paths[frame_num])
        if img_bgr is None:
            return idx_
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if img_artist is None:
            img_artist = ax_img.imshow(img_rgb)
        else:
            img_artist.set_data(img_rgb)

        # 👇 show the actual filename + keep the (i/total)
        fname = os.path.basename(paths[frame_num])
        ax_img.set_title(f"{fname}  ({idx_+1}/{len(frames)})")

        ax_img.axis("off")
        fig.canvas.draw_idle()
        return idx_


    idx = _show(idx)

    # Buttons
    btn_l10 = Button(ax_left10, "<<")   # -10
    btn_l1  = Button(ax_left1,  "<")    # -1
    btn_r1  = Button(ax_right1, ">")    # +1
    btn_r10 = Button(ax_right10, ">>")  # +10

    def move(delta):
        nonlocal idx
        idx = max(0, min(idx + delta, len(frames) - 1))
        idx = _show(idx)

    btn_l10.on_clicked(lambda _: move(-10))
    btn_l1.on_clicked(lambda _: move(-1))
    btn_r1.on_clicked(lambda _: move(+1))
    btn_r10.on_clicked(lambda _: move(+10))

    # Labels + text boxes
    ax_start_label.text(0, 0.5, "Start frame:", va="center", fontsize=10)
    ax_end_label.text(0, 0.5, "End frame:", va="center", fontsize=10)

    start_box = TextBox(ax_start_box, "", initial=str(frames[idx]))
    end_box   = TextBox(ax_end_box,   "", initial=str(frames[idx]))

    def _normalize_to_existing(textbox):
        """Clamp/snap textbox value to nearest valid frame; write back as int."""
        try:
            val = int(round(float(textbox.text)))
        except Exception:
            val = frames[idx]
        snapped = _nearest_existing(frames, val)
        if str(snapped) != textbox.text:
            textbox.set_val(str(snapped))
        return snapped

    def on_submit_start(_):
        _normalize_to_existing(start_box)

    def on_submit_end(_):
        _normalize_to_existing(end_box)

    start_box.on_submit(on_submit_start)
    end_box.on_submit(on_submit_end)

    # Keyboard navigation
    def on_key(event):
        k = (event.key or "").lower()
        if k == "left":
            move(-1)
        elif k == "right":
            move(+1)
        elif k == "shift+left":
            move(-10)
        elif k == "shift+right":
            move(+10)
        elif k == "enter":
            finish()  # allow Enter to finish

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Done
    result = {"done": False, "pair": None}

    def finish(_=None):
        s = _normalize_to_existing(start_box)
        e = _normalize_to_existing(end_box)
        if s > e:
            s, e = e, s
            # reflect the swap visually
            start_box.set_val(str(s))
            end_box.set_val(str(e))
        result["pair"] = [s, e]
        result["done"] = True
        try:
            plt.close(fig)
        except Exception:
            pass

    btn_done = Button(ax_done, "Done")
    btn_done.on_clicked(finish)

    # Help text
    fig.text(
        0.05, 0.05,
        "Navigation: <, > = ±1   |   <<, >> = ±10   |   Shift+Arrow = ±10   |   Enter = Done",
        fontsize=9,
    )

    # Block until closed
    plt.show()

    # If window closed without Done, still return current (snapped) values
    if not result["done"]:
        s = _nearest_existing(frames, int(start_box.text) if start_box.text.strip() else frames[idx])
        e = _nearest_existing(frames, int(end_box.text) if end_box.text.strip() else frames[idx])
        if s > e:
            s, e = e, s
        return [s, e]

    return result["pair"]


if __name__ == "__main__":
    img_dir = "/Users/jacobgoldberg/Documents/Stanford/REALab/record_3d_pcd_generation/data/basement_test_2/multiview/object_15/input"
    start = int(input("Starting frame number to display: ").strip())
    out = run_frame_range_gui(img_dir, start)
    print("\nSelected range:", out)
