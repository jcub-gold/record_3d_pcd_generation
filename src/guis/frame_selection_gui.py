# frame_range_gui.py
import os
import re
import cv2
import bisect
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons
import json

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# digits immediately before extension: 00012.jpg or frame_00012.png -> "00012"
_DIGITS_BEFORE_EXT = re.compile(r"(\d+)(?=\.(?:jpe?g|png)$)", re.IGNORECASE)

_CONFIG_DIR = "Configs"


def load_labels():
    labels_path = os.path.join(_CONFIG_DIR, "label_config.json")
    with open(labels_path, "r") as f:
        data = json.load(f)
    return data["labels"]

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


def run_frame_range_gui(img_dir: str, start_frame_num: int, title="Title"):
    frames, paths = _collect_frames(img_dir)

    labels = load_labels()
    if not labels:
        raise ValueError("No labels provided/found for selection.")

    current_frame = _nearest_existing(frames, int(start_frame_num))
    idx = frames.index(current_frame)
    current_label = labels[0]

    # Figure + layout constants
    fig = plt.figure(figsize=(10, 7))

    # Left: image viewport
    ax_img = fig.add_axes([0.05, 0.13, 0.67, 0.82])

    # Right: control column (use consistent positions)
    PANEL_LEFT = 0.65
    PANEL_W    = 0.22
    SP         = 0.02
    BTN_H      = 0.06
    BTN_W      = (PANEL_W - SP) / 2.0

    # Arrow rows
    row1_y = 0.80
    row2_y = 0.72
    ax_left10 = fig.add_axes([PANEL_LEFT,             row1_y, BTN_W, BTN_H])
    ax_left1  = fig.add_axes([PANEL_LEFT+BTN_W+SP,    row1_y, BTN_W, BTN_H])
    ax_right1 = fig.add_axes([PANEL_LEFT,             row2_y, BTN_W, BTN_H])
    ax_right10= fig.add_axes([PANEL_LEFT+BTN_W+SP,    row2_y, BTN_W, BTN_H])

    # Place a centered, bold title just above the first arrow row
    ax_ctrl_title = fig.add_axes([PANEL_LEFT, row1_y + BTN_H + SP*0.5, PANEL_W, 0.06])
    ax_ctrl_title.axis("off")
    ax_ctrl_title.text(
        0.5, 0.5, title,
        ha="center", va="center",
        fontsize=16, fontweight="bold"
    )

    # Start/End fields
    ax_start_label = fig.add_axes([PANEL_LEFT, 0.63, PANEL_W, 0.03]); ax_start_label.axis("off")
    ax_start_box   = fig.add_axes([PANEL_LEFT, 0.58, PANEL_W, 0.05])
    ax_end_label   = fig.add_axes([PANEL_LEFT, 0.52, PANEL_W, 0.03]); ax_end_label.axis("off")
    ax_end_box     = fig.add_axes([PANEL_LEFT, 0.47, PANEL_W, 0.05])

    # Label dropdown header + button
    ax_label_hdr   = fig.add_axes([PANEL_LEFT, 0.41, PANEL_W, 0.03]); ax_label_hdr.axis("off")
    ax_label_btn   = fig.add_axes([PANEL_LEFT, 0.36, PANEL_W, 0.05])

    # Dropdown menu size (adapt to number of labels, capped)
    # Each radio item ~0.03 high; add a little padding
    menu_items_h = 0.03 * max(1, len(labels))
    menu_h = min(0.20, menu_items_h + 0.02)
    ax_label_menu  = fig.add_axes([PANEL_LEFT, 0.16, PANEL_W, menu_h])  # sits above Done
    ax_done        = fig.add_axes([PANEL_LEFT, 0.08, PANEL_W, 0.07])

    # ---- image show ----
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

        fname = os.path.basename(paths[frame_num])
        ax_img.set_title(f"{fname}  ({idx_+1}/{len(frames)})")
        ax_img.axis("off")
        fig.canvas.draw_idle()
        return idx_
    idx = _show(idx)

    # ---- navigation buttons ----
    btn_l10 = Button(ax_left10, "<<");   btn_l1  = Button(ax_left1,  "<")
    btn_r1  = Button(ax_right1, ">");    btn_r10 = Button(ax_right10, ">>")
    def move(delta):
        nonlocal idx
        idx = max(0, min(idx + delta, len(frames) - 1))
        idx = _show(idx)
    btn_l10.on_clicked(lambda _: move(-10))
    btn_l1 .on_clicked(lambda _: move(-1))
    btn_r1 .on_clicked(lambda _: move(+1))
    btn_r10.on_clicked(lambda _: move(+10))

    # ---- start/end text boxes ----
    ax_start_label.text(0, 0.5, "Start frame:", va="center", fontsize=10)
    ax_end_label.text(0, 0.5, "End frame:", va="center", fontsize=10)

    start_box = TextBox(ax_start_box, "", initial=str(frames[idx]))
    end_box   = TextBox(ax_end_box,   "", initial=str(frames[idx]))

    def _nearest_existing_int(txt):
        try:
            val = int(round(float(txt)))
        except Exception:
            return frames[idx]
        return _nearest_existing(frames, val)

    def _normalize_to_existing(textbox):
        snapped = _nearest_existing_int(textbox.text)
        if str(snapped) != textbox.text:
            textbox.set_val(str(snapped))
        return snapped

    start_box.on_submit(lambda _: _normalize_to_existing(start_box))
    end_box  .on_submit(lambda _: _normalize_to_existing(end_box))

    # ---- label dropdown (radio menu that toggles) ----
    ax_label_hdr.text(0, 0.5, "Label:", va="center", fontsize=10)
    label_btn = Button(ax_label_btn, current_label)

    rb = RadioButtons(ax_label_menu, labels, active=0)
    ax_label_menu.set_visible(False)
    ax_label_menu.set_title("Choose label")
    menu_open = [False]

    def toggle_menu(_=None):
        menu_open[0] = not menu_open[0]
        ax_label_menu.set_visible(menu_open[0])
        fig.canvas.draw_idle()

    def on_label_selected(lbl):
        nonlocal current_label
        current_label = lbl
        label_btn.label.set_text(lbl)
        ax_label_menu.set_visible(False)
        menu_open[0] = False
        fig.canvas.draw_idle()

    label_btn.on_clicked(toggle_menu)
    rb.on_clicked(on_label_selected)

    # ---- keyboard ----
    def on_key(event):
        k = (event.key or "").lower()
        if   k == "left":         move(-1)
        elif k == "right":        move(+1)
        elif k == "shift+left":   move(-10)
        elif k == "shift+right":  move(+10)
        elif k == "enter":        finish()
        elif k == "escape" and menu_open[0]:
            toggle_menu()
    fig.canvas.mpl_connect("key_press_event", on_key)

    # ---- finish / return ----
    result = {"done": False, "pair": None}
    def finish(_=None):
        s = _normalize_to_existing(start_box)
        e = _normalize_to_existing(end_box)
        if s > e:
            s, e = e, s
            start_box.set_val(str(s)); end_box.set_val(str(e))
        result["pair"] = [s, e]
        result["done"] = True
        try: plt.close(fig)
        except Exception: pass

    btn_done = Button(ax_done, "Done")
    btn_done.on_clicked(finish)


    fig.text(
        0.05, 0.05,
        "Navigation: <, > = ±1   |   <<, >> = ±10   |   Shift+Arrow = ±10   |   Enter = Done   |   Click label to select",
        fontsize=9,
    )

    plt.show()

    if not result["done"]:
        s = _nearest_existing(frames, int(start_box.text) if start_box.text.strip() else frames[idx])
        e = _nearest_existing(frames, int(end_box.text) if end_box.text.strip() else frames[idx])
        if s > e: s, e = e, s
        return [s, e], current_label
    return result["pair"], current_label


if __name__ == "__main__":
    img_dir = "/Users/jacobgoldberg/Documents/Stanford/REALab/record_3d_pcd_generation/data/basement_test_2/multiview/object_15/input"
    start = int(input("Starting frame number to display: ").strip())
    out = run_frame_range_gui(img_dir, start)
    print("\nSelected range:", out)
