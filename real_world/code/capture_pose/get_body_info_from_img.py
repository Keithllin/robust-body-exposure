# importing the module
import cv2

import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import argparse
import os.path as osp
import pickle
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from pickle_compat import load_pickle

pixel_to_bed_affine = None
MIN_DIAMETER_M = 0.01
# Chest and waist are the two upper-body spheres. A click-slip (~1 cm) used
# to pass the global min and collapse the waist disk to a point.
TORSO_MIN_DIAMETER_M = 0.12

LIMBS = [
    "head",
    "upperchest",
    "waist",
    "upperarm",
    "forearm",
    "hand",
    "thigh",
    "shin",
    "foot",
]

LIMB_HELP = {
    "head": "HEAD — drag across the head (left–right).",
    "upperchest": (
        "UPPER CHEST  (sphere 1 of 2) — drag across the chest at the "
        "shoulders / armpits. This is not the waist."
    ),
    "waist": (
        "WAIST  (sphere 2 of 2) — drag across the waist / hips. "
        "The upper body needs TWO full spheres, not one chest disk and one point."
    ),
    "upperarm": "UPPER ARM — drag across the upper arm (width, not length).",
    "forearm": "FOREARM — drag across the forearm width.",
    "hand": "HAND — drag across the hand width.",
    "thigh": "THIGH — drag across the thigh width.",
    "shin": "SHIN — drag across the shin width.",
    "foot": "FOOT — drag across the foot width.",
}


def min_diameter_for(limb: str) -> float:
    if limb in ("upperchest", "waist"):
        return TORSO_MIN_DIAMETER_M
    return MIN_DIAMETER_M


def create_circle(x, y, r, canvas):  # center coordinates, radius
    return canvas.create_oval(x - r, y - r, x + r, y + r, fill="red")


def get_diameter(points_i_px, points_f_px):
    if pixel_to_bed_affine is not None:
        points_i_h = np.append(np.asarray(points_i_px, dtype=np.float64), 1.0)
        points_f_h = np.append(np.asarray(points_f_px, dtype=np.float64), 1.0)
        points_i_m = points_i_h @ pixel_to_bed_affine.T
        points_f_m = points_f_h @ pixel_to_bed_affine.T
        return abs(np.linalg.norm(points_i_m - points_f_m))

    points_i_m = (points_i_px - origin_px) * m2px_scale
    points_f_m = (points_f_px - origin_px) * m2px_scale

    return abs(np.linalg.norm(np.array(points_i_m) - np.array(points_f_m)))


def refresh_prompt():
    n = len(canvas.limbs)
    if canvas.index < n:
        limb = canvas.limbs[canvas.index]
        step = f"{canvas.index + 1}/{n}"
        help_text = LIMB_HELP[limb]
        banner = f"{step}   {help_text}"
        print(banner)
        print(
            "  (chest and waist are two separate widths; "
            "do not skip WAIST after UPPERCHEST)"
        )
        canvas.itemconfigure(canvas.prompt_id, text=banner)
        root.title(f"Body diameters  {step}  {limb.upper()}")
        return

    invalid = [
        limb
        for limb in canvas.limbs
        if canvas.body_info[limb][1] < min_diameter_for(limb) / 2
    ]
    if invalid:
        canvas.index = canvas.limbs.index(invalid[0])
        print(
            "Invalid measurement; retry "
            f"{invalid[0].upper()} with a visible diameter "
            f"(min {min_diameter_for(invalid[0]):.2f} m)."
        )
        refresh_prompt()
        return
    chest_r = float(canvas.body_info["upperchest"][1])
    waist_r = float(canvas.body_info["waist"][1])
    print(
        "MEASUREMENTS DONE!  "
        f"upperchest radius={chest_r:.3f} m, waist radius={waist_r:.3f} m"
    )
    filename = "body_info.pkl"
    dest = osp.join(args.pose_dir, filename)
    with open(dest, "wb") as f:
        pickle.dump(canvas.body_info, f)
    print(f"Saved {dest}")
    canvas.winfo_toplevel().destroy()


def draw_line(event):
    if canvas.index >= len(canvas.limbs):
        return
    limb = canvas.limbs[canvas.index]
    if str(event.type) == "ButtonPress":
        canvas.old_coords = event.x, event.y
        create_circle(event.x, event.y, 2, canvas)

    elif str(event.type) == "Motion":
        x, y = event.x, event.y
        x1, y1 = canvas.old_coords
        canvas.delete(canvas.old_line_id)
        canvas.old_line_id = canvas.create_line(x, y, x1, y1)

    elif str(event.type) == "ButtonRelease":
        canvas.old_line_id = None
        create_circle(event.x, event.y, 2, canvas)

        diameter = get_diameter(canvas.old_coords, (event.x, event.y))
        print(f"{limb} diameter = {diameter:.4f} m")
        need = min_diameter_for(limb)
        if diameter < need:
            print(
                f"Measurement too small ({diameter:.4f} m); "
                f"need at least {need:.2f} m across the {limb}. Retry."
            )
            refresh_prompt()
            return
        canvas.body_info[limb][1] = diameter / 2
        canvas.index += 1
        refresh_prompt()


def reset_coords(event):
    canvas.old_coords = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Click-drag nine body diameters on the uncovered RGB. "
            "Upper body is two spheres: UPPERCHEST then WAIST."
        )
    )
    parser.add_argument("--subject-dir", type=str, default="TEST")
    parser.add_argument("--pose-dir", type=str, default="TEST")
    args = parser.parse_args()

    data = load_pickle(osp.join(args.pose_dir, "sim_origin_data.pkl"))
    dist = data["dist"]
    mtx = data["mtx"]
    centers_px = data["centers_px"]
    centers_m = data["centers_m"]
    origin_px = data["origin_px"]
    origin_m = data["origin_m"]
    m2px_scale = data["m2px_scale"]
    pixel_to_bed_affine = data.get("pixel_to_bed_xy")

    print("Nine diameter measurements, in order:")
    for i, name in enumerate(LIMBS, start=1):
        print(f"  {i}. {LIMB_HELP[name]}")
    print("Look at the YELLOW text on the image — not only this terminal.")

    root = tk.Tk()
    root.title("Body diameters  1/9  HEAD")

    scale = 1
    canvas = tk.Canvas(root, width=1280 * scale, height=720 * scale)
    canvas.pack()

    bg = ImageTk.PhotoImage(file=osp.join(args.pose_dir, "uncovered_rgb.png"))

    canvas.create_image(0, 0, image=bg, anchor="nw")
    canvas.prompt_id = canvas.create_text(
        24,
        24,
        anchor="nw",
        fill="yellow",
        font=("Helvetica", 16, "bold"),
        width=1230,
        text="",
    )
    canvas.old_coords = None
    canvas.old_line_id = None
    canvas.body_info = {
        "head": [None, 0],
        "upperchest": [None, 0],
        "waist": [None, 0],
        "upperarm": [None, 0],
        "forearm": [None, 0],
        "hand": [None, 0],
        "thigh": [None, 0],
        "shin": [None, 0],
        "foot": [None, 0],
    }
    canvas.limbs = list(LIMBS)
    canvas.index = 0

    root.bind("<ButtonPress-1>", draw_line)
    root.bind("<ButtonRelease-1>", draw_line)
    root.bind("<B1-Motion>", draw_line)

    refresh_prompt()
    root.mainloop()
