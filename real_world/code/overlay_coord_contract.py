#!/usr/bin/env python3
"""Draw canonical origin and ±X/±Y contract points on the ceiling RGB.

Use this before any cloth pull to freeze the bed-frame convention:

  (0,0)   visual bed center
  (+0.20, 0) / (-0.20, 0)
  (0, +0.20)

Actions and graphs must already be in the 4-marker canonical frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from conda_python import drop_foreign_site_packages  # noqa: E402

drop_foreign_site_packages()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from canonical_bed import maybe_load_canonical_frame  # noqa: E402
from overlay_action_on_ceiling import (  # noqa: E402
    _points_in_layout,
    _load_sim_origin,
    bed_xyz_to_pixel,
)


CONTRACT_POINTS = {
    "O": (0.0, 0.0),
    "+X": (0.20, 0.0),
    "-X": (-0.20, 0.0),
    "+Y": (0.0, 0.20),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--sim-origin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, default=None)
    parser.add_argument("--bed-z", type=float, default=0.0)
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Failed to read {args.image}")
    sim_origin = _load_sim_origin(args.sim_origin)
    pose_dir = args.pose_dir or args.sim_origin.parent
    frame = maybe_load_canonical_frame(pose_dir)
    dist = sim_origin.get("dist")
    distortion = (
        np.zeros(5, dtype=np.float64)
        if dist is None
        else np.asarray(dist, dtype=np.float64).reshape(-1)
    )
    names = list(CONTRACT_POINTS)
    xy = np.array([CONTRACT_POINTS[name] for name in names], dtype=np.float64)
    points = np.column_stack([xy, np.full(len(xy), args.bed_z)])
    layout_pts = _points_in_layout(points, frame)
    pixels = bed_xyz_to_pixel(
        layout_pts,
        bed_from_camera=sim_origin["bed_from_camera"],
        camera_matrix=sim_origin["mtx"],
        distortion=distortion,
    )
    colors = {
        "O": (0, 255, 255),
        "+X": (0, 0, 255),
        "-X": (0, 128, 255),
        "+Y": (0, 200, 0),
    }
    for name, px in zip(names, pixels):
        pt = (int(round(px[0])), int(round(px[1])))
        color = colors[name]
        cv2.circle(image, pt, 12, color, -1)
        cv2.putText(
            image,
            name,
            (pt[0] + 14, pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    origin = pixels[names.index("O")]
    plus_x = pixels[names.index("+X")]
    plus_y = pixels[names.index("+Y")]
    cv2.arrowedLine(
        image,
        (int(origin[0]), int(origin[1])),
        (int(plus_x[0]), int(plus_x[1])),
        colors["+X"],
        3,
        tipLength=0.15,
    )
    cv2.arrowedLine(
        image,
        (int(origin[0]), int(origin[1])),
        (int(plus_y[0]), int(plus_y[1])),
        colors["+Y"],
        3,
        tipLength=0.15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), image):
        raise SystemExit(f"Failed to write {args.output}")
    print(f"Wrote coordinate contract overlay: {args.output}")
    if frame is None:
        print("WARN: no canonical_bed_frame.json; overlay is layout-bed metres")


if __name__ == "__main__":
    main()
