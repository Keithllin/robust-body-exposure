#!/usr/bin/env python3
"""Overlay grasp/release actions onto a ceiling RGB image for manual execution.

Projection uses the same 3D camera model as PCD-on-RGB overlays
(``bed_from_camera`` + intrinsics from ``sim_origin_data.pkl``), not the
legacy planar ``pixel_to_bed_xy`` affine.  Actions are lifted to a blanket
surface height (default 0.08 m, or median Z from an optional PCD) before
``cv2.projectPoints``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from canonical_bed import (  # noqa: E402
    maybe_load_canonical_frame,
    canonical_points_to_layout,
    canonical_xy_to_layout,
)
from marker_utils import invert_transform, transform_points  # noqa: E402
from pickle_compat import load_pickle  # noqa: E402


ACTION_COLORS = {
    "A": (40, 180, 40),  # BGR green
    "B": (40, 40, 220),  # BGR red
    "uncover": (0, 165, 255),  # BGR orange
    "ee-now": (255, 255, 0),  # BGR cyan: live Stretch gripper in bed frame
    "sensor": (40, 180, 40),
    "density-1": (200, 180, 0),
    "density-50": (0, 140, 255),
    "density-full": (180, 0, 255),
    "pred": (40, 40, 220),
    "snap": (180, 80, 220),
    "gt-pred": (40, 40, 220),
    "pred-FT": (40, 180, 40),
}
PCD_BGR = (0, 0, 255)  # red, same as visualize_pcd_on_rgb merged
DEFAULT_BED_Z_M = 0.08
# Match Recover voxel grasp_thres (max(0.05, voxel_size)).
DEFAULT_SNAP_RADIUS_M = 0.05


def _points_in_layout(points, canonical_frame):
    """Overlay projection uses layout-bed T_bed_camera from sim_origin."""

    if points is None or canonical_frame is None:
        return points
    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return canonical_points_to_layout(cloud, canonical_frame)


def _action_in_layout(action_bed, canonical_frame, *, bed_z: float):
    action = np.asarray(action_bed, dtype=np.float64).reshape(4)
    if canonical_frame is None:
        return action
    xy = canonical_xy_to_layout(
        np.vstack([action[:2], action[2:]]),
        canonical_frame,
        z=bed_z,
    )
    return np.concatenate([xy[0], xy[1]])


def _load_sim_origin(sim_origin_path: Path) -> dict:
    data = load_pickle(sim_origin_path)
    required = ("bed_from_camera", "mtx")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{sim_origin_path} missing {missing}")
    return data


def _load_bed_action(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if "scaled_action" in payload:
            action = payload["scaled_action"]
        elif "action_bed" in payload:
            action = payload["action_bed"]
        else:
            raise KeyError(f"{path} missing scaled_action/action_bed")
        return np.asarray(action, dtype=np.float64).reshape(4)
    with path.open("rb") as handle:
        try:
            action = pickle.load(handle)
        except Exception:
            action = load_pickle(path)
    return np.asarray(action, dtype=np.float64).reshape(4)


def _read_pcd_xyz(path: Path) -> np.ndarray:
    """Read XYZ from ASCII/binary PCD without Open3D (robe env may lack it)."""

    header_lines: list[str] = []
    data_format = None
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            header_lines.append(line.decode("ascii").strip())
            if line.upper().startswith(b"DATA"):
                parts = line.decode("ascii").strip().split()
                data_format = parts[1].lower() if len(parts) > 1 else None
                break
        payload = handle.read()

    if data_format not in {"ascii", "binary"}:
        raise RuntimeError(f"Unsupported PCD header: {path}")

    fields = sizes = types = counts = points_count = None
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        key = parts[0].upper()
        if key == "FIELDS":
            fields = parts[1:]
        elif key == "SIZE":
            sizes = [int(v) for v in parts[1:]]
        elif key == "TYPE":
            types = parts[1:]
        elif key == "COUNT":
            counts = [int(v) for v in parts[1:]]
        elif key == "POINTS":
            points_count = int(parts[1])
    if fields is None or sizes is None or types is None:
        raise RuntimeError(f"PCD header missing field metadata: {path}")
    if counts is None:
        counts = [1] * len(fields)
    if any(f not in fields for f in ("x", "y", "z")):
        raise RuntimeError(f"PCD missing XYZ: {path}")

    if data_format == "ascii":
        rows = np.fromstring(payload.decode("ascii"), sep=" ").reshape(-1, sum(counts))
        cols = []
        offset = 0
        for count, field in zip(counts, fields):
            if field in {"x", "y", "z"}:
                cols.append(rows[:, offset])
            offset += count
        return np.column_stack(cols).astype(np.float64, copy=False)

    dtype_map = {
        ("F", 4): np.dtype("<f4"),
        ("F", 8): np.dtype("<f8"),
        ("U", 1): np.dtype("u1"),
        ("U", 2): np.dtype("<u2"),
        ("U", 4): np.dtype("<u4"),
        ("I", 1): np.dtype("i1"),
        ("I", 2): np.dtype("<i2"),
        ("I", 4): np.dtype("<i4"),
    }
    structured_fields = []
    for field, size, kind, count in zip(fields, sizes, types, counts):
        base = dtype_map.get((kind, size))
        if base is None:
            raise RuntimeError(f"Unsupported binary PCD field {field}: {kind}{size}")
        structured_fields.append(
            (field, base, (count,)) if count != 1 else (field, base)
        )
    structured = np.frombuffer(
        payload,
        dtype=np.dtype(structured_fields),
        count=points_count if points_count is not None else -1,
    )
    return np.column_stack(
        [structured[f].reshape(-1) for f in ("x", "y", "z")]
    ).astype(np.float64, copy=False)


def _load_blanket_points(blanket_pcd: Path | None) -> np.ndarray | None:
    if blanket_pcd is None or not blanket_pcd.is_file():
        return None
    try:
        points = _read_pcd_xyz(blanket_pcd)
    except Exception as exc:
        print(f"WARN: could not read blanket PCD {blanket_pcd}: {exc}")
        return None
    return points if len(points) else None


def _resolve_bed_z(
    bed_z: float | None,
    blanket_pcd: Path | None,
    points: np.ndarray | None = None,
) -> float:
    if bed_z is not None:
        return float(bed_z)
    cloud = points if points is not None else _load_blanket_points(blanket_pcd)
    if cloud is not None and len(cloud):
        return float(np.median(cloud[:, 2]))
    return DEFAULT_BED_Z_M


def _lift_bed_xy(
    bed_xy: np.ndarray,
    *,
    points: np.ndarray | None,
    default_z: float,
    snap_radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift bed XY to 3D. Snap onto a nearby blanket point when one exists.

    CMA only requires the grasp to be within the voxel threshold (~5 cm) of
    cloth, so the raw action XY can sit in a hole that the Recover plot still
    paints as on-PCD.  Using the nearest measured point puts the overlay
    marker on the same surface the planner used.
    """

    bed_xy = np.asarray(bed_xy, dtype=np.float64).reshape(-1, 2)
    lifted = np.column_stack(
        [bed_xy, np.full(len(bed_xy), float(default_z), dtype=np.float64)]
    )
    snap_cm = np.full(len(bed_xy), np.nan, dtype=np.float64)
    if points is None or not len(points) or snap_radius_m <= 0:
        return lifted, snap_cm
    for i, xy in enumerate(bed_xy):
        dxy = np.linalg.norm(points[:, :2] - xy[None, :], axis=1)
        nearest = int(np.argmin(dxy))
        if dxy[nearest] <= snap_radius_m:
            lifted[i] = points[nearest]
            snap_cm[i] = dxy[nearest] * 100.0
    return lifted, snap_cm


def bed_xyz_to_pixel(
    points_bed: np.ndarray,
    *,
    bed_from_camera: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Project bed-frame XYZ into image pixels."""

    points_bed = np.asarray(points_bed, dtype=np.float64).reshape(-1, 3)
    camera_from_bed = invert_transform(np.asarray(bed_from_camera, dtype=np.float64))
    points_cam = transform_points(points_bed, camera_from_bed)
    if np.any(points_cam[:, 2] <= 1e-4):
        raise ValueError(
            "Action point(s) project behind the camera; check bed frame / Z"
        )
    projected, _ = cv2.projectPoints(
        points_cam.reshape(-1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
        np.asarray(distortion, dtype=np.float64).reshape(-1),
    )
    return projected.reshape(-1, 2)


def bed_xy_to_pixel_3d(
    bed_xy: np.ndarray,
    *,
    bed_from_camera: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    bed_z: float,
) -> np.ndarray:
    """Project bed-frame XY at height ``bed_z`` into image pixels."""

    bed_xy = np.asarray(bed_xy, dtype=np.float64).reshape(-1, 2)
    points_bed = np.column_stack(
        [bed_xy, np.full(len(bed_xy), float(bed_z), dtype=np.float64)]
    )
    return bed_xyz_to_pixel(
        points_bed,
        bed_from_camera=bed_from_camera,
        camera_matrix=camera_matrix,
        distortion=distortion,
    )


def _draw_pcd(
    image: np.ndarray,
    sim_origin: dict,
    points: np.ndarray,
    *,
    max_points: int = 25000,
    color: tuple[int, int, int] = PCD_BGR,
    radius: int = 1,
    alpha: float = 0.55,
) -> None:
    """Paint blanket XYZ onto the RGB using the same 3D model as the action."""

    if points is None or not len(points):
        return
    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if max_points > 0 and len(cloud) > max_points:
        rng = np.random.default_rng(0)
        cloud = cloud[rng.choice(len(cloud), size=max_points, replace=False)]
    dist = sim_origin.get("dist", None)
    if dist is None:
        distortion = np.zeros(5, dtype=np.float64)
    else:
        distortion = np.asarray(dist, dtype=np.float64).reshape(-1)
    pixels = bed_xyz_to_pixel(
        cloud,
        bed_from_camera=sim_origin["bed_from_camera"],
        camera_matrix=sim_origin["mtx"],
        distortion=distortion,
    )
    height, width = image.shape[:2]
    xs = np.round(pixels[:, 0]).astype(np.int32)
    ys = np.round(pixels[:, 1]).astype(np.int32)
    keep = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs[keep]
    ys = ys[keep]
    overlay = image.copy()
    for x, y in zip(xs, ys):
        cv2.circle(overlay, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    blended = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)
    image[:] = blended


def _draw_action(
    image: np.ndarray,
    sim_origin: dict,
    action_bed: Sequence[float],
    *,
    label: str,
    color: tuple[int, int, int],
    bed_z: float,
    points: np.ndarray | None = None,
    snap_radius_m: float = DEFAULT_SNAP_RADIUS_M,
    caption_mode: str = "full",
) -> None:
    action = np.asarray(action_bed, dtype=np.float64).reshape(4)
    grasp_bed = action[:2]
    release_bed = action[2:]
    dist = sim_origin.get("dist", None)
    if dist is None:
        distortion = np.zeros(5, dtype=np.float64)
    else:
        distortion = np.asarray(dist, dtype=np.float64).reshape(-1)
    raw_xyz = np.column_stack(
        [
            np.vstack([grasp_bed, release_bed]),
            np.full(2, float(bed_z), dtype=np.float64),
        ]
    )
    lifted, snap_cm = _lift_bed_xy(
        np.vstack([grasp_bed, release_bed]),
        points=points,
        default_z=bed_z,
        snap_radius_m=snap_radius_m,
    )
    # Draw the pickle XY the robot will execute. Snapping onto PCD is
    # annotation only — a 4 cm hole still counts as CMA on-cloth.
    pixels = bed_xyz_to_pixel(
        raw_xyz,
        bed_from_camera=sim_origin["bed_from_camera"],
        camera_matrix=sim_origin["mtx"],
        distortion=distortion,
    )
    snap_px = bed_xyz_to_pixel(
        lifted,
        bed_from_camera=sim_origin["bed_from_camera"],
        camera_matrix=sim_origin["mtx"],
        distortion=distortion,
    )
    grasp_px, release_px = pixels[0], pixels[1]
    height, width = image.shape[:2]
    g = (int(round(grasp_px[0])), int(round(grasp_px[1])))
    r = (int(round(release_px[0])), int(round(release_px[1])))

    def _inside(pt: tuple[int, int]) -> bool:
        return 0 <= pt[0] < width and 0 <= pt[1] < height

    g_in = _inside(g)
    r_in = _inside(r)
    if g_in and r_in:
        cv2.arrowedLine(image, g, r, color, 4, tipLength=0.12)
    elif g_in or r_in:
        cv2.line(image, g, r, color, 4)
    cv2.circle(image, g, 14, (255, 255, 255), -1)
    cv2.circle(image, g, 12, color, -1)
    cv2.circle(image, r, 12, (255, 255, 255), 2)
    cv2.circle(image, r, 10, color, -1)
    cv2.drawMarker(
        image, g, color, markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2
    )
    if np.isfinite(snap_cm[0]) and float(snap_cm[0]) > 1.0:
        sg = (
            int(round(float(snap_px[0, 0]))),
            int(round(float(snap_px[0, 1]))),
        )
        if _inside(sg):
            cv2.line(image, g, sg, (80, 80, 80), 1, cv2.LINE_AA)
            cv2.drawMarker(
                image, sg, (80, 80, 80), cv2.MARKER_TILTED_CROSS, 16, 1
            )
    if caption_mode == "short" and g_in:
        cv2.putText(
            image,
            label,
            (g[0] + 16, max(24, g[1] - 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    elif caption_mode == "full":
        snap_note = ""
        if np.isfinite(snap_cm[0]):
            snap_note = f" cloth {snap_cm[0]:.1f}cm away"
            if float(snap_cm[0]) >= 2.0:
                snap_note += " WARN"
                print(
                    f"WARN: {label} grasp is {snap_cm[0]:.1f} cm from the "
                    "nearest blanket point. Overlay used to snap the marker "
                    "onto cloth; the robot executes the hole. Do not YES."
                )
        else:
            snap_note = f" z={bed_z:.3f}"
        text = (
            f"{label}: grasp=({grasp_bed[0]:+.3f},{grasp_bed[1]:+.3f}) "
            f"release=({release_bed[0]:+.3f},{release_bed[1]:+.3f}) "
            f"px=({g[0]},{g[1]})->({r[0]},{r[1]}){snap_note}"
        )
        if not (g_in and r_in):
            text += " [OUT OF FRAME]"
        y = 36 if label in ("A", "uncover") else 72
        if label == "B":
            y = 72
        cv2.putText(
            image,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    if not (g_in and r_in):
        print(
            f"WARN: overlay {label} projects outside the ceiling image "
            f"(grasp_px={g}, release_px={r}, image={width}x{height})"
        )


def render_overlay(
    image_path: Path,
    sim_origin_path: Path,
    actions: list[tuple[str, np.ndarray]],
    output_path: Path,
    title: str | None = None,
    *,
    bed_z: float | None = None,
    blanket_pcd: Path | None = None,
    snap_radius_m: float = DEFAULT_SNAP_RADIUS_M,
    draw_pcd: bool = False,
    pcd_max_points: int = 25000,
    caption_mode: str = "full",
) -> Path:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read ceiling RGB: {image_path}")
    sim_origin = _load_sim_origin(sim_origin_path)
    canonical_frame = maybe_load_canonical_frame(sim_origin_path.parent)
    points = _load_blanket_points(blanket_pcd)
    surface_z = _resolve_bed_z(bed_z, blanket_pcd, points)
    layout_points = _points_in_layout(points, canonical_frame)
    canvas = image.copy()
    if draw_pcd:
        _draw_pcd(canvas, sim_origin, layout_points, max_points=pcd_max_points)
    if title:
        cv2.putText(
            canvas,
            title,
            (20, canvas.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    for label, action in actions:
        color = ACTION_COLORS.get(label, (255, 255, 0))
        _draw_action(
            canvas,
            sim_origin,
            _action_in_layout(action, canonical_frame, bed_z=surface_z),
            label=label,
            color=color,
            bed_z=surface_z,
            points=layout_points,
            snap_radius_m=snap_radius_m,
            caption_mode=caption_mode,
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to write overlay: {output_path}")
    print(
        f"Overlay projection bed_z={surface_z:.3f} m "
        f"snap_radius={snap_radius_m:.3f} m "
        f"pcd={'yes' if points is not None else 'no'} "
        f"draw_pcd={draw_pcd}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--sim-origin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument(
        "--bed-z",
        type=float,
        default=None,
        help="Bed-frame Z for grasp/release (meters). Default: median of "
        "--blanket-pcd, else 0.08",
    )
    parser.add_argument(
        "--blanket-pcd",
        type=Path,
        default=None,
        help="Optional merged blanket PCD used to pick surface height "
        "and snap grasp/release onto nearby cloth points",
    )
    parser.add_argument(
        "--snap-radius",
        type=float,
        default=DEFAULT_SNAP_RADIUS_M,
        help="Snap overlay markers onto a PCD point within this XY radius "
        f"(meters). Default {DEFAULT_SNAP_RADIUS_M} matches voxel grasp_thres.",
    )
    parser.add_argument(
        "--draw-pcd",
        action="store_true",
        help="Paint blanket_pcd onto the RGB before drawing actions",
    )
    parser.add_argument(
        "--pcd-max-points",
        type=int,
        default=25000,
        help="Subsample for --draw-pcd; 0 keeps all points",
    )
    parser.add_argument(
        "--action",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="Repeatable: LABEL action.pkl|json  (e.g. A recover_pred/scaled_action.pkl)",
    )
    args = parser.parse_args()
    actions = [
        (label, _load_bed_action(Path(path))) for label, path in args.action
    ]
    out = render_overlay(
        args.image,
        args.sim_origin,
        actions,
        args.output,
        title=args.title,
        bed_z=args.bed_z,
        blanket_pcd=args.blanket_pcd,
        snap_radius_m=args.snap_radius,
        draw_pcd=bool(args.draw_pcd),
        pcd_max_points=args.pcd_max_points,
    )
    print(f"Wrote action overlay: {out}")


if __name__ == "__main__":
    main()
