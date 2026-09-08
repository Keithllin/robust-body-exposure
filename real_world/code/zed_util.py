"""Shared ZED helpers for real_world capture and calibration scripts.

Requires the ``robe-zed`` conda env (Python 3.10 + pyzed + open3d + opencv).
"""

from __future__ import annotations

import json
import os
import os.path as osp
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import pyzed.sl as sl

REAL_WORLD_DIR = Path(__file__).resolve().parents[1]
ROBE_ROOT = REAL_WORLD_DIR.parent
DEFAULT_FILTER_JSON = REAL_WORLD_DIR / "calibration" / "zed_blanket_filter.json"
# Fixed production lighting for HSV filter tuning and blanket PCD capture.
# Marker calibration uses camera_settings stored in zed_extrinsics.json instead.
PCD_EXPOSURE = 55
PCD_GAIN = 40


@dataclass
class BlanketFilterParams:
    h_min: int = 2
    h_max: int = 26
    s_min: int = 138
    s_max: int = 255
    v_min: int = 0
    v_max: int = 255
    depth_min_m: float = 0.93
    depth_max_m: float = 1.43
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    keep_largest_component: bool = False
    min_component_area: int = 0


def resolution_from_name(name: str):
    """Return a ZED resolution enum from a CLI-friendly name."""

    key = str(name).upper()
    if not hasattr(sl.RESOLUTION, key):
        available = sorted(
            value for value in dir(sl.RESOLUTION) if value.isupper()
        )
        raise ValueError(f"Unknown ZED resolution {name!r}; choose from {available}")
    return getattr(sl.RESOLUTION, key)


def load_filter_params(path: Optional[str] = None) -> BlanketFilterParams:
    path = path or str(DEFAULT_FILTER_JSON)
    if not osp.exists(path):
        raise FileNotFoundError(
            f"Blanket filter JSON not found: {path}. "
            "Provide the locally measured calibration filter JSON first."
        )
    with open(path) as f:
        data = json.load(f)
    base = BlanketFilterParams()
    return BlanketFilterParams(**{**base.__dict__, **data})


def open_zed(
    serial: int = 0,
    resolution=sl.RESOLUTION.HD720,
    depth_mode=sl.DEPTH_MODE.NEURAL,
    coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
) -> sl.Camera:
    init = sl.InitParameters()
    init.camera_resolution = resolution
    init.depth_mode = depth_mode
    init.coordinate_units = sl.UNIT.METER
    # Keep all ZED XYZ captures in the same right-handed bed-calibration
    # convention. ArUco's OpenCV image-frame pose is converted explicitly
    # before it is written to zed_extrinsics.json.
    init.coordinate_system = coordinate_system
    init.depth_minimum_distance = 0.3
    init.depth_maximum_distance = 4.0
    if serial:
        init.set_from_serial_number(int(serial))

    cam = sl.Camera()
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED: {err}")
    return cam


def open_ceiling_zed(
    serial: int = 0,
    resolution=sl.RESOLUTION.HD720,
    depth_mode=sl.DEPTH_MODE.NEURAL,
) -> sl.Camera:
    """Backward-compatible name for opening the current ZED role."""

    return open_zed(
        serial=serial,
        resolution=resolution,
        depth_mode=depth_mode,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
    )


def warm_up(camera: sl.Camera, n: int = 30) -> None:
    runtime = sl.RuntimeParameters()
    for _ in range(n):
        camera.grab(runtime)


def _get_video_setting(camera: sl.Camera, setting) -> int:
    status, value = camera.get_camera_settings(setting)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to read ZED setting {setting}: {status}")
    return int(value)


def set_auto_exposure_gain(camera: sl.Camera, enabled: bool) -> None:
    """Enable or disable ZED's coupled automatic exposure/gain controller."""

    status = camera.set_camera_settings(
        sl.VIDEO_SETTINGS.AEC_AGC, int(bool(enabled))
    )
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(
            f"Failed to {'enable' if enabled else 'disable'} "
            f"ZED auto exposure/gain: {status}"
        )


def configure_exposure_gain(
    camera: sl.Camera,
    exposure: Optional[int] = None,
    gain: Optional[int] = None,
    lock: bool = False,
) -> dict:
    """Optionally lock exposure/gain after auto-exposure warm-up.

    With ``lock=True`` and no explicit values, the current auto-selected
    values are frozen. Explicit values are in the ZED 0--100 percentage
    ranges and automatically disable AEC/AGC.
    """

    if exposure is not None and not 0 <= int(exposure) <= 100:
        raise ValueError("exposure must be between 0 and 100")
    if gain is not None and not 0 <= int(gain) <= 100:
        raise ValueError("gain must be between 0 and 100")

    auto_exposure = _get_video_setting(camera, sl.VIDEO_SETTINGS.AEC_AGC)
    current_exposure = _get_video_setting(camera, sl.VIDEO_SETTINGS.EXPOSURE)
    current_gain = _get_video_setting(camera, sl.VIDEO_SETTINGS.GAIN)
    should_lock = bool(lock or exposure is not None or gain is not None)
    if not should_lock:
        return {
            "locked": False,
            "auto_exposure": bool(auto_exposure),
            "exposure": current_exposure,
            "gain": current_gain,
        }

    target_exposure = current_exposure if exposure is None else int(exposure)
    target_gain = current_gain if gain is None else int(gain)
    set_auto_exposure_gain(camera, False)
    for setting, value in (
        (sl.VIDEO_SETTINGS.EXPOSURE, target_exposure),
        (sl.VIDEO_SETTINGS.GAIN, target_gain),
    ):
        status = camera.set_camera_settings(setting, value)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to set ZED setting {setting}={value}: {status}")

    applied_auto = _get_video_setting(camera, sl.VIDEO_SETTINGS.AEC_AGC)
    applied_exposure = _get_video_setting(camera, sl.VIDEO_SETTINGS.EXPOSURE)
    applied_gain = _get_video_setting(camera, sl.VIDEO_SETTINGS.GAIN)
    if applied_auto or applied_exposure != target_exposure or applied_gain != target_gain:
        raise RuntimeError(
            "ZED rejected the locked exposure/gain "
            f"(wanted AEC=0 exposure={target_exposure} gain={target_gain}, "
            f"got AEC={applied_auto} exposure={applied_exposure} "
            f"gain={applied_gain})"
        )

    return {
        "locked": True,
        "auto_exposure": False,
        "exposure": applied_exposure,
        "gain": applied_gain,
    }


def grab_frame(
    camera: sl.Camera,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return BGR uint8, depth meters (NaN invalid), XYZ meters (NaN invalid)."""
    runtime = sl.RuntimeParameters()
    if camera.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        return None, None, None

    image_m = sl.Mat()
    depth_m = sl.Mat()
    xyz_m = sl.Mat()
    camera.retrieve_image(image_m, sl.VIEW.LEFT)
    camera.retrieve_measure(depth_m, sl.MEASURE.DEPTH)
    camera.retrieve_measure(xyz_m, sl.MEASURE.XYZ)

    # ZED retrieve_image returns BGRA (OpenCV order). Do NOT RGB2BGR-swap.
    bgra = image_m.get_data()
    bgr = np.ascontiguousarray(bgra[:, :, :3])
    depth = depth_m.get_data().astype(np.float32)
    depth[~np.isfinite(depth)] = np.nan
    xyz = xyz_m.get_data()[:, :, :3].astype(np.float32)
    xyz[~np.isfinite(xyz)] = np.nan
    return bgr, depth, xyz


def grab_rgb_bgr(camera: sl.Camera) -> Optional[np.ndarray]:
    bgr, _, _ = grab_frame(camera)
    return bgr


def get_left_intrinsics(camera: sl.Camera) -> Tuple[np.ndarray, np.ndarray]:
    """Return (camera_matrix 3x3, dist_coeffs) for the left image."""
    info = camera.get_camera_information()
    # SDK 4/5: calibration_parameters.left_cam
    cal = info.camera_configuration.calibration_parameters.left_cam
    fx, fy, cx, cy = cal.fx, cal.fy, cal.cx, cal.cy
    mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(
        [getattr(cal, "disto", [0, 0, 0, 0, 0])[i] if hasattr(cal, "disto") else 0.0 for i in range(5)],
        dtype=np.float64,
    )
    if hasattr(cal, "disto"):
        dist = np.asarray(cal.disto, dtype=np.float64).reshape(-1)[:5]
        if dist.size < 5:
            dist = np.pad(dist, (0, 5 - dist.size))
    return mtx, dist


def apply_depth_roi(
    mask: np.ndarray,
    depth: Optional[np.ndarray],
    params: BlanketFilterParams,
) -> np.ndarray:
    """Keep mask pixels inside the tuned image ROI and depth band."""

    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask[:, :, 0]
    out = mask.copy()
    h, w = out.shape
    x0 = int(params.x_min * w)
    x1 = max(x0 + 1, int(params.x_max * w))
    y0 = int(params.y_min * h)
    y1 = max(y0 + 1, int(params.y_max * h))
    roi = np.zeros_like(out)
    roi[y0:y1, x0:x1] = 255
    out = cv2.bitwise_and(out, roi)
    if depth is not None:
        valid = np.isfinite(depth) & (depth > 0)
        zmask = (
            valid
            & (depth >= params.depth_min_m)
            & (depth <= params.depth_max_m)
        )
        out = out.copy()
        out[~zmask] = 0
    return out


def build_blanket_mask(
    bgr: np.ndarray,
    depth: np.ndarray,
    params: BlanketFilterParams,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([params.h_min, params.s_min, params.v_min], dtype=np.uint8)
    upper = np.array([params.h_max, params.s_max, params.v_max], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = apply_depth_roi(mask, depth, params)

    if params.keep_largest_component or params.min_component_area > 0:
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8),
            connectivity=8,
        )
        if component_count > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep_labels = np.flatnonzero(
                areas >= max(1, int(params.min_component_area))
            ) + 1
            if params.keep_largest_component and len(keep_labels):
                keep_labels = np.asarray(
                    [keep_labels[np.argmax(areas[keep_labels - 1])]]
                )
            mask = np.where(
                np.isin(labels, keep_labels),
                np.uint8(255),
                np.uint8(0),
            )
    return mask


def masked_pcd(
    bgr: np.ndarray,
    xyz: np.ndarray,
    mask: np.ndarray,
) -> o3d.geometry.PointCloud:
    pts = xyz[mask > 0]
    cols = bgr[mask > 0][:, ::-1] / 255.0  # BGR -> RGB
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    cols = cols[valid]
    pcd = o3d.geometry.PointCloud()
    if len(pts):
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    return pcd


def capture_filtered_blanket(
    camera: sl.Camera,
    filter_json: Optional[str] = None,
    warm_frames: int = 45,
    grab_attempts: int = 15,
    mask_backend: str = "sam2",
) -> Tuple[
    o3d.geometry.PointCloud,
    np.ndarray,
    np.ndarray,
    BlanketFilterParams,
    np.ndarray,
    np.ndarray,
]:
    """Warm up, grab frames, keep the best HSV seed, optionally SAM2-refine.

    ``mask_backend``:
      - ``hsv``: production mask is the HSV+depth filter
      - ``sam2``: HSV box prompts SAM2; production mask is SAM ∩ depth/ROI

    Returns ``(pcd, bgr, mask, params, xyz, hsv_mask)``. ``xyz`` is HxWx3
    camera-frame metres so ArUco centers can be lifted without the mask.
    """
    backend = str(mask_backend or "sam2").strip().lower()
    if backend not in {"hsv", "sam2"}:
        raise ValueError(f"Unknown mask_backend {mask_backend!r}")
    params = load_filter_params(filter_json)
    warm_up(camera, warm_frames)

    best = None  # (keep_count, bgr, depth, xyz, hsv_mask)
    for _ in range(grab_attempts):
        bgr, depth, xyz = grab_frame(camera)
        if bgr is None:
            continue
        hsv_mask = build_blanket_mask(bgr, depth, params)
        keep = int(np.count_nonzero(hsv_mask))
        if best is None or keep > best[0]:
            best = (keep, bgr, depth, xyz, hsv_mask)

    if best is None:
        raise RuntimeError("Failed to grab ZED frame")
    keep, bgr, depth, xyz, hsv_mask = best
    mask = hsv_mask
    if backend == "sam2":
        from sam_blanket import refine_mask_sam2

        sam = refine_mask_sam2(bgr, hsv_mask)
        mask = apply_depth_roi(sam, depth, params)
        if int(np.count_nonzero(mask)) == 0:
            raise RuntimeError(
                "SAM2 produced an empty blanket mask after depth/ROI. "
                "Check the HSV seed box or use --mask-backend hsv."
            )
    pcd = masked_pcd(bgr, xyz, mask)
    return pcd, bgr, mask, params, xyz, hsv_mask


def list_zed_serials() -> list:
    devices = sl.Camera.get_device_list()
    out = []
    for d in devices:
        out.append(int(d.serial_number))
    return out
