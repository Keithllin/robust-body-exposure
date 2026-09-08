"""ID-aware ArUco geometry, PnP, and bed-frame transform helpers.

The marker layout is deliberately stored in a file rather than inferred from
detection order.  A camera pose is represented as:

    T_camera_bed: bed-frame points -> camera frame
    T_bed_camera: camera points -> bed frame

ZED capture is configured with COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP. OpenCV
solvePnP uses the optical image frame (x right, y down, z forward); the two
frames differ by a 180-degree rotation around X and are converted explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import open3d as o3d
except ModuleNotFoundError:  # Action / overlay envs may not ship Open3D.
    o3d = None


DEFAULT_ARUCO_DICTIONARY = "DICT_5X5_100"
DEFAULT_LAYOUT_PATH = (
    Path(__file__).resolve().parents[1] / "calibration" / "marker_layout.json"
)
EXPECTED_MARKERS = (0, 1, 2, 3, 10, 11, 12, 13)
ROLE_MARKERS = {
    "ceiling": (0, 1, 2, 3),
    "side_left": (10, 11, 12, 13),
    "side_right": (10, 11, 12, 13),
}
# p_image = IMAGE_FROM_ZED @ p_zed for ZED RIGHT_HANDED_Y_UP coordinates.
IMAGE_FROM_ZED = np.diag([1.0, -1.0, -1.0, 1.0])


class MarkerLayoutError(ValueError):
    """Raised when a marker layout is incomplete or geometrically invalid."""


@dataclass(frozen=True)
class MarkerSpec:
    marker_id: int
    name: str
    center_m: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray


def _unit_vector(value: Sequence[float], field: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        raise MarkerLayoutError(f"{field} must be a non-zero 3-vector")
    return vector / norm


def load_layout(
    path: Optional[str] = None,
    required_ids: Optional[Iterable[int]] = None,
) -> Tuple[dict, Dict[int, MarkerSpec]]:
    """Load and validate a marker layout.

    ``center_m``, ``u_axis``, and ``v_axis`` are intentionally required at
    calibration time. The layout dictionary is the source of truth for the
    physically printed tags, so changing print dictionaries requires changing
    the layout and all generated tags together.
    """

    layout_path = Path(path or DEFAULT_LAYOUT_PATH)
    if not layout_path.is_file():
        raise MarkerLayoutError(f"Marker layout not found: {layout_path}")

    with layout_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    dictionary = data.get("dictionary")
    if not isinstance(dictionary, str) or not hasattr(cv2.aruco, dictionary):
        raise MarkerLayoutError(
            f"OpenCV ArUco dictionary is unavailable: {dictionary!r}"
        )

    marker_size = float(data.get("marker_size_m", 0.0))
    if abs(marker_size - 0.07) > 1e-6:
        raise MarkerLayoutError(
            f"marker_size_m must be 0.07 for the installed 7 cm tags, got {marker_size}"
        )

    markers = data.get("markers")
    if not isinstance(markers, Mapping):
        raise MarkerLayoutError("layout.markers must be an object")

    ids = tuple(
        int(marker_id)
        for marker_id in (
            EXPECTED_MARKERS if required_ids is None else required_ids
        )
    )
    specs: Dict[int, MarkerSpec] = {}
    missing = []
    for marker_id in ids:
        raw = markers.get(str(marker_id))
        if not isinstance(raw, Mapping):
            missing.append(f"{marker_id}: entry")
            continue
        if raw.get("center_m") is None:
            missing.append(f"{marker_id}: center_m")
            continue
        if raw.get("u_axis") is None:
            missing.append(f"{marker_id}: u_axis")
            continue
        if raw.get("v_axis") is None:
            missing.append(f"{marker_id}: v_axis")
            continue

        center = np.asarray(raw["center_m"], dtype=np.float64).reshape(3)
        u_axis = _unit_vector(raw["u_axis"], f"{marker_id}.u_axis")
        v_axis = _unit_vector(raw["v_axis"], f"{marker_id}.v_axis")
        if abs(float(np.dot(u_axis, v_axis))) > 1e-3:
            raise MarkerLayoutError(
                f"{marker_id}.u_axis and v_axis must be perpendicular"
            )
        if not np.isfinite(center).all():
            raise MarkerLayoutError(f"{marker_id}.center_m contains non-finite values")

        specs[marker_id] = MarkerSpec(
            marker_id=marker_id,
            name=str(raw.get("name", marker_id)),
            center_m=center,
            u_axis=u_axis,
            v_axis=v_axis,
        )

    if missing:
        raise MarkerLayoutError(
            "Marker geometry is incomplete; measure and fill "
            f"{layout_path}: {', '.join(missing)}"
        )

    return data, specs


def _aruco_dictionary(name: str = DEFAULT_ARUCO_DICTIONARY):
    if not hasattr(cv2.aruco, name):
        raise MarkerLayoutError(f"OpenCV ArUco dictionary is unavailable: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _make_aruco_detector(dictionary_name: str):
    dictionary = _aruco_dictionary(dictionary_name)
    parameters = cv2.aruco.DetectorParameters()
    # The 7 cm tags can occupy a small fraction of a 1280-pixel image when
    # the ceiling camera is mounted high. OpenCV's default perimeter gate is
    # unnecessarily strict for that setup.
    parameters.minMarkerPerimeterRate = 0.01
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.minCornerDistanceRate = 0.01
    parameters.minDistanceToBorder = 1
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def detect_markers(
    image_bgr: np.ndarray,
    dictionary_name: str = DEFAULT_ARUCO_DICTIONARY,
) -> Dict[int, np.ndarray]:
    """Detect markers and return {ID: 4x2 image corners}.

    The image corner order is OpenCV's top-left, top-right, bottom-right,
    bottom-left order. Duplicate IDs are rejected rather than silently
    replacing one detection. A 3x fallback is used for small 7 cm tags in
    HD720 frames; returned corners are always in the original image pixels.
    """

    detector = _make_aruco_detector(dictionary_name)
    corners, ids, _rejected = detector.detectMarkers(image_bgr)
    scale = 1.0
    if ids is None:
        scale = 3.0
        enlarged = cv2.resize(
            image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        corners, ids, _rejected = detector.detectMarkers(enlarged)
    if ids is None:
        return {}

    detections: Dict[int, np.ndarray] = {}
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        marker_id = int(marker_id)
        if marker_id in detections:
            raise MarkerLayoutError(f"Duplicate ArUco ID detected: {marker_id}")
        detections[marker_id] = (
            np.asarray(marker_corners, dtype=np.float64).reshape(4, 2) / scale
        )
    return detections


def marker_object_corners(spec: MarkerSpec, marker_size_m: float) -> np.ndarray:
    """Return 3D corners matching OpenCV's detected-corner ordering."""

    half = float(marker_size_m) / 2.0
    local = np.asarray(
        [
            [-half, -half],
            [half, -half],
            [half, half],
            [-half, half],
        ],
        dtype=np.float64,
    )
    return (
        spec.center_m[None, :]
        + local[:, 0, None] * spec.u_axis[None, :]
        + local[:, 1, None] * spec.v_axis[None, :]
    )


def collect_correspondences(
    detections: Mapping[int, np.ndarray],
    specs: Mapping[int, MarkerSpec],
    marker_size_m: float,
    required_ids: Optional[Iterable[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    """Build stacked 3D/2D corner correspondences in stable ID order."""

    allowed = set(
        int(marker_id)
        for marker_id in (specs.keys() if required_ids is None else required_ids)
    )
    visible = tuple(sorted(set(detections).intersection(allowed).intersection(specs)))
    if not visible:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            visible,
        )

    object_points = np.concatenate(
        [marker_object_corners(specs[marker_id], marker_size_m) for marker_id in visible],
        axis=0,
    )
    image_points = np.concatenate(
        [np.asarray(detections[marker_id], dtype=np.float64) for marker_id in visible],
        axis=0,
    )
    return object_points, image_points, visible


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def bed_from_image_camera_to_bed_from_zed_camera(
    t_bed_image_camera: np.ndarray,
) -> np.ndarray:
    """Convert a bed<-OpenCV-image pose to a bed<-ZED-RIGHT_HANDED_Y_UP pose."""

    return np.asarray(t_bed_image_camera, dtype=np.float64).reshape(4, 4) @ IMAGE_FROM_ZED


def bed_from_zed_camera_to_bed_from_image_camera(
    t_bed_zed_camera: np.ndarray,
) -> np.ndarray:
    """Convert a bed<-ZED pose to a bed<-OpenCV-image pose."""

    return np.asarray(t_bed_zed_camera, dtype=np.float64).reshape(4, 4) @ IMAGE_FROM_ZED


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    flat = points.reshape(-1, 3)
    homogeneous = np.concatenate(
        [flat, np.ones((len(flat), 1), dtype=np.float64)], axis=1
    )
    transformed = (np.asarray(transform) @ homogeneous.T).T[:, :3]
    return transformed.reshape(points.shape)


def transform_pcd(
    pcd: o3d.geometry.PointCloud, transform: np.ndarray
) -> o3d.geometry.PointCloud:
    result = o3d.geometry.PointCloud(pcd)
    result.transform(np.asarray(transform, dtype=np.float64))
    return result


def crop_pcd_to_bed_xy(
    pcd: o3d.geometry.PointCloud,
    x_limits: Sequence[float] = (-0.55, 0.55),
    y_limits: Sequence[float] = (-1.10, 1.10),
) -> o3d.geometry.PointCloud:
    """Remove bed-frame points outside the physical blanket footprint."""

    points = np.asarray(pcd.points, dtype=np.float64)
    if points.size == 0:
        return o3d.geometry.PointCloud()
    x_limits = tuple(float(value) for value in x_limits)
    y_limits = tuple(float(value) for value in y_limits)
    if x_limits[0] >= x_limits[1] or y_limits[0] >= y_limits[1]:
        raise MarkerLayoutError(
            f"Invalid bed crop limits: x={x_limits}, y={y_limits}"
        )
    keep = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= x_limits[0])
        & (points[:, 0] <= x_limits[1])
        & (points[:, 1] >= y_limits[0])
        & (points[:, 1] <= y_limits[1])
    )
    return pcd.select_by_index(np.flatnonzero(keep).tolist())


def _pnp_reprojection_rms(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> Tuple[float, np.ndarray]:
    projected, _ = cv2.projectPoints(
        object_points,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        camera_matrix,
        distortion,
    )
    projected = projected.reshape(-1, 2)
    residuals = np.linalg.norm(projected - image_points, axis=1)
    return float(np.sqrt(np.mean(residuals**2))), residuals


def solve_bed_to_camera(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> dict:
    """Solve bed->camera PnP and return both transform directions."""

    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64)
    if len(object_points) < 4:
        raise MarkerLayoutError(
            f"At least 4 marker corners are required, got {len(object_points)}"
        )

    centered = object_points - object_points.mean(axis=0, keepdims=True)
    rank = int(np.linalg.matrix_rank(centered, tol=1e-8))

    # IPPE is attractive for planar targets but can return a mirrored solution
    # with a low reprojection error. A valid camera pose must put every marker
    # corner in front of the OpenCV camera, so reject candidates with any
    # non-positive camera-frame depth before comparing reprojection errors.
    candidates = []
    flag_names = []
    if hasattr(cv2, "SOLVEPNP_SQPNP"):
        flag_names.append(("SQPNP", cv2.SOLVEPNP_SQPNP))
    flag_names.append(("ITERATIVE", cv2.SOLVEPNP_ITERATIVE))
    flag_names.append(("EPNP", cv2.SOLVEPNP_EPNP))
    if rank <= 2 and hasattr(cv2, "SOLVEPNP_IPPE"):
        flag_names.append(("IPPE", cv2.SOLVEPNP_IPPE))

    for name, flags in flag_names:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points, image_points, camera_matrix, distortion, flags=flags
            )
        except cv2.error:
            continue
        if not ok:
            continue
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        camera_points = (
            rotation @ object_points.T
            + np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        )
        if not np.isfinite(camera_points).all() or np.min(camera_points[2]) <= 1e-6:
            continue
        rms, _ = _pnp_reprojection_rms(
            object_points, image_points, camera_matrix, distortion, rvec, tvec
        )
        candidates.append((rms, name, rvec, tvec))

    if not candidates:
        raise MarkerLayoutError(
            "No valid PnP candidate; all candidates either failed or place "
            "a marker corner behind the OpenCV camera."
        )

    candidates.sort(key=lambda item: item[0])
    best_rms, method, rvec, tvec = candidates[0]
    try:
        ok, rvec_ref, tvec_ref = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            refined_rms, _ = _pnp_reprojection_rms(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                rvec_ref,
                tvec_ref,
            )
            refined_rotation, _ = cv2.Rodrigues(
                np.asarray(rvec_ref, dtype=np.float64).reshape(3, 1)
            )
            refined_camera_points = (
                refined_rotation @ object_points.T
                + np.asarray(tvec_ref, dtype=np.float64).reshape(3, 1)
            )
            if (
                refined_rms <= best_rms + 1e-9
                and np.isfinite(refined_camera_points).all()
                and np.min(refined_camera_points[2]) > 1e-6
            ):
                rvec, tvec, best_rms, method = (
                    rvec_ref,
                    tvec_ref,
                    refined_rms,
                    f"{method}+ITERATIVE",
                )
    except cv2.error:
        pass

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t_camera_bed = make_transform(
        rotation, np.asarray(tvec, dtype=np.float64).reshape(3)
    )
    t_bed_camera = invert_transform(t_camera_bed)
    _, residuals = _pnp_reprojection_rms(
        object_points, image_points, camera_matrix, distortion, rvec, tvec
    )
    return {
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3).tolist(),
        "tvec_m": np.asarray(tvec, dtype=np.float64).reshape(3).tolist(),
        "T_camera_bed": t_camera_bed.tolist(),
        "T_bed_camera": t_bed_camera.tolist(),
        "reprojection_rms_px": float(np.sqrt(np.mean(residuals**2))),
        "reprojection_max_px": float(np.max(residuals)),
        "num_points": int(len(object_points)),
        "rank": rank,
        "pnp_method": method,
    }


def draw_detections(
    image_bgr: np.ndarray,
    detections: Mapping[int, np.ndarray],
    expected_ids: Optional[Iterable[int]] = None,
) -> np.ndarray:
    output = image_bgr.copy()
    expected = set(expected_ids or detections.keys())
    for marker_id, corners in detections.items():
        corners_int = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
        color = (0, 220, 0) if marker_id in expected else (0, 140, 255)
        cv2.polylines(output, [corners_int], True, color, 2)
        center = np.round(corners.mean(axis=0)).astype(int)
        cv2.putText(
            output,
            str(marker_id),
            tuple(center + np.array([8, -8])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def project_marker_corners(
    specs: Mapping[int, MarkerSpec],
    marker_size_m: float,
    pose: Mapping[str, object],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    marker_ids: Iterable[int],
) -> Dict[int, np.ndarray]:
    """Project layout corners into the image using a solved bed->camera pose."""

    rvec = np.asarray(pose["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(pose["tvec_m"], dtype=np.float64).reshape(3, 1)
    projected: Dict[int, np.ndarray] = {}
    for marker_id in marker_ids:
        object_points = marker_object_corners(specs[int(marker_id)], marker_size_m)
        points, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(distortion, dtype=np.float64),
        )
        projected[int(marker_id)] = points.reshape(4, 2)
    return projected


def draw_reprojection_overlay(
    image_bgr: np.ndarray,
    detections: Mapping[int, np.ndarray],
    projected: Mapping[int, np.ndarray],
    marker_ids: Optional[Iterable[int]] = None,
    arrow_scale: float = 2.0,
) -> np.ndarray:
    """Overlay detected vs projected corners and residual arrows.

    Legend:
      green outline / circles = detected corners
      magenta outline / crosses = projected layout corners
      yellow arrows = detected -> projected (2x by default)
    """

    output = image_bgr.copy()
    ids = tuple(
        sorted(
            int(marker_id)
            for marker_id in (
                marker_ids
                if marker_ids is not None
                else set(detections).intersection(projected)
            )
        )
    )
    for marker_id in ids:
        if marker_id not in detections or marker_id not in projected:
            continue
        detected = np.asarray(detections[marker_id], dtype=np.float64).reshape(4, 2)
        projected_corners = np.asarray(projected[marker_id], dtype=np.float64).reshape(
            4, 2
        )
        residuals = projected_corners - detected
        residual_norms = np.linalg.norm(residuals, axis=1)
        marker_rms = float(np.sqrt(np.mean(residual_norms**2)))

        detected_int = np.round(detected).astype(np.int32)
        projected_int = np.round(projected_corners).astype(np.int32)
        cv2.polylines(output, [detected_int.reshape(-1, 1, 2)], True, (0, 220, 0), 2)
        cv2.polylines(
            output, [projected_int.reshape(-1, 1, 2)], True, (255, 0, 255), 2
        )

        for corner_idx, (det_pt, proj_pt, residual) in enumerate(
            zip(detected, projected_corners, residuals)
        ):
            det_i = tuple(np.round(det_pt).astype(int))
            proj_i = tuple(np.round(proj_pt).astype(int))
            cv2.circle(output, det_i, 4, (0, 220, 0), -1, cv2.LINE_AA)
            cv2.drawMarker(
                output,
                proj_i,
                (255, 0, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=10,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            tip = det_pt + float(arrow_scale) * residual
            tip_i = tuple(np.round(tip).astype(int))
            cv2.arrowedLine(
                output,
                det_i,
                tip_i,
                (0, 255, 255),
                2,
                tipLength=0.25,
                line_type=cv2.LINE_AA,
            )
            cv2.putText(
                output,
                str(corner_idx),
                (det_i[0] + 6, det_i[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        center = np.round(detected.mean(axis=0)).astype(int)
        label = f"{marker_id}:{marker_rms:.1f}px"
        cv2.putText(
            output,
            label,
            (int(center[0]) + 10, int(center[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            (int(center[0]) + 10, int(center[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    legend = [
        "green=detected  magenta=projected",
        f"yellow arrow=detected->projected x{arrow_scale:g}",
        "corner 0=TL 1=TR 2=BR 3=BL",
    ]
    y = 28
    for line in legend:
        cv2.putText(
            output,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return output


def _pixel_to_bed_affine(
    centers_px: Mapping[int, Sequence[float]],
    specs: Mapping[int, MarkerSpec],
) -> np.ndarray:
    ids = sorted(set(centers_px).intersection(specs))
    if len(ids) < 3:
        return np.full((2, 3), np.nan, dtype=np.float64)
    pixels = np.asarray(
        [[*np.asarray(centers_px[i], dtype=np.float64), 1.0] for i in ids]
    )
    world_xy = np.asarray([specs[i].center_m[:2] for i in ids], dtype=np.float64)
    # world_xy.T = affine @ pixels.T
    return (np.linalg.pinv(pixels) @ world_xy).T


def pixel_to_bed_xy(
    pixels_px: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    """Map image pixels to bed-frame XY using saved marker geometry."""

    pixels_px = np.asarray(pixels_px, dtype=np.float64).reshape(-1, 2)
    affine = np.asarray(affine, dtype=np.float64).reshape(2, 3)
    if not np.isfinite(affine).all():
        raise MarkerLayoutError("pixel_to_bed_xy affine contains non-finite values")
    homogeneous = np.concatenate(
        [pixels_px, np.ones((len(pixels_px), 1), dtype=np.float64)],
        axis=1,
    )
    return homogeneous @ affine.T


def bed_xy_to_pixel(
    bed_xy: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    """Map bed-frame XY back to image pixels for diagnostic overlays.

    ``affine`` is the 2x3 map ``bed = A @ [u, v, 1]``. Invert the linear
    block explicitly; a homogeneous pinv on the 2x3 matrix is ill-posed and
    projects grasp/release far outside the image.
    """

    bed_xy = np.asarray(bed_xy, dtype=np.float64).reshape(-1, 2)
    affine = np.asarray(affine, dtype=np.float64).reshape(2, 3)
    if not np.isfinite(affine).all():
        raise MarkerLayoutError("pixel_to_bed_xy affine contains non-finite values")
    linear = affine[:, :2]
    translation = affine[:, 2]
    if abs(float(np.linalg.det(linear))) < 1e-12:
        raise MarkerLayoutError("pixel_to_bed_xy linear block is singular")
    pixels = (np.linalg.inv(linear) @ (bed_xy.T - translation[:, None])).T
    if not np.isfinite(pixels).all():
        raise MarkerLayoutError("bed_xy_to_pixel produced non-finite pixels")
    return pixels


def build_sim_origin_data(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    detections: Mapping[int, np.ndarray],
    specs: Mapping[int, MarkerSpec],
    pose: Mapping[str, object],
    marker_size_m: float,
) -> dict:
    """Build current legacy fields plus accurate bed-frame metadata."""

    centers_px = {
        int(marker_id): np.asarray(corners).mean(axis=0).tolist()
        for marker_id, corners in detections.items()
    }
    camera_from_bed = np.asarray(pose["T_camera_bed"], dtype=np.float64)
    origin_projection, _ = cv2.projectPoints(
        np.zeros((1, 3), dtype=np.float64),
        np.asarray(pose["rvec"], dtype=np.float64),
        np.asarray(pose["tvec_m"], dtype=np.float64),
        camera_matrix,
        distortion,
    )
    origin_px = origin_projection.reshape(2)

    camera_centers = {
        int(marker_id): transform_points(
            specs[marker_id].center_m, camera_from_bed
        ).tolist()
        for marker_id in set(detections).intersection(specs)
    }

    affine = _pixel_to_bed_affine(centers_px, specs)
    # Legacy code multiplies pixel deltas by [scale_y, scale_x] and swaps
    # the resulting columns. Keep a coarse scale while exposing the full
    # affine mapping for new consumers.
    scale_x = np.linalg.norm(affine[0, :2]) if np.isfinite(affine).all() else np.nan
    scale_y = np.linalg.norm(affine[1, :2]) if np.isfinite(affine).all() else np.nan
    legacy_scale = np.asarray([scale_y, scale_x], dtype=np.float64)

    return {
        "dist": np.asarray(distortion, dtype=np.float64),
        "mtx": np.asarray(camera_matrix, dtype=np.float64),
        "centers_px": centers_px,
        "centers_m": camera_centers,
        "origin_px": origin_px,
        "origin_m": np.zeros(3, dtype=np.float64),
        "m2px_scale": legacy_scale,
        "pixel_to_bed_xy": affine,
        "bed_from_camera": np.asarray(pose["T_bed_camera"], dtype=np.float64),
        "camera_from_bed": np.asarray(pose["T_camera_bed"], dtype=np.float64),
        "camera_coordinate_system": "OPENCV_IMAGE",
        "marker_size_m": float(marker_size_m),
        "image_shape_hw": list(image_bgr.shape[:2]),
        "detected_ids": sorted(int(i) for i in detections),
        "reprojection_rms_px": float(pose["reprojection_rms_px"]),
    }
