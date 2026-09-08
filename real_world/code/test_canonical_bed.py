#!/usr/bin/env python3
"""Unit tests for 4-marker canonical bed frame (no ZED / ROS required)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from canonical_bed import (  # noqa: E402
    CanonicalBedError,
    canonical_bed_frame,
    canonical_frame_from_sim_origin,
    canonical_points_to_layout,
    canonicalize_body_xy,
    layout_points_to_canonical,
    layout_xy_to_canonical,
    load_canonical_frame,
    physical_posts,
    sample_xyz_at_pixels,
    save_canonical_frame,
    transform_action_canonical_to_layout,
    write_canonical_from_sim_origin,
)


def _axis_angle(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    theta = np.deg2rad(degrees)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


class CanonicalBedTests(unittest.TestCase):
    def test_physical_posts_collapses_top_and_side(self):
        self.assertEqual(physical_posts((0, 10, 1)), frozenset({0, 1}))
        self.assertEqual(physical_posts((0, 1, 2, 3)), frozenset({0, 1, 2, 3}))

    def test_centroid_and_axes_on_axis_aligned_rectangle(self):
        centers = {
            0: [-0.4, -0.9, 0.0],
            1: [0.4, -0.9, 0.0],
            2: [0.4, 0.9, 0.0],
            3: [-0.4, 0.9, 0.0],
        }
        frame = canonical_bed_frame(centers)
        np.testing.assert_allclose(frame.origin, [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(frame.x_axis, [1.0, 0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(frame.y_axis, [0.0, 1.0, 0.0], atol=1e-8)
        converted = layout_points_to_canonical(
            np.array(list(centers.values()), dtype=np.float64),
            frame,
        )
        np.testing.assert_allclose(
            converted[:, :2],
            [[-0.4, -0.9], [0.4, -0.9], [0.4, 0.9], [-0.4, 0.9]],
            atol=1e-8,
        )
        # No stretch onto the sim 0.88 x 2.10 mattress.
        self.assertLess(np.max(np.abs(converted[:, 0])), 0.41)

    def test_yaw_offset_is_absorbed_without_scaling(self):
        base = np.array(
            [
                [-0.4225, -0.925, 0.0],
                [0.4225, -0.925, 0.0],
                [0.4225, 0.925, 0.0],
                [-0.4225, 0.925, 0.0],
            ]
        )
        yaw = _axis_angle(np.array([0.0, 0.0, 1.0]), 7.0)
        shift = np.array([0.03, -0.02, 0.01])
        rotated = (yaw @ base.T).T + shift
        centers = {i: rotated[k] for k, i in enumerate((0, 1, 2, 3))}
        frame = canonical_bed_frame(centers)
        back = layout_points_to_canonical(rotated, frame)
        np.testing.assert_allclose(back[:, :2], base[:, :2], atol=1e-9)
        cloud = np.array([[0.1, 0.2, 0.0]])
        world = (yaw @ cloud.T).T + shift
        model = layout_points_to_canonical(world, frame)
        np.testing.assert_allclose(model, cloud, atol=1e-9)

    def test_roundtrip_layout_canonical(self):
        centers = {
            0: [-0.4, -0.9, 0.0],
            1: [0.42, -0.91, 0.01],
            2: [0.41, 0.93, -0.01],
            3: [-0.39, 0.92, 0.0],
        }
        frame = canonical_bed_frame(centers)
        points = np.array([[0.05, -0.2, 0.08], [-0.1, 0.4, 0.12]])
        layout = canonical_points_to_layout(points, frame)
        back = layout_points_to_canonical(layout, frame)
        np.testing.assert_allclose(back, points, atol=1e-12)

    def test_action_mapping_preserves_metres(self):
        centers = {
            0: [-0.4, -0.9, 0.0],
            1: [0.4, -0.9, 0.0],
            2: [0.4, 0.9, 0.0],
            3: [-0.4, 0.9, 0.0],
        }
        frame = canonical_bed_frame(centers)
        action = np.array([0.12, -0.3, 0.20, 0.1])
        layout = transform_action_canonical_to_layout(action, frame)
        np.testing.assert_allclose(layout, action + np.array([0, 0, 0, 0]), atol=1e-12)

    def test_body_xy_only(self):
        centers = {
            0: [-0.4, -0.9, 0.0],
            1: [0.4, -0.9, 0.0],
            2: [0.4, 0.9, 0.0],
            3: [-0.4, 0.9, 0.0],
        }
        frame = canonical_bed_frame(centers)
        body = np.array([[0.1, 0.2, 1.0], [-0.05, -0.4, 0.0]])
        converted = canonicalize_body_xy(body, frame)
        np.testing.assert_allclose(converted[:, :2], body[:, :2], atol=1e-12)
        np.testing.assert_array_equal(converted[:, 2], body[:, 2])

    def test_missing_marker_raises(self):
        with self.assertRaises(CanonicalBedError):
            canonical_bed_frame({0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 0]})

    def test_json_roundtrip(self):
        centers = {
            0: [-0.4, -0.9, 0.0],
            1: [0.4, -0.9, 0.0],
            2: [0.4, 0.9, 0.0],
            3: [-0.4, 0.9, 0.0],
        }
        frame = canonical_bed_frame(centers)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "canonical_bed_frame.json"
            save_canonical_frame(path, frame)
            loaded = load_canonical_frame(path)
            payload = json.loads(path.read_text())
        np.testing.assert_allclose(loaded.origin, frame.origin)
        self.assertEqual(payload["metres"], "identity")
        self.assertEqual(payload["convention"]["scale"], "none")

    def test_sample_xyz_median_window(self):
        xyz = np.full((10, 10, 3), np.nan, dtype=np.float64)
        xyz[4:7, 4:7] = (0.1, -0.2, 1.3)
        xyz[5, 5] = (0.8, 0.8, 0.8)
        sampled = sample_xyz_at_pixels(xyz, np.array([[5.0, 5.0]]), radius_px=1)
        np.testing.assert_allclose(sampled[0], [0.1, -0.2, 1.3], atol=1e-12)

    def test_canonical_from_uncovered_sim_origin(self) -> None:
        layout = {
            0: np.array([-0.4, -0.9, 0.0]),
            1: np.array([0.4, -0.9, 0.0]),
            2: np.array([0.4, 0.9, 0.0]),
            3: np.array([-0.4, 0.9, 0.0]),
        }
        t_bed_cam = np.eye(4)
        t_bed_cam[:3, 3] = [0.05, -0.02, 0.01]
        t_cam_bed = np.linalg.inv(t_bed_cam)
        centers_cam = {
            marker_id: (t_cam_bed @ np.append(point, 1.0))[:3]
            for marker_id, point in layout.items()
        }
        frame = canonical_frame_from_sim_origin(
            {
                "centers_m": centers_cam,
                "bed_from_camera": t_bed_cam,
                "detected_ids": [0, 1, 2, 3],
            }
        )
        np.testing.assert_allclose(frame.origin, [0.0, 0.0, 0.0], atol=1e-12)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sim_origin_data.pkl"
            dest_dir = Path(tmp) / "initial"
            dest_dir.mkdir()
            import pickle

            src.write_bytes(
                pickle.dumps(
                    {
                        "centers_m": centers_cam,
                        "bed_from_camera": t_bed_cam,
                        "detected_ids": [0, 1, 2, 3],
                    }
                )
            )
            wrote = write_canonical_from_sim_origin(dest_dir)
            self.assertEqual(wrote, Path(tmp) / "canonical_bed_frame.json")
            self.assertTrue(wrote.is_file())


if __name__ == "__main__":
    unittest.main()
