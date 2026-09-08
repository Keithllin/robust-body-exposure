#!/usr/bin/env python3
"""Upper body is two full XY spheres, matching the sim 2D GNN torso."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ASSISTIVE = Path(__file__).resolve().parents[1] / "assistive-gym-fem"
if str(ASSISTIVE) not in sys.path:
    sys.path.insert(0, str(ASSISTIVE))

from assistive_gym.envs.bu_gnn_util import (  # noqa: E402
    get_circular_limb_points,
    get_torso_points,
)


def _pose() -> np.ndarray:
    pose = np.zeros((14, 2), dtype=np.float64)
    pose[2] = [-0.2, -0.4]  # right shoulder
    pose[8] = [0.2, -0.4]  # left shoulder
    pose[5] = [-0.15, 0.2]  # right hip
    pose[11] = [0.15, 0.2]  # left hip
    pose[12] = [0.0, -0.1]  # torso center
    return pose


class TorsoSphereTests(unittest.TestCase):
    def test_two_full_spheres_same_count(self):
        points = get_torso_points(
            _pose(),
            radius_upperchest=0.16,
            radius_waist=0.14,
            num_rings=4,
        )
        n_one = len(get_circular_limb_points([0.0, 0.0], radius=0.16, num_rings=4))
        self.assertEqual(len(points), 2 * n_one)
        self.assertGreater(n_one, 1)

    def test_tiny_waist_is_copied_not_a_point(self):
        points = get_torso_points(
            _pose(),
            radius_upperchest=0.16,
            radius_waist=0.0,
            num_rings=4,
        )
        n_one = len(get_circular_limb_points([0.0, 0.0], radius=0.16, num_rings=4))
        self.assertEqual(len(points), 2 * n_one)
        spread = np.ptp(points[:n_one], axis=0)
        spread_w = np.ptp(points[n_one:], axis=0)
        self.assertGreater(float(np.linalg.norm(spread)), 0.05)
        self.assertGreater(float(np.linalg.norm(spread_w)), 0.05)

    def test_num_rings_one_becomes_a_disk(self):
        disk = get_circular_limb_points([0.0, 0.0], radius=0.1, num_rings=1)
        self.assertGreater(len(disk), 1)


if __name__ == "__main__":
    unittest.main()
