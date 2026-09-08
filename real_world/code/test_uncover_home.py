"""Along-bed return math: no rotate, session home not overwritten."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from uncover_home import along_bed_return, record_uncover_home  # noqa: E402


def _t(x: float, y: float, yaw: float = 0.0) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    t = np.eye(4)
    t[0, 0], t[0, 1], t[1, 0], t[1, 1] = c, -s, s, c
    t[0, 3], t[1, 3] = x, y
    return t


class UncoverHomeTest(unittest.TestCase):
    def test_along_track_only(self) -> None:
        home = _t(-0.80, -0.01)
        now = _t(0.53, -0.01)
        cmd = along_bed_return(now, home)
        self.assertAlmostEqual(cmd["translate_mobile_base"], -1.33, places=2)
        self.assertLess(cmd["lateral_m"], 1e-6)
        self.assertLess(abs(cmd["yaw_err_rad"]), 1e-6)

    def test_lateral_is_not_eaten(self) -> None:
        home = _t(-0.80, 0.00)
        now = _t(-0.80, 0.06)
        cmd = along_bed_return(now, home)
        self.assertAlmostEqual(cmd["translate_mobile_base"], 0.0, places=3)
        self.assertAlmostEqual(cmd["lateral_m"], 0.06, places=3)

    def test_session_home_not_overwritten(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "20260901_exp00"
            stretch = session / "stretch"
            stretch.mkdir(parents=True)
            pose = root / "pose_1_TL2_1"
            pose.mkdir()
            first = {"T_odom_base": _t(-0.80, -0.01).tolist()}
            (pose / "snap.json").write_text(json.dumps(first))
            record_uncover_home(pose / "snap.json", pose, session)
            later = root / "pose_1_TL2_2"
            later.mkdir()
            (later / "snap.json").write_text(
                json.dumps({"T_odom_base": _t(0.53, 0.04).tolist()})
            )
            record_uncover_home(later / "snap.json", later, session)
            kept = json.loads((stretch / "uncover_home.json").read_text())
            self.assertAlmostEqual(kept["T_odom_base"][0][3], -0.80, places=2)
            trial2 = json.loads((later / "uncover_home.json").read_text())
            self.assertAlmostEqual(trial2["T_odom_base"][0][3], 0.53, places=2)


if __name__ == "__main__":
    unittest.main()
