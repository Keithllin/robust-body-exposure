#!/usr/bin/env python3
"""Foreign site-packages must not leak into robe / robe-zed children."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from conda_python import env_for_python  # noqa: E402


class EnvForPythonTests(unittest.TestCase):
    def test_strips_other_conda_site_packages(self) -> None:
        foreign = "/tmp/fake-conda-env/lib/python3.11/site-packages"
        code = str(CODE_DIR)
        previous = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = os.pathsep.join([code, foreign])
        try:
            env = env_for_python(sys.executable)
        finally:
            if previous is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous
        path = env.get("PYTHONPATH", "")
        self.assertNotIn("fake-conda-env", path)
        self.assertIn(code, path.split(os.pathsep))

    def test_drops_foreign_pythonhome(self) -> None:
        previous = os.environ.get("PYTHONHOME")
        os.environ["PYTHONHOME"] = "/tmp/fake-conda-env"
        try:
            env = env_for_python(sys.executable)
        finally:
            if previous is None:
                os.environ.pop("PYTHONHOME", None)
            else:
                os.environ["PYTHONHOME"] = previous
        self.assertNotIn("PYTHONHOME", env)


if __name__ == "__main__":
    unittest.main()
