"""Cross-environment pickle loading helpers.

The ZED and action environments currently use different NumPy versions.
NumPy 2 serializes some array internals under ``numpy._core`` while the
legacy NumPy in ``robe`` exposes them under ``numpy.core``.  Keep the
on-disk artifacts portable instead of requiring both environments to share a
NumPy installation.
"""

from __future__ import annotations

import pickle
from pathlib import Path


class NumpyCompatUnpickler(pickle.Unpickler):
    """Resolve NumPy 2 private module names on older NumPy versions."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core") :]
        elif module == "numpy._globals":
            module = "numpy.core._globals"
        return super().find_class(module, name)


def load_pickle(path: str | Path):
    """Load a pickle produced by either the ZED or action environment."""

    with Path(path).open("rb") as handle:
        return NumpyCompatUnpickler(handle).load()
