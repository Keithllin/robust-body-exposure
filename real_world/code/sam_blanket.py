"""SAM2 blanket refine: HSV mask supplies the box, SAM supplies pixels.

Used by live ZED capture in robe-zed. HSV filter JSON is unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
REAL_WORLD_DIR = CODE_DIR.parent
ROBE_ROOT = REAL_WORLD_DIR.parent
DEFAULT_WEIGHTS = "sam2.1_b.pt"

_MODEL = None
_DEVICE = None
_WEIGHTS = None


def bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def resolve_sam2_weights() -> str:
    explicit = os.environ.get("ROBE_SAM2_WEIGHTS", "").strip()
    candidates = [
        Path(explicit) if explicit else None,
        ROBE_ROOT / DEFAULT_WEIGHTS,
        REAL_WORLD_DIR / DEFAULT_WEIGHTS,
        CODE_DIR / DEFAULT_WEIGHTS,
        Path(DEFAULT_WEIGHTS),
    ]
    for path in candidates:
        if path is not None and path.is_file():
            return str(path.resolve())
    return DEFAULT_WEIGHTS


def resolve_sam_device() -> str:
    explicit = os.environ.get("ROBE_SAM_DEVICE", "").strip()
    if explicit:
        device = explicit
        if device in ("0", "cuda"):
            device = "cuda:0"
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_sam2(weights: str | None = None, device: str | None = None):
    global _MODEL, _DEVICE, _WEIGHTS
    weights = weights or resolve_sam2_weights()
    device = device or resolve_sam_device()
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            print("WARN: CUDA not available; SAM2 using cpu")
            device = "cpu"
    if _MODEL is not None and _WEIGHTS == weights and _DEVICE == device:
        return _MODEL, _DEVICE, _WEIGHTS
    from ultralytics import SAM

    _MODEL = SAM(weights)
    _MODEL.predict_params = getattr(_MODEL, "predict_params", {})
    _DEVICE = device
    _WEIGHTS = weights
    print(f"SAM2 loaded weights={weights} device={device}")
    return _MODEL, _DEVICE, _WEIGHTS


def refine_mask_sam2(
    bgr: np.ndarray,
    hsv_mask: np.ndarray,
    *,
    weights: str | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Segment RGB with SAM2, boxed by the HSV seed. Empty seed → empty mask."""

    if hsv_mask.shape[:2] != bgr.shape[:2]:
        hsv_mask = cv2.resize(
            hsv_mask,
            (bgr.shape[1], bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    box = bbox_xyxy(hsv_mask)
    if box is None:
        return np.zeros(bgr.shape[:2], dtype=np.uint8)
    model, device, _weights = load_sam2(weights, device)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = model.predict(rgb, bboxes=[box], verbose=False, device=device)
    masks = result[0].masks
    if masks is None or masks.data is None or len(masks.data) == 0:
        return np.zeros(bgr.shape[:2], dtype=np.uint8)
    m = np.asarray(masks.data[0].detach().cpu().numpy(), dtype=np.float32)
    if m.ndim > 2:
        m = np.squeeze(m)
    m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(m > 0.5, np.uint8(255), np.uint8(0))
