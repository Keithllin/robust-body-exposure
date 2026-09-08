"""Runtime helpers for the Recover 2D radius dynamics model.

The ZED capture pipeline produces points in the measured ``bed`` frame.  The
Recover checkpoint was trained with the same centred XY convention, but with
the legacy simulation surface height used by the draping preprocessor.  This
module keeps that conversion explicit and provides a small runtime graph
builder instead of depending on the old external ``build_bm_graph`` module.
"""

from __future__ import annotations

import configparser
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgba

try:
    import open3d as o3d
except ModuleNotFoundError:  # The action environment does not require ZED SDK.
    o3d = None


REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = REPO_ROOT / "code"
ASSISTIVE_ROOT = REPO_ROOT / "assistive-gym-fem"
for import_root in (CODE_ROOT, ASSISTIVE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from models_graph_res import GNNModel  # noqa: E402
from voxel_ops import xyz_voxel_representatives  # noqa: E402


DEFAULT_MODEL_DIR = (
    REPO_ROOT
    / "trained_models"
    / "FINAL_MODELS"
    / "Recover"
    / "LowDrag_FT_70k_voxelxyz05_unfold_20260819"
)
DEFAULT_UNCOVER_MODEL_DIR = (
    REPO_ROOT
    / "trained_models"
    / "GNN"
    / "GNN"
    / "standard_2D_10k_epochs=250_batch=100_workers=4_1668718872"
)

ACTION_SCALE = np.asarray([0.44, 1.05, 0.44, 1.05], dtype=np.float64)
DEFAULT_RECOVER_POINTS = 1061
# Half-width of the sim bed, and the draping rotation pivot.  Used to keep the
# draping preprocessor restricted to the side flaps; see rotate_draping_points.
DRAPING_EDGE_X = 0.44
GEN_IMAGES_COLORS = {
    "blanket": "#63BEF2",
    "prediction": "#263CC9",
    "body": "#FFBA47",
    "recover_good": "#0CD870",
    "recover_uncovered": "#D7C7EF",
    "recover_bad": "#D80C0C",
    "head_bad": "#ECA5A5",
}


def scale_action(action: Sequence[float]) -> np.ndarray:
    """Convert four normalized action values to model/world metres."""

    values = np.asarray(action, dtype=np.float64).reshape(4)
    return values * ACTION_SCALE


def normalize_action(action: Sequence[float]) -> np.ndarray:
    """Convert model/world action metres back to normalized CMA coordinates."""

    values = np.asarray(action, dtype=np.float64).reshape(4)
    return values / ACTION_SCALE


@dataclass(frozen=True)
class RecoverConfig:
    model_dir: Path
    checkpoint_path: Path
    device: str
    proc_layer_num: int
    global_size: int
    output_size: int
    node_dim: int
    edge_dim: int
    edge_threshold: float
    action_to_all: bool
    cloth_dim: int
    use_displacement: bool
    rot_draping: bool
    voxel_size: float


def _parse_float(value: str) -> float:
    if value.strip().lower() == "nan":
        return float("nan")
    return float(value)


def _checkpoint_path(model_dir: Path, checkpoint_number: int | None) -> Path:
    checkpoint_dir = model_dir / "checkpoints"
    if checkpoint_number is not None:
        path = checkpoint_dir / f"model_{checkpoint_number}.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Recover checkpoint does not exist: {path}")
        return path
    # Prefer held-out best checkpoints when present (LowDrag FT / 70k voxel).
    for preferred in (
        "model_best_lowdrag_heldout.pth",
        "model_best_heldout.pth",
    ):
        preferred_path = checkpoint_dir / preferred
        if preferred_path.is_file():
            return preferred_path
    checkpoints = []
    for path in checkpoint_dir.glob("model_*.pth"):
        match = re.fullmatch(r"model_(\d+)\.pth", path.name)
        if match is not None:
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        raise FileNotFoundError(f"No Recover checkpoints found in {checkpoint_dir}")
    return max(checkpoints, key=lambda item: item[0])[1]


def load_recover_model(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    checkpoint_number: int | None = None,
    device: str = "cpu",
    residual_model_dir: str | Path | None = None,
    residual_checkpoint_number: int | None = None,
) -> tuple[torch.nn.Module, RecoverConfig]:
    """Load Recover, optionally wrapped with a score/state residual bundle."""

    model, config = load_dynamics_model(
        model_dir,
        checkpoint_number=checkpoint_number,
        device=device,
        expected_edge_threshold=None,
        role="Recover",
    )
    if residual_model_dir is not None:
        from recover_score_corrector import (
            is_score_corrector_bundle,
            load_score_corrector_bundle,
        )

        if is_score_corrector_bundle(str(residual_model_dir)):
            model = load_score_corrector_bundle(
                base_model=model,
                bundle_dir=str(residual_model_dir),
                checkpoint_number=residual_checkpoint_number,
                device=device,
            )
        else:
            from recover_residual import load_residual_bundle

            model = load_residual_bundle(
                base_model=model,
                residual_dir=str(residual_model_dir),
                checkpoint_number=residual_checkpoint_number,
                device=device,
            )
    return model, config


def load_uncover_model(
    model_dir: str | Path = DEFAULT_UNCOVER_MODEL_DIR,
    checkpoint_number: int | None = None,
    device: str = "cpu",
) -> tuple[torch.nn.Module, RecoverConfig]:
    """Load the official GitHub Uncover 2D 10k checkpoint (model_249)."""

    return load_dynamics_model(
        model_dir,
        checkpoint_number=checkpoint_number,
        device=device,
        expected_edge_threshold=0.06,
        role="Uncover",
    )


def load_dynamics_model(
    model_dir: str | Path,
    checkpoint_number: int | None = None,
    device: str = "cpu",
    *,
    expected_edge_threshold: float | None,
    role: str,
) -> tuple[torch.nn.Module, RecoverConfig]:
    """Load a 2D radius dynamics checkpoint and validate its graph contract."""

    model_dir = Path(model_dir).expanduser().resolve()
    config_path = model_dir / "config.ini"
    if not config_path.is_file():
        raise FileNotFoundError(f"{role} config does not exist: {config_path}")

    parser = configparser.ConfigParser()
    parser.read(config_path)
    if "Model" not in parser or "Dataset" not in parser:
        raise ValueError(f"{role} config must contain Model and Dataset: {config_path}")

    model_config = parser["Model"]
    dataset_config = parser["Dataset"]
    required_model_keys = (
        "proc_layer_num",
        "global_size",
        "output_size",
        "node_dim",
        "edge_dim",
    )
    missing = [key for key in required_model_keys if key not in model_config]
    if missing:
        raise ValueError(f"{role} config is missing model fields: {missing}")

    cloth_dim = int(dataset_config.get("cloth_dim", "2"))
    use_3d = dataset_config.getboolean("use_3d", fallback=cloth_dim == 3)
    if use_3d or cloth_dim != 2:
        raise ValueError(f"This runtime only supports the {role} 2D checkpoint")

    edge_mode = dataset_config.get("edge_mode", "radius").lower()
    if edge_mode != "radius":
        raise ValueError(f"Expected {role} radius graph, got edge_mode={edge_mode!r}")

    config = RecoverConfig(
        model_dir=model_dir,
        checkpoint_path=_checkpoint_path(model_dir, checkpoint_number),
        device=device,
        proc_layer_num=int(model_config["proc_layer_num"]),
        global_size=int(model_config["global_size"]),
        output_size=int(model_config["output_size"]),
        node_dim=int(model_config["node_dim"]),
        edge_dim=int(model_config["edge_dim"]),
        edge_threshold=float(dataset_config.get("edge_threshold", "0.04")),
        action_to_all=dataset_config.getboolean("action_to_all", fallback=True),
        cloth_dim=cloth_dim,
        use_displacement=model_config.getboolean(
            "use_displacement", fallback=True
        ),
        rot_draping=dataset_config.getboolean("rot_draping", fallback=True),
        voxel_size=_parse_float(dataset_config.get("voxel_size", "nan")),
    )
    if (
        expected_edge_threshold is not None
        and not math.isclose(
            config.edge_threshold,
            expected_edge_threshold,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError(
            f"{role} runtime expects edge_threshold={expected_edge_threshold}, "
            f"got {config.edge_threshold}"
        )
    if not math.isnan(config.voxel_size) and config.voxel_size <= 0.0:
        raise ValueError(
            f"{role} voxel_size must be positive when subsampling is enabled, "
            f"got {config.voxel_size}"
        )
    if config.node_dim != 6 or config.edge_dim != 1 or config.output_size != 2:
        raise ValueError(
            f"Unexpected {role} architecture: "
            f"node_dim={config.node_dim}, edge_dim={config.edge_dim}, "
            f"output_size={config.output_size}"
        )

    args = SimpleNamespace(
        node_dim=config.node_dim,
        edge_dim=config.edge_dim,
    )
    model = GNNModel(
        args,
        config.proc_layer_num,
        config.global_size,
        config.output_size,
    )
    model.to(torch.device(device))

    checkpoint = torch.load(config.checkpoint_path, map_location=torch.device(device))
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise ValueError(
            f"Checkpoint must contain a 'model' state dict: {config.checkpoint_path}"
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config


@dataclass(frozen=True)
class ModelFrameAdapter:
    """Convert calibrated bed XYZ into the Recover model convention.

    XY is deliberately identity: the measured bed frame is centred at the
    marker rectangle and uses metres, matching the Recover action scale.
    Z is shifted only for the legacy draping preprocessor; Recover itself is
    2D and consumes X/Y after that preprocessing.
    """

    x_limits: tuple[float, float] = (-0.55, 0.55)
    y_limits: tuple[float, float] = (-1.10, 1.10)
    model_surface_z: float = 0.58
    mirror_x: bool = False

    def bed_to_model(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        converted = points.copy()
        if self.mirror_x:
            converted[:, 0] *= -1.0
        converted[:, 2] += self.model_surface_z
        return converted

    def model_to_bed(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        converted = points.copy()
        if self.mirror_x:
            converted[:, 0] *= -1.0
        converted[:, 2] -= self.model_surface_z
        return converted

    def filter_bed_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(points).all(axis=1)
        inside = (
            (points[:, 0] >= self.x_limits[0])
            & (points[:, 0] <= self.x_limits[1])
            & (points[:, 1] >= self.y_limits[0])
            & (points[:, 1] <= self.y_limits[1])
        )
        filtered = points[finite & inside]
        if len(filtered) < 3:
            raise ValueError(
                "Fewer than three blanket points remain after bed-frame crop; "
                f"x_limits={self.x_limits}, y_limits={self.y_limits}"
            )
        return filtered

    def voxel_sample(
        self,
        points: np.ndarray,
        max_points: int = DEFAULT_RECOVER_POINTS,
        voxel_size: float = float("nan"),
    ) -> np.ndarray:
        """Reduce a point cloud using the model's voxel contract.

        Voxel-aware checkpoints use the fixed XYZ representative operator
        shared with ``BMDataset``.  The occupied-cell count is the graph
        size: ``max_points`` is warn-only and must not rewrite the
        representation.  Legacy ``voxel_size=nan`` checkpoints keep the
        previous adaptive XY-centroid fallback so they remain runnable.
        """

        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if np.isfinite(voxel_size):
            sampled, _ = xyz_voxel_representatives(points, voxel_size)
            if max_points > 0 and len(sampled) > max_points:
                print(
                    "WARN: voxel graph has "
                    f"{len(sampled)} nodes > max_points={max_points}; "
                    "keeping the full XYZ voxel set (no extra subsample)."
                )
            return sampled

        return self._xy_centroid_sample(points, max_points)

    @staticmethod
    def _xy_centroid_sample(
        points: np.ndarray,
        max_points: int,
    ) -> np.ndarray:
        """Legacy adaptive XY-centroid fallback for ``voxel_size=nan``."""

        if max_points <= 0 or len(points) <= max_points:
            return points

        xy = points[:, :2]
        lower = np.min(xy, axis=0)
        span = np.maximum(np.ptp(xy, axis=0), 1e-6)
        voxel_size = max(
            1e-3,
            math.sqrt(float(np.prod(span)) / float(max_points)),
        )
        sampled = points
        for _ in range(12):
            keys = np.floor((xy - lower) / voxel_size).astype(np.int64)
            _, inverse, counts = np.unique(
                keys,
                axis=0,
                return_inverse=True,
                return_counts=True,
            )
            sampled = np.zeros((len(counts), 3), dtype=np.float64)
            np.add.at(sampled, inverse, points)
            sampled /= counts[:, None]
            if len(sampled) <= max_points:
                return sampled
            voxel_size *= math.sqrt(len(sampled) / float(max_points)) * 1.05

        # Extremely irregular masks can still leave a few extra cells after
        # the adaptive loop.  Keep a deterministic spatially distributed set.
        order = np.lexsort((sampled[:, 1], sampled[:, 0]))
        selected = np.linspace(
            0,
            len(order) - 1,
            num=max_points,
            dtype=np.int64,
        )
        return sampled[order[selected]]

    def bed_xy_to_model(self, points_xy: np.ndarray) -> np.ndarray:
        points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        converted = points_xy.copy()
        if self.mirror_x:
            converted[:, 0] *= -1.0
        return converted

    def model_xy_to_bed(self, points_xy: np.ndarray) -> np.ndarray:
        points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        converted = points_xy.copy()
        if self.mirror_x:
            converted[:, 0] *= -1.0
        return converted

    def mirror_body_points(self, body_points: np.ndarray) -> np.ndarray:
        """Mirror pose-derived body points into the same frame as the cloth."""

        body_points = np.asarray(body_points, dtype=np.float64)
        if not self.mirror_x:
            return body_points
        mirrored = body_points.copy()
        mirrored[:, 0] *= -1.0
        return mirrored

    def model_action_to_bed(self, action_model: Sequence[float]) -> np.ndarray:
        """Convert normalized-model output in metres to bed-frame XY action."""

        action_model = np.asarray(action_model, dtype=np.float64).reshape(4)
        grasp = self.model_xy_to_bed(action_model[:2][None, :])[0]
        release = self.model_xy_to_bed(action_model[2:][None, :])[0]
        return np.concatenate([grasp, release])


def _rotation_matrix(axis: Sequence[float], theta: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.asarray(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ],
        dtype=np.float64,
    )


def rotate_draping_points(
    points: np.ndarray, *, edge_x: float = DRAPING_EDGE_X
) -> np.ndarray:
    """Match the Recover training draping preprocessor in model coordinates.

    ``bm_dataset.rotate_draping_cloth_points`` selects points by ``z < 0.575``
    alone, which is safe in simulation only because the rigid bed plane keeps
    every on-bed cloth point above that height: across 800 sampled sim cloth
    states, 99.95% of the selected points sit at ``|x| > 0.44``, i.e. they are
    exactly the side flaps the rotation is meant to unfold.  Measured clouds
    straddle the calibrated bed plane instead, so a z-only test also catches
    mid-bed points and teleports them to ``|x| = 1.02 - z``.  ``edge_x``
    restricts the rotation to the bed edge; pass ``0.0`` for the literal
    training rule.
    """

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3).copy()
    right = _rotation_matrix([0.0, 1.0, 0.0], -math.pi / 2.0)
    left = _rotation_matrix([0.0, 1.0, 0.0], math.pi / 2.0)
    edge_x = abs(float(edge_x))
    for index, point in enumerate(points):
        if point[2] < 0.575 and abs(point[0]) > edge_x:
            if point[0] > 0:
                pivot = np.asarray([0.44, 0.0, 0.58])
                points[index] = right @ (point - pivot) + pivot
            elif point[0] < 0:
                pivot = np.asarray([-0.44, 0.0, 0.58])
                points[index] = left @ (point - pivot) + pivot
    return points


def _radius_edges(points_xy: np.ndarray, radius: float) -> torch.Tensor:
    try:
        from scipy.spatial import cKDTree

        pairs = cKDTree(points_xy).query_pairs(radius, output_type="ndarray")
    except (ImportError, TypeError):
        pairs_list = []
        for start in range(0, len(points_xy), 1024):
            block = points_xy[start : start + 1024]
            distances = np.linalg.norm(
                block[:, None, :] - points_xy[None, :, :], axis=2
            )
            rows, cols = np.where(distances <= radius)
            for row, col in zip(rows, cols):
                left = start + int(row)
                right = int(col)
                if left < right:
                    pairs_list.append((left, right))
        pairs = np.asarray(pairs_list, dtype=np.int64).reshape(-1, 2)

    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs) == 0:
        raise ValueError(
            f"No radius edges found for {len(points_xy)} points and radius {radius}"
        )
    directed = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    return torch.as_tensor(directed.T, dtype=torch.long).contiguous()


def compute_graph_stats(
    points_xy: np.ndarray,
    edge_index: torch.Tensor | np.ndarray,
    *,
    voxel_size: float | None = None,
    max_points: int | None = None,
) -> dict[str, float | int | bool | None]:
    """Local geometry / topology diagnostics for a 2D radius graph."""

    points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    n = int(len(points_xy))
    ei = np.asarray(edge_index)
    if ei.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got shape {ei.shape}")
    if ei.shape[0] == 2:
        src = ei[0].astype(np.int64)
        dst = ei[1].astype(np.int64)
    elif ei.shape[1] == 2:
        src = ei[:, 0].astype(np.int64)
        dst = ei[:, 1].astype(np.int64)
    else:
        raise ValueError(f"Unexpected edge_index shape: {ei.shape}")

    deg = np.bincount(src, minlength=n).astype(np.float64)
    n_edges_undirected = int(len(src) // 2)
    isolated = float(np.mean(deg == 0.0)) if n else 0.0
    mean_degree = float(deg.mean()) if n else 0.0
    p95_degree = float(np.percentile(deg, 95)) if n else 0.0

    # Connected components via BFS on undirected adjacency.
    adj: list[list[int]] = [[] for _ in range(n)]
    for left, right in zip(src.tolist(), dst.tolist()):
        adj[left].append(right)
    seen = np.zeros(n, dtype=bool)
    n_components = 0
    largest = 0
    for start in range(n):
        if seen[start]:
            continue
        n_components += 1
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in adj[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        largest = max(largest, size)
    largest_frac = float(largest / n) if n else 0.0

    if n >= 2:
        try:
            from scipy.spatial import cKDTree

            nn = cKDTree(points_xy).query(points_xy, k=2)[0][:, 1]
        except ImportError:
            distances = np.linalg.norm(
                points_xy[:, None, :] - points_xy[None, :, :], axis=2
            )
            np.fill_diagonal(distances, np.inf)
            nn = distances.min(axis=1)
        nn_med = float(np.median(nn))
        nn_p05 = float(np.percentile(nn, 5))
    else:
        nn_med = float("nan")
        nn_p05 = float("nan")

    pts_per_col = None
    n_cols = None
    if voxel_size is not None and np.isfinite(voxel_size) and voxel_size > 0 and n:
        keys = np.floor(points_xy / float(voxel_size)).astype(np.int64)
        n_cols = int(len(np.unique(keys, axis=0)))
        pts_per_col = float(n / n_cols) if n_cols else float("nan")

    max_points_triggered = bool(
        max_points is not None and max_points > 0 and n > int(max_points)
    )
    return {
        "num_nodes": n,
        "num_edges": n_edges_undirected,
        "edges_per_node": float(n_edges_undirected / n) if n else 0.0,
        "mean_degree": mean_degree,
        "degree_p95": p95_degree,
        "isolated_frac": isolated,
        "num_components": int(n_components),
        "largest_component_frac": largest_frac,
        "nn_median": nn_med,
        "nn_p05": nn_p05,
        "occupied_columns": n_cols,
        "pts_per_column": pts_per_col,
        "max_points": None if max_points is None else int(max_points),
        "max_points_triggered": max_points_triggered,
        "voxel_size": (
            None
            if voxel_size is None or not np.isfinite(voxel_size)
            else float(voxel_size)
        ),
    }


class RecoverRuntimeGraph:
    """Build the exact 2D runtime graph contract expected by Recover."""

    def __init__(
        self,
        points_model: np.ndarray,
        config: RecoverConfig,
        apply_draping: bool = True,
        action_feature_mode: str = "scaled",
        draping_edge_x: float = DRAPING_EDGE_X,
    ):
        if action_feature_mode not in ("scaled", "normalized"):
            raise ValueError(
                "action_feature_mode must be 'scaled' or 'normalized', "
                f"got {action_feature_mode!r}"
            )
        # 'scaled' matches code/evaluate_recover_state_sources.py and
        # code/bm_dataset.py; 'normalized' matches the simulation CMA runtime
        # code/build_runtime_graph.py used by run_robe_sim.
        self.action_feature_mode = action_feature_mode
        points_model = np.asarray(points_model, dtype=np.float64).reshape(-1, 3)
        if apply_draping and config.rot_draping:
            points_model = rotate_draping_points(points_model, edge_x=draping_edge_x)
        self.points_model = points_model
        self.initial_state = points_model[:, :2].copy()
        self.edge_index = _radius_edges(self.initial_state, config.edge_threshold)
        self.edge_attr = torch.zeros(
            (self.edge_index.shape[1], config.edge_dim), dtype=torch.float32
        )
        self.batch = torch.zeros(len(self.initial_state), dtype=torch.long)
        self.u = torch.zeros((1, config.global_size), dtype=torch.float32)
        if not config.action_to_all:
            raise ValueError("Recover checkpoint requires action_to_all/broadcast=True")

    def build_graph(self, normalized_action: Sequence[float]) -> dict[str, torch.Tensor]:
        if self.action_feature_mode == "scaled":
            action_values = scale_action(normalized_action)
        else:
            action_values = np.asarray(normalized_action, dtype=np.float64).reshape(4)
        action_features = np.broadcast_to(
            action_values,
            (len(self.initial_state), len(action_values)),
        )
        node_features = np.concatenate(
            [self.initial_state, action_features],
            axis=1,
        )
        return {
            "x": torch.as_tensor(node_features, dtype=torch.float32),
            "edge_attr": self.edge_attr,
            "edge_index": self.edge_index,
            "batch": self.batch,
            "u": self.u,
        }

    def predict(
        self,
        model: torch.nn.Module,
        normalized_action: Sequence[float],
        device: str,
        use_displacement: bool = True,
    ) -> np.ndarray:
        detail = self.predict_detail(
            model,
            normalized_action,
            device=device,
            use_displacement=use_displacement,
        )
        return detail["prediction"]

    def predict_detail(
        self,
        model: torch.nn.Module,
        normalized_action: Sequence[float],
        device: str,
        use_displacement: bool = True,
    ) -> dict[str, Any]:
        """Forward Recover; preserve latents when a score corrector is attached."""

        data = {
            key: value.to(torch.device(device))
            for key, value in self.build_graph(normalized_action).items()
        }
        with torch.no_grad():
            if hasattr(model, "forward_dynamics") and hasattr(
                model, "correct_single_score"
            ):
                output = model.forward_dynamics(data)
            else:
                output = model(data)
            delta = output["target"].detach()
            prediction = delta.cpu().numpy()
            latent = output.get("n_nxt")
            if latent is not None:
                latent = latent.detach()
        if use_displacement:
            prediction = self.initial_state + prediction
        return {
            "prediction": prediction,
            "base_delta": delta,
            "latent": latent,
            "action_policy": np.asarray(normalized_action, dtype=np.float64).reshape(4),
        }


def read_pcd_points(
    path: str | Path,
    adapter: ModelFrameAdapter,
    max_points: int = DEFAULT_RECOVER_POINTS,
    *,
    voxel_size: float = float("nan"),
    apply_draping: bool = False,
    draping_edge_x: float = DRAPING_EDGE_X,
) -> tuple[np.ndarray, np.ndarray]:
    """Read, transform, drape, and voxelize a measured point cloud.

    The voxel-aware path deliberately follows the simulation order:
    bed-frame crop -> model-frame conversion -> draping -> XYZ voxel
    representatives.  ``points_bed`` is the inverse-transformed sampled
    representation and therefore stays index-aligned with ``points_model``.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Point cloud does not exist: {path}")
    if o3d is not None:
        cloud = o3d.io.read_point_cloud(str(path))
        points_bed = np.asarray(cloud.points, dtype=np.float64)
    else:
        points_bed = _read_pcd_xyz(path)
    points_bed = adapter.filter_bed_points(points_bed)
    points_model = adapter.bed_to_model(points_bed)
    if apply_draping:
        points_model = rotate_draping_points(points_model, edge_x=draping_edge_x)
    points_model = adapter.voxel_sample(
        points_model,
        max_points=max_points,
        voxel_size=voxel_size,
    )
    return adapter.model_to_bed(points_model), points_model


def _read_pcd_xyz(path: Path) -> np.ndarray:
    """Read XYZ from ASCII or binary PCD without requiring Open3D."""

    header_lines = []
    data_offset = None
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
                data_offset = handle.tell()
                break
        payload = handle.read()

    if data_offset is None or data_format not in {"ascii", "binary"}:
        raise RuntimeError(f"Unsupported PCD header: {path}")

    fields = None
    sizes = None
    types = None
    counts = None
    points_count = None
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        key = parts[0].upper()
        if key == "FIELDS":
            fields = parts[1:]
        elif key == "SIZE":
            sizes = [int(value) for value in parts[1:]]
        elif key == "TYPE":
            types = parts[1:]
        elif key == "COUNT":
            counts = [int(value) for value in parts[1:]]
        elif key == "POINTS":
            points_count = int(parts[1])
    if fields is None or sizes is None or types is None:
        raise RuntimeError(f"PCD header is missing field metadata: {path}")
    if counts is None:
        counts = [1] * len(fields)

    missing = [field for field in ("x", "y", "z") if field not in fields]
    if missing:
        raise RuntimeError(f"PCD is missing XYZ fields {missing}: {path}")

    if data_format == "ascii":
        rows = np.fromstring(payload.decode("ascii"), sep=" ")
        rows = rows.reshape(-1, sum(counts))
        columns = []
        offset = 0
        for count, field in zip(counts, fields):
            if field in {"x", "y", "z"}:
                columns.append(rows[:, offset])
            offset += count
        return np.column_stack(columns).astype(np.float64, copy=False)

    dtype_map = {
        ("F", 4): np.dtype("<f4"),
        ("F", 8): np.dtype("<f8"),
        ("U", 1): np.dtype("u1"),
        ("U", 2): np.dtype("<u2"),
        ("U", 4): np.dtype("<u4"),
        ("U", 8): np.dtype("<u8"),
        ("I", 1): np.dtype("i1"),
        ("I", 2): np.dtype("<i2"),
        ("I", 4): np.dtype("<i4"),
        ("I", 8): np.dtype("<i8"),
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
        [structured[field].reshape(-1) for field in ("x", "y", "z")]
    ).astype(np.float64, copy=False)


def _read_ascii_pcd_points(path: Path) -> np.ndarray:
    """Compatibility alias for callers that used the old private helper."""

    return _read_pcd_xyz(path)


def _draw_recover_prediction_matplotlib_legacy(
    output_path: str | Path,
    original_xy: np.ndarray,
    intermediate_xy: np.ndarray,
    predicted_xy: np.ndarray,
    body_points: np.ndarray,
    action_model: Sequence[float],
    initial_covered_status: np.ndarray | None = None,
    intermediate_covered_status: np.ndarray | None = None,
    final_covered_status: np.ndarray | None = None,
    title: str = "Recover action",
) -> None:
    """Save a Recover result using the simulation ``gen_images`` appearance.

    The real-world runtime has no measured post-action blanket, so the third
    panel shows the intermediate state with the planned action and the fourth
    panel shows the model prediction after that action.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    action_model = np.asarray(action_model, dtype=np.float64).reshape(4)
    original_xy = np.asarray(original_xy, dtype=np.float64).reshape(-1, 2)
    intermediate_xy = np.asarray(intermediate_xy, dtype=np.float64).reshape(-1, 2)
    predicted_xy = np.asarray(predicted_xy, dtype=np.float64).reshape(-1, 2)
    body_points = np.asarray(body_points, dtype=np.float64)

    light_blue = GEN_IMAGES_COLORS["blanket"]
    dark_blue = GEN_IMAGES_COLORS["prediction"]
    tan = GEN_IMAGES_COLORS["body"]
    green = GEN_IMAGES_COLORS["recover_good"]
    purple = GEN_IMAGES_COLORS["recover_uncovered"]
    red = GEN_IMAGES_COLORS["recover_bad"]
    pink = GEN_IMAGES_COLORS["head_bad"]

    def valid_status(status: np.ndarray | None) -> np.ndarray | None:
        if status is None:
            return None
        array = np.asarray(status)
        if (
            array.ndim != 2
            or array.shape[0] != len(body_points)
            or array.shape[1] < 2
        ):
            return None
        return array

    initial_status = valid_status(initial_covered_status)
    intermediate_status = valid_status(intermediate_covered_status)
    final_status = valid_status(final_covered_status)

    def status_colors() -> list[tuple[float, float, float, float]]:
        """Match get_body_point_colors_recovering from cma_gnn_util.py."""

        if (
            initial_status is None
            or intermediate_status is None
            or final_status is None
        ):
            return [to_rgba(tan) for _ in body_points]

        colors = []
        for index in range(len(body_points)):
            is_target = final_status[index, 0]
            is_covered = bool(final_status[index, 1])
            is_initially_covered = bool(initial_status[index, 1])
            is_intermediately_covered = bool(intermediate_status[index, 1])

            if is_target != -1:
                if not is_intermediately_covered and is_initially_covered:
                    color = green if is_covered else purple
                elif is_intermediately_covered and not is_covered:
                    color = red
                else:
                    color = tan
            elif is_covered and not is_initially_covered:
                color = pink
            else:
                color = tan
            colors.append(
                to_rgba(
                    color,
                    0.8 if color in {green, purple, red, pink} else 1.0,
                )
            )
        return colors

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16.0, 8.0),
        dpi=100,
        sharex=True,
        sharey=True,
    )
    panel_titles = (
        "Initial blanket",
        "Manual uncover",
        "Recover action",
        "Predicted final",
    )
    grasp = action_model[:2]
    release = action_model[2:]

    def draw_blanket(axis, points, color, alpha):
        if len(points):
            axis.scatter(
                points[:, 0],
                points[:, 1],
                s=9,
                c=color,
                alpha=alpha,
                linewidths=0,
                zorder=2,
            )

    def draw_body(axis, colors):
        if body_points.ndim != 2 or body_points.shape[1] < 2 or not len(body_points):
            return
        axis.scatter(
            body_points[:, 0],
            body_points[:, 1],
            s=10,
            c=colors,
            linewidths=0,
            zorder=4,
        )

    def draw_action(axis):
        axis.scatter(
            [grasp[0]],
            [grasp[1]],
            s=36,
            c="black",
            zorder=8,
        )
        axis.annotate(
            "",
            xy=(release[0], release[1]),
            xytext=(grasp[0], grasp[1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                linewidth=4.0,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=7,
        )

    body_base_colors = [to_rgba(tan)] * len(body_points)
    body_state_colors = status_colors()
    for index, axis in enumerate(axes):
        axis.set_facecolor("white")
        axis.set_aspect("equal", adjustable="box")
        axis.invert_xaxis()
        axis.set_title(panel_titles[index], fontsize=11, pad=8)
        axis.axis("off")

    # Match the four semantic states while keeping the real-world distinction
    # between the measured intermediate state and the predicted final state.
    draw_blanket(axes[0], original_xy, light_blue, 0.10)

    draw_blanket(axes[1], original_xy, light_blue, 0.10)
    draw_blanket(axes[1], intermediate_xy, light_blue, 0.50)
    draw_body(axes[1], body_base_colors)

    draw_blanket(axes[2], intermediate_xy, light_blue, 0.70)
    draw_body(axes[2], body_base_colors)
    draw_action(axes[2])

    draw_blanket(axes[3], intermediate_xy, light_blue, 0.25)
    draw_blanket(axes[3], predicted_xy, dark_blue, 0.50)
    draw_body(axes[3], body_state_colors)
    draw_action(axes[3])

    fig.text(
        0.5,
        0.045,
        title,
        ha="center",
        va="bottom",
        fontsize=14,
    )
    fig.subplots_adjust(
        left=0.015,
        right=0.985,
        bottom=0.09,
        top=0.94,
        wspace=0.03,
    )
    fig.savefig(output_path, dpi=100, facecolor="white")
    plt.close(fig)


def draw_recover_prediction(
    output_path: str | Path,
    original_xy: np.ndarray,
    intermediate_xy: np.ndarray,
    predicted_xy: np.ndarray,
    body_points: np.ndarray,
    action_model: Sequence[float],
    initial_covered_status: np.ndarray | None = None,
    intermediate_covered_status: np.ndarray | None = None,
    final_covered_status: np.ndarray | None = None,
    title: str = "Recover action",
) -> None:
    """Render the Recover figure with the original Plotly gen-images axes."""

    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_xy = np.asarray(original_xy, dtype=np.float64).reshape(-1, 2)
    intermediate_xy = np.asarray(intermediate_xy, dtype=np.float64).reshape(-1, 2)
    predicted_xy = np.asarray(predicted_xy, dtype=np.float64).reshape(-1, 2)
    body_points = np.asarray(body_points, dtype=np.float64)
    action_model = np.asarray(action_model, dtype=np.float64).reshape(4)

    colors = {
        "blanket": "rgba(99, 190, 242, 0.5)",
        "blanket_initial": "rgba(99, 190, 242, 0.1)",
        "prediction": "rgba(38, 60, 201, 0.5)",
        "body": "rgba(255, 186, 71, 1)",
        "recover_good": "rgba(12, 216, 112, 0.8)",
        "recover_uncovered": "rgba(215, 199, 239, 0.8)",
        "recover_bad": "rgba(216, 12, 12, 0.8)",
        "head_bad": "rgba(236, 165, 165, 0.8)",
    }

    def valid_status(status: np.ndarray | None) -> np.ndarray | None:
        if status is None:
            return None
        array = np.asarray(status)
        if (
            array.ndim != 2
            or array.shape[0] != len(body_points)
            or array.shape[1] < 2
        ):
            return None
        return array

    initial_status = valid_status(initial_covered_status)
    intermediate_status = valid_status(intermediate_covered_status)
    final_status = valid_status(final_covered_status)

    def status_colors() -> list[str]:
        """Match get_body_point_colors_recovering from cma_gnn_util.py."""

        if (
            initial_status is None
            or intermediate_status is None
            or final_status is None
        ):
            return [colors["body"]] * len(body_points)

        result = []
        for index in range(len(body_points)):
            is_target = final_status[index, 0]
            is_covered = bool(final_status[index, 1])
            is_initially_covered = bool(initial_status[index, 1])
            is_intermediately_covered = bool(intermediate_status[index, 1])
            if is_target != -1:
                if not is_intermediately_covered and is_initially_covered:
                    color = (
                        colors["recover_good"]
                        if is_covered
                        else colors["recover_uncovered"]
                    )
                elif is_intermediately_covered and not is_covered:
                    color = colors["recover_bad"]
                else:
                    color = colors["body"]
            elif is_covered and not is_initially_covered:
                color = colors["head_bad"]
            else:
                color = colors["body"]
            result.append(color)
        return result

    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=(
            "Initial blanket",
            "Manual uncover",
            "Recover action",
            "Predicted final",
        ),
    )

    def add_blanket(column: int, points: np.ndarray, color: str) -> None:
        if len(points):
            fig.add_trace(
                go.Scatter(
                    mode="markers",
                    x=points[:, 0],
                    y=points[:, 1],
                    showlegend=False,
                    marker=dict(color=color, size=9),
                ),
                row=1,
                col=column,
            )

    def add_body(column: int, point_colors: str | list[str]) -> None:
        if body_points.ndim != 2 or body_points.shape[1] < 2 or not len(body_points):
            return
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=body_points[:, 0],
                y=body_points[:, 1],
                showlegend=False,
                marker=dict(color=point_colors, size=10),
            ),
            row=1,
            col=column,
        )

    def add_action(column: int) -> None:
        grasp = action_model[:2]
        release = action_model[2:]
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=[grasp[0]],
                y=[grasp[1]],
                showlegend=False,
                marker=dict(color="rgba(0, 0, 0, 1)", size=12),
            ),
            row=1,
            col=column,
        )
        axis_name = "" if column == 1 else str(column)
        fig.add_annotation(
            dict(
                ax=grasp[0],
                ay=grasp[1],
                xref=f"x{axis_name}",
                yref=f"y{axis_name}",
                text="",
                showarrow=True,
                axref=f"x{axis_name}",
                ayref=f"y{axis_name}",
                x=release[0],
                y=release[1],
                arrowhead=3,
                arrowwidth=4,
                arrowcolor="rgb(0, 0, 0)",
            )
        )

    add_blanket(1, original_xy, colors["blanket_initial"])
    add_body(1, colors["body"])

    add_blanket(2, intermediate_xy, colors["blanket"])
    add_body(2, colors["body"])

    add_blanket(3, intermediate_xy, colors["blanket"])
    add_body(3, colors["body"])
    add_action(3)

    add_blanket(4, intermediate_xy, colors["blanket"])
    add_blanket(4, predicted_xy, colors["prediction"])
    add_body(4, status_colors())
    add_action(4)

    for column in range(1, 5):
        fig.update_xaxes(
            autorange="reversed",
            visible=False,
            row=1,
            col=column,
        )
        fig.update_yaxes(
            autorange="reversed",
            visible=False,
            row=1,
            col=column,
        )

    fig.update_layout(
        width=1600,
        height=800,
        plot_bgcolor="rgba(255, 255, 255, 1)",
        paper_bgcolor="rgba(255, 255, 255, 1)",
        margin=dict(l=0, r=0, t=45, b=70),
        title=dict(
            text=title,
            y=0.08,
            x=0.5,
            xanchor="center",
            yanchor="bottom",
        ),
    )
    fig.write_image(str(output_path), width=1600, height=800)
