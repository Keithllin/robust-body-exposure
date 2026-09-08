"""Oracle grasp attachment helpers for recover/uncover dynamics models.

Prefer stored simulator `anchor_idx` when present. If missing, recompute using the
same singulate/nearest rules as `robe_bm_reversible.recover_step` /
`uncover_step` so existing PKLs can still train an Oracle-attachment model.
"""

from __future__ import print_function

import numpy as np

from cloth_mesh_edges import DEFAULT_CLOTH_MESH_PATH


ACTION_SCALE = np.asarray([0.44, 1.05], dtype=np.float64)
CLIPPING_THRES = 0.028

_MESH_NEIGHBORS = None


def _scale_action(action, scale=ACTION_SCALE):
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    multiplier = len(action) // 2
    return action * np.tile(np.asarray(scale, dtype=np.float64), multiplier)


def _mesh_neighbors():
    global _MESH_NEIGHBORS
    if _MESH_NEIGHBORS is None:
        import trimesh
        mesh = trimesh.load(str(DEFAULT_CLOTH_MESH_PATH), process=False)
        neighbors = {int(i): [] for i in range(len(mesh.vertices))}
        for v1, v2 in np.asarray(mesh.edges, dtype=np.int64):
            v1 = int(v1)
            v2 = int(v2)
            neighbors[v1].append(v2)
            neighbors[v2].append(v1)
        _MESH_NEIGHBORS = neighbors
    return _MESH_NEIGHBORS


def _singulate_layer_height(grasp_loc, cloth, radius=0.028):
    del radius  # kept for API parity with bu_gnn_util
    cloth = np.asarray(cloth, dtype=np.float64)
    dist = np.linalg.norm(cloth[:, 0:2] - np.asarray(grasp_loc, dtype=np.float64)[0:2], axis=1)
    closest_points = np.argpartition(dist, 10)[:5]
    highest_local = int(np.argmax(cloth[closest_points, 2]))
    return int(closest_points[highest_local])


def recompute_anchor_idx(action, cloth_points, singulate_layers=True, clipping_thres=CLIPPING_THRES):
    """Reproduce simulator attachment selection for a pick/place action."""
    cloth = np.asarray(cloth_points, dtype=np.float64)
    if cloth.ndim != 2 or cloth.shape[1] < 2:
        return []

    action_scaled = _scale_action(action)
    grasp_loc = action_scaled[0:2]
    dist = np.linalg.norm(cloth[:, 0:2] - grasp_loc, axis=1)
    if dist.size == 0 or not np.any(dist < float(clipping_thres)):
        return []

    if singulate_layers:
        if cloth.shape[1] < 3:
            raise ValueError('singulate_layers oracle anchors require 3D cloth points')
        highest_vertex = _singulate_layer_height(grasp_loc, cloth)
        neighbors = list(_mesh_neighbors().get(highest_vertex, []))
        for i, v1 in enumerate(neighbors):
            rest = neighbors[:i] + neighbors[i + 1:]
            for v2 in rest:
                if v1 in _mesh_neighbors().get(v2, []):
                    return [int(highest_vertex), int(v1), int(v2)]
        # Fallback matches uncover_step when a triangle cannot be found.
        nearest = list(np.argpartition(dist, min(4, len(dist) - 1))[:4])
        return [int(v) for v in nearest]

    nearest = list(np.argpartition(dist, min(4, len(dist) - 1))[:4])
    return [int(v) for v in nearest]


def extract_stored_anchor_idx(raw_data):
    """Return stored simulator anchors if present, else None."""
    if not isinstance(raw_data, dict):
        return None

    info = raw_data.get('info') or {}
    dci = raw_data.get('data_collection_info') or {}

    for candidate in (
        info.get('anchor_idx'),
        info.get('recover_anchor_idx'),
        dci.get('anchor_idx'),
        dci.get('recover_anchor_idx'),
    ):
        parsed = _as_int_list(candidate)
        if parsed is not None:
            return parsed

    trajectory = info.get('cloth_trajectory')
    if isinstance(trajectory, dict):
        frames = trajectory.get('frames') or []
        for frame in reversed(frames):
            if not isinstance(frame, dict):
                continue
            phase = str(frame.get('phase', ''))
            if 'recover' not in phase and 'uncover' not in phase:
                continue
            parsed = _as_int_list(frame.get('anchor_idx'))
            if parsed is not None:
                return parsed
    return None


def resolve_oracle_anchor_idx(
    raw_data,
    action,
    cloth_points,
    singulate_layers=True,
    allow_recompute=True,
):
    """Prefer stored anchors; optionally recompute from cloth+action."""
    stored = extract_stored_anchor_idx(raw_data)
    if stored is not None:
        return stored, 'stored'
    if not allow_recompute:
        return [], 'missing'
    recomputed = recompute_anchor_idx(
        action,
        cloth_points,
        singulate_layers=singulate_layers,
    )
    return recomputed, 'recomputed'


def anchor_mask(num_nodes, anchor_idx):
    mask = np.zeros((int(num_nodes), 1), dtype=np.float32)
    for idx in _as_int_list(anchor_idx) or []:
        if 0 <= int(idx) < int(num_nodes):
            mask[int(idx), 0] = 1.0
    return mask


def build_gated_action_rows(num_nodes, action_scaled, anchor_idx):
    """Return (N, A) action features: action only on attached particles, else zeros."""
    action_scaled = np.asarray(action_scaled, dtype=np.float32).reshape(-1)
    rows = np.zeros((int(num_nodes), int(action_scaled.size)), dtype=np.float32)
    for idx in _as_int_list(anchor_idx) or []:
        i = int(idx)
        if 0 <= i < int(num_nodes):
            rows[i] = action_scaled
    return rows


def build_node_xyz_action_features(cloth_coords, action_scaled, action_mode, cloth_dim, anchor_idx=None):
    """Build [xyz_or_xy | action] node features for broadcast or oracle_triangle gating."""
    cloth = np.asarray(cloth_coords, dtype=np.float32)
    if cloth.ndim != 2:
        raise ValueError('cloth_coords must be (N, D), got %s' % (cloth.shape,))
    n = cloth.shape[0]
    pos = cloth[:, :int(cloth_dim)]
    action_scaled = np.asarray(action_scaled, dtype=np.float32).reshape(-1)
    mode = str(action_mode).strip().lower()
    if mode == 'broadcast':
        actions = np.tile(action_scaled, (n, 1))
    elif mode == 'oracle_triangle':
        actions = build_gated_action_rows(n, action_scaled, anchor_idx)
    else:
        raise ValueError("action_mode must be 'broadcast' or 'oracle_triangle', got %r" % (action_mode,))
    return np.concatenate([pos, actions], axis=1).astype(np.float32, copy=False)


def _as_int_list(values):
    if values is None:
        return None
    try:
        arr = np.asarray(values).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return []
    out = []
    for value in arr.tolist():
        try:
            out.append(int(value))
        except Exception:
            return None
    return out
