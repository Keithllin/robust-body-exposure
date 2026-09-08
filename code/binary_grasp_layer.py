"""Oracle binary grasp-layer indicator via mesh-aware vertical ray ranking.

At the realized anchor triangle, ask: is that local sheet the topmost hit along
vertical rays? Mesh adjacency only merges duplicate face hits on the same sheet;
upper/bottom is decided by z-rank among sheet clusters.
"""

from __future__ import print_function

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ACTION_SCALE = np.asarray([0.44, 1.05], dtype=np.float64)
CLIPPING_THRES = 0.028

# Legacy aliases (old XY+z path); ray ranking uses z_margin_min / z_merge_eps.
DEFAULT_RADIUS = 0.05
DEFAULT_MARGIN = 0.005

DEFAULT_NUM_QUERIES = 7
DEFAULT_MIN_NZ = 0.2
DEFAULT_Z_MERGE_EPS = 0.003
DEFAULT_Z_MARGIN_MIN = 0.005
DEFAULT_MIN_CONSISTENT = 0.8

_MESH_CACHE = None


def _scale_action(action, scale=ACTION_SCALE):
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    multiplier = len(action) // 2
    return action * np.tile(np.asarray(scale, dtype=np.float64), multiplier)


def extract_recover_cloth_action(raw_data):
    """Return (cloth_xyz Nx3+, recover_action, error_str)."""
    if not isinstance(raw_data, dict):
        return None, None, 'not_dict'
    info = raw_data.get('info') or {}
    intermediate = info.get('cloth_intermediate')
    if not intermediate or len(intermediate) < 2 or intermediate[1] is None:
        return None, None, 'missing_cloth_intermediate'
    cloth = np.asarray(intermediate[1], dtype=np.float64)
    if cloth.ndim != 2 or cloth.shape[0] == 0 or cloth.shape[1] < 3:
        return None, None, 'bad_cloth_shape'
    action = raw_data.get('recover_action')
    if action is None:
        return None, None, 'missing_recover_action'
    return cloth, np.asarray(action, dtype=np.float64).reshape(-1), ''


def state_group_key_from_raw(raw_data, raw_path=None):
    """Group key for state-grouped splits (prefer collection_seed)."""
    dci = (raw_data or {}).get('data_collection_info') or {}
    for key in ('collection_seed', 'source_collection_seed', 'uncover_collection_seed',
                'source_seed', 'env_seed', 'uncover_seed'):
        val = dci.get(key)
        if val is not None:
            try:
                return 'cseed_%s' % int(val)
            except Exception:
                return 'cseed_%s' % val

    info = (raw_data or {}).get('info') or {}
    for key in ('collection_seed', 'env_seed'):
        val = info.get(key)
        if val is not None:
            try:
                return 'cseed_%s' % int(val)
            except Exception:
                return 'cseed_%s' % val

    stem = Path(raw_path).stem if raw_path else ''
    m = re.search(r'_c_(\d+)_(\d+)_(\d+)$', stem)
    if m:
        return 'cseed_%s' % m.group(3)
    m = re.search(r'_(\d+)$', stem)
    if m:
        return 'stem_tail_%s' % m.group(1)
    return 'stem_%s' % (stem or 'unknown')


def sample_id_from_raw_path(raw_path):
    return Path(raw_path).stem


def _load_mesh_topology(mesh_path=None):
    """Cache faces, face adjacency, and face->vertex lookups from blanket mesh."""
    global _MESH_CACHE
    if _MESH_CACHE is not None and mesh_path is None:
        return _MESH_CACHE

    import trimesh

    default_mesh = (
        Path(__file__).resolve().parents[1]
        / 'assistive-gym-fem'
        / 'assistive_gym'
        / 'envs'
        / 'assets'
        / 'clothing'
        / 'blanket_1061v.obj'
    )
    path = Path(mesh_path) if mesh_path is not None else default_mesh
    mesh = trimesh.load(str(path), process=False)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('expected triangular faces, got %s' % (faces.shape,))

    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        for u, v in ((a, b), (b, c), (c, a)):
            edge_to_faces[tuple(sorted((u, v)))].append(fi)

    adjacency = [set() for _ in range(len(faces))]
    for face_list in edge_to_faces.values():
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                adjacency[face_list[i]].add(face_list[j])
                adjacency[face_list[j]].add(face_list[i])

    # Map unordered vertex triple -> face index for exact triangle match.
    face_key_to_idx = {}
    for fi, face in enumerate(faces):
        face_key_to_idx[frozenset(int(v) for v in face.tolist())] = int(fi)

    cache = {
        'path': str(path),
        'faces': faces,
        'adjacency': adjacency,
        'face_key_to_idx': face_key_to_idx,
        'n_vertices': int(len(mesh.vertices)),
    }
    if mesh_path is None:
        _MESH_CACHE = cache
    return cache


def resolve_anchor_face_idx(anchor_idx, faces, face_key_to_idx):
    """Map realized anchor vertices to a mesh face index."""
    anchors = [int(a) for a in (anchor_idx or [])]
    if len(anchors) >= 3:
        key = frozenset(anchors[:3])
        if key in face_key_to_idx:
            return int(face_key_to_idx[key])
        # Try all triples if >3 anchors (nearest-4 fallback).
        from itertools import combinations
        for triple in combinations(anchors, 3):
            key = frozenset(triple)
            if key in face_key_to_idx:
                return int(face_key_to_idx[key])

    # Fallback: face containing the most anchors (prefer exact 3).
    best_fi, best_score = None, -1
    anchor_set = set(anchors)
    for fi, face in enumerate(faces):
        score = len(anchor_set.intersection(int(v) for v in face.tolist()))
        if score > best_score:
            best_score = score
            best_fi = fi
    if best_score >= 2:
        return int(best_fi)
    return None


def sample_barycentric_points_on_triangle(tri_xyz, num_queries=DEFAULT_NUM_QUERIES, rng=None):
    """Sample points inside a triangle (includes centroid + near-vertex samples)."""
    tri = np.asarray(tri_xyz, dtype=np.float64).reshape(3, 3)
    rng = np.random.RandomState(0) if rng is None else rng
    n = max(1, int(num_queries))
    points = []

    # Centroid
    points.append(tri.mean(axis=0))
    # Near vertices (barycentric 0.8 / 0.1 / 0.1 permutations)
    for i in range(3):
        w = np.full(3, 0.1, dtype=np.float64)
        w[i] = 0.8
        points.append(w[0] * tri[0] + w[1] * tri[1] + w[2] * tri[2])

    while len(points) < n:
        r1 = float(rng.random())
        r2 = float(rng.random())
        if r1 + r2 > 1.0:
            r1, r2 = 1.0 - r1, 1.0 - r2
        r3 = 1.0 - r1 - r2
        points.append(r1 * tri[0] + r2 * tri[1] + r3 * tri[2])

    return np.asarray(points[:n], dtype=np.float64)


def _point_in_triangle_xy(q_xy, tri_xy, eps=1e-10):
    """Barycentric point-in-triangle in XY. Returns (inside, (w0,w1,w2))."""
    a, b, c = tri_xy
    v0 = c - a
    v1 = b - a
    v2 = q_xy - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < eps:
        return False, None
    inv = 1.0 / den
    u = (v2[0] * v1[1] - v1[0] * v2[1]) * inv
    v = (v0[0] * v2[1] - v2[0] * v0[1]) * inv
    w = 1.0 - u - v
    # Small tolerance for edge queries.
    if u < -1e-8 or v < -1e-8 or w < -1e-8:
        return False, None
    return True, (float(w), float(v), float(u))  # weights for a,b,c


def _face_normal_z(tri_xyz):
    tri = np.asarray(tri_xyz, dtype=np.float64)
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        return 0.0
    return float(n[2] / norm)


def vertical_triangle_hits(q_xy, vertices, faces, min_abs_normal_z=DEFAULT_MIN_NZ, candidate_face_ids=None):
    """Return hits [{face_idx, z, normal_z}] for faces whose XY contains q."""
    q = np.asarray(q_xy, dtype=np.float64).reshape(2)
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if candidate_face_ids is None:
        candidate_face_ids = range(len(faces))

    hits = []
    for fi in candidate_face_ids:
        fi = int(fi)
        tri = verts[faces[fi]]
        nz = _face_normal_z(tri)
        if abs(nz) < float(min_abs_normal_z):
            continue
        inside, weights = _point_in_triangle_xy(q, tri[:, :2])
        if not inside:
            continue
        z = float(weights[0] * tri[0, 2] + weights[1] * tri[1, 2] + weights[2] * tri[2, 2])
        hits.append({'face_idx': fi, 'z': z, 'normal_z': nz})
    hits.sort(key=lambda h: h['z'], reverse=True)
    return hits


def _candidate_faces_near_xy(q_xy, face_xy_aabb, pad=0.02):
    q = np.asarray(q_xy, dtype=np.float64).reshape(2)
    mins = face_xy_aabb[:, 0, :]
    maxs = face_xy_aabb[:, 1, :]
    mask = (
        (mins[:, 0] - pad <= q[0]) & (q[0] <= maxs[:, 0] + pad)
        & (mins[:, 1] - pad <= q[1]) & (q[1] <= maxs[:, 1] + pad)
    )
    return np.flatnonzero(mask)


def _build_face_xy_aabb(vertices, faces):
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tris = verts[faces][:, :, :2]
    mins = tris.min(axis=1)
    maxs = tris.max(axis=1)
    return np.stack([mins, maxs], axis=1)


class _UF(object):
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def merge_adjacent_face_hits(hits, face_adjacency, z_eps=DEFAULT_Z_MERGE_EPS):
    """Merge hits on adjacent faces with similar z into sheet clusters."""
    if not hits:
        return []
    n = len(hits)
    uf = _UF(n)
    face_to_local = {int(h['face_idx']): i for i, h in enumerate(hits)}
    for i, h in enumerate(hits):
        fi = int(h['face_idx'])
        for nb in face_adjacency[fi]:
            j = face_to_local.get(int(nb))
            if j is None:
                continue
            if abs(float(h['z']) - float(hits[j]['z'])) <= float(z_eps):
                uf.union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    sheet_hits = []
    for members in clusters.values():
        zs = [float(hits[i]['z']) for i in members]
        face_ids = [int(hits[i]['face_idx']) for i in members]
        sheet_hits.append({
            'face_ids': face_ids,
            'mean_z': float(np.mean(zs)),
            'max_z': float(np.max(zs)),
            'min_z': float(np.min(zs)),
            'hit_indices': members,
        })
    sheet_hits.sort(key=lambda c: c['mean_z'], reverse=True)
    return sheet_hits


def find_cluster_containing_face(sheet_hits, face_idx):
    face_idx = int(face_idx)
    for i, cluster in enumerate(sheet_hits):
        if face_idx in cluster['face_ids']:
            return i
    return None


def majority_vote(votes):
    if not votes:
        return None, 0.0
    counts = Counter(votes)
    label, n = counts.most_common(1)[0]
    return label, float(n) / float(len(votes))


def _faces_incident_to_vertices(faces, vertex_ids):
    wanted = set(int(v) for v in vertex_ids)
    out = []
    for fi, face in enumerate(faces):
        if wanted.intersection(int(v) for v in face.tolist()):
            out.append(int(fi))
    return out


def classify_binary_grasp_layer(
    vertices,
    faces,
    face_adjacency,
    anchor_face_idx,
    anchor_vertex_ids=None,
    num_queries=DEFAULT_NUM_QUERIES,
    min_nz=DEFAULT_MIN_NZ,
    z_merge_eps=DEFAULT_Z_MERGE_EPS,
    z_margin_min=DEFAULT_Z_MARGIN_MIN,
    min_consistent_fraction=DEFAULT_MIN_CONSISTENT,
    rng=None,
):
    """Ray-rank whether the anchor sheet is locally topmost (not clean-2-layer)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    anchor_face_idx = int(anchor_face_idx)
    tri = vertices[faces[anchor_face_idx]]
    anchor_nz = _face_normal_z(tri)
    empty = {
        'valid': False,
        'label': 0,
        'is_topmost_sheet': 0,
        'reason': 'ambiguous',
        'confidence': 0.0,
        'z_margin': 0.0,
        'z_margin_above': 0.0,
        'z_margin_below': 0.0,
        'anchor_sheet_rank': None,
        'num_sheet_clusters_median': 0.0,
        'num_valid_queries': 0,
        'anchor_face_idx': anchor_face_idx,
        'anchor_face_normal_z': float(anchor_nz),
        'query_cluster_counts': [],
        'skip_reasons': {},
    }
    if abs(anchor_nz) < float(min_nz):
        empty['reason'] = 'ambiguous'
        return empty

    queries = sample_barycentric_points_on_triangle(tri, num_queries=num_queries, rng=rng)
    face_aabb = _build_face_xy_aabb(vertices, faces)
    attach_faces = set([anchor_face_idx])
    if anchor_vertex_ids:
        attach_faces.update(_faces_incident_to_vertices(faces, anchor_vertex_ids))

    votes = []
    margins_above = []
    margins_below = []
    ranks = []
    cluster_counts = []
    reasons_skip = Counter()

    for q_xyz in queries:
        q_xy = q_xyz[:2]
        cand = _candidate_faces_near_xy(q_xy, face_aabb, pad=0.01)
        if anchor_face_idx not in cand:
            cand = np.concatenate([cand, np.asarray([anchor_face_idx], dtype=np.int64)])
        hits = vertical_triangle_hits(
            q_xy, vertices, faces, min_abs_normal_z=min_nz, candidate_face_ids=cand,
        )
        if not hits:
            reasons_skip['no_hits'] += 1
            continue

        sheet_hits = merge_adjacent_face_hits(hits, face_adjacency, z_eps=z_merge_eps)
        sheet_hits = sorted(sheet_hits, key=lambda c: c['mean_z'], reverse=True)
        cluster_counts.append(len(sheet_hits))

        # multi_attach: gripper-associated faces land in >=2 sheet clusters
        attach_clusters = set()
        for fi in attach_faces:
            ci = find_cluster_containing_face(sheet_hits, fi)
            if ci is not None:
                attach_clusters.add(int(ci))
        if len(attach_clusters) >= 2:
            reasons_skip['multi_attach'] += 1
            continue

        anchor_rank = find_cluster_containing_face(sheet_hits, anchor_face_idx)
        if anchor_rank is None:
            reasons_skip['anchor_not_hit'] += 1
            continue

        if len(sheet_hits) == 1:
            reasons_skip['non_overlap'] += 1
            continue

        # margin to sheet above (if any) and below (if any)
        z_a = float(sheet_hits[anchor_rank]['mean_z'])
        if anchor_rank == 0:
            z_next = float(sheet_hits[1]['mean_z'])
            z_above = z_a - z_next
            z_below = 0.0
            if z_above < float(z_margin_min):
                reasons_skip['small_margin'] += 1
                continue
            votes.append('topmost')
            margins_above.append(z_above)
            margins_below.append(z_below)
            ranks.append(0)
        else:
            z_above_sheet = float(sheet_hits[anchor_rank - 1]['mean_z'])
            z_above = z_above_sheet - z_a
            if anchor_rank + 1 < len(sheet_hits):
                z_below = z_a - float(sheet_hits[anchor_rank + 1]['mean_z'])
            else:
                z_below = 0.0
            if z_above < float(z_margin_min):
                reasons_skip['small_margin'] += 1
                continue
            votes.append('non_topmost')
            margins_above.append(z_above)
            margins_below.append(z_below)
            ranks.append(int(anchor_rank))

    result = dict(empty)
    result['num_sheet_clusters_median'] = float(np.median(cluster_counts)) if cluster_counts else 0.0
    result['query_cluster_counts'] = cluster_counts
    result['skip_reasons'] = dict(reasons_skip)
    result['num_valid_queries'] = int(len(votes))

    if not votes:
        if reasons_skip.get('non_overlap', 0) >= max(1, int(0.5 * num_queries)):
            result['reason'] = 'non_overlap'
        elif reasons_skip.get('multi_attach', 0) >= max(1, int(0.5 * num_queries)):
            result['reason'] = 'multi_attach'
        else:
            result['reason'] = 'ambiguous'
        return result

    majority_label, majority_fraction = majority_vote(votes)
    result['confidence'] = float(majority_fraction)
    result['z_margin_above'] = float(np.median(margins_above)) if margins_above else 0.0
    result['z_margin_below'] = float(np.median(margins_below)) if margins_below else 0.0
    result['z_margin'] = result['z_margin_above']
    result['anchor_sheet_rank'] = int(np.median(ranks)) if ranks else None
    if majority_fraction < float(min_consistent_fraction):
        result['reason'] = 'inconsistent_across_queries'
        return result

    is_top = 1 if majority_label == 'topmost' else 0
    result['valid'] = True
    result['label'] = int(is_top)
    result['is_topmost_sheet'] = int(is_top)
    result['reason'] = 'topmost' if is_top else 'non_topmost'
    return result


def label_binary_grasp_layer(
    raw_data,
    radius=DEFAULT_RADIUS,
    margin=DEFAULT_MARGIN,
    clipping_thres=CLIPPING_THRES,
    singulate_layers=True,
    raw_path=None,
    num_queries=DEFAULT_NUM_QUERIES,
    min_nz=DEFAULT_MIN_NZ,
    z_merge_eps=DEFAULT_Z_MERGE_EPS,
    z_margin_min=None,
    min_consistent_fraction=DEFAULT_MIN_CONSISTENT,
):
    """Compute topmost-sheet oracle label for one recover PKL."""
    if z_margin_min is None:
        z_margin_min = margin if margin is not None else DEFAULT_Z_MARGIN_MIN

    out = {
        'layer_label': 0,
        'layer_label_valid': False,
        'is_topmost_sheet': 0,
        'layer_valid': 0,
        'anchor_sheet_rank': None,
        'num_sheet_clusters': 0.0,
        'num_sheet_clusters_median': 0.0,
        'z_margin_above': 0.0,
        'z_margin_below': 0.0,
        'layer_margin': 0.0,
        'layer_z_margin': 0.0,
        'layer_reason': 'unknown',
        'layer_confidence': 0.0,
        'query_consistency': 0.0,
        'num_valid_queries': 0,
        'anchor_face_idx': None,
        'anchor_face_normal_z': 0.0,
        'anchor_idx': [],
        'grasp_xy': [],
        'anchor_source': '',
        'state_group_key': state_group_key_from_raw(raw_data, raw_path),
        'label_method': 'topmost_ray_rank',
        'local_count': 0,
        'z_range': 0.0,
    }

    cloth, action, err = extract_recover_cloth_action(raw_data)
    if err:
        out['layer_reason'] = err
        return out

    action_scaled = _scale_action(action)
    grasp_xy = action_scaled[0:2].astype(np.float64)
    out['grasp_xy'] = [float(grasp_xy[0]), float(grasp_xy[1])]

    dist = np.linalg.norm(cloth[:, 0:2] - grasp_xy[None, :], axis=1)
    if dist.size == 0 or not np.any(dist < float(clipping_thres)):
        out['layer_reason'] = 'miss'
        return out

    from oracle_anchor import resolve_oracle_anchor_idx
    anchors, source = resolve_oracle_anchor_idx(
        raw_data,
        action,
        cloth,
        singulate_layers=bool(singulate_layers),
        allow_recompute=True,
    )
    out['anchor_source'] = source
    anchors = [int(a) for a in (anchors or []) if 0 <= int(a) < len(cloth)]
    out['anchor_idx'] = anchors
    if not anchors:
        out['layer_reason'] = 'miss'
        return out

    topo = _load_mesh_topology()
    if len(cloth) != int(topo['n_vertices']):
        out['layer_reason'] = 'mesh_vertex_mismatch'
        return out

    faces = topo['faces']
    anchor_face = resolve_anchor_face_idx(anchors, faces, topo['face_key_to_idx'])
    if anchor_face is None:
        out['layer_reason'] = 'no_anchor_face'
        return out
    out['anchor_face_idx'] = int(anchor_face)

    classified = classify_binary_grasp_layer(
        vertices=cloth,
        faces=faces,
        face_adjacency=topo['adjacency'],
        anchor_face_idx=anchor_face,
        anchor_vertex_ids=anchors,
        num_queries=num_queries,
        min_nz=min_nz,
        z_merge_eps=z_merge_eps,
        z_margin_min=z_margin_min,
        min_consistent_fraction=min_consistent_fraction,
    )

    valid = bool(classified.get('valid'))
    topmost = int(classified.get('is_topmost_sheet') or classified.get('label') or 0)
    out['layer_label_valid'] = valid
    out['layer_valid'] = 1 if valid else 0
    out['is_topmost_sheet'] = int(topmost) if valid else 0
    out['layer_label'] = out['is_topmost_sheet']  # alias for loaders
    out['layer_reason'] = str(classified.get('reason') or 'ambiguous')
    out['layer_confidence'] = float(classified.get('confidence') or 0.0)
    out['query_consistency'] = out['layer_confidence']
    out['z_margin_above'] = float(classified.get('z_margin_above') or 0.0)
    out['z_margin_below'] = float(classified.get('z_margin_below') or 0.0)
    out['layer_z_margin'] = out['z_margin_above']
    out['layer_margin'] = out['z_margin_above']
    out['anchor_sheet_rank'] = classified.get('anchor_sheet_rank')
    out['num_sheet_clusters_median'] = float(classified.get('num_sheet_clusters_median') or 0.0)
    out['num_sheet_clusters'] = out['num_sheet_clusters_median']
    out['num_valid_queries'] = int(classified.get('num_valid_queries') or 0)
    out['anchor_face_normal_z'] = float(classified.get('anchor_face_normal_z') or 0.0)
    out['skip_reasons'] = classified.get('skip_reasons') or {}
    out['local_count'] = int(out['num_valid_queries'])
    out['z_range'] = out['layer_z_margin']
    return out


def append_broadcast_layer_bit(action_scaled, layer_bit):
    """Append one or more layer feature scalars to scaled action (broadcast later)."""
    action_scaled = np.asarray(action_scaled, dtype=np.float32).reshape(-1)
    bits = np.atleast_1d(np.asarray(layer_bit, dtype=np.float32)).reshape(-1)
    return np.concatenate([action_scaled, bits], axis=0)


def encoding_features_from_record(rec, shuffled_topmost=None):
    """Return (layer_valid, is_topmost) floats for model input."""
    valid = 1.0 if rec.get('layer_label_valid') or int(rec.get('layer_valid') or 0) == 1 else 0.0
    if valid < 0.5:
        return np.asarray([0.0, 0.0], dtype=np.float32)
    if shuffled_topmost is not None:
        top = float(int(shuffled_topmost))
    else:
        top = float(int(rec.get('is_topmost_sheet', rec.get('layer_label', 0)) or 0))
    return np.asarray([1.0, top], dtype=np.float32)


def load_layer_labels_jsonl(path):
    """Return dict sample_id -> record."""
    path = Path(path).expanduser().resolve()
    records = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = str(rec.get('sample_id') or Path(rec.get('raw_path', '')).stem)
            records[sid] = rec
    return records


def load_shuffle_map(path):
    """Return dict sample_id -> int layer_label (0/1)."""
    path = Path(path).expanduser().resolve()
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'labels' in data:
        data = data['labels']
    return {str(k): int(v) for k, v in data.items()}


def resolve_layer_bit_for_sample(sample_id, layer_bit_mode, labels_by_id, shuffle_map=None):
    """Return (feature_vector or None, use_layer_feature bool).

    feature_vector is length-2: [layer_valid, is_topmost].
    layer_bit_mode: 'none' | 'oracle' | 'shuffled'
    For full-50k training every sample_id must exist in labels_by_id.
    shuffle_map maps sample_id -> shuffled is_topmost (valid samples only).
    """
    mode = str(layer_bit_mode or 'none').strip().lower()
    if mode in ('', 'none', 'off', 'false', '0'):
        return None, False
    sid = str(sample_id)
    rec = labels_by_id.get(sid)
    if rec is None:
        raise KeyError('missing layer label for sample_id=%s' % sid)
    if mode == 'oracle':
        return encoding_features_from_record(rec), True
    if mode == 'shuffled':
        if not shuffle_map:
            raise ValueError('layer_bit_mode=shuffled requires shuffle_map')
        valid = bool(rec.get('layer_label_valid') or int(rec.get('layer_valid') or 0) == 1)
        if not valid:
            return encoding_features_from_record(rec), True
        if sid not in shuffle_map:
            raise KeyError('missing shuffled topmost for valid sample_id=%s' % sid)
        return encoding_features_from_record(rec, shuffled_topmost=shuffle_map[sid]), True
    raise ValueError("layer_bit_mode must be 'none', 'oracle', or 'shuffled', got %r" % mode)


def make_proportion_preserving_shuffle(sample_ids, labels_by_id, rng):
    """Shuffle is_topmost among *valid* sample_ids; preserve topmost/non-topmost counts."""
    ids = []
    values = []
    for sid in sample_ids:
        sid = str(sid)
        rec = labels_by_id[sid]
        if not (rec.get('layer_label_valid') or int(rec.get('layer_valid') or 0) == 1):
            continue
        ids.append(sid)
        values.append(int(rec.get('is_topmost_sheet', rec.get('layer_label', 0)) or 0))
    if not ids:
        return {}
    values = np.asarray(values, dtype=np.int64)
    perm = values.copy()
    rng.shuffle(perm)
    if len(perm) > 1 and np.array_equal(perm, values):
        for i in range(len(perm) - 1):
            if perm[i] != perm[i + 1]:
                perm[i], perm[i + 1] = perm[i + 1], perm[i]
                break
    return {sid: int(perm[i]) for i, sid in enumerate(ids)}
