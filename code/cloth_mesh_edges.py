from pathlib import Path

import numpy as np
import torch
import trimesh


DEFAULT_CLOTH_MESH_PATH = (
    Path(__file__).resolve().parents[1]
    / 'assistive-gym-fem'
    / 'assistive_gym'
    / 'envs'
    / 'assets'
    / 'clothing'
    / 'blanket_1061v.obj'
)


def _unique_undirected_edges(edges):
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def load_cloth_mesh_edge_indices(num_vertices=None, mesh_path=None):
    """Return bidirectional GT mesh edges as a torch LongTensor with shape [E, 2]."""
    mesh_path = Path(mesh_path) if mesh_path is not None else DEFAULT_CLOTH_MESH_PATH
    mesh = trimesh.load(str(mesh_path), process=False)

    if num_vertices is not None and int(num_vertices) != len(mesh.vertices):
        raise ValueError(
            'GT mesh edge mode requires cloth vertex count to match %s: got %d, expected %d. '
            'Do not use mesh edges after voxel subsampling or with a different blanket mesh.'
            % (mesh_path, int(num_vertices), len(mesh.vertices))
        )

    undirected_edges = _unique_undirected_edges(np.asarray(mesh.edges, dtype=np.int64))
    bidirectional_edges = np.vstack([undirected_edges, undirected_edges[:, ::-1]]).astype(np.int64)
    return torch.tensor(bidirectional_edges, dtype=torch.long)
