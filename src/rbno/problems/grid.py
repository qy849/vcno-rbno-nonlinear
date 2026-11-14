"""Structured-grid views of CG1 fields (network inputs and grid baselines)."""

from __future__ import annotations

import numpy as np


def grid_flat_index(V, num_x: int, num_y: int, length_x: float, length_y: float) -> np.ndarray:
    """Flat index ``iy * (num_x + 1) + ix`` of every (scalar) dof of a vertex-based space."""
    coords = V.tabulate_dof_coordinates()[:, :2]
    fx = coords[:, 0] / length_x * num_x
    fy = coords[:, 1] / length_y * num_y
    ix, iy = np.rint(fx).astype(np.int64), np.rint(fy).astype(np.int64)
    if max(np.abs(fx - ix).max(), np.abs(fy - iy).max()) > 1e-6:
        raise ValueError("Dofs do not lie on the structured grid.")
    flat = iy * (num_x + 1) + ix
    if np.unique(flat).size != flat.size:
        raise ValueError("Two dofs share a grid vertex.")
    return flat


def cg1_grid_images(problem, p_dof: np.ndarray) -> np.ndarray:
    """``(N, n_p)`` CG1 dofs -> ``(N, 1, num_y + 1, num_x + 1)`` float32 images."""
    p_dof = np.atleast_2d(np.asarray(p_dof))
    flat = getattr(problem, "_p_grid_flat", None)
    if flat is None:
        flat = grid_flat_index(problem.Vh["p"], problem.num_x, problem.num_y,
                               problem.length_x, problem.length_y)
        problem._p_grid_flat = flat
    img = np.empty((p_dof.shape[0], (problem.num_y + 1) * (problem.num_x + 1)), dtype=np.float32)
    img[:, flat] = p_dof
    return img.reshape(p_dof.shape[0], 1, problem.num_y + 1, problem.num_x + 1)
