"""Small shared helpers: seeding, I/O, dof bookkeeping, point evaluation, MPI splits."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

SNES_REASONS = {
    2: "CONVERGED_FNORM_ABS",
    3: "CONVERGED_FNORM_RELATIVE",
    4: "CONVERGED_SNORM_RELATIVE",
    5: "CONVERGED_ITS",
    -1: "DIVERGED_FUNCTION_DOMAIN",
    -2: "DIVERGED_FUNCTION_COUNT",
    -3: "DIVERGED_LINEAR_SOLVE",
    -4: "DIVERGED_FNORM_NAN",
    -5: "DIVERGED_MAX_IT",
    -6: "DIVERGED_LINE_SEARCH",
    -8: "DIVERGED_LOCAL_MIN",
    -9: "DIVERGED_DTOL",
    0: "ITERATING",
}


def snes_reason_name(reason: int) -> str:
    return SNES_REASONS.get(int(reason), f"UNKNOWN({reason})")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return {} if cfg is None else cfg


def save_yaml(path: str | Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# dolfinx dof bookkeeping
# ---------------------------------------------------------------------------
def num_dofs(V) -> int:
    return int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)


def assign_array(fun, values) -> None:
    """Copy a dof array into a dolfinx Function (owned or owned+ghost length)."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    owned = fun.function_space.dofmap.index_map.size_local * fun.function_space.dofmap.index_map_bs
    if arr.size == fun.x.array.size:
        fun.x.array[:] = arr
    elif arr.size == owned:
        fun.x.array[:owned] = arr
    else:
        raise ValueError(f"Array of size {arr.size} does not fit function of size {fun.x.array.size}.")
    fun.x.scatter_forward()


def function_from_array(V, values):
    from dolfinx import fem

    fun = fem.Function(V)
    assign_array(fun, values)
    return fun


def collapse_with_map(V_sub):
    V, sub_to_parent = V_sub.collapse()
    return V, np.asarray(sub_to_parent, dtype=np.int64)


def sigma_u_permutation(Vh) -> np.ndarray:
    """Map concat order ``[sigma dofs, u dofs]`` to the mixed ``sigma_u`` dof order.

    ``y_mixed[perm] = y_concat`` and ``y_concat = y_mixed[perm]``.
    """
    _, sigma_map = collapse_with_map(Vh["sigma_u"].sub(0))
    _, u_map = collapse_with_map(Vh["sigma_u"].sub(1))
    perm = np.concatenate([sigma_map, u_map])
    if np.unique(perm).size != perm.size or perm.size != num_dofs(Vh["sigma_u"]):
        raise ValueError("sigma/u sub-space maps do not form a permutation of the mixed dofs.")
    return perm


def evaluate_at_points(mesh, fun, points) -> np.ndarray:
    """Evaluate a Function (including RT and mixed sub-functions) at physical points.

    Returns an array of shape ``(n_points, value_size)`` in the order of ``points``.
    Points outside the mesh get zeros.
    """
    import dolfinx
    from dolfinx import default_scalar_type
    from mpi4py import MPI

    pts = np.asarray(points, dtype=mesh.geometry.x.dtype)
    if pts.shape[1] != 3:
        pts3 = np.zeros((pts.shape[0], 3), dtype=pts.dtype)
        pts3[:, : pts.shape[1]] = pts
        pts = pts3
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, pts)
    colliding = dolfinx.geometry.compute_colliding_cells(mesh, candidates, pts)
    cells = np.full(pts.shape[0], -1, dtype=np.int32)
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            cells[i] = int(links[0])
    valid = np.flatnonzero(cells >= 0)
    value_size = max(1, int(np.prod(fun.ufl_shape)))
    values = np.zeros((pts.shape[0], value_size), dtype=default_scalar_type)
    if valid.size == 0:
        return values
    geom = mesh.geometry
    ref = np.zeros((valid.size, geom.dim), dtype=geom.x.dtype)
    for k, (i, c) in enumerate(zip(valid, cells[valid])):
        ref[k] = geom.cmap.pull_back(pts[i : i + 1], geom.x[geom.dofmap[c]])[0]
    expr = dolfinx.fem.Expression(fun, ref, comm=MPI.COMM_SELF)
    raw = np.asarray(expr.eval(mesh, cells[valid]))
    diag = np.arange(valid.size)
    vals = raw[diag, diag]
    values[valid] = vals.reshape(valid.size, -1)
    return values


# ---------------------------------------------------------------------------
# MPI work splitting
# ---------------------------------------------------------------------------
def contiguous_split(n: int, size: int, rank: int) -> tuple[int, int]:
    base, rem = divmod(int(n), int(size))
    counts = [base + (1 if r < rem else 0) for r in range(size)]
    start = sum(counts[:rank])
    return start, start + counts[rank]


def gather_rows(comm, local: np.ndarray):
    """Gather variable-length row blocks to rank 0 (``None`` elsewhere)."""
    parts = comm.gather(np.asarray(local), root=0)
    if comm.Get_rank() != 0:
        return None
    return np.concatenate(parts, axis=0)


def summary_stats(values) -> dict:
    """Mean, std, median, min, max and count of the finite entries of ``values``."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "median": float(np.median(v)),
        "min": float(v.min()),
        "max": float(v.max()),
        "q05": float(np.quantile(v, 0.05)),
        "q95": float(np.quantile(v, 0.95)),
    }
