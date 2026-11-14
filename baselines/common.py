"""Shared infrastructure of the baselines (PCA-Net, FNO, FC-PINO, RBNO data-loss ablations).

Grid representation. Grid-based operators work on images of shape ``(C, ny + 1, nx + 1)``
(axis 0 is ``x2``). A finite element state is moved to the grid by interpolating the flux
or stress (RT_1, row-wise) and the primal variable (CG_2) into CG_1, i.e. by sampling them
at the mesh vertices; channel order: flux/stress components (``s1, s2`` or
``s11, s12, s21, s22``) then ``u`` (or ``u1, u2``). A grid prediction is mapped back by
reading it as a CG_1 x CG_1 function and interpolating it into RT_1 x CG_2, which
represents it exactly (CG_1 vectors are contained in RT_1 and CG_1 in CG_2); boundary
values are not modified, so a prediction violating the homogeneous boundary conditions
is evaluated as it is. The result is written in the prediction format of
``rbno.evaluate`` (``results/<experiment>/<method>/pred_sigma_u_dof.npy``).

Native grid metrics (``tab:native_grid``) compare a prediction with the finite element
solution sampled at the same vertices, both as CG_1 x CG_1 functions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem

from rbno.config import Experiment
from rbno.problems import build_problem
from rbno.problems.grid import grid_flat_index
from rbno.utils import REPO_ROOT, collapse_with_map, load_yaml, save_json


# ---------------------------------------------------------------------------------------
# configuration and row selection
# ---------------------------------------------------------------------------------------
def load_baseline_config(experiment: str, method: str) -> tuple[Experiment, dict]:
    exp = Experiment.load(REPO_ROOT / "configs" / f"{experiment}.yaml")
    cfg = load_yaml(REPO_ROOT / "configs" / "baselines" / f"{experiment}.yaml")[method]
    return exp, cfg


def select_rows(spec: dict, valid: np.ndarray, *, exclude: np.ndarray | None = None) -> np.ndarray:
    """Rows from ``{"first": n, "offset": k}`` (k-th to (k+n)-th converged snapshot) or
    ``{"range": [a, b]}`` (all converged snapshots with ``a <= row < b``)."""
    valid = np.asarray(valid)
    if "range" in spec:
        a, b = spec["range"]
        rows = valid[(valid >= a) & (valid < b)]
    else:
        k = int(spec.get("offset", 0))
        rows = valid[k:k + int(spec["first"])]
        if rows.size < int(spec["first"]):
            raise ValueError(f"only {rows.size} converged snapshots available for {spec}")
    if exclude is not None:
        rows = rows[~np.isin(rows, exclude)]
    return rows


def physics_rows(exp: Experiment, n: int, exclude: np.ndarray) -> np.ndarray:
    """First ``n`` parameter rows outside the test block and ``exclude`` (no labels needed)."""
    total = np.load(exp.parameter_dir / "p_dof.npy", mmap_mode="r").shape[0]
    banned = np.union1d(exp.test_rows, exclude)
    rows = np.setdiff1d(np.arange(total), banned)[:n]
    if rows.size < n:
        raise ValueError(f"only {rows.size} parameter rows available for the physics stage")
    return rows


# ---------------------------------------------------------------------------------------
# grid context
# ---------------------------------------------------------------------------------------
@dataclass
class GridContext:
    exp: Experiment
    problem: object
    V1: fem.FunctionSpace            # scalar CG1
    flat: np.ndarray                 # CG1 dof -> flat grid index
    sigma_map: np.ndarray            # mixed dofs of the flux block
    u_map: np.ndarray                # mixed dofs of the primal block
    sigma_leaves: list               # [(space, map into Vh["sigma"])]
    u_leaves: list                   # [(space, map into Vh["u"])]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.problem.num_y + 1, self.problem.num_x + 1)

    @property
    def n_sigma_channels(self) -> int:
        return 2 * len(self.sigma_leaves)

    @property
    def n_channels(self) -> int:
        return self.n_sigma_channels + len(self.u_leaves)

    @property
    def lengths(self) -> tuple[float, float]:
        return self.problem.length_x, self.problem.length_y


def build_context(exp: Experiment) -> GridContext:
    problem = build_problem(exp)
    Vh = problem.Vh
    V1 = fem.functionspace(problem.domain, basix.ufl.element("CG", problem.cell_name, 1))
    flat = grid_flat_index(V1, problem.num_x, problem.num_y, problem.length_x, problem.length_y)
    _, sigma_map = collapse_with_map(Vh["sigma_u"].sub(0))
    _, u_map = collapse_with_map(Vh["sigma_u"].sub(1))
    if problem.vector_valued:
        sigma_leaves = [collapse_with_map(Vh["sigma"].sub(i)) for i in range(2)]
        u_leaves = [collapse_with_map(Vh["u"].sub(i)) for i in range(2)]
    else:
        sigma_leaves, u_leaves = [(Vh["sigma"], None)], [(Vh["u"], None)]
    return GridContext(exp, problem, V1, flat, sigma_map, u_map, sigma_leaves, u_leaves)


def _vec_space(ctx: GridContext):
    return fem.functionspace(ctx.problem.domain, basix.ufl.element("CG", ctx.problem.cell_name, 1, shape=(2,)))


def states_to_images(ctx: GridContext, states: np.ndarray) -> np.ndarray:
    """Mixed dof vectors ``(N, n_state)`` -> vertex images ``(N, C, ny+1, nx+1)`` (float32)."""
    Vh = ctx.problem.Vh
    V1v = _vec_space(ctx)
    ny, nx = ctx.shape
    out = np.zeros((states.shape[0], ctx.n_channels, ny * nx), dtype=np.float32)
    sig, u = fem.Function(Vh["sigma"]), fem.Function(Vh["u"])
    s1, u1 = fem.Function(V1v), fem.Function(ctx.V1)
    for i in range(states.shape[0]):
        row = np.asarray(states[i], dtype=np.float64)
        sig.x.array[:] = row[ctx.sigma_map]
        u.x.array[:] = row[ctx.u_map]
        ch = 0
        for space, m in ctx.sigma_leaves:
            leaf = fem.Function(space)
            leaf.x.array[:] = sig.x.array if m is None else sig.x.array[m]
            s1.interpolate(leaf)
            v = s1.x.array.reshape(-1, 2)
            out[i, ch, ctx.flat] = v[:, 0]
            out[i, ch + 1, ctx.flat] = v[:, 1]
            ch += 2
        for space, m in ctx.u_leaves:
            leaf = fem.Function(space)
            leaf.x.array[:] = u.x.array if m is None else u.x.array[m]
            u1.interpolate(leaf)
            out[i, ch, ctx.flat] = u1.x.array
            ch += 1
    return out.reshape(states.shape[0], ctx.n_channels, ny, nx)


def images_to_states(ctx: GridContext, images: np.ndarray) -> np.ndarray:
    """Vertex images -> RT_1 x CG_2 mixed dof vectors (exact representation of CG_1 x CG_1)."""
    Vh = ctx.problem.Vh
    V1v = _vec_space(ctx)
    n = images.shape[0]
    flat_imgs = np.asarray(images, dtype=np.float64).reshape(n, ctx.n_channels, -1)
    out = np.zeros((n, ctx.problem.n_state))
    s1, u1 = fem.Function(V1v), fem.Function(ctx.V1)
    for i in range(n):
        ch = 0
        sig_arr = np.zeros(fem.Function(Vh["sigma"]).x.array.size)
        for space, m in ctx.sigma_leaves:
            v = np.empty((ctx.flat.size, 2))
            v[:, 0] = flat_imgs[i, ch, ctx.flat]
            v[:, 1] = flat_imgs[i, ch + 1, ctx.flat]
            s1.x.array[:] = v.reshape(-1)
            leaf = fem.Function(space)
            leaf.interpolate(s1)
            if m is None:
                sig_arr[:] = leaf.x.array
            else:
                sig_arr[m] = leaf.x.array
            ch += 2
        u_arr = np.zeros(fem.Function(Vh["u"]).x.array.size)
        for space, m in ctx.u_leaves:
            u1.x.array[:] = flat_imgs[i, ch, ctx.flat]
            leaf = fem.Function(space)
            leaf.interpolate(u1)
            if m is None:
                u_arr[:] = leaf.x.array
            else:
                u_arr[m] = leaf.x.array
            ch += 1
        out[i, ctx.sigma_map] = sig_arr
        out[i, ctx.u_map] = u_arr
    return out


def parameter_images(ctx: GridContext, rows: np.ndarray) -> np.ndarray:
    p_all = np.load(ctx.exp.parameter_dir / "p_dof.npy", mmap_mode="r")
    out = []
    for s in range(0, len(rows), 1000):
        out.append(ctx.problem.parameter_images(np.asarray(p_all[np.asarray(rows[s:s + 1000])])))
    return np.concatenate(out)


# ---------------------------------------------------------------------------------------
# grid caches
# ---------------------------------------------------------------------------------------
def grid_cache(exp: Experiment, store: str) -> tuple[np.ndarray, np.ndarray]:
    """Vertex images of a solution store (``snapshots`` or ``test``), cached in data/<exp>/grid/."""
    out = exp.data_dir / "grid"
    img_path, row_path = out / f"{store}_images.npy", out / f"{store}_rows.npy"
    if img_path.exists() and row_path.exists():
        return np.load(img_path, mmap_mode="r"), np.load(row_path)
    ctx = build_context(exp)
    src = exp.snapshot_dir if store == "snapshots" else exp.test_dir
    rows = np.load(src / "valid_indices.npy")
    data = np.load(src / "sigma_u_dof.npy", mmap_mode="r")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tmp = out / f"{store}_images.tmp.npy"
    imgs = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                     shape=(rows.size, ctx.n_channels, *ctx.shape))
    for s in range(0, rows.size, 100):
        imgs[s:s + 100] = states_to_images(ctx, np.asarray(data[s:s + 100]))
    imgs.flush()
    del imgs
    tmp.replace(img_path)  # rows are written last: a cache without them is incomplete
    np.save(row_path, rows)
    print(f"[{exp.name}] rasterized {rows.size} {store} solutions in {time.perf_counter() - t0:.0f}s")
    return np.load(img_path, mmap_mode="r"), rows


def images_for_rows(exp: Experiment, rows: np.ndarray) -> np.ndarray:
    """Solution images of parameter rows (looked up in the snapshot and test caches)."""
    out = None
    for store in ("snapshots", "test"):
        imgs, srows = grid_cache(exp, store)
        pos = {int(r): k for k, r in enumerate(srows)}
        hit = [k for k, r in enumerate(rows) if int(r) in pos]
        if out is None:
            out = np.full((len(rows), *imgs.shape[1:]), np.nan, dtype=np.float32)
        for k in hit:
            out[k] = imgs[pos[int(rows[k])]]
    if np.isnan(out).any():
        raise KeyError("some rows have no converged solution")
    return out


# ---------------------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------------------
def save_grid_predictions(exp: Experiment, method: str, rows: np.ndarray, pred_images: np.ndarray,
                          extra: dict | None = None) -> Path:
    """Store grid predictions and their RT_1 x CG_2 representation for ``rbno.evaluate``."""
    ctx = build_context(exp)
    out = exp.results_dir(method)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_grid.npy", pred_images.astype(np.float32))
    states = images_to_states(ctx, pred_images)
    np.save(out / "pred_sigma_u_dof.npy", states)
    np.save(out / "pred_indices.npy", np.asarray(rows))
    if extra:
        save_json(out / "pred_meta.json", extra)
    return out


def native_grid_metrics(exp: Experiment, method: str) -> dict:
    """Errors of grid predictions against the FE solution sampled at the same vertices,
    both as CG_1 x CG_1 functions: relative L2 x L2 and H(div) x H^1 errors."""
    ctx = build_context(exp)
    out = exp.results_dir(method)
    pred = np.load(out / "pred_grid.npy", mmap_mode="r")
    rows = np.load(out / "pred_indices.npy")
    truth = images_for_rows(exp, rows)
    P = ctx.problem
    V1v = _vec_space(ctx)
    nl = len(ctx.sigma_leaves)
    import basix.ufl as bu

    sig_el = bu.element("CG", P.cell_name, 1, shape=(2,))
    Vs = fem.functionspace(P.domain, sig_el if nl == 1 else bu.mixed_element([sig_el] * nl))
    Vu = fem.functionspace(P.domain, bu.element("CG", P.cell_name, 1) if nl == 1 else
                           bu.mixed_element([bu.element("CG", P.cell_name, 1)] * 2))
    s_maps = [None] if nl == 1 else [collapse_with_map(Vs.sub(i))[1] for i in range(2)]
    u_maps = [None] if nl == 1 else [collapse_with_map(Vu.sub(i))[1] for i in range(2)]

    def to_fun(img):
        s, u = fem.Function(Vs), fem.Function(Vu)
        flat = img.reshape(ctx.n_channels, -1)
        for k, m in enumerate(s_maps):
            v = np.stack([flat[2 * k, ctx.flat], flat[2 * k + 1, ctx.flat]], axis=1).reshape(-1)
            if m is None:
                s.x.array[:] = v
            else:
                tmp = fem.Function(V1v)
                tmp.x.array[:] = v
                sub = fem.Function(collapse_with_map(Vs.sub(k))[0])
                sub.interpolate(tmp)
                s.x.array[m] = sub.x.array
        for k, m in enumerate(u_maps):
            v = flat[ctx.n_sigma_channels + k, ctx.flat]
            if m is None:
                u.x.array[:] = v
            else:
                u.x.array[m] = v
        return s, u

    errH, refH, errL, refL = [], [], [], []
    for i in range(rows.size):
        sp, up = to_fun(np.asarray(pred[i], dtype=np.float64))
        st, ut = to_fun(np.asarray(truth[i], dtype=np.float64))
        e = P.norm_components(sp - st if nl == 1 else _sub(sp, st), up - ut if nl == 1 else _sub(up, ut))
        r = P.norm_components(st, ut)
        errH.append(sum(e.values()))
        refH.append(sum(r.values()))
        errL.append(e["sigma_l2"] + e["u_l2"])
        refL.append(r["sigma_l2"] + r["u_l2"])
    relH = np.sqrt(np.array(errH) / np.array(refH))
    relL = np.sqrt(np.array(errL) / np.array(refL))
    res = {"rel_L2_mean": float(relL.mean()), "rel_L2_std": float(relL.std()),
           "rel_H_mean": float(relH.mean()), "rel_H_std": float(relH.std()), "n": int(rows.size)}
    save_json(out / "native_grid_metrics.json", res)
    print(f"[{exp.name}/{method}] native grid: rel L2 {res['rel_L2_mean']:.3e}, rel H {res['rel_H_mean']:.3e}")
    return res


def _sub(a, b):
    """Difference of two functions on a mixed CG1 space as a flattened UFL vector."""
    return ufl.as_vector([a[i] - b[i] for i in range(a.ufl_shape[0])])
