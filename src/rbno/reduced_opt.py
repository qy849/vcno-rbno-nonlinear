"""Reduced minimizer ``s_r^dagger(p)`` of the loss over the POD space (``lem:reduced_minimizer``).

The coefficients ``c`` of ``s = Pi_r c`` minimize ``J_{h,q}(Pi_r c; p)``. They are computed by
the Gauss--Newton method in the ``r`` coefficients, warm-started at the H-orthogonal projection
``P_r s_h`` of the finite element solution (so that the iteration stays in the neighbourhood
of the lemma): with the Gauss--Newton matrix ``K = 2 (A_s ., A_s .)`` and the gradient ``F`` of
the loss form at the current state (the forms of the finite element solve, same quadrature),

    (Pi_r^T K Pi_r) d = -Pi_r^T F,      c <- c + t d,

with backtracking (Armijo) on ``J``. Stop when the reduced gradient has decreased by ``rtol``
or the relative loss decrease of a step is below ``ftol``.

Two products:

* ``results/<exp>/reduced_opt/``: ``s_r^dagger`` for every test parameter at the ``r`` of the
  RBNO, in the prediction format (``rbno.evaluate --method reduced_opt`` gives its loss and
  error; its mean loss is the reference line of the training histories), plus ``stats.npz``;
* ``analysis/<exp>/reduced_opt/sweep.npz``: for the first ``--sweep-num`` test parameters and
  ``r = 1, 2, 4, ...`` (and the ``r`` of the RBNO): squared H errors and losses of ``P_r s_h``
  and of ``s_r^dagger`` (``fig:pod_record``). The longer POD basis of the sweep is recomputed
  from the snapshots (same construction as ``rbno.pod``) and cached in that directory.

    mpirun -n 60 python -m rbno.reduced_opt configs/quasilinear.yaml
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc

from .config import Experiment
from .pod import compute_pod, load_basis, load_snapshots
from .problems import build_problem
from .utils import assign_array, save_json


class ReducedGaussNewton:
    """Gauss--Newton iteration for ``min_c J(Pi c; p)`` on one parameter (COMM_SELF problem)."""

    def __init__(self, problem, rtol: float = 1e-8, ftol: float = 1e-13, max_it: int = 50):
        self.problem = problem
        self.rtol, self.ftol, self.max_it = rtol, ftol, max_it
        V = problem.Vh["sigma_u"]
        self.state = fem.Function(V)
        self.p = fem.Function(problem.Vh["p"])
        sigma, u = ufl.split(self.state)
        J = problem.ls_functional(sigma, u, self.p)
        F = ufl.derivative(J, self.state, ufl.TestFunction(V))
        ds, dv = ufl.split(ufl.TrialFunction(V))
        ts, tv = ufl.split(ufl.TestFunction(V))
        a1, a2 = problem.linearized_residual(ds, dv, self.p, u)
        b1, b2 = problem.linearized_residual(ts, tv, self.p, u)
        K = 2.0 * (ufl.inner(a1, b1) + ufl.inner(a2, b2)) * problem.dx
        self.J, self.F, self.K = fem.form(J), fem.form(F), fem.form(K)

    def set_parameter(self, p_dof) -> None:
        assign_array(self.p, p_dof)
        self.p.x.scatter_forward()

    def _set_state(self, s) -> None:
        assign_array(self.state, s)
        self.state.x.scatter_forward()

    def loss(self, s) -> float:
        self._set_state(s)
        return float(fem.assemble_scalar(self.J))

    def _gradient(self) -> np.ndarray:
        vec = fem_petsc.assemble_vector(self.F)
        vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        F = vec.array.copy()
        vec.destroy()
        return F

    def _gradient_and_matrix(self):
        F = self._gradient()
        A = fem_petsc.assemble_matrix(self.K)
        A.assemble()
        ai, aj, av = A.getValuesCSR()
        K = sp.csr_matrix((av, aj, ai), shape=A.getSize())
        A.destroy()
        return F, K

    def solve(self, basis: np.ndarray, c0: np.ndarray) -> tuple[np.ndarray, dict]:
        c = np.array(c0, dtype=np.float64)
        s = basis @ c
        J = self.loss(s)
        J0, g0, it, steps = J, None, 0, []
        t_start = time.perf_counter()
        converged = False
        for it in range(1, self.max_it + 1):
            F, K = self._gradient_and_matrix()          # at the state set by the last loss(s)
            g = basis.T @ F
            gnorm = float(np.linalg.norm(g))
            g0 = gnorm if g0 is None else g0
            if gnorm <= self.rtol * g0 or gnorm == 0.0:
                converged, it = True, it - 1
                break
            H = basis.T @ (K @ basis)
            try:
                d = sla.solve(H, -g, assume_a="pos")
            except (np.linalg.LinAlgError, sla.LinAlgError):
                d = np.linalg.lstsq(H, -g, rcond=None)[0]
            slope = float(g @ d)
            t = 1.0
            while True:
                s_new = basis @ (c + t * d)
                J_new = self.loss(s_new)
                if J_new <= J + 1e-4 * t * slope or t < 1e-10:
                    break
                t *= 0.5
            steps.append(t)
            c, s = c + t * d, s_new
            decrease = (J - J_new) / max(J, 1e-300)
            J = J_new
            if 0.0 <= decrease < self.ftol:
                converged = True
                break
        self._set_state(s)
        return c, {"loss": J, "loss_init": J0, "iterations": it, "converged": converged,
                   "grad_reduction": float(np.linalg.norm(basis.T @ self._gradient())) / g0
                   if g0 else 0.0, "min_step": float(min(steps)) if steps else 1.0,
                   "seconds": time.perf_counter() - t_start}


def _sweep_ranks(r_used: int, r_max: int) -> list[int]:
    rs = [2 ** k for k in range(int(np.log2(r_max)) + 1)]
    return sorted(set(rs + [r_used]))


def _sweep_basis(exp: Experiment, r_max: int, comm) -> np.ndarray:
    d = exp.analysis_dir / "reduced_opt"
    path = d / f"basis_{r_max}.npy"
    if comm.Get_rank() == 0 and not path.exists():
        d.mkdir(parents=True, exist_ok=True)
        X = build_problem(exp).h_gram_matrix()
        snaps, _ = load_snapshots(exp, int(exp.section("pod")["num_snapshots"]))
        _, basis = compute_pod(snaps, X, r_max)
        tmp = d / f"basis_{r_max}.tmp.npy"
        np.save(tmp, basis)
        tmp.rename(path)
    comm.Barrier()
    return np.load(path, mmap_mode="r")


def run(exp: Experiment, sweep_num: int, sweep_rmax: int, skip_main: bool, comm=MPI.COMM_WORLD) -> None:
    rank, size = comm.Get_rank(), comm.Get_size()
    problem = build_problem(exp)
    X = problem.h_gram_matrix()
    solver = ReducedGaussNewton(problem)
    p_all = np.load(exp.parameter_dir / "p_dof.npy", mmap_mode="r")
    rows = np.load(exp.test_dir / "valid_indices.npy")
    truth = np.load(exp.test_dir / "sigma_u_dof.npy", mmap_mode="r")
    r_used = int(exp.section("pod")["num_basis"])

    def one(basis, k):
        s_h = np.asarray(truth[k], dtype=np.float64)
        solver.set_parameter(p_all[rows[k]])
        Xs = X @ s_h
        c0 = basis.T @ Xs
        s_proj = basis @ c0
        J_proj = solver.loss(s_proj)
        c, info = solver.solve(basis, c0)
        s_opt = basis @ c
        n2 = float(s_h @ Xs)
        e_p, e_o = s_proj - s_h, s_opt - s_h
        rec = dict(proj_err2=float(e_p @ (X @ e_p)), opt_err2=float(e_o @ (X @ e_o)), norm2=n2,
                   proj_loss=J_proj, opt_loss=info["loss"], iterations=info["iterations"],
                   converged=int(info["converged"]), grad_reduction=info["grad_reduction"],
                   seconds=info["seconds"])
        return s_opt, rec

    # -- r of the RBNO, all test parameters ------------------------------------------
    out = exp.results_dir("reduced_opt")
    if not skip_main:
        basis = load_basis(exp)
        mine = np.arange(rank, rows.size, size)
        local_pred, local_rec = [], []
        for k in mine:
            s_opt, rec = one(basis, k)
            local_pred.append(s_opt.astype(np.float64))
            local_rec.append(rec)
            if rank == 0:
                print(f"[{exp.name}/reduced_opt] rank 0: {len(local_rec)}/{mine.size}, {rec['iterations']} it, "
                      f"loss {rec['proj_loss']:.3e} -> {rec['opt_loss']:.3e}, {rec['seconds']:.1f}s", flush=True)
        parts = comm.gather((mine, local_pred, local_rec), root=0)
        if rank == 0:
            pred = np.empty((rows.size, basis.shape[0]))
            recs = [None] * rows.size
            for idx, preds, rr in parts:
                for k, s, r in zip(idx, preds, rr):
                    pred[k], recs[k] = s, r
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "pred_sigma_u_dof.npy", pred)
            np.save(out / "pred_indices.npy", rows)
            stats = {key: np.array([r[key] for r in recs]) for key in recs[0]}
            np.savez(out / "stats.npz", rows=rows, **stats)
            save_json(out / "summary.json", {
                "experiment": exp.name, "r": r_used, "n": int(rows.size),
                "converged": int(stats["converged"].sum()),
                "iterations_median": float(np.median(stats["iterations"])),
                "iterations_max": int(stats["iterations"].max()),
                "loss_mean": float(stats["opt_loss"].mean()), "proj_loss_mean": float(stats["proj_loss"].mean()),
                "rel_err_mean": float(np.mean(np.sqrt(stats["opt_err2"] / stats["norm2"]))),
                "proj_rel_err_mean": float(np.mean(np.sqrt(stats["proj_err2"] / stats["norm2"]))),
                "seconds_mean": float(stats["seconds"].mean())})
            print(f"[{exp.name}/reduced_opt] r={r_used}: converged {int(stats['converged'].sum())}/{rows.size}, "
                  f"loss {stats['proj_loss'].mean():.3e} (P_r s_h) -> {stats['opt_loss'].mean():.3e}, "
                  f"median {np.median(stats['iterations']):.0f} it -> {out}", flush=True)

    # -- sweep over r, first sweep_num test parameters -----------------------------------
    if sweep_num <= 0:
        return
    n_pos = int(np.load(exp.pod_dir / "eigenvalues.npy").size)
    r_max = min(sweep_rmax, n_pos)
    rs = [r for r in _sweep_ranks(r_used, r_max) if r <= r_max]
    full = _sweep_basis(exp, max(rs), comm)
    ks = np.arange(min(sweep_num, rows.size))
    mine = ks[rank::size]
    local = []
    for k in mine:
        for r in rs:
            _, rec = one(np.ascontiguousarray(full[:, :r]), k)
            local.append((k, r, rec))
    parts = comm.gather(local, root=0)
    if rank == 0:
        recs = [x for part in parts for x in part]
        keys = list(recs[0][2])
        arr = {key: np.full((len(rs), ks.size), np.nan) for key in keys}
        for k, r, rec in recs:
            for key in keys:
                arr[key][rs.index(r), k] = rec[key]
        d = exp.analysis_dir / "reduced_opt"
        np.savez(d / "sweep.npz", r=np.array(rs), rows=rows[ks], r_used=r_used, **arr)
        print(f"[{exp.name}/reduced_opt] sweep r={rs} on {ks.size} test parameters -> {d / 'sweep.npz'}")
        for i, r in enumerate(rs):
            print(f"  r={r:4d}: err2 {np.mean(arr['proj_err2'][i]):.3e} -> {np.mean(arr['opt_err2'][i]):.3e}, "
                  f"loss {np.mean(arr['proj_loss'][i]):.3e} -> {np.mean(arr['opt_loss'][i]):.3e}, "
                  f"it {np.median(arr['iterations'][i]):.0f}, conv {int(np.sum(arr['converged'][i]))}/{ks.size}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--sweep-num", type=int, default=100, help="test parameters of the r sweep (0: none)")
    ap.add_argument("--sweep-rmax", type=int, default=512)
    ap.add_argument("--skip-main", action="store_true")
    args = ap.parse_args()
    run(Experiment.load(args.config), args.sweep_num, args.sweep_rmax, args.skip_main)


if __name__ == "__main__":
    main()
