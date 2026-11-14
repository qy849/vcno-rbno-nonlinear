"""Discrete stability test (Sections 3.5 and 6.5): smallest generalized eigenvalue

    lambda_min,h(ubar, p) = min_t  (A t, A t)_{L2} / ||t||_H^2 ,   t in H_h (homogeneous),

of the linearized residual ``A = A_{ubar,p}``. Representation and quadrature: the residual
is *not* projected; ``K = (A psi_j, A psi_i)`` is assembled exactly as the Gauss-Newton
part of the loss Hessian, with the quadrature rule of the loss (degree 6 semilinear,
8 quasilinear and hyperelastic), and ``M`` is the H(div) x H^1 Gram matrix (exact
quadrature). Constrained dofs are removed before the eigensolve. ``lambda_min`` is
computed as ``1 / mu_max`` of ``M x = mu K x`` with the randomized double-pass
generalized eigensolver of hIPPYlib-X (MUMPS factorization of ``K``).

States, for the first ``stability.num_params`` test parameters:
  * ``resolution`` -- finite element solutions on the meshes ``stability.resolutions``
    (each mesh with its own converged state, same solver and initial guess as the data);
  * ``newton``     -- all Newton iterates of the solve on the finest mesh (last load step
    for hyperelasticity);
  * ``rbno``       -- the RBNO prediction on the finest mesh.

    mpirun -n 25 python -m rbno.stability configs/quasilinear.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc

from .config import Experiment
from .problems import build_problem
from .problems.base import collect_bc_dofs
from .solve import StateSolver
from .utils import save_json


def lambda_min(problem, p_fun, state_dof, *, oversample: int = 40, power: int = 8, seed: int = 1) -> float:
    import hippylibx as hpx

    s = problem.state_function(state_dof)
    ubar = s.sub(1).collapse()
    K_form, M_form = problem.stability_forms(p_fun, ubar)
    bcs = problem.bcs["state"]
    K = fem_petsc.assemble_matrix(fem.form(K_form), bcs=bcs, diag=1.0)
    K.assemble()
    M = fem_petsc.assemble_matrix(fem.form(M_form), bcs=bcs, diag=1.0)
    M.assemble()
    keep = np.ones(K.getSize()[0], dtype=bool)
    keep[collect_bc_dofs(bcs)] = False
    iset = PETSc.IS().createGeneral(np.flatnonzero(keep).astype(PETSc.IntType), comm=K.getComm())
    Kf, Mf = K.createSubMatrix(iset, iset), M.createSubMatrix(iset, iset)
    solver = hpx.PETScLUSolver(Kf.getComm(), "mumps")
    solver.set_operator(Kf)
    Omega = hpx.MultiVector(Kf.createVecRight(), 1 + oversample)
    hpx.Random(seed, 1).normal(1.0, Omega)
    mu, _ = hpx.doublePassG(Mf, Kf, solver, Omega, 1, s=power)
    for obj in (K, M, Kf, Mf, iset):
        obj.destroy()
    return 1.0 / float(np.max(mu))


class MeshLevel:
    """Problem, solver and parameter transfer on one mesh of the resolution sweep."""

    def __init__(self, exp: Experiment, n: int):
        base_x, base_y = exp.mesh["num_x"], exp.mesh["num_y"]
        self.n = n
        self.problem = build_problem(exp, num_x=n, num_y=n * base_y // base_x)
        self.solver = StateSolver(exp, self.problem)
        self.fine = n == base_x

    def parameter(self, exp: Experiment, row: int, fine_level: "MeshLevel", p_all, mu_all):
        if self.fine:
            return self.problem.p_function(p_all[row])
        if exp.problem_type == "semilinear":  # squares are aligned with every mesh of size 16 k
            return self.problem.p_function(self.problem.conductivity_dofs(mu_all[row:row + 1])[0])
        p_fine = fine_level.problem.p_function(p_all[row])
        p = fem.Function(self.problem.Vh["p"])
        cells = np.arange(self.problem.domain.topology.index_map(2).size_local, dtype=np.int32)
        data = fem.create_interpolation_data(self.problem.Vh["p"], fine_level.problem.Vh["p"], cells)
        p.interpolate_nonmatching(p_fine, cells, data)
        return p

    def prepare_seed(self, exp: Experiment, fine_level: "MeshLevel", p_all, mu_all):
        """Mean-parameter seed on this mesh (data seed on the finest mesh)."""
        mode = self.solver.cfg.get("initial_guess", "zero")
        if mode not in ("mean_parameter", "mean_parameter_continuation"):
            return
        if self.fine:
            self.solver.seed = np.load(exp.data_dir / "seed_state.npy")
            return
        p_mean_fine = fine_level.problem.p_function(np.asarray(p_all).mean(axis=0))
        p = fem.Function(self.problem.Vh["p"])
        cells = np.arange(self.problem.domain.topology.index_map(2).size_local, dtype=np.int32)
        data = fem.create_interpolation_data(self.problem.Vh["p"], fine_level.problem.Vh["p"], cells)
        p.interpolate_nonmatching(p_mean_fine, cells, data)
        if mode == "mean_parameter":
            self.solver.seed = self.problem.newton_solve(p).x.array.copy()
        else:
            loads = np.asarray(self.solver.cfg["seed_loads"], dtype=float)
            self.solver.seed = self.problem.continuation_solve(p, None, loads).x.array.copy()


def run(exp: Experiment, comm=MPI.COMM_WORLD, method: str = "rbno") -> None:
    rank, size = comm.Get_rank(), comm.Get_size()
    scfg = exp.section("stability")
    rows = exp.test_rows[: int(scfg.get("num_params", 50))]
    resolutions = [int(n) for n in scfg.get("resolutions", [exp.mesh["num_x"]])]
    p_all = np.load(exp.parameter_dir / "p_dof.npy", mmap_mode="r")
    mu_all = np.load(exp.parameter_dir / "mu.npy") if exp.problem_type == "semilinear" else None
    pred = np.load(exp.results_dir(method) / "pred_sigma_u_dof.npy", mmap_mode="r")
    pred_pos = {int(r): k for k, r in enumerate(np.load(exp.results_dir(method) / "pred_indices.npy"))}

    levels = {n: MeshLevel(exp, n) for n in resolutions}
    fine = levels[exp.mesh["num_x"]]
    with contextlib.redirect_stdout(io.StringIO()):
        for lvl in levels.values():
            lvl.prepare_seed(exp, fine, p_all, mu_all)

    records = []
    t0 = time.perf_counter()
    for row in rows[rank::size]:
        row = int(row)
        for n, lvl in levels.items():
            p = lvl.parameter(exp, row, fine, p_all, mu_all)
            with contextlib.redirect_stdout(io.StringIO()):
                state, info = lvl.solver.solve(p.x.array, capture_iterates=lvl.fine)
            ok = state is not None
            records.append(dict(row=row, population="resolution", n=n, k=-1, converged=int(ok),
                                lam_min=lambda_min(lvl.problem, p, state) if ok else np.nan))
            if lvl.fine and ok:
                for k, it in enumerate(lvl.problem.newton_iterates):
                    records.append(dict(row=row, population="newton", n=n, k=k, converged=1,
                                        lam_min=lambda_min(lvl.problem, p, it)))
                if row in pred_pos:
                    records.append(dict(row=row, population="rbno", n=n, k=-1, converged=1,
                                        lam_min=lambda_min(lvl.problem, p, np.asarray(pred[pred_pos[row]]))))
        if rank == 0:
            print(f"[{exp.name}] stability: row {row} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    gathered = comm.gather(records, root=0)
    if rank != 0:
        return
    import pandas as pd

    df = pd.DataFrame([r for part in gathered for r in part]).sort_values(["row", "population", "n", "k"])
    out = exp.analysis_dir / "stability"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "lambda_min.csv", index=False)
    finest = max(resolutions)
    conv_fine = set(df[(df.population == "resolution") & (df.n == finest) & (df.converged == 1)].row)
    summary = {"experiment": exp.name, "rows": [int(r) for r in rows], "resolutions": resolutions,
               "representation": "unprojected residual, K assembled with the loss quadrature",
               "quadrature_degree": int(exp.cfg["problem"]["quadrature_degree"]),
               "groups": {}}
    for name, sub in (
        *[(f"fe_{n}", df[(df.population == "resolution") & (df.n == n) & (df.converged == 1)]) for n in resolutions],
        ("newton", df[(df.population == "newton") & df.row.isin(conv_fine)]),
        ("rbno", df[(df.population == "rbno") & df.row.isin(conv_fine)]),
    ):
        v = sub.lam_min.to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        summary["groups"][name] = {"n": int(v.size), "min": float(v.min()) if v.size else None,
                                   "median": float(np.median(v)) if v.size else None,
                                   "max": float(v.max()) if v.size else None}
    allv = [g["min"] for g in summary["groups"].values() if g["min"] is not None]
    summary["overall_min"] = float(min(allv))
    summary["n_states"] = int(sum(g["n"] for g in summary["groups"].values()))
    summary["chi_bound"] = float(summary["overall_min"] ** -0.5)
    summary["seconds"] = time.perf_counter() - t0
    save_json(out / "summary.json", summary)
    print(summary, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--method", default="rbno", help="predictor whose states are tested")
    ap.add_argument("--num-params", type=int, default=None, help="override stability.num_params")
    args = ap.parse_args()
    exp = Experiment.load(args.config)
    if args.num_params is not None:
        exp.cfg["stability"]["num_params"] = args.num_params
    run(exp, method=args.method)


if __name__ == "__main__":
    main()
