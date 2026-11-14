"""Influence of the quadrature degree on the finite element loss, its gradient and the solution.

For the first ``--num`` test parameters of an experiment and every degree ``q`` in
``--degrees``, compared with a reference degree ``--ref`` (exact for the semilinear benchmark,
where all integrands are polynomials of degree at most 12; high for the other two, whose
integrands contain ``exp`` sources or ``log det F``):

* loss: relative difference ``|J_q(s) - J_ref(s)| / J_ref(s)`` at two fixed states, the stored
  finite element solution ``s_h`` (small loss, close to the floor) and the RBNO prediction
  (the states seen in training and at inference);
* gradient: ``||g_q - g_ref|| / ||g_ref||`` at the prediction (what training sees);
* cost: wall-clock of one ``loss_and_grad`` call (mean of 3 after compilation);
* solution: the production solve with degree ``q`` (same initial guess and continuation),
  its relative H distance to the solution with the reference degree, its loss evaluated
  with the reference degree relative to that of the reference solution, iterations, time.

One MPI rank per parameter (``COMM_SELF`` problems); rank 0 writes
``analysis/<experiment>/quadrature/{rows.csv,summary.json,quadrature.png}``.

    mpirun -n 16 python -m rbno.quadrature_study configs/semilinear_k1000.yaml --degrees 2 4 6 8 10 12 --ref 12
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time

import numpy as np
from mpi4py import MPI

from .config import Experiment
from .problems import build_problem
from .solve import StateSolver
from .utils import save_json, sigma_u_permutation


def _problem(exp: Experiment, degree: int):
    problem = build_problem(exp)
    problem.quadrature_degree = int(degree)
    return problem


def _loss_grad_time(problem, y, p, reps: int = 3):
    problem.loss_and_grad(y, p)  # compiles the forms of this degree
    t0 = time.perf_counter()
    for _ in range(reps):
        J, g = problem.loss_and_grad(y, p)
    return J, g, (time.perf_counter() - t0) / reps


def study_row(exp: Experiment, row: int, degrees: list[int], ref: int, solve: bool) -> list[dict]:
    p_all = np.load(exp.parameter_dir / "p_dof.npy", mmap_mode="r")
    p = np.asarray(p_all[row])
    valid = list(np.load(exp.test_dir / "valid_indices.npy"))
    s_h = np.asarray(np.load(exp.test_dir / "sigma_u_dof.npy", mmap_mode="r")[valid.index(row)])
    pred_rows = list(np.load(exp.results_dir("rbno") / "pred_indices.npy"))
    s_pred = np.asarray(np.load(exp.results_dir("rbno") / "pred_sigma_u_dof.npy", mmap_mode="r")[pred_rows.index(row)])

    P_ref = _problem(exp, ref)
    perm = sigma_u_permutation(P_ref.Vh)
    X = P_ref.h_gram_matrix()
    J_h_ref, _, _ = _loss_grad_time(P_ref, s_h[perm], p, reps=1)
    J_p_ref, g_ref, t_ref = _loss_grad_time(P_ref, s_pred[perm], p)
    s_ref = None
    if solve:
        solver = StateSolver(exp, P_ref)
        with contextlib.redirect_stdout(io.StringIO()):
            solver.prepare(MPI.COMM_SELF)
            s_ref, info_ref = solver.solve(p)
        J_sref = P_ref.loss(s_ref[perm], p)

    rows = []
    for q in degrees:
        P = _problem(exp, q)
        J_h = P.loss(s_h[perm], p)
        J_p, g, t = _loss_grad_time(P, s_pred[perm], p)
        rec = dict(row=row, degree=q, ref=ref,
                   loss_fe_rel=abs(J_h - J_h_ref) / J_h_ref,
                   loss_pred_rel=abs(J_p - J_p_ref) / J_p_ref,
                   grad_rel=float(np.linalg.norm(g - g_ref) / np.linalg.norm(g_ref)),
                   loss_grad_s=t, loss_grad_ref_s=t_ref)
        if solve:
            solver = StateSolver(exp, P)
            with contextlib.redirect_stdout(io.StringIO()):
                solver.prepare(MPI.COMM_SELF)
                t0 = time.perf_counter()
                s_q, info = solver.solve(p)
                rec["solve_s"] = time.perf_counter() - t0
            rec["solve_converged"] = int(s_q is not None)
            if s_q is not None:
                d = s_q - s_ref
                rec["solution_rel_H"] = float(np.sqrt(d @ (X @ d) / (s_ref @ (X @ s_ref))))
                rec["solution_loss_excess_rel"] = (P_ref.loss(s_q[perm], p) - J_sref) / J_sref
                rec["iterations"] = info.get("iterations")
            rec["solve_ref_s"] = info_ref.get("total_s")
        rows.append(rec)
    return rows


def plot(df, out, title: str) -> None:
    from .plots.style import plt

    g = df.groupby("degree")
    deg = np.array(sorted(df.degree.unique()))
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for key, label, mk in (("loss_fe_rel", r"loss at $s_h$", "o"), ("loss_pred_rel", "loss at prediction", "s"),
                           ("grad_rel", "gradient at prediction", "^"),
                           ("solution_rel_H", r"solution, rel. $\mathbb{H}$", "D"),
                           ("solution_loss_excess_rel", r"loss of $s_{h,q}$ (ref. rule)", "v")):
        if key in df:
            v = g[key].max().reindex(deg).to_numpy(dtype=float)
            v = np.where(v > 0, v, np.nan)
            a.semilogy(deg, v, marker=mk, lw=2, label=label)
    a.set_xlabel("quadrature degree $q$", fontsize=18)
    a.set_ylabel("max relative difference to reference", fontsize=16)
    a.legend(fontsize=12)
    a.grid(alpha=0.3, which="both")
    b.plot(deg, g["loss_grad_s"].mean().reindex(deg), marker="o", lw=2, label="loss + gradient")
    if "solve_s" in df:
        b2 = b.twinx()
        b2.plot(deg, g["solve_s"].mean().reindex(deg), marker="s", lw=2, color="C1", label="FE solve")
        b2.set_ylabel("FE solve (s)", fontsize=16, color="C1")
    b.set_xlabel("quadrature degree $q$", fontsize=18)
    b.set_ylabel("loss + gradient (s)", fontsize=16)
    b.grid(alpha=0.3)
    fig.suptitle(title, fontsize=18)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--degrees", type=int, nargs="+", required=True)
    ap.add_argument("--ref", type=int, required=True)
    ap.add_argument("--num", type=int, default=None, help="test parameters (default: one per rank)")
    ap.add_argument("--no-solve", action="store_true")
    args = ap.parse_args()
    comm = MPI.COMM_WORLD
    exp = Experiment.load(args.config)
    rows = exp.test_rows[: args.num or comm.Get_size()]
    out = []
    for row in rows[comm.Get_rank()::comm.Get_size()]:
        out.extend(study_row(exp, int(row), args.degrees, args.ref, not args.no_solve))
    parts = comm.gather(out, root=0)
    if comm.Get_rank() != 0:
        return
    import pandas as pd

    df = pd.DataFrame([r for part in parts for r in part]).sort_values(["degree", "row"])
    d = exp.analysis_dir / "quadrature"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "rows.csv", index=False)
    agg = {}
    for q, grp in df.groupby("degree"):
        rec = {"n": int(len(grp))}
        for k in grp.columns:
            if k in ("row", "degree", "ref"):
                continue
            v = grp[k].to_numpy(dtype=float)
            rec[k] = {"max": float(np.nanmax(v)), "mean": float(np.nanmean(v))}
        agg[int(q)] = rec
    save_json(d / "summary.json", {"experiment": exp.name, "ref_degree": args.ref,
                                   "configured_degree": int(exp.cfg["problem"]["quadrature_degree"]),
                                   "rows": [int(r) for r in rows], "by_degree": agg})
    plot(df, d / "quadrature.png", f"{exp.name}: reference degree {args.ref}")
    cols = [c for c in ("loss_fe_rel", "loss_pred_rel", "grad_rel", "solution_rel_H",
                        "solution_loss_excess_rel", "loss_grad_s", "solve_s", "iterations") if c in df]
    print(df.groupby("degree")[cols].max().to_string(float_format=lambda x: f"{x:.2e}"))


if __name__ == "__main__":
    main()
