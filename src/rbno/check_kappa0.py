"""Regression check: the nonlinear code at kappa = 0 against the linear FOSLS code of QCD.

For ``--num`` test parameters, compare the state computed by the nonlinear pipeline
(Gauss--Newton on the degree-12 loss with kappa = 0) with the solution of QCD's linear
FOSLS normal equations, assembled here exactly as in
``data_generation/differential_equations/poisson_setup1_least_squares.py`` of
https://github.com/qy849/vcno-rbno
(bilinear form ``(sigma - p grad u, tau - p grad v) + (div sigma, div tau)``, right-hand side
``(p grad w - z + f1, tau - p grad v) - (f2, div tau)``, default quadrature, MUMPS), and the
loss of both states evaluated with QCD's two loss functionals (default quadrature)
and with the degree-12 evaluator. Reports the largest relative differences.

    python -m rbno.check_kappa0 configs/semilinear_k0.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc

from .config import Experiment
from .problems import build_problem
from .solve import StateSolver
from .utils import save_json


def qcd_linear_solve(P, p) -> fem.Function:
    V = P.Vh["sigma_u"]
    sigma, u = ufl.TrialFunctions(V)
    tau, v = ufl.TestFunctions(V)
    z = ufl.grad(P.q)
    dx = ufl.dx(domain=P.domain)
    a = ufl.inner(sigma - p * ufl.grad(u), tau - p * ufl.grad(v)) * dx + ufl.inner(ufl.div(sigma), ufl.div(tau)) * dx
    L = ufl.inner(p * ufl.grad(P.w) - z + P.f1(), tau - p * ufl.grad(v)) * dx - ufl.inner(P.f2(), ufl.div(tau)) * dx
    prob = fem_petsc.LinearProblem(a, L, bcs=P.bcs["state"], petsc_options_prefix="qcd_",
                                   petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                  "pc_factor_mat_solver_type": "mumps"})
    return prob.solve()


def qcd_loss(P, sigma, u, p) -> float:
    """QCD's ``compute_physical_loss_1 + compute_physical_loss_2`` (default quadrature)."""
    t1 = sigma - (p * ufl.grad(u) + p * ufl.grad(P.w) - ufl.grad(P.q) + P.f1())
    t2 = ufl.div(sigma) + P.f2()
    dx = ufl.dx(domain=P.domain)
    return P._assemble(ufl.inner(t1, t1) * dx) + P._assemble(ufl.inner(t2, t2) * dx)


def run(exp: Experiment, num: int) -> dict:
    P = build_problem(exp)
    assert float(P.kappa) == 0.0, "run the check with the kappa = 0 configuration"
    solver = StateSolver(exp, P)
    X = P.h_gram_matrix()
    p_all = np.load(exp.parameter_dir / "p_dof.npy", mmap_mode="r")
    rows = exp.test_rows[:num]
    d_state, d_loss_qcd, d_loss_q12 = [], [], []
    for row in rows:
        p_dof = np.asarray(p_all[row])
        p = P.p_function(p_dof)
        s_new, _ = solver.solve(p_dof)
        s_lin = qcd_linear_solve(P, p).x.array
        e = s_new - s_lin
        d_state.append(float(np.sqrt(e @ (X @ e) / (s_lin @ (X @ s_lin)))))
        lq_new = qcd_loss(P, *P.split_state(s_new), p)
        lq_lin = qcd_loss(P, *P.split_state(s_lin), p)
        l12_new = sum(P.loss_of_state(s_new, p_dof))
        d_loss_qcd.append(abs(lq_new - lq_lin) / lq_lin)
        d_loss_q12.append(abs(l12_new - lq_lin) / lq_lin)
    out = {"rows": [int(r) for r in rows],
           "max_rel_state_difference_H": max(d_state),
           "max_rel_loss_difference_qcd_functional": max(d_loss_qcd),
           "max_rel_loss_difference_degree12_vs_qcd": max(d_loss_q12),
           "per_row": {"state": d_state, "loss_qcd": d_loss_qcd, "loss_q12": d_loss_q12}}
    exp.analysis_dir.mkdir(parents=True, exist_ok=True)
    save_json(exp.analysis_dir / "check_kappa0.json", out)
    print({k: v for k, v in out.items() if k.startswith("max")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--num", type=int, default=20)
    args = ap.parse_args()
    run(Experiment.load(args.config), args.num)


if __name__ == "__main__":
    main()
