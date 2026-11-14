"""Common finite element evaluation of predictions on the test set (Section 6.1).

Any predictor is evaluated in the same way: its prediction ``s_i`` is a dof vector of
RT_1 x CG_2 (mixed order), stored as ``pred_sigma_u_dof.npy`` with the parameter rows in
``pred_indices.npy`` in ``results/<experiment>/<method>/``. For every test parameter:

    eta_i^2   = J_{h,q}(s_i; p_i)            loss of the prediction (and its two components)
    eta_h,i^2 = J_{h,q}(s_h,i; p_i)          loss of the finite element solution (floor)
    E_h,i     = ||s_i - s_h,i||_H            error, split into ||e_sigma||^2, ||div e_sigma||^2,
                                             ||e_u||^2 and ||grad e_u||^2
    chi_i     = E_h,i / (eta_i^2 - eta_h,i^2)^{1/2}     floor-corrected ratio
    relative H error E_h,i / ||s_h,i||_H and relative L2 x L2 error;
    POD split (eq. state_pythagoras): ||(I - P_r) s_h,i||_H and ||s_i - P_r s_h,i||_H.

Parameters whose finite element solve failed keep their loss ``eta_i`` and get NaN
errors; they are counted, never dropped silently.

    mpirun -n 60 python -m rbno.evaluate configs/quasilinear.yaml --method rbno
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from mpi4py import MPI

from .config import Experiment
from .problems import build_problem
from .utils import save_json, summary_stats

FIELDS = ("loss", "loss_r1", "loss_r2", "loss_h", "loss_h_r1", "loss_h_r2",
          "err_sigma_l2", "err_div_sigma_l2", "err_u_l2", "err_grad_u_l2",
          "ref_sigma_l2", "ref_div_sigma_l2", "ref_u_l2", "ref_grad_u_l2",
          "basis_part2", "coef_part2")


def _quad(M, x):
    return float(x @ (M @ x))


def evaluate(exp: Experiment, method: str, *, pod_split: bool = True, comm=MPI.COMM_WORLD) -> dict | None:
    rank, size = comm.Get_rank(), comm.Get_size()
    res_dir = exp.results_dir(method)
    pred = np.load(res_dir / "pred_sigma_u_dof.npy", mmap_mode="r")
    pred_rows = np.load(res_dir / "pred_indices.npy")
    test = exp.test_dir
    truth = np.load(test / "sigma_u_dof.npy", mmap_mode="r")
    truth_row = {int(r): k for k, r in enumerate(np.load(test / "valid_indices.npy"))}
    p_all = np.load(exp.parameter_dir / "p_dof.npy", mmap_mode="r")

    problem = build_problem(exp)
    mats = problem.norm_matrices()
    basis = gram = None
    if pod_split and (exp.pod_dir / "basis.npy").exists():
        basis = np.load(exp.pod_dir / "basis.npy")
        gram = mats["sigma_l2"] + mats["div_sigma_l2"] + mats["u_l2"] + mats["grad_u_l2"]

    mine = np.arange(rank, pred_rows.size, size)
    local = {k: np.full(mine.size, np.nan) for k in FIELDS}
    t0 = time.perf_counter()
    for j, pos in enumerate(mine):
        row = int(pred_rows[pos])
        p_dof = np.asarray(p_all[row])
        s = np.asarray(pred[pos], dtype=np.float64)
        l1, l2 = problem.loss_of_state(s, p_dof)
        local["loss_r1"][j], local["loss_r2"][j], local["loss"][j] = l1, l2, l1 + l2
        if row not in truth_row:
            continue
        sh = np.asarray(truth[truth_row[row]], dtype=np.float64)
        h1, h2 = problem.loss_of_state(sh, p_dof)
        local["loss_h_r1"][j], local["loss_h_r2"][j], local["loss_h"][j] = h1, h2, h1 + h2
        e = s - sh
        for key, M in mats.items():
            local[f"err_{key}"][j] = _quad(M, e)
            local[f"ref_{key}"][j] = _quad(M, sh)
        if basis is not None:
            Xsh = gram @ sh
            c = basis.T @ Xsh
            local["basis_part2"][j] = float(sh @ Xsh - c @ c)
            d = s - basis @ c
            local["coef_part2"][j] = _quad(gram, d)
    seconds = time.perf_counter() - t0

    gathered = comm.gather((mine, local), root=0)
    if rank != 0:
        return None
    data = {k: np.full(pred_rows.size, np.nan) for k in FIELDS}
    for pos, loc in gathered:
        for k in FIELDS:
            data[k][pos] = loc[k]
    return _finish(exp, method, pred_rows, data, seconds, size)


def derived(data: dict) -> dict:
    """Errors, norms and ratios from the raw squared components."""
    err_H2 = data["err_sigma_l2"] + data["err_div_sigma_l2"] + data["err_u_l2"] + data["err_grad_u_l2"]
    ref_H2 = data["ref_sigma_l2"] + data["ref_div_sigma_l2"] + data["ref_u_l2"] + data["ref_grad_u_l2"]
    err_L2 = data["err_sigma_l2"] + data["err_u_l2"]
    ref_L2 = data["ref_sigma_l2"] + data["ref_u_l2"]
    eta = np.sqrt(data["loss"])
    excess = data["loss"] - data["loss_h"]
    E = np.sqrt(err_H2)
    with np.errstate(invalid="ignore", divide="ignore"):
        chi = np.where(excess > 0, E / np.sqrt(np.where(excess > 0, excess, 1.0)), np.nan)
        return {
            "E_h": E, "eta": eta, "eta_h": np.sqrt(data["loss_h"]),
            "loss_excess": excess,
            "ratio_E_over_eta": E / eta,
            "chi": chi,
            "rel_H": E / np.sqrt(ref_H2),
            "rel_L2": np.sqrt(err_L2 / ref_L2),
            "rel_basis_part": np.sqrt(data["basis_part2"]) / np.sqrt(ref_H2),
            "rel_coef_part": np.sqrt(data["coef_part2"]) / np.sqrt(ref_H2),
            "err_H1_u2": data["err_u_l2"] + data["err_grad_u_l2"],
        }


def _finish(exp, method, rows, data, seconds, ranks) -> dict:
    out_dir = exp.results_dir(method)
    der = derived(data)
    has_truth = np.isfinite(data["loss_h"])
    np.savez(out_dir / "metrics.npz", indices=rows, has_truth=has_truth, **data, **der)
    m = has_truth
    bochner = lambda num, den: float(np.sqrt(np.nansum(num[m]) / np.nansum(den[m])))  # noqa: E731
    err_H2 = der["E_h"] ** 2
    ref_H2 = (der["E_h"] / der["rel_H"]) ** 2
    summary = {
        "experiment": exp.name, "method": method,
        "n_total": int(rows.size), "n_with_truth": int(m.sum()),
        "rows_without_truth": [int(r) for r in rows[~m]],
        "n_loss_below_floor": int(np.sum(der["loss_excess"][m] <= 0)),
        "loss": summary_stats(data["loss"]),
        "loss_on_truth_cohort": summary_stats(data["loss"][m]),
        "loss_without_truth": summary_stats(data["loss"][~m]),
        "loss_r1": summary_stats(data["loss_r1"][m]), "loss_r2": summary_stats(data["loss_r2"][m]),
        "loss_h": summary_stats(data["loss_h"][m]),
        "loss_h_r1": summary_stats(data["loss_h_r1"][m]), "loss_h_r2": summary_stats(data["loss_h_r2"][m]),
        "rel_H": summary_stats(der["rel_H"][m]), "rel_L2": summary_stats(der["rel_L2"][m]),
        "rel_H_bochner": bochner(err_H2, ref_H2),
        "rel_basis_part": summary_stats(der["rel_basis_part"][m]),
        "rel_coef_part": summary_stats(der["rel_coef_part"][m]),
        "E_h": summary_stats(der["E_h"][m]),
        "ratio_E_over_eta": summary_stats(der["ratio_E_over_eta"][m]),
        "chi": summary_stats(der["chi"][m]),
        "error_components": {k: summary_stats(data[k][m]) for k in
                             ("err_sigma_l2", "err_div_sigma_l2", "err_u_l2", "err_grad_u_l2")},
        "err_H1_u2": summary_stats(der["err_H1_u2"][m]),
        "seconds": seconds, "mpi_ranks": ranks,
    }
    save_json(out_dir / "metrics_summary.json", summary)
    print(f"[{exp.name}/{method}] n={m.sum()}/{rows.size}: rel H {summary['rel_H']['mean']:.3e} "
          f"({summary['rel_H']['std']:.2e}), loss {summary['loss']['mean']:.3e}, floor "
          f"{summary['loss_h']['mean']:.3e}, E/eta median {summary['ratio_E_over_eta']['median']:.3f} "
          f"[{summary['ratio_E_over_eta']['min']:.3f}, {summary['ratio_E_over_eta']['max']:.3f}], chi median "
          f"{summary['chi']['median']:.3f} [{summary['chi']['min']:.3f}, {summary['chi']['max']:.3f}]", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--method", default="rbno")
    ap.add_argument("--no-pod-split", action="store_true")
    args = ap.parse_args()
    evaluate(Experiment.load(args.config), args.method, pod_split=not args.no_pod_split)


if __name__ == "__main__":
    main()
