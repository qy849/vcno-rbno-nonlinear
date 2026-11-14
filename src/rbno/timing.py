"""Wall-clock costs per parameter for Table ``tab:cost``, measured in one CPU process.

* loss and gradient: one call of ``problem.loss_and_grad`` (assembly of ``J`` and of the
  derivative form) at the RBNO prediction, after a warm-up call that compiles the forms;
* finite element solve: the production solve of :class:`rbno.solve.StateSolver`
  (initial guess and, for hyperelasticity, load continuation and fallbacks included);
  for the semilinear benchmark the linear kappa = 0 solve used as initial guess is
  reported separately and included in ``solve_total_s``;
* prediction and training time are read from ``pred_meta.json`` / ``train_summary.json``.

Run it on an otherwise idle node; the store means (``meta.json`` of the test store)
were measured with many concurrent MPI ranks and are reported alongside.

    python -m rbno.timing configs/quasilinear.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time

import numpy as np

from .config import Experiment
from .solve import StateSolver
from .utils import load_json, save_json, sigma_u_permutation


def measure(exp: Experiment, n_loss: int = 20, n_solve: int = 5, method: str = "rbno") -> dict:
    solver = StateSolver(exp)
    problem = solver.problem
    perm = sigma_u_permutation(problem.Vh)
    p_all = np.load(exp.parameter_dir / "p_dof.npy", mmap_mode="r")
    rows = exp.test_rows
    pred = np.load(exp.results_dir(method) / "pred_sigma_u_dof.npy", mmap_mode="r")

    problem.loss_and_grad(np.asarray(pred[0])[perm], np.asarray(p_all[rows[0]]))  # compile
    t = []
    for k in range(1, n_loss + 1):
        y, p = np.asarray(pred[k])[perm], np.asarray(p_all[rows[k]])
        t0 = time.perf_counter()
        problem.loss_and_grad(y, p)
        t.append(time.perf_counter() - t0)

    with contextlib.redirect_stdout(io.StringIO()):
        solver.prepare()
        solver.solve(np.asarray(p_all[rows[0]]))  # compile
    solve_s, total_s, iters = [], [], []
    for k in range(1, n_solve + 1):
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            _, info = solver.solve(np.asarray(p_all[rows[k]]))
        total_s.append(time.perf_counter() - t0)
        solve_s.append(info.get("elapsed_s", np.nan))
        iters.append(info.get("iterations", np.nan))

    out = {
        "experiment": exp.name,
        "loss_grad_s": float(np.mean(t)), "loss_grad_s_std": float(np.std(t)), "n_loss": n_loss,
        "solve_s": float(np.nanmean(solve_s)), "solve_total_s": float(np.mean(total_s)),
        "solve_iterations": [float(i) for i in iters], "n_solve": n_solve,
        "note": "single CPU process (MPI COMM_SELF), wall-clock; solve_total_s includes the initial guess "
                "(kappa=0 solve for the semilinear benchmark) and all continuation steps/attempts",
    }
    for name, path in (("prediction", exp.results_dir(method) / "pred_meta.json"),
                       ("training", exp.results_dir(method) / "train_summary.json"),
                       ("test_store", exp.test_dir / "meta.json")):
        if path.exists():
            out[name] = load_json(path)
    out["prediction_s"] = out.get("prediction", {}).get("seconds_per_parameter")
    out["training_h"] = out.get("training", {}).get("train_hours")
    out["store_mean_solve_s"] = out.get("test_store", {}).get("mean_elapsed_s")
    exp.analysis_dir.mkdir(parents=True, exist_ok=True)
    save_json(exp.analysis_dir / "timing.json", out)
    print(f"[{exp.name}] loss+grad {out['loss_grad_s']:.3f}s | solve {out['solve_s']:.2f}s "
          f"(total {out['solve_total_s']:.2f}s) | prediction {out['prediction_s']} s | training {out['training_h']} h")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--n-loss", type=int, default=20)
    ap.add_argument("--n-solve", type=int, default=5)
    args = ap.parse_args()
    measure(Experiment.load(args.config), args.n_loss, args.n_solve)


if __name__ == "__main__":
    main()
