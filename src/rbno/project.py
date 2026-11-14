"""The H-orthogonal projection ``P_r s_h`` of the test solutions as a pseudo-predictor.

Written in the prediction format to ``results/<experiment>/projection/`` so that the common
evaluator reports its loss and error; its mean loss is the reference line of the training
histories, and its error is the basis part of ``eq:state_pythagoras``.

    python -m rbno.project configs/quasilinear.yaml
"""

from __future__ import annotations

import argparse

import numpy as np

from .config import Experiment
from .pod import load_basis
from .problems import build_problem


def project(exp: Experiment) -> None:
    basis = load_basis(exp)
    X = build_problem(exp).h_gram_matrix()
    rows = np.load(exp.test_dir / "valid_indices.npy")
    truth = np.load(exp.test_dir / "sigma_u_dof.npy", mmap_mode="r")
    out = exp.results_dir("projection")
    out.mkdir(parents=True, exist_ok=True)
    pred = np.empty((rows.size, basis.shape[0]))
    for s in range(0, rows.size, 100):
        Y = np.asarray(truth[s:s + 100], dtype=np.float64)
        pred[s:s + 100] = (basis @ (basis.T @ (X @ Y.T))).T
    np.save(out / "pred_sigma_u_dof.npy", pred)
    np.save(out / "pred_indices.npy", rows)
    print(f"[{exp.name}] projected {rows.size} test solutions onto r={basis.shape[1]} POD modes -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    project(Experiment.load(ap.parse_args().config))


if __name__ == "__main__":
    main()
