"""H-orthonormal proper orthogonal decomposition of the snapshot store.

With the snapshot matrix ``S`` (columns ``s_h(p_i)``, ``i = 1..N_POD``) and the Gram matrix
``X_h`` of the H(div) x H^1 inner product, the eigenpairs ``(lambda_k, v_k)`` of
``C = S^T X_h S / N_POD`` give the POD vectors ``pi_k = S v_k / sqrt(N_POD lambda_k)``,
which are orthonormal in ``X_h`` (QCD, Section 5). The first ``r`` of them form ``Pi_r``.

Outputs in ``data/<experiment>/pod/``::

    basis.npy          (n_state, r)   Pi_r in mixed sigma_u dof order
    eigenvalues.npy    (N_POD,)       lambda_k, descending (all positive eigenvalues of C)
    snapshot_rows.npy  parameter rows of the snapshots used
    meta.json

    python -m rbno.pod configs/quasilinear.yaml
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .config import Experiment
from .problems import build_problem
from .utils import load_json, save_json


def compute_pod(snapshots: np.ndarray, gram, num_basis: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(eigenvalues, basis)`` for snapshots of shape ``(N, n_state)``.

    ``num_basis=None`` returns every mode with a positive eigenvalue."""
    n = snapshots.shape[0]
    XS = gram @ snapshots.T                          # (n_state, N)
    C = (snapshots @ XS) / n                         # (N, N)
    C = 0.5 * (C + C.T)
    lam, V = np.linalg.eigh(C)
    lam, V = lam[::-1], V[:, ::-1]
    tol = max(1e-14, 1e-12 * lam[0])
    keep = lam > tol
    num_basis = int(keep.sum()) if num_basis is None else num_basis
    if int(keep.sum()) < num_basis:
        raise ValueError(f"Only {int(keep.sum())} positive POD eigenvalues, {num_basis} requested.")
    basis = (snapshots.T @ V[:, :num_basis]) / np.sqrt(n * lam[:num_basis])[None, :]
    return lam[keep], basis


def load_snapshots(exp: Experiment, count: int) -> tuple[np.ndarray, np.ndarray]:
    """The first ``count`` converged snapshots (ascending parameter row)."""
    store = exp.snapshot_dir
    valid = np.load(store / "valid_indices.npy")
    if valid.size < count:
        raise ValueError(f"Snapshot store has {valid.size} converged solves, {count} needed.")
    data = np.load(store / "sigma_u_dof.npy", mmap_mode="r")
    return np.asarray(data[:count], dtype=np.float64), valid[:count]


def build(exp: Experiment, force: bool = False) -> None:
    out = exp.pod_dir
    if (out / "basis.npy").exists() and not force:
        print(f"{out / 'basis.npy'} exists (frozen with the trained models); use --force to rebuild.")
        return
    cfg = exp.section("pod")
    n_pod, r = int(cfg["num_snapshots"]), int(cfg["num_basis"])
    problem = build_problem(exp)
    t0 = time.perf_counter()
    S, rows = load_snapshots(exp, n_pod)
    out.mkdir(parents=True, exist_ok=True)
    X = problem.h_gram_matrix()
    lam, full = compute_pod(S, X, None)
    basis = full[:, :r]
    test_projection(exp, full, X, lam)
    del full
    # Orthonormality check in X_h.
    G = basis.T @ (X @ basis)
    ortho = float(np.abs(G - np.eye(r)).max())
    mean_norm2 = float(np.mean(np.einsum("ij,ji->i", S, X @ S.T)))
    np.save(out / "basis.npy", basis)
    np.save(out / "eigenvalues.npy", lam)
    np.save(out / "snapshot_rows.npy", rows)
    save_json(out / "meta.json", {
        "num_snapshots": n_pod, "num_basis": r, "orthonormality_error": ortho,
        "mean_snapshot_norm2": mean_norm2,
        "tail_sum_after_r": float(mean_norm2 - lam[:r].sum()),
        "seconds": time.perf_counter() - t0,
    })
    print(f"POD: N={n_pod}, r={r}, max|Pi^T X Pi - I| = {ortho:.2e}, "
          f"tail after r = {mean_norm2 - lam[:r].sum():.3e} -> {out}")


def test_projection(exp: Experiment, full_basis: np.ndarray, gram, lam: np.ndarray) -> None:
    """Mean ``||(I - P_r) s_h||_H^2`` over the converged test solutions for r = 1..N_POD-1.

    Uses ``||(I - P_r) s||^2 = ||s||^2 - sum_{k<=r} c_k^2`` with ``c = Pi^T X_h s``.
    """
    store = exp.test_dir
    if not (store / "valid_indices.npy").exists():
        print("no test store yet; skipping the test projection error")
        return
    data = np.load(store / "sigma_u_dof.npy", mmap_mode="r")
    norm2, cum = [], []
    for s in range(0, data.shape[0], 100):
        Y = np.asarray(data[s:s + 100], dtype=np.float64)
        XY = (gram @ Y.T)
        C = full_basis.T @ XY                                   # (K, batch)
        norm2.append(np.einsum("ij,ji->i", Y, XY))
        cum.append(np.cumsum(C ** 2, axis=0).T)                 # (batch, K)
    norm2 = np.concatenate(norm2)
    cum = np.concatenate(cum)
    proj_err = norm2[:, None] - cum                             # (n_test, K)
    snap_tail = np.cumsum(lam[::-1])[::-1]                      # sum_{k>=r} lambda_k
    np.savez(exp.pod_dir / "projection_errors.npz",
             r=np.arange(1, proj_err.shape[1] + 1), test_mean=proj_err.mean(axis=0),
             test_median=np.median(proj_err, axis=0), test_rel_mean=(proj_err / norm2[:, None]).mean(axis=0),
             test_norm2_mean=norm2.mean(), snapshot_tail=np.append(snap_tail[1:], 0.0), eigenvalues=lam)


def load_basis(exp: Experiment) -> np.ndarray:
    path = exp.pod_dir / "basis.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing POD basis {path}; run `python -m rbno.pod {exp.path}` first.")
    basis = np.load(path)
    r = int(exp.section("pod")["num_basis"])
    if basis.shape[1] != r:
        raise ValueError(f"POD basis has {basis.shape[1]} modes, the configuration asks for {r}.")
    return basis


def pod_meta(exp: Experiment) -> dict:
    return load_json(exp.pod_dir / "meta.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    build(Experiment.load(args.config), force=args.force)


if __name__ == "__main__":
    main()
