"""Offline finite element solves: snapshot and test stores.

Every parameter row is solved independently of the others, from a fixed initial guess,
so a store does not depend on the order of the solves or on the number of MPI ranks:

* ``semilinear``  -- Newton for ``kappa`` started from the solution of the linear problem
  with ``kappa = 0`` for the same parameter (one linear solve). The relative change
  ``||s_kappa - s_0||_H / ||s_kappa||_H`` is recorded.
* ``quasilinear`` -- Newton started from the solution for the mean parameter.
* ``hyperelastic`` -- load continuation. A seed is computed once for the mean parameter at
  the first load factor (cold ramp ``solver.seed_loads``); every sample then walks the
  load through ``solver.attempts[0].loads`` from that seed. If an attempt fails, the next
  attempt of ``solver.attempts`` is tried (more and smaller load steps, possibly from a
  cold start). The attempt that converged is recorded; samples for which every attempt
  fails are listed as failed, never dropped silently.

A store directory holds::

    sigma_u_dof.npy     (n_valid, n_state)  converged states, mixed dof order, rows of valid_indices
    valid_indices.npy   parameter rows that converged (ascending)
    failed_indices.npy  parameter rows without a solution
    indices.npy         every row that was attempted
    solve_stats.npz     per attempted row: success, elapsed_s (successful attempt), total_s
                        (all attempts), iterations, reason, attempt, rel_change, linear_s
    meta.json           experiment, solver settings, counts, ranks

Solves are written to ``<store>/parts/`` as they finish, so an interrupted run resumes
where it stopped when the same command is repeated.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

from .config import Experiment
from .problems import SolveError, build_problem
from .utils import save_json


class StateSolver:
    """Per-sample solve of one experiment, with its initial guess and fallbacks."""

    def __init__(self, exp: Experiment, problem=None):
        self.exp = exp
        self.problem = problem if problem is not None else build_problem(exp)
        self.cfg = exp.section("solver")
        self.seed: np.ndarray | None = None
        self._gram = None

    # -- seed ---------------------------------------------------------------------------
    @property
    def seed_path(self) -> Path:
        return self.exp.data_dir / "seed_state.npy"

    def prepare(self, comm=MPI.COMM_WORLD) -> None:
        """Compute (rank 0) or load the shared seed state, then broadcast it."""
        mode = self.cfg.get("initial_guess", "zero")
        if mode not in ("mean_parameter", "mean_parameter_continuation"):
            return
        seed = None
        if comm.Get_rank() == 0:
            if self.seed_path.exists():
                seed = np.load(self.seed_path)
            else:
                p_all = np.load(self.exp.parameter_dir / "p_dof.npy", mmap_mode="r")
                p_mean = np.zeros(p_all.shape[1])
                for s in range(0, p_all.shape[0], 1000):
                    p_mean += np.asarray(p_all[s:s + 1000]).sum(axis=0)
                p_mean /= p_all.shape[0]
                p = self.problem.p_function(p_mean)
                if mode == "mean_parameter":
                    state = self.problem.newton_solve(p)
                else:
                    loads = np.asarray(self.cfg["seed_loads"], dtype=float)
                    state = self.problem.continuation_solve(p, None, loads,
                                                            snes_max_it=int(self.cfg.get("snes_max_it", 120)))
                seed = state.x.array.copy()
                self.seed_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(self.seed_path, seed)
        self.seed = comm.bcast(seed, root=0)

    # -- one sample --------------------------------------------------------------------
    def solve(self, p_dof: np.ndarray, *, capture_iterates: bool = False) -> tuple[np.ndarray | None, dict]:
        mode = self.cfg.get("initial_guess", "zero")
        P = self.problem
        p = P.p_function(p_dof)
        if mode == "linear":
            return self._solve_from_linear(p, capture_iterates)
        if mode == "mean_parameter_continuation":
            return self._solve_continuation(p, capture_iterates)
        guess = self.seed if mode == "mean_parameter" else None
        t0 = time.perf_counter()
        try:
            state = P.newton_solve(p, guess, snes_max_it=self.cfg.get("snes_max_it"),
                                   capture_iterates=capture_iterates)
        except SolveError as exc:
            return None, {**exc.info, "attempt": -1, "total_s": time.perf_counter() - t0}
        return state.x.array.copy(), {**P.solve_info, "attempt": 0, "total_s": time.perf_counter() - t0}

    def _solve_from_linear(self, p, capture_iterates):
        """semilinear: Newton for kappa from the kappa = 0 solution of the same parameter."""
        P = self.problem
        kappa = float(P.kappa)
        t0 = time.perf_counter()
        P.set_kappa(0.0)
        try:
            lin = P.newton_solve(p)
        finally:
            P.set_kappa(kappa)
        lin_info = dict(P.solve_info)
        if kappa == 0.0:
            return lin.x.array.copy(), {**lin_info, "attempt": 0, "linear_s": lin_info["elapsed_s"],
                                        "total_s": time.perf_counter() - t0, "rel_change": 0.0}
        try:
            state = P.newton_solve(p, lin.x.array, snes_max_it=self.cfg.get("snes_max_it"),
                                   capture_iterates=capture_iterates)
        except SolveError as exc:
            return None, {**exc.info, "attempt": -1, "linear_s": lin_info["elapsed_s"],
                          "total_s": time.perf_counter() - t0}
        diff = state.x.array - lin.x.array
        gram = self.gram()
        rel = float(np.sqrt(diff @ (gram @ diff) / (state.x.array @ (gram @ state.x.array))))
        return state.x.array.copy(), {**P.solve_info, "attempt": 0, "linear_s": lin_info["elapsed_s"],
                                      "total_s": time.perf_counter() - t0, "rel_change": rel}

    def _solve_continuation(self, p, capture_iterates):
        """hyperelastic: load continuation from the seed, then the fallback attempts."""
        P = self.problem
        t0 = time.perf_counter()
        history = []
        for k, attempt in enumerate(self.cfg["attempts"]):
            loads = np.asarray(attempt["loads"], dtype=float)
            if "num_steps" in attempt:
                loads = np.linspace(float(attempt["loads"][0]), float(attempt["loads"][1]), int(attempt["num_steps"]))
            start = self.seed if attempt.get("start", "seed") == "seed" else None
            try:
                state = P.continuation_solve(p, start, loads, snes_max_it=int(self.cfg.get("snes_max_it", 120)),
                                             capture_iterates=capture_iterates)
            except SolveError as exc:
                history.append({"attempt": k, "reason": exc.info["reason"], "elapsed_s": exc.info["elapsed_s"]})
                continue
            info = {**P.solve_info, "attempt": k, "total_s": time.perf_counter() - t0, "failed_attempts": history}
            info.pop("steps", None)
            return state.x.array.copy(), info
        last = history[-1] if history else {"reason": 0, "elapsed_s": 0.0}
        return None, {"success": False, "attempt": -1, "reason": last["reason"], "iterations": -1,
                      "elapsed_s": float("nan"), "total_s": time.perf_counter() - t0, "failed_attempts": history}

    def gram(self):
        if self._gram is None:
            self._gram = self.problem.h_gram_matrix()
        return self._gram


# ---------------------------------------------------------------------------------------
# block solves
# ---------------------------------------------------------------------------------------
STAT_KEYS = ("success", "elapsed_s", "total_s", "iterations", "reason", "attempt", "rel_change", "linear_s")


def solve_block(exp: Experiment, rows: np.ndarray, out_dir: Path, *, comm=MPI.COMM_WORLD, label: str = "") -> None:
    """Solve the parameter ``rows`` into the store ``out_dir`` (collective)."""
    rank, size = comm.Get_rank(), comm.Get_size()
    out_dir = Path(out_dir)
    parts = out_dir / "parts"
    if rank == 0:
        parts.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    solver = StateSolver(exp)
    solver.prepare(comm)
    p_all = np.load(exp.parameter_dir / "p_dof.npy", mmap_mode="r")
    rows = np.asarray(rows, dtype=np.int64)
    mine = rows[rank::size]  # round-robin: solve times vary strongly between samples
    t_start = time.perf_counter()
    for k, idx in enumerate(mine):
        part = parts / f"{int(idx):06d}.npz"
        if part.exists():
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            state, info = solver.solve(np.asarray(p_all[idx]))
        stats = {key: float(info.get(key, np.nan)) if info.get(key) is not None else np.nan for key in STAT_KEYS}
        stats["success"] = float(state is not None)
        tmp = parts / f"{int(idx):06d}.tmp.npz"
        np.savez(tmp, state=state if state is not None else np.empty(0), **stats)
        os.replace(tmp, part)
        if rank == 0:
            done = k + 1
            eta = (time.perf_counter() - t_start) / done * (len(mine) - done)
            print(f"[{label}] rank 0: {done}/{len(mine)} solved, last {info.get('elapsed_s', np.nan):.1f}s "
                  f"({info.get('iterations')} it, attempt {info.get('attempt')}), eta {eta / 60:.1f} min", flush=True)
    comm.Barrier()
    wall = time.perf_counter() - t_start
    if rank == 0:
        merge_store(exp, rows, out_dir, mpi_ranks=size, wall_s=wall, label=label)
    comm.Barrier()


def merge_store(exp: Experiment, rows: np.ndarray, out_dir: Path, *, mpi_ranks: int, wall_s: float, label: str) -> None:
    parts = out_dir / "parts"
    stats = {k: np.full(rows.size, np.nan) for k in STAT_KEYS}
    valid, failed = [], []
    n_state = None
    for i, idx in enumerate(rows):
        with np.load(parts / f"{int(idx):06d}.npz") as z:
            for k in STAT_KEYS:
                stats[k][i] = float(z[k])
            if z["state"].size:
                valid.append(int(idx))
                n_state = z["state"].size
            else:
                failed.append(int(idx))
    valid = np.asarray(valid, dtype=np.int64)
    out = np.lib.format.open_memmap(out_dir / "sigma_u_dof.npy", mode="w+", dtype=np.float64,
                                    shape=(valid.size, n_state or 0))
    for r, idx in enumerate(valid):
        with np.load(parts / f"{int(idx):06d}.npz") as z:
            out[r] = z["state"]
    out.flush()
    del out
    np.save(out_dir / "valid_indices.npy", valid)
    np.save(out_dir / "failed_indices.npy", np.asarray(failed, dtype=np.int64))
    np.save(out_dir / "indices.npy", rows)
    np.savez(out_dir / "solve_stats.npz", indices=rows, **stats)
    ok = stats["success"] > 0
    meta = {
        "experiment": exp.name,
        "label": label,
        "problem": exp.cfg["problem"],
        "solver": exp.section("solver"),
        "num_attempted": int(rows.size),
        "num_valid": int(valid.size),
        "num_failed": int(len(failed)),
        "failed_indices": [int(i) for i in failed],
        "attempt_counts": {str(int(a)): int(np.sum(stats["attempt"] == a)) for a in np.unique(stats["attempt"][np.isfinite(stats["attempt"])])},
        "median_iterations": float(np.median(stats["iterations"][ok])) if ok.any() else None,
        "mean_elapsed_s": float(np.mean(stats["elapsed_s"][ok])) if ok.any() else None,
        "mean_total_s": float(np.mean(stats["total_s"])) if rows.size else None,
        "mpi_ranks": int(mpi_ranks),
        "wall_s": float(wall_s),
        "timing_note": "elapsed_s: wall-clock of the successful Newton solve(s) of one sample on one MPI "
                       "rank (COMM_SELF), all load steps included; total_s adds failed attempts and, for the "
                       "semilinear benchmark, the linear kappa=0 solve used as initial guess. "
                       f"{mpi_ranks} ranks ran concurrently.",
    }
    save_json(out_dir / "meta.json", meta)
    for f in parts.glob("*.npz"):
        f.unlink()
    parts.rmdir()
    print(f"[{label}] {valid.size}/{rows.size} converged, {len(failed)} failed -> {out_dir}", flush=True)
    if failed:
        print(f"[{label}] failed rows: {failed}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Solve the snapshot or test block of an experiment.")
    ap.add_argument("config")
    ap.add_argument("--store", choices=["snapshots", "test"], required=True)
    ap.add_argument("--force", action="store_true", help="re-solve even if the store exists")
    args = ap.parse_args()
    exp = Experiment.load(args.config)
    comm = MPI.COMM_WORLD
    out_dir = exp.snapshot_dir if args.store == "snapshots" else exp.test_dir
    rows = exp.snapshot_rows if args.store == "snapshots" else exp.test_rows
    done = comm.bcast((out_dir / "meta.json").exists() and not args.force, root=0)
    if done:
        if comm.Get_rank() == 0:
            print(f"{out_dir} exists; use --force to recompute.")
        return
    solve_block(exp, rows, out_dir, comm=comm, label=f"{exp.name}/{args.store}")


if __name__ == "__main__":
    main()
