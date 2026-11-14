"""Sample the parameters of an experiment into ``<parameters.dir>/p_dof.npy``.

Deterministic: the same configuration produces the same file. Experiments that share a
parameter directory (``semilinear_k0`` and ``semilinear_k1000``) reuse it.

    python -m rbno.generate_parameters configs/quasilinear.yaml
"""

from __future__ import annotations

import argparse

import numpy as np

from .config import Experiment
from .parameters import lognormal_field_max, sample_mini_square_exponents, sample_shifted_lognormal
from .problems import build_problem
from .utils import save_json


def generate(exp: Experiment, force: bool = False) -> None:
    out = exp.parameter_dir
    if (out / "p_dof.npy").exists() and not force:
        print(f"{out / 'p_dof.npy'} exists; use --force to resample.")
        return
    out.mkdir(parents=True, exist_ok=True)
    pcfg = exp.section("parameters")
    problem = build_problem(exp)
    kind = pcfg["type"]
    n = int(pcfg["num_samples"])
    meta = {"type": kind, "num_samples": n, "config": pcfg}
    if kind == "mini_squares_qcd":
        mu = sample_mini_square_exponents(n, int(pcfg.get("qcd_ranks", 50)))
        p_dof = problem.conductivity_dofs(mu)
        np.save(out / "mu.npy", mu)
        meta.update(p_min=float(p_dof.min()), p_max=float(p_dof.max()))
    elif kind == "shifted_lognormal":
        p_dof = sample_shifted_lognormal(
            n, problem.Vh["p"], gamma=float(pcfg["gamma"]), delta=float(pcfg["delta"]),
            shift=float(pcfg["shift"]), seed=int(pcfg.get("seed", 0)), robin_bc=bool(pcfg.get("robin_bc", True)),
        )
        meta.update(p_min=float(p_dof.min()), p_max=float(p_dof.max()),
                    m_hat=lognormal_field_max(p_dof, float(pcfg["shift"])))
    else:
        raise ValueError(f"Unknown parameter type {kind!r}.")
    np.save(out / "p_dof.npy", p_dof)
    save_json(out / "meta.json", meta)
    print(f"saved {p_dof.shape} -> {out / 'p_dof.npy'}  ({meta})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    generate(Experiment.load(args.config), force=args.force)


if __name__ == "__main__":
    main()
