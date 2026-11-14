"""FOSLS discretizations of the three benchmarks."""

from __future__ import annotations

from mpi4py import MPI

from .base import FOSLSProblem, SolveError
from .hyperelastic import HyperelasticProblem
from .quasilinear import QuasilinearProblem
from .semilinear import SemilinearProblem

PROBLEMS = {
    "semilinear": SemilinearProblem,
    "quasilinear": QuasilinearProblem,
    "hyperelastic": HyperelasticProblem,
}


def build_problem(exp, comm=None, *, num_x: int | None = None, num_y: int | None = None) -> FOSLSProblem:
    """Problem of an :class:`~rbno.config.Experiment` (optionally on another mesh)."""
    pcfg = exp.cfg["problem"]
    mesh = exp.mesh
    kwargs = dict(
        comm=MPI.COMM_SELF if comm is None else comm,
        num_x=int(num_x if num_x is not None else mesh["num_x"]),
        num_y=int(num_y if num_y is not None else mesh["num_y"]),
        length_x=float(mesh.get("length_x", 1.0)),
        length_y=float(mesh.get("length_y", 1.0)),
        quadrature_degree=int(pcfg["quadrature_degree"]),
    )
    kwargs["jacobian"] = str(exp.cfg.get("solver", {}).get("jacobian", "newton"))
    if exp.problem_type == "semilinear":
        kwargs["kappa"] = float(pcfg.get("kappa", 0.0))
    return PROBLEMS[exp.problem_type](**kwargs)


__all__ = ["FOSLSProblem", "SolveError", "PROBLEMS", "build_problem",
           "SemilinearProblem", "QuasilinearProblem", "HyperelasticProblem"]
