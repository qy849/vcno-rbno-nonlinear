"""Reduced basis neural operators (RBNO) trained on nonlinear FOSLS residuals.

Subpackages
-----------
problems   FOSLS discretizations of the three benchmarks (semilinear, quasilinear,
           hyperelastic): spaces, lifts, residual, Newton solver, norms.
models     Network encoders (the QCD heat-conduction CNN and the wide CNN encoder).
optim      SOAP and Muon optimizers and the optimizer/scheduler factories.
plots      Figure scripts for the paper.

Top-level modules implement the pipeline: parameter sampling (``parameters``),
offline solves (``solve``), POD (``pod``), residual training (``train``),
prediction (``predict``), common finite element evaluation (``evaluate``),
timing (``timing``), stability test (``stability``) and reporting (``report``).
"""

__all__ = ["__version__"]
__version__ = "1.0.0"
