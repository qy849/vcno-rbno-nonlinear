# Copyright (c) 2025-, Georgia Institute of Technology.
# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

"""PETSc linear solver wrappers and helpers."""

from petsc4py import PETSc


_PC_MAP = {
    "ml_amg": ("ml", None),
    "hypre_amg": ("hypre", "boomeramg"),
    "petsc_amg": ("gamg", None),
}


def _pc_type_from_preconditioner(preconditioner):
    """Return the PETSc PC type for a preconditioner spec."""
    if preconditioner is None:
        return "none", None
    if preconditioner in _PC_MAP:
        return _PC_MAP[preconditioner]
    return preconditioner, None


class _PETScSolverWrapper:
    """Class _PETScSolverWrapper."""
    def __init__(self, comm, ksp_type, pc_type, pc_factor=None):
        """Initialize the instance."""
        self._ksp = PETSc.KSP().create(comm)
        self._ksp.setType(ksp_type)
        pc = self._ksp.getPC()
        pc.setType(pc_type)
        if pc_factor:
            pc.setFactorSolverType(pc_factor)
        if pc_type == "hypre":
            try:
                pc.setHYPREType("boomeramg")
            except PETSc.Error:
                pass
        self.parameters = {
            "maximum_iterations": None,
            "relative_tolerance": None,
            "absolute_tolerance": None,
            "error_on_nonconvergence": False,
            "nonzero_initial_guess": False,
        }

    def set_operator(self, A):
        """Compute set_operator."""
        if hasattr(A, "mat"):
            self._ksp.setOperators(A.mat())
        else:
            self._ksp.setOperators(A)
        self._apply_parameters()

    def solve(self, x, b):
        """Compute solve."""
        self._apply_parameters()
        self._ksp.solve(b, x)
        return self._ksp.getIterationNumber()

    def ksp(self):
        """Compute ksp."""
        return self._ksp

    def _apply_parameters(self):
        """Internal helper."""
        params = self.parameters
        if (
            params.get("maximum_iterations") is not None
            or params.get("relative_tolerance") is not None
            or params.get("absolute_tolerance") is not None
        ):
            rtol = params.get("relative_tolerance")
            atol = params.get("absolute_tolerance")
            max_it = params.get("maximum_iterations")
            self._ksp.setTolerances(
                rtol=PETSc.DEFAULT if rtol is None else rtol,
                atol=PETSc.DEFAULT if atol is None else atol,
                max_it=PETSc.DEFAULT if max_it is None else max_it,
            )
        self._ksp.setErrorIfNotConverged(bool(params.get("error_on_nonconvergence")))
        self._ksp.setInitialGuessNonzero(bool(params.get("nonzero_initial_guess")))


def PETScKrylovSolver(comm, method, preconditioner):
    """Create a PETSc KSP solver wrapper."""
    pc_type, pc_factor = _pc_type_from_preconditioner(preconditioner)
    return _PETScSolverWrapper(comm, method, pc_type, pc_factor)


def PETScLUSolver(comm, method="default"):
    """Create a PETSc LU solver wrapper."""
    pc_factor = None if method == "default" else method
    return _PETScSolverWrapper(comm, "preonly", "lu", pc_factor)
