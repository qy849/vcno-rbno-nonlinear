# Copyright (c) 2025-, Georgia Institute of Technology.
# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
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
# Software Foundation) version 2.1 dated February 1999.

"""Variational QoI implementations and assembly utilities."""

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
import ufl
from petsc4py import PETSc

from ..modeling.variables import STATE, PARAMETER
from ..utils.vector2function import vector2Function
from .qoi import Qoi


def _assemble_vec(form, out, kind=None):
    """Assemble a linear form into an output vector."""
    tmp = fem_petsc.assemble_vector(fem.form(form), kind=kind)
    try:
        tmp.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    except PETSc.Error:
        pass
    out.set(0.0)
    out.axpy(1.0, tmp)
    out.assemblyBegin()
    out.assemblyEnd()

class VariationalQoi(Qoi):
    """QoI defined by a variational form of state and parameter."""
    def __init__(self, Vh, qoi_varf, state_bcs=None):
        """Initialize a variational QoI defined by a form."""
        self.Vh = Vh
        self.qoi_varf = qoi_varf
        if state_bcs is None:
            self._state_bcs = []
        elif isinstance(state_bcs, (list, tuple)):
            self._state_bcs = list(state_bcs)
        else:
            self._state_bcs = [state_bcs]
        self._state_bc_dofs = None
        
        self.L = {}
        self.use_hessian = True

    def eval(self, x):
        """
        Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter a are accessed.
        """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        return fem.assemble_scalar(fem.form(self.qoi_varf(u, m)))
    
    def grad(self, i, x, g):
        """Evaluate gradient with respect to state or parameter."""
        if i == STATE:
            self.grad_state(x, g)
        elif i==PARAMETER:
            self.grad_param(x, g)
        else:
            raise i
                
    def grad_state(self,x,g):
        """Evaluate the gradient with respect to the state.
        Only the state u and (possibly) the parameter m are accessed. """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        g.set(0.0)
        form = ufl.derivative(self.qoi_varf(u, m), u)
        _assemble_vec(form, g)
        self._zero_state_bc_entries(g)
        
    def grad_param(self,x,g):
        """Evaluate the gradient with respect to the state.
        Only the state u and (possibly) the parameter m are accessed. """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        g.set(0.0)
        form = ufl.derivative(self.qoi_varf(u, m), m)
        _assemble_vec(form, g)
                
    def apply_ij(self,i,j, dir, out):
        r"""Apply the second variation \delta_ij (i,j = STATE,PARAMETER) of the cost in direction dir."""
        if not self.use_hessian:
            out.set(0.0)
            return
        form = self.L.get((i, j))
        if form is None:
            raise RuntimeError(f"VariationalQoi missing form for ({i}, {j})")
        if len(form.arguments()) < 2:
            out.set(0.0)
            return
        dir_fun = vector2Function(dir, self.Vh[j])
        try:
            action_form = ufl.action(form, dir_fun)
        except Exception:
            out.set(0.0)
            return
        _assemble_vec(action_form, out)
        if i == STATE:
            self._zero_state_bc_entries(out)


    def setLinearizationPoint(self, x):
        """Precompute second-derivative forms at the linearization point."""
        if not self.use_hessian:
            return
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        x = [u,m]
        for i in range(2):
            di_form = ufl.derivative(self.qoi_varf(*x), x[i])
            for j in range(2):
                dij_form = ufl.derivative(di_form, x[j])
                self.L[i, j] = dij_form

    def _zero_state_bc_entries(self, vec):
        """Zero entries on state Dirichlet dofs to reflect fixed boundary values."""
        if not self._state_bcs:
            return
        if self._state_bc_dofs is None:
            dofs = []
            for bc in self._state_bcs:
                bc_dofs = None
                try:
                    bc_dofs = bc.dof_indices()
                    if isinstance(bc_dofs, tuple):
                        bc_dofs = bc_dofs[0]
                except Exception:
                    pass
                if bc_dofs is None:
                    bc_dofs = getattr(bc, "dofs", None)
                if bc_dofs is None:
                    bc_dofs = getattr(bc, "_dofs", None)
                if bc_dofs is None:
                    continue
                dofs.append(np.asarray(bc_dofs, dtype=np.int32))
            if dofs:
                self._state_bc_dofs = np.unique(np.concatenate(dofs))
            else:
                self._state_bc_dofs = np.array([], dtype=np.int32)
        if self._state_bc_dofs.size == 0:
            return
        if hasattr(vec, "array"):
            vec.array[self._state_bc_dofs] = 0.0
        elif hasattr(vec, "getArray"):
            arr = vec.getArray()
            arr[self._state_bc_dofs] = 0.0
        try:
            vec.assemblyBegin()
            vec.assemblyEnd()
        except Exception:
            pass
