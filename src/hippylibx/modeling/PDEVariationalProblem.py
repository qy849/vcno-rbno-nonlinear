# Copyright (c) 2025-, Georgia Institute of Technology.
# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2022, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
# Copyright (c) 2023-2024, The University of Texas at Austin 
# & University of California--Merced.
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

"""Variational PDE problem interface for stationary models."""

import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from .PDEProblem import PDEProblem
from .variables import STATE, PARAMETER, ADJOINT
from ..algorithms.linSolvers import PETScLUSolver
from ..utils.vector2function import vector2Function
from ..utils.vector2function import assign_vec_to_function


def _as_bcs(bcs):
    """Normalize a boundary condition argument to a list."""
    if bcs is None:
        return []
    if isinstance(bcs, fem.DirichletBC):
        return [bcs]
    return list(bcs)


def _assemble_system(a_form, L_form, bcs, kind=None):
    """Assemble a linear system and apply Dirichlet lifting if present."""
    a = fem.form(a_form)
    L = fem.form(L_form)
    A = fem_petsc.assemble_matrix(a, bcs, kind=kind)
    A.assemble()

    b = fem_petsc.assemble_vector(L, kind=kind)
    if bcs:
        fem_petsc.apply_lifting(b, [a], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b, bcs)
    else:
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return A, b


def _assemble_matrix(a_form, bcs, kind=None):
    """Assemble a bilinear form into a PETSc matrix."""
    a = fem.form(a_form)
    A = fem_petsc.assemble_matrix(a, bcs, kind=kind)
    A.assemble()
    return A


def _assemble_vector_into(out, form, bcs=None, kind=None):
    """Assemble a form into an existing vector with optional BC handling."""
    tmp = fem_petsc.assemble_vector(form, kind=kind)
    try:
        tmp.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    except PETSc.Error:
        pass
    if bcs:
        fem_petsc.set_bc(tmp, bcs)

    out.set(0.0)
    out.axpy(1.0, tmp)
    out.assemblyBegin()
    out.assemblyEnd()


def _collect_bc_dofs(bcs):
    """Collect unique dof indices from a list of DirichletBC objects."""
    if not bcs:
        return None
    dofs = []
    for bc in bcs:
        try:
            dof_info = bc.dof_indices()
        except AttributeError:
            continue
        if isinstance(dof_info, tuple):
            dof_array = dof_info[0]
        else:
            dof_array = dof_info
        if dof_array is not None and len(dof_array) > 0:
            dofs.append(np.asarray(dof_array, dtype=np.int64))
    if not dofs:
        return None
    return np.unique(np.concatenate(dofs))

class PDEVariationalProblem(PDEProblem):
    def __init__(self, Vh, varf_handler, bc, bc0, is_fwd_linear = False):
        self.Vh = Vh
        self.varf_handler = varf_handler
        self.bc = _as_bcs(bc)
        self.bc0 = _as_bcs(bc0)
        
        self.A  = None
        self.At = None
        self.C = None
        self.Wmu = None
        self.Wmm = None
        self.Wuu = None
        
        self.solver = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None
        
        self.is_fwd_linear = is_fwd_linear
        self._A_fwd = None
        self._fwd_linear_problem = None
        self._m_coeff = None
        self._u_sol = None
        self._x_fun = None
        self._C_form = None
        self._Ct_form = None
        self._dm_fun = None
        self._dp_fun = None
        self._du_fun = None
        self._Wuu_form = None
        self._Wmu_form = None
        self._Wum_form = None
        self._Wmm_form = None
        self._bc0_dofs = _collect_bc_dofs(self.bc0)
        self._dir_state_bc = None
        self.n_calls = {"forward": 0,
                        "adjoint":0 ,
                        "incremental_forward":0,
                        "incremental_adjoint":0}
    def generate_state(self):
        """ Return a vector in the shape of the state. """
        V = self.Vh[STATE]
        imap = V.dofmap.index_map
        bs = V.dofmap.index_map_bs
        local = imap.size_local * bs
        global_size = imap.size_global * bs
        return PETSc.Vec().createMPI((local, global_size), comm=V.mesh.comm)
    
    def generate_parameter(self):
        """ Return a vector in the shape of the parameter. """
        V = self.Vh[PARAMETER]
        imap = V.dofmap.index_map
        bs = V.dofmap.index_map_bs
        local = imap.size_local * bs
        global_size = imap.size_global * bs
        return PETSc.Vec().createMPI((local, global_size), comm=V.mesh.comm)
    
    def init_parameter(self, m):
        """ Initialize the parameter. """
        dummy = self.generate_parameter()
        if m.getSize() == 0:
            local, global_size = dummy.getSizes()
            m.setSizes((local, global_size))
            m.setUp()
    
    def solveFwd(self, state, x):
        """ Solve the possibly nonlinear forward problem:
        Given :math:`m`, find :math:`u` such that
        
            .. math:: \\delta_p F(u, m, p;\\hat{p}) = 0,\\quad \\forall \\hat{p}."""
        self.n_calls["forward"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()
        if self.is_fwd_linear:
            if self._fwd_linear_problem is None:
                self._m_coeff = fem.Function(self.Vh[PARAMETER], name="parameter_coeff")
                self._u_sol = fem.Function(self.Vh[STATE], name="state_sol")

                u = ufl.TrialFunction(self.Vh[STATE])
                p = ufl.TestFunction(self.Vh[ADJOINT])
                res_form = self.varf_handler(u, self._m_coeff, p)
                A_form = ufl.lhs(res_form)
                b_form = ufl.rhs(res_form)
                if len(b_form.integrals()) == 0:
                    b_form = ufl.ZeroBaseForm((p,))

                self._fwd_linear_problem = fem_petsc.LinearProblem(
                    A_form,
                    b_form,
                    bcs=self.bc,
                    u=self._u_sol,
                    petsc_options_prefix="hippylibx_fwd",
                    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                )
                self._A_fwd = self._fwd_linear_problem.A

            # Update coefficient from current parameter vector.
            m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            self._m_coeff.x.array[:] = m.x.array
            self._m_coeff.x.scatter_forward()

            self._fwd_linear_problem.solve()
            owned = state.getLocalSize()
            state.array[:] = self._u_sol.x.array[:owned]
            state.assemblyBegin()
            state.assemblyEnd()
        else:
            u = vector2Function(x[STATE], self.Vh[STATE])
            m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = ufl.TestFunction(self.Vh[ADJOINT])
            res_form = self.varf_handler(u, m, p)
            nl_problem = fem_petsc.NonlinearProblem(
                res_form, u, bcs=self.bc, petsc_options_prefix="hippylibx_fwd"
            )
            nl_problem.solve()
            state.set(0.0)
            state.axpy(1.0, u.x.petsc_vec)
        
    def solveAdj(self, adj, x, adj_rhs):
        """ Solve the linear adjoint problem: 
            Given :math:`m, u`; find :math:`p` such that
            
                .. math:: \\delta_u F(u, m, p;\\hat{u}) = 0, \\quad \\forall \\hat{u}.
        """
        self.n_calls["adjoint"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()

        if self.is_fwd_linear and self._fwd_linear_problem is not None:
            tmp_rhs = self._fwd_linear_problem.A.createVecRight()
            tmp_adj = self._fwd_linear_problem.A.createVecRight()
            owned = adj_rhs.getLocalSize()
            tmp_rhs.array[:owned] = adj_rhs.array[:owned]
            tmp_rhs.assemblyBegin()
            tmp_rhs.assemblyEnd()
            if self.bc0:
                fem_petsc.set_bc(tmp_rhs, self.bc0)
            # For linear self-adjoint PDEs, the adjoint operator matches the
            # operator already set in the most recent forward solve.
            self._fwd_linear_problem.solver.solve(tmp_rhs, tmp_adj)
            adj.array[:] = tmp_adj.array[:owned]
            adj.assemblyBegin()
            adj.assemblyEnd()
            return
            
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = fem.Function(self.Vh[ADJOINT])
        du = ufl.TestFunction(self.Vh[STATE])
        dp = ufl.TrialFunction(self.Vh[ADJOINT])
        varf = self.varf_handler(u, m, p)
        adj_form = ufl.derivative(ufl.derivative(varf, u, du), p, dp)
        Aadj = _assemble_matrix(adj_form, self.bc0)
        if self.bc0:
            fem_petsc.set_bc(adj_rhs, self.bc0)
        self.solver.set_operator(Aadj)
        self.solver.solve(adj, adj_rhs)
     
    def evalGradientParameter(self, x, out):
        """Given :math:`u, m, p`; evaluate :math:`\\delta_m F(u, m, p; \\hat{m}),\\, \\forall \\hat{m}.` """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        dm = ufl.TestFunction(self.Vh[PARAMETER])
        res_form = self.varf_handler(u, m, p)
        form = fem.form(ufl.derivative(res_form, m, dm))
        _assemble_vector_into(out, form)
         
    def setLinearizationPoint(self,x, gauss_newton_approx):
        """ Set the values of the state and parameter
            for the incremental forward and adjoint solvers. """
        if self._x_fun is None:
            self._x_fun = [fem.Function(self.Vh[i]) for i in range(3)]
        x_fun = self._x_fun
        for i in range(3):
            assign_vec_to_function(x_fun[i], x[i])
        
        f_form = self.varf_handler(*x_fun)
        
        g_form = [None,None,None]
        for i in range(3):
            g_form[i] = ufl.derivative(f_form, x_fun[i])
        if self.is_fwd_linear and self._fwd_linear_problem is not None:
            self.A = self._fwd_linear_problem.A
            self.At = self._fwd_linear_problem.A
        else:
            self.A = _assemble_matrix(ufl.derivative(g_form[ADJOINT], x_fun[STATE]), self.bc0)
            self.At = _assemble_matrix(ufl.derivative(g_form[STATE], x_fun[ADJOINT]), self.bc0)
        dm_trial = ufl.TrialFunction(self.Vh[PARAMETER])
        du_trial = ufl.TrialFunction(self.Vh[STATE])
        self._C_form = ufl.derivative(g_form[ADJOINT], x_fun[PARAMETER], dm_trial)
        self._Ct_form = ufl.adjoint(self._C_form)
        # Avoid assembling C (rectangular) to sidestep dolfinx create_matrix crashes.
        self.C = None
                
        if self.solver_fwd_inc is None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()
        
        self.solver_fwd_inc.set_operator(self.A)
        self.solver_adj_inc.set_operator(self.At)

        if gauss_newton_approx:
            self.Wuu = None
            self.Wmu = None
            self.Wmm = None
            self._Wuu_form = None
            self._Wmu_form = None
            self._Wum_form = None
            self._Wmm_form = None
        else:
            # Use matrix-free forms for Hessian blocks to avoid sparsity creation crashes.
            self.Wuu = None
            self.Wmu = None
            self.Wmm = None
            self._Wuu_form = ufl.derivative(g_form[STATE], x_fun[STATE], du_trial)
            self._Wmu_form = ufl.derivative(g_form[PARAMETER], x_fun[STATE], du_trial)
            self._Wum_form = ufl.derivative(g_form[STATE], x_fun[PARAMETER], dm_trial)
            self._Wmm_form = ufl.derivative(g_form[PARAMETER], x_fun[PARAMETER], dm_trial)
        
    def solveIncremental(self, out, rhs, is_adj):
        """ If :code:`is_adj == False`:

            Solve the forward incremental system:
            Given :math:`u, m`, find :math:`\\tilde{u}` such that
            
                .. math:: \\delta_{pu} F(u, m, p; \\hat{p}, \\tilde{u}) = \\mbox{rhs},\\quad \\forall \\hat{p}.
            
            If :code:`is_adj == True`:

            Solve the adjoint incremental system:
            Given :math:`u, m`, find :math:`\\tilde{p}` such that
            
                .. math:: \\delta_{up} F(u, m, p; \\hat{u}, \\tilde{p}) = \\mbox{rhs},\\quad \\forall \\hat{u}.
        """
        if self.bc0:
            fem_petsc.set_bc(rhs, self.bc0)
        if is_adj:
            self.n_calls["incremental_adjoint"] += 1
            self.solver_adj_inc.solve(out, rhs)
        else:
            self.n_calls["incremental_forward"] += 1
            self.solver_fwd_inc.solve(out, rhs)
    
    def apply_ij(self,i,j, dir, out):   
        """
            Given :math:`u, m, p`; compute 
            :math:`\\delta_{ij} F(u, m, p; \\hat{i}, \\tilde{j})` in the direction :math:`\\tilde{j} =` :code:`dir`,
            :math:`\\forall \\hat{i}`.
        """
        if i == STATE and j == STATE and self.Wuu is None and self._Wuu_form is not None:
            self._apply_form_state(self._Wuu_form, dir, out, bcs=self.bc0, zero_bc_dofs=True)
            return
        if i == PARAMETER and j == STATE and self.Wmu is None and self._Wmu_form is not None:
            self._apply_form_state(self._Wmu_form, dir, out, zero_bc_dofs=True)
            return
        if i == PARAMETER and j == PARAMETER and self.Wmm is None and self._Wmm_form is not None:
            self._apply_form_param(self._Wmm_form, dir, out)
            return
        if i == STATE and j == PARAMETER and self._Wum_form is not None:
            self._apply_form_param(self._Wum_form, dir, out)
            if self.bc0:
                fem_petsc.set_bc(out, self.bc0)
            return

        KKT = {}
        KKT[STATE,STATE] = self.Wuu
        KKT[PARAMETER, STATE] = self.Wmu
        KKT[PARAMETER, PARAMETER] = self.Wmm
        KKT[ADJOINT, STATE] = self.A
        KKT[ADJOINT, PARAMETER] = self.C
        
        if i >= j:
            if i == ADJOINT and j == PARAMETER and self.C is None:
                self._apply_C(dir, out)
            elif KKT[i,j] is None:
                out.set(0.0)
            else:
                KKT[i,j].mult(dir, out)
        else:
            if j == ADJOINT and i == PARAMETER and self.C is None:
                self._apply_Ct(dir, out)
            elif KKT[j,i] is None:
                out.set(0.0)
            else:
                mat = KKT[j,i]
                if hasattr(mat, "multTranspose"):
                    mat.multTranspose(dir, out)
                else:
                    mat.transpmult(dir, out)

    def zero_bc_vec(self, vec):
        if not self.bc0:
            return
        fem_petsc.set_bc(vec, self.bc0)
        vec.assemblyBegin()
        vec.assemblyEnd()

    def _apply_form_state(self, form, trial_vec, out, bcs=None, zero_bc_dofs=False, zero_bc_rows=False):
        if len(form.arguments()) == 0:
            out.set(0.0)
            return
        if self._du_fun is None:
            self._du_fun = fem.Function(self.Vh[STATE])
        if zero_bc_dofs and self.bc0:
            if self._dir_state_bc is None:
                self._dir_state_bc = trial_vec.duplicate()
            self._dir_state_bc.array[:] = trial_vec.array
            fem_petsc.set_bc(self._dir_state_bc, self.bc0)
            assign_vec_to_function(self._du_fun, self._dir_state_bc)
        else:
            assign_vec_to_function(self._du_fun, trial_vec)
        try:
            action_form = ufl.action(form, self._du_fun)
        except IndexError:
            out.set(0.0)
            return
        _assemble_vector_into(out, fem.form(action_form), bcs=bcs)
        if zero_bc_rows:
            self.zero_bc_vec(out)

    def _apply_form_param(self, form, trial_vec, out, zero_bc_rows=False):
        if len(form.arguments()) == 0:
            out.set(0.0)
            return
        if self._dm_fun is None:
            self._dm_fun = fem.Function(self.Vh[PARAMETER])
        assign_vec_to_function(self._dm_fun, trial_vec)
        try:
            action_form = ufl.action(form, self._dm_fun)
        except IndexError:
            out.set(0.0)
            return
        _assemble_vector_into(out, fem.form(action_form))
        if zero_bc_rows:
            self.zero_bc_vec(out)

    def _apply_C(self, dm_vec, out):
        if self._C_form is None:
            raise RuntimeError("C form not initialized")
        if self._dm_fun is None:
            self._dm_fun = fem.Function(self.Vh[PARAMETER])
        assign_vec_to_function(self._dm_fun, dm_vec)
        form = fem.form(ufl.action(self._C_form, self._dm_fun))
        _assemble_vector_into(out, form)
        self.zero_bc_vec(out)

    def _apply_Ct(self, dp_vec, out):
        if self._Ct_form is None:
            raise RuntimeError("Ct form not initialized")
        if self._dp_fun is None:
            self._dp_fun = fem.Function(self.Vh[ADJOINT])
        if self.bc0:
            if self._dir_state_bc is None:
                self._dir_state_bc = dp_vec.duplicate()
            self._dir_state_bc.array[:] = dp_vec.array
            fem_petsc.set_bc(self._dir_state_bc, self.bc0)
            assign_vec_to_function(self._dp_fun, self._dir_state_bc)
        else:
            assign_vec_to_function(self._dp_fun, dp_vec)
        form = fem.form(ufl.action(self._Ct_form, self._dp_fun))
        _assemble_vector_into(out, form)
                
    def apply_ijk(self,i,j,k, x, jdir, kdir, out):
        x_fun = [vector2Function(x[ii], self.Vh[ii]) for ii in range(3)]
        idir_fun = ufl.TestFunction(self.Vh[i])
        jdir_fun = vector2Function(jdir, self.Vh[j])
        kdir_fun = vector2Function(kdir, self.Vh[k])
        
        res_form = self.varf_handler(*x_fun)
        form = ufl.derivative(
            ufl.derivative(
                ufl.derivative(res_form, x_fun[i], idir_fun),
                x_fun[j],
                jdir_fun,
            ),
            x_fun[k],
            kdir_fun,
        )
        
        _assemble_vector_into(out, fem.form(form), bcs=self.bc0 if i in (STATE, ADJOINT) else None)
                   
    def _createLUSolver(self):   
        return PETScLUSolver(self.Vh[STATE].mesh.comm)
