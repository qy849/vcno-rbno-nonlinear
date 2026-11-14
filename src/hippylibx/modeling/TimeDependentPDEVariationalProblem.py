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

"""Time-dependent variational PDE problems with one-step schemes."""

import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.fem import petsc as fem_petsc

from .PDEProblem import PDEProblem
from .variables import STATE, PARAMETER, ADJOINT
from ..algorithms.linSolvers import PETScLUSolver
from ..utils.vector2function import vector2Function, assign_vec_to_function
from .timeDependentVector import TimeDependentVector
from ..utils.deprecate import deprecated


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
    tmp.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
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




def _create_vec_from_space(V):
    """Create a PETSc Vec compatible with the given function space."""
    imap = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    local = imap.size_local * bs
    global_size = imap.size_global * bs
    return PETSc.Vec().createMPI((local, global_size), comm=V.mesh.comm)


@deprecated(
    "ImplicitEulerTimeDependentPDEVariationalProblem",
    "3.3.0",
    "Please use TimeDependentPDEVariationalProblem instead unless you specifically need this class.",
)
class ImplicitEulerTimeDependentPDEVariationalProblem(PDEProblem):
    """Compatibility wrapper for a time-dependent PDE interface."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._impl = TimeDependentPDEVariationalProblem(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._impl, name)


class TimeDependentPDEVariationalProblem(PDEProblem):
    def __init__(self, Vh, varf_handler, bc, bc0, u0, t_init, t_final, is_fwd_linear=False):
        """
        Time-dependent PDE problem class for generic one-step methods.
        varf_handler defines r(u_n, u_{n-1}, m, v) = 0.
        """
        self.Vh = Vh
        self.varf = varf_handler
        self.fwd_bc = _as_bcs(bc)
        self.adj_bc = _as_bcs(bc0)
        self.mesh = Vh[STATE].mesh
        self.init_cond = u0
        self.t_init = t_init
        self.t_final = t_final
        self.dt = varf_handler.dt
        self.times = np.arange(self.t_init, self.t_final + 0.5 * self.dt, self.dt)

        self.linearize_x = None
        self.gauss_newton_approx = False
        self.solverA = None
        self.solverAadj = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None
        self.is_fwd_linear = is_fwd_linear
        self._bc0_dofs = _collect_bc_dofs(self.adj_bc)

    def generate_vector(self, component="ALL"):
        if component == "ALL":
            u = TimeDependentVector(self.times, mpi_comm=self.mesh.comm)
            u.initialize(self.Vh[STATE])
            a = _create_vec_from_space(self.Vh[PARAMETER])
            p = TimeDependentVector(self.times, mpi_comm=self.mesh.comm)
            p.initialize(self.Vh[ADJOINT])
            return [u, a, p]
        if component == STATE:
            u = TimeDependentVector(self.times, mpi_comm=self.mesh.comm)
            u.initialize(self.Vh[STATE])
            return u
        if component == PARAMETER:
            return _create_vec_from_space(self.Vh[PARAMETER])
        if component == ADJOINT:
            p = TimeDependentVector(self.times, mpi_comm=self.mesh.comm)
            p.initialize(self.Vh[ADJOINT])
            return p
        raise ValueError("Unknown component")

    def generate_state(self):
        return self.generate_vector(component=STATE)

    def generate_parameter(self):
        return self.generate_vector(component=PARAMETER)

    def generate_adjoint(self):
        return self.generate_vector(component=ADJOINT)

    def generate_static_state(self):
        return _create_vec_from_space(self.Vh[STATE])

    def generate_static_adjoint(self):
        return _create_vec_from_space(self.Vh[ADJOINT])

    def init_parameter(self, m):
        dummy = self.generate_parameter()
        if m.getSize() == 0:
            local, global_size = dummy.getSizes()
            m.setSizes((local, global_size))
            m.setUp()

    def _set_time(self, bcs, t):
        for bc in bcs:
            try:
                bc.value.t = t
            except Exception:
                pass

    def solveFwd(self, out, x):
        out.zero()
        if self.solverA is None:
            self.solverA = self._createLUSolver()

        u_old = fem.Function(self.Vh[STATE])
        u_old.x.array[:] = self.init_cond.x.array
        u_old.x.scatter_forward()
        out.store(u_old.x.petsc_vec, self.t_init)

        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])

        if not self.is_fwd_linear:
            u = fem.Function(self.Vh[STATE])
            v = ufl.TestFunction(self.Vh[ADJOINT])
            du = ufl.TrialFunction(self.Vh[STATE])
            for t in self.times[1:]:
                self._set_time(self.fwd_bc, t)
                u.x.array[:] = u_old.x.array
                u.x.scatter_forward()
                form = self.varf(u, u_old, m, v, t)
                jac_form = ufl.derivative(form, u, du)
                problem = fem_petsc.NonlinearProblem(
                    form,
                    u,
                    bcs=self.fwd_bc,
                    J=jac_form,
                    petsc_options_prefix="td_fwd_",
                    petsc_options={
                        "snes_type": "newtonls",
                        "snes_rtol": 1e-8,
                        "snes_atol": 1e-10,
                        "snes_max_it": 50,
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                    },
                )
                problem.solve()
                out.store(u.x.petsc_vec, t)
                u_old.x.array[:] = u.x.array
                u_old.x.scatter_forward()
            return

        du = ufl.TrialFunction(self.Vh[STATE])
        dp = ufl.TestFunction(self.Vh[ADJOINT])
        u_vec = fem.Function(self.Vh[STATE])
        A = None
        b = None

        for t in self.times[1:]:
            self._set_time(self.fwd_bc, t)
            A_form = ufl.lhs(self.varf(du, u_old, m, dp, t))
            b_form = ufl.rhs(self.varf(du, u_old, m, dp, t))
            A, b = _assemble_system(A_form, b_form, self.fwd_bc)
            self.solverA.set_operator(A)
            self.solverA.solve(u_vec.x.petsc_vec, b)
            out.store(u_vec.x.petsc_vec, t)
            u_old.x.array[:] = u_vec.x.array
            u_old.x.scatter_forward()

    def solveAdj(self, out, x, adj_rhs):
        out.zero()
        if self.solverAadj is None:
            self.solverAadj = self._createLUSolver()

        u = fem.Function(self.Vh[STATE])
        u_old = fem.Function(self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = fem.Function(self.Vh[ADJOINT])
        p_old = fem.Function(self.Vh[ADJOINT])

        du = ufl.TestFunction(self.Vh[STATE])
        dp = ufl.TrialFunction(self.Vh[ADJOINT])

        p_vec = fem.Function(self.Vh[ADJOINT])
        rhs_t = self.generate_static_state()

        for k in reversed(range(1, len(self.times))):
            t = self.times[k]
            t_old = self.times[k - 1]
            adj_rhs.retrieve(rhs_t, t)
            if self.adj_bc:
                fem_petsc.set_bc(rhs_t, self.adj_bc)
            assign_vec_to_function(u, x[STATE].view(t))
            assign_vec_to_function(u_old, x[STATE].view(t_old))

            form = self.varf(u, u_old, m, p, t)
            adj_form = ufl.derivative(ufl.derivative(form, u, du), p, dp)
            b_form = -ufl.derivative(ufl.derivative(form, u_old, du), p, p_old)
            A_adj = _assemble_matrix(adj_form, self.adj_bc)
            b = self.generate_static_state()
            _assemble_vector_into(b, fem.form(b_form), bcs=self.adj_bc)
            b.axpy(1.0, rhs_t)
            self.solverAadj.set_operator(A_adj)
            self.solverAadj.solve(p_vec.x.petsc_vec, b)
            out.store(p_vec.x.petsc_vec, t)
            p_old.x.array[:] = p_vec.x.array
            p_old.x.scatter_forward()

    def evalGradientParameter(self, x, out):
        out.set(0.0)
        out_t = out.copy()
        dm = ufl.TestFunction(self.Vh[PARAMETER])
        u = fem.Function(self.Vh[STATE])
        u_old = fem.Function(self.Vh[STATE])
        p = fem.Function(self.Vh[ADJOINT])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])

        assign_vec_to_function(u_old, x[STATE].view(self.times[0]))
        for t in self.times[1:]:
            assign_vec_to_function(u, x[STATE].view(t))
            assign_vec_to_function(p, x[ADJOINT].view(t))
            form = self.varf(u, u_old, m, p, t)
            out_t.set(0.0)
            _assemble_vector_into(out_t, fem.form(ufl.derivative(form, m, dm)))
            out.axpy(1.0, out_t)
            u_old.x.array[:] = u.x.array
            u_old.x.scatter_forward()

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        self.linearize_x = x
        self.gauss_newton_approx = gauss_newton_approx
        if self.solver_fwd_inc is None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()

    def _solveIncrementalFwd(self, out, rhs):
        out.zero()
        u = fem.Function(self.Vh[STATE])
        u_old = fem.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        dp = ufl.TestFunction(self.Vh[ADJOINT])
        du = ufl.TrialFunction(self.Vh[STATE])

        uhat = fem.Function(self.Vh[STATE])
        uhat_old = fem.Function(self.Vh[STATE])

        assign_vec_to_function(u_old, self.linearize_x[STATE].view(self.times[0]))
        for t in self.times[1:]:
            assign_vec_to_function(u, self.linearize_x[STATE].view(t))
            form = self.varf(u, u_old, m, dp, t)
            Ainc_form = ufl.derivative(form, u, du)
            binc_form = -ufl.derivative(form, u_old, uhat_old)
            Ainc = _assemble_matrix(Ainc_form, self.adj_bc)
            binc = self.generate_static_state()
            _assemble_vector_into(binc, fem.form(binc_form), bcs=self.adj_bc)
            binc.axpy(1.0, rhs.view(t))
            self.solver_fwd_inc.set_operator(Ainc)
            self.solver_fwd_inc.solve(uhat.x.petsc_vec, binc)
            out.store(uhat.x.petsc_vec, t)
            uhat_old.x.array[:] = uhat.x.array
            uhat_old.x.scatter_forward()
            u_old.x.array[:] = u.x.array
            u_old.x.scatter_forward()

    def _solveIncrementalAdj(self, out, rhs):
        out.zero()
        u = fem.Function(self.Vh[STATE])
        u_old = fem.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = fem.Function(self.Vh[ADJOINT])
        p_old = fem.Function(self.Vh[ADJOINT])

        dp = ufl.TrialFunction(self.Vh[ADJOINT])
        du = ufl.TestFunction(self.Vh[STATE])
        du_old = ufl.TestFunction(self.Vh[STATE])

        phat = fem.Function(self.Vh[ADJOINT])
        phat_old = fem.Function(self.Vh[ADJOINT])
        rhs_t = self.generate_static_state()

        for k in reversed(range(1, len(self.times))):
            t = self.times[k]
            t_old = self.times[k - 1]
            assign_vec_to_function(u, self.linearize_x[STATE].view(t))
            assign_vec_to_function(u_old, self.linearize_x[STATE].view(t_old))
            assign_vec_to_function(p, self.linearize_x[ADJOINT].view(t))
            rhs.retrieve(rhs_t, t)
            if self.adj_bc:
                fem_petsc.set_bc(rhs_t, self.adj_bc)

            form = self.varf(u, u_old, m, p, t)
            A_adj_form = ufl.derivative(ufl.derivative(form, u, du), p, dp)
            b_adj_form = -ufl.derivative(ufl.derivative(form, u_old, du_old), p, phat_old)
            A_adj = _assemble_matrix(A_adj_form, self.adj_bc)
            b_adj = self.generate_static_state()
            _assemble_vector_into(b_adj, fem.form(b_adj_form), bcs=self.adj_bc)
            b_adj.axpy(1.0, rhs_t)
            self.solver_adj_inc.set_operator(A_adj)
            self.solver_adj_inc.solve(phat.x.petsc_vec, b_adj)
            out.store(phat.x.petsc_vec, t)
            phat_old.x.array[:] = phat.x.array
            phat_old.x.scatter_forward()
            p_old.x.array[:] = p.x.array
            p_old.x.scatter_forward()

    def solveIncremental(self, out, rhs, is_adj):
        if is_adj:
            return self._solveIncrementalAdj(out, rhs)
        return self._solveIncrementalFwd(out, rhs)

    def applyC(self, dm, out):
        out.zero()
        out_t = self.generate_static_adjoint()
        u = fem.Function(self.Vh[STATE])
        u_old = fem.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = fem.Function(self.Vh[ADJOINT])
        dp = ufl.TestFunction(self.Vh[ADJOINT])
        dm_fun = vector2Function(dm, self.Vh[PARAMETER])

        assign_vec_to_function(u_old, self.linearize_x[STATE].view(self.times[0]))
        for t in self.times[1:]:
            assign_vec_to_function(u, self.linearize_x[STATE].view(t))
            assign_vec_to_function(p, self.linearize_x[ADJOINT].view(t))
            form = self.varf(u, u_old, m, p, t)
            cvarf = ufl.derivative(ufl.derivative(form, p, dp), m, dm_fun)
            out_t.set(0.0)
            _assemble_vector_into(out_t, fem.form(cvarf), bcs=self.adj_bc)
            out.store(out_t, t)
            u_old.x.array[:] = u.x.array
            u_old.x.scatter_forward()

    def applyCt(self, dp, out):
        out.set(0.0)
        out_t = self.generate_parameter()
        u = fem.Function(self.Vh[STATE])
        u_old = fem.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = fem.Function(self.Vh[ADJOINT])
        dm = ufl.TestFunction(self.Vh[PARAMETER])
        dp_fun = fem.Function(self.Vh[ADJOINT])

        assign_vec_to_function(u_old, self.linearize_x[STATE].view(self.times[0]))
        for t in self.times[1:]:
            assign_vec_to_function(u, self.linearize_x[STATE].view(t))
            assign_vec_to_function(p, self.linearize_x[ADJOINT].view(t))
            assign_vec_to_function(dp_fun, dp.view(t))
            form = self.varf(u, u_old, m, p, t)
            cvarf_adj = ufl.derivative(ufl.derivative(form, p, dp_fun), m, dm)
            out_t.set(0.0)
            _assemble_vector_into(out_t, fem.form(cvarf_adj))
            out.axpy(1.0, out_t)
            u_old.x.array[:] = u.x.array
            u_old.x.scatter_forward()

    def applyWuu(self, du, out):
        out.zero()
        if self.gauss_newton_approx:
            return
        u = fem.Function(self.Vh[STATE])
        u_old = fem.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = fem.Function(self.Vh[ADJOINT])
        du_fun = fem.Function(self.Vh[STATE])
        du_old = fem.Function(self.Vh[STATE])
        du_test = ufl.TestFunction(self.Vh[STATE])
        du_old_test = ufl.TestFunction(self.Vh[STATE])

        assign_vec_to_function(u_old, self.linearize_x[STATE].view(self.times[0]))
        assign_vec_to_function(du_old, du.view(self.times[0]))
        for t in self.times[1:]:
            assign_vec_to_function(u, self.linearize_x[STATE].view(t))
            assign_vec_to_function(p, self.linearize_x[ADJOINT].view(t))
            assign_vec_to_function(du_fun, du.view(t))
            form = self.varf(u, u_old, m, p, t)
            varf = ufl.derivative(ufl.derivative(form, u, du_fun), u, du_test)
            varf += ufl.derivative(ufl.derivative(form, u_old, du_old), u_old, du_old_test)
            out_t = self.generate_static_state()
            _assemble_vector_into(out_t, fem.form(varf), bcs=self.adj_bc)
            out.store(out_t, t)
            assign_vec_to_function(u_old, self.linearize_x[STATE].view(t))
            assign_vec_to_function(du_old, du.view(t))

    def applyWum(self, dm, out):
        out.zero()
        if self.gauss_newton_approx:
            return
        u = fem.Function(self.Vh[STATE])
        u_old = fem.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = fem.Function(self.Vh[ADJOINT])
        dm_fun = vector2Function(dm, self.Vh[PARAMETER])
        du_test = ufl.TestFunction(self.Vh[STATE])
        du_old_test = ufl.TestFunction(self.Vh[STATE])

        assign_vec_to_function(u_old, self.linearize_x[STATE].view(self.times[0]))
        for t in self.times[1:]:
            assign_vec_to_function(u, self.linearize_x[STATE].view(t))
            assign_vec_to_function(p, self.linearize_x[ADJOINT].view(t))
            form = self.varf(u, u_old, m, p, t)
            varf = ufl.derivative(ufl.derivative(form, m, dm_fun), u, du_test)
            varf += ufl.derivative(ufl.derivative(form, m, dm_fun), u_old, du_old_test)
            out_t = self.generate_static_adjoint()
            _assemble_vector_into(out_t, fem.form(varf), bcs=self.adj_bc)
            out.store(out_t, t)
            assign_vec_to_function(u_old, self.linearize_x[STATE].view(t))

    def applyWmu(self, du, out):
        out.set(0.0)
        if self.gauss_newton_approx:
            return
        u = fem.Function(self.Vh[STATE])
        u_old = fem.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = fem.Function(self.Vh[ADJOINT])
        du_fun = fem.Function(self.Vh[STATE])
        du_old = fem.Function(self.Vh[STATE])
        dm_test = ufl.TestFunction(self.Vh[PARAMETER])

        assign_vec_to_function(u_old, self.linearize_x[STATE].view(self.times[0]))
        assign_vec_to_function(du_old, du.view(self.times[0]))
        for t in self.times[1:]:
            assign_vec_to_function(u, self.linearize_x[STATE].view(t))
            assign_vec_to_function(p, self.linearize_x[ADJOINT].view(t))
            assign_vec_to_function(du_fun, du.view(t))
            form = self.varf(u, u_old, m, p, t)
            varf = ufl.derivative(ufl.derivative(form, u, du_fun), m, dm_test)
            varf += ufl.derivative(ufl.derivative(form, u_old, du_old), m, dm_test)
            out_t = self.generate_parameter()
            _assemble_vector_into(out_t, fem.form(varf))
            out.axpy(1.0, out_t)
            assign_vec_to_function(u_old, self.linearize_x[STATE].view(t))
            assign_vec_to_function(du_old, du.view(t))

    def applyWmm(self, dm, out):
        out.set(0.0)
        if self.gauss_newton_approx:
            return
        out_t = self.generate_parameter()
        u = fem.Function(self.Vh[STATE])
        u_old = fem.Function(self.Vh[STATE])
        m = vector2Function(self.linearize_x[PARAMETER], self.Vh[PARAMETER])
        p = fem.Function(self.Vh[ADJOINT])
        dm_fun = vector2Function(dm, self.Vh[PARAMETER])
        dm_test = ufl.TestFunction(self.Vh[PARAMETER])

        assign_vec_to_function(u_old, self.linearize_x[STATE].view(self.times[0]))
        for t in self.times[1:]:
            assign_vec_to_function(u, self.linearize_x[STATE].view(t))
            assign_vec_to_function(p, self.linearize_x[ADJOINT].view(t))
            form = self.varf(u, u_old, m, p, t)
            varf = ufl.derivative(ufl.derivative(form, m, dm_fun), m, dm_test)
            out_t.set(0.0)
            _assemble_vector_into(out_t, fem.form(varf))
            out.axpy(1.0, out_t)
            assign_vec_to_function(u_old, self.linearize_x[STATE].view(t))

    def apply_ij(self, i, j, dir, out):
        KKT = {
            (STATE, STATE): self.applyWuu,
            (PARAMETER, STATE): self.applyWmu,
            (STATE, PARAMETER): self.applyWum,
            (PARAMETER, PARAMETER): self.applyWmm,
            (ADJOINT, PARAMETER): self.applyC,
            (PARAMETER, ADJOINT): self.applyCt,
        }
        op = KKT.get((i, j))
        if op is None:
            if hasattr(out, "set"):
                out.set(0.0)
            else:
                out.zero()
            return
        op(dir, out)

    def zero_bc_vec(self, vec):
        if not self.adj_bc:
            return
        if isinstance(vec, TimeDependentVector):
            for snapshot in vec.data:
                fem_petsc.set_bc(snapshot, self.adj_bc)
                snapshot.assemblyBegin()
                snapshot.assemblyEnd()
            return
        fem_petsc.set_bc(vec, self.adj_bc)
        vec.assemblyBegin()
        vec.assemblyEnd()

    def exportState(self, u, fname):
        ufun = fem.Function(self.Vh[STATE])
        mesh_degree = self.mesh.geometry.cmap.degree
        Vout = fem.functionspace(self.mesh, ("Lagrange", mesh_degree))
        fout = fem.Function(Vout, name="state")
        with io.XDMFFile(self.mesh.comm, fname, "w") as fid:
            fid.write_mesh(self.mesh)
            assign_vec_to_function(ufun, self.init_cond.x.petsc_vec)
            fout.interpolate(ufun)
            fid.write_function(fout, self.times[0])
            for t in self.times[1:]:
                assign_vec_to_function(ufun, u.view(t))
                fout.interpolate(ufun)
                fid.write_function(fout, t)

    def _createLUSolver(self):
        return PETScLUSolver(self.mesh.comm)
