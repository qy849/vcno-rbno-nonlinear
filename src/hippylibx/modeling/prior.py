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

"""Prior models and utilities for Bayesian inverse problems."""

import math
import numbers
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
import basix.ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc

from ..algorithms.linalg import get_diagonal, estimate_diagonal_inv2, Solver2Operator, Operator2Solver, amg_method, MatMatMult
from ..algorithms.linSolvers import PETScKrylovSolver, PETScLUSolver
from ..algorithms.traceEstimator import TraceEstimator
from ..algorithms.multivector import MultiVector
from ..algorithms.randomizedEigensolver import doublePass, doublePassG
from ..utils.random import parRandom
from ..utils.vector2function import vector2Function

from .expression import ExpressionModule

def _dense_mat(arr, comm):
    """Create a PETSc dense matrix from a local NumPy array."""
    n, m = arr.shape
    mat = PETSc.Mat().createDense([n, m], comm=comm)
    mat.setUp()
    mat.setValues(range(n), range(m), arr)
    mat.assemble()
    return mat


def _identity_mat(n, comm):
    """Create an identity PETSc matrix on the given communicator."""
    mat = PETSc.Mat().createAIJ([n, n], comm=comm)
    mat.setUp()
    diag = PETSc.Vec().createMPI(n, comm=comm)
    diag.set(1.0)
    diag.assemblyBegin()
    diag.assemblyEnd()
    mat.setDiagonal(diag)
    mat.assemble()
    return mat


def _assemble_matrix(form, kind=None):
    """Assemble a PETSc matrix from a UFL form with optional kind."""
    A = fem_petsc.assemble_matrix(fem.form(form), kind=kind)
    A.assemble()
    return A


class _RinvM:
    """
    Operator that models the action of R^{-1}M.
    """

    def __init__(self, Rsolver, M):
        self.Rsolver = Rsolver
        self.M = M
        self._tmp = None

    def init_vector(self, x, dim):
        if hasattr(self.M, "init_vector"):
            self.M.init_vector(x, dim)
        else:
            tmp = self.M.createVecLeft() if dim == 0 else self.M.createVecRight()
            x.setSizes(tmp.getSizes())
            x.setUp()

    def mult(self, x, y):
        if self._tmp is None:
            self._tmp = y.duplicate()
        self.M.mult(x, self._tmp)
        self.Rsolver.solve(y, self._tmp)


class _Prior:
    """
    Abstract class to describe the prior model.
    """

    def trace(self, method="Exact", tol=1e-1, min_iter=20, max_iter=100, r=200):
        op = _RinvM(self.Rsolver, self.M)
        if method == "Exact":
            marginal_variance = PETSc.Vec().createMPI(self.R.getSize()[0], comm=self.R.getComm())
            self.init_vector(marginal_variance, 0)
            get_diagonal(op, marginal_variance)
            return marginal_variance.sum()
        elif method == "Estimator":
            tr_estimator = TraceEstimator(op, False, tol)
            tr_exp, _ = tr_estimator(min_iter, max_iter)
            return tr_exp
        elif method == "Randomized":
            dummy = PETSc.Vec().createMPI(self.R.getSize()[0], comm=self.R.getComm())
            self.init_vector(dummy, 0)
            Omega = MultiVector(dummy, r)
            parRandom.normal(1.0, Omega)
            d, _ = doublePassG(
                Solver2Operator(self.Rsolver),
                Solver2Operator(self.Msolver),
                Operator2Solver(self.M),
                Omega,
                r,
                s=1,
                check=False,
            )
            return d.sum()
        else:
            raise NameError("Unknown method")

    def pointwise_variance(self, method, k=1000000, r=200):
        pw_var = PETSc.Vec().createMPI(self.R.getSize()[0], comm=self.R.getComm())
        self.init_vector(pw_var, 0)
        if method == "Exact":
            get_diagonal(Solver2Operator(self.Rsolver, init_vector=self.init_vector), pw_var)
        elif method == "Estimator":
            estimate_diagonal_inv2(self.Rsolver, k, pw_var)
        elif method == "Randomized":
            Omega = MultiVector(pw_var, r)
            parRandom.normal(1.0, Omega)
            d, U = doublePass(Solver2Operator(self.Rsolver), Omega, r, s=1, check=False)
            tmp = pw_var.duplicate()
            for i in np.arange(U.nvec()):
                ui = U[i]
                tmp.set(0.0)
                tmp.array[:] = ui.array * ui.array
                pw_var.axpy(float(d[i]), tmp)
        else:
            raise NameError("Unknown method")
        return pw_var

    def cost(self, m):
        d = m.copy()
        try:
            m_sizes = m.getSizes()
        except PETSc.Error:
            m_sizes = None
        try:
            mean_sizes = self.mean.getSizes()
        except PETSc.Error:
            mean_sizes = None
        if m_sizes is None or mean_sizes != m_sizes:
            self.mean = m.duplicate()
            self.mean.set(0.0)
        d.axpy(-1.0, self.mean)

        Rd = PETSc.Vec().create(comm=self.R.getComm())
        self.init_vector(Rd, 0)
        self.R.mult(d, Rd)
        return 0.5 * Rd.dot(d)

    def grad(self, m, out):
        d = m.copy()
        try:
            m_sizes = m.getSizes()
        except PETSc.Error:
            m_sizes = None
        try:
            mean_sizes = self.mean.getSizes()
        except PETSc.Error:
            mean_sizes = None
        if m_sizes is None or mean_sizes != m_sizes:
            self.mean = m.duplicate()
            self.mean.set(0.0)
        d.axpy(-1.0, self.mean)
        self.R.mult(d, out)

    def init_vector(self, x, dim):
        raise NotImplementedError("Child class should implement method init_vector")

    def sample(self, noise, s, add_mean=True):
        raise NotImplementedError("Child class should implement method sample")

    def getHessianPreconditioner(self):
        return self.Rsolver


class LaplacianPrior(_Prior):
    def __init__(
        self,
        Vh,
        gamma,
        delta,
        mean=None,
        rel_tol=1e-12,
        max_iter=100,
        solver_type="krylov",
        lu_method="default",
    ):
        assert delta != 0.0, "Intrinsic Gaussian prior is not supported"
        self.Vh = Vh

        trial = ufl.TrialFunction(Vh)
        test = ufl.TestFunction(Vh)
        varfL = ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
        varfM = ufl.inner(trial, test) * ufl.dx
        self.M = _assemble_matrix(varfM)
        self.R = _assemble_matrix(gamma * varfL + delta * varfM)

        comm = Vh.mesh.comm
        if solver_type == "krylov":
            self.Rsolver = PETScKrylovSolver(comm, "cg", amg_method())
            self.Rsolver.set_operator(self.R)
            self.Msolver = PETScKrylovSolver(comm, "cg", "jacobi")
            self.Msolver.set_operator(self.M)
        elif solver_type == "lu":
            self.Rsolver = PETScLUSolver(comm, method=lu_method)
            self.Rsolver.set_operator(self.R)
            self.Msolver = PETScLUSolver(comm, method=lu_method)
            self.Msolver.set_operator(self.M)
        else:
            raise ValueError(f"Unknown solver type {solver_type}")

        for solver in (self.Rsolver, self.Msolver):
            solver.parameters["maximum_iterations"] = max_iter
            solver.parameters["relative_tolerance"] = rel_tol
            solver.parameters["error_on_nonconvergence"] = True
            solver.parameters["nonzero_initial_guess"] = False

        # Quadrature-based sqrtR construction
        ndim = Vh.mesh.topology.dim
        qdegree = 2 * Vh.ufl_element().degree
        metadata = {"quadrature_degree": qdegree, "quadrature_scheme": "default"}
        dx_q = ufl.dx(metadata=metadata)
        
        Qe = basix.ufl.quadrature_element(Vh.mesh.topology.cell_name(), value_shape=(ndim+1,), degree=qdegree, scheme="default")
        Qh = fem.functionspace(Vh.mesh, Qe)
        
        ph = ufl.TrialFunction(Qh)
        qh = ufl.TestFunction(Qh)
        
        Mqh = _assemble_matrix(ufl.inner(ph, qh) * dx_q)
        diag_vec = Mqh.getDiagonal()
        # dMqh = 1/sqrt(weights)
        vals = diag_vec.array
        np.sqrt(vals, out=vals)
        np.reciprocal(vals, out=vals)
        Mqh.setDiagonal(diag_vec)
        
        pph = ufl.split(ph)
        sqrtdelta = math.sqrt(delta)
        sqrtgamma = math.sqrt(gamma)
        
        varfGG = sqrtdelta * pph[0] * test * dx_q
        for i in range(ndim):
            varfGG = varfGG + sqrtgamma * pph[i+1] * test.dx(i) * dx_q
            
        GG = _assemble_matrix(varfGG)
        self.sqrtR = MatMatMult(GG, Mqh)

        if mean is None:
            self.mean = self.M.createVecLeft()
            self.mean.set(0.0)
            self.mean.assemblyBegin()
            self.mean.assemblyEnd()
        else:
            self.mean = mean

    def init_vector(self, x, dim):
        if dim == "noise":
             template = self.sqrtR.createVecRight()
             try:
                 sizes = x.getSizes()
             except PETSc.Error:
                 sizes = None
             if sizes != template.getSizes():
                 try:
                     x.destroy()
                 except Exception:
                     pass
                 x.create(comm=template.getComm())
                 x.setSizes(template.getSizes())
                 x.setUp()
        else:
             ptr = self.M.createVecLeft()
             try:
                 sizes = x.getSizes()
             except PETSc.Error:
                 sizes = None
             if sizes != ptr.getSizes():
                 try:
                     x.destroy()
                 except Exception:
                     pass
                 x.create(comm=ptr.getComm())
                 x.setSizes(ptr.getSizes())
                 x.setUp()
             x.set(0.0)
             x.assemblyBegin()
             x.assemblyEnd()

    def sample(self, noise, s, add_mean=True):
        rhs = self.sqrtR.createVecLeft()
        self.sqrtR.mult(noise, rhs)
        self.Rsolver.solve(s, rhs)
        if add_mean:
            s.axpy(1.0, self.mean)


class _BilaplacianR:
    def __init__(self, A, Msolver):
        self.A = A
        self.Msolver = Msolver
        self.help1 = A.createVecRight()
        self.help2 = A.createVecRight()

    def init_vector(self, x, dim):
        temp = self.A.createVecRight()
        try:
            sizes = x.getSizes()
        except PETSc.Error:
            sizes = None
        if sizes != temp.getSizes():
            try:
                x.destroy()
            except Exception:
                pass
            x.create(comm=temp.getComm())
            x.setSizes(temp.getSizes())
            x.setUp()


    def getSize(self):
        return self.A.getSize()

    def getComm(self):
        return self.A.getComm()

    def createVecRight(self):
        return self.A.createVecRight()

    def createVecLeft(self):
        return self.A.createVecLeft()

    def mult(self, x, y):
        self.A.mult(x, self.help1)
        self.Msolver.solve(self.help2, self.help1)
        self.A.mult(self.help2, y)


class _BilaplacianRsolver:
    def __init__(self, Asolver, M):
        self.Asolver = Asolver
        self.M = M
        self.help1 = M.createVecRight()
        self.help2 = M.createVecRight()

    def init_vector(self, x, dim):
        temp = self.M.createVecRight()
        try:
            sizes = x.getSizes()
        except PETSc.Error:
            sizes = None
        if sizes != temp.getSizes():
            try:
                x.destroy()
            except Exception:
                pass
            x.create(comm=temp.getComm())
            x.setSizes(temp.getSizes())
            x.setUp()

    def createVecRight(self):
        return self.M.createVecRight()

    def createVecLeft(self):
        return self.M.createVecLeft()

    def solve(self, x, b):
        self.Asolver.solve(self.help1, b)
        self.M.mult(self.help1, self.help2)
        return self.Asolver.solve(x, self.help2)


class SqrtPrecisionPDE_Prior(_Prior):
    def __init__(
        self,
        Vh,
        sqrt_precision_varf_handler,
        mean=None,
        rel_tol=1e-12,
        max_iter=1000,
        solver_type="krylov",
        lu_method="default",
    ):
        self.Vh = Vh
        trial = ufl.TrialFunction(Vh)
        test = ufl.TestFunction(Vh)
        self.M = _assemble_matrix(ufl.inner(trial, test) * ufl.dx)
        self.A = _assemble_matrix(sqrt_precision_varf_handler(trial, test))

        comm = Vh.mesh.comm
        if solver_type == "krylov":
            self.Msolver = PETScKrylovSolver(comm, "cg", "jacobi")
            self.Msolver.set_operator(self.M)
            self.Asolver = PETScKrylovSolver(comm, "cg", amg_method())
            self.Asolver.set_operator(self.A)
        elif solver_type == "lu":
            self.Msolver = PETScLUSolver(comm, method=lu_method)
            self.Msolver.set_operator(self.M)
            self.Asolver = PETScLUSolver(comm, method=lu_method)
            self.Asolver.set_operator(self.A)
        else:
            raise ValueError(f"Unknown solver type {solver_type}")

        for solver in (self.Msolver, self.Asolver):
            solver.parameters["maximum_iterations"] = max_iter
            solver.parameters["relative_tolerance"] = rel_tol
            solver.parameters["error_on_nonconvergence"] = True
            solver.parameters["nonzero_initial_guess"] = False

        # Quadrature-based sqrtM construction
        qdegree = 2 * Vh.ufl_element().degree
        metadata = {"quadrature_degree": qdegree, "quadrature_scheme": "default"}
        dx_q = ufl.dx(metadata=metadata)
        
        num_sub_spaces = Vh.num_sub_spaces
        # For scalar field, num_sub_spaces might be 0 or 1 depending on version.
        # dolfinx functionspace.num_sub_spaces is not always direct attribute.
        # check ufl element
        ufl_elt = Vh.ufl_element()
        if ufl_elt.num_sub_elements > 0:
             # Mixed element
             val_shape = (ufl_elt.num_sub_elements,)
             dim_q = val_shape[0]
        else:
             # Scalar or Vector
             ref_shape = ufl_elt.reference_value_shape
             if len(ref_shape) == 0:
                 val_shape = ()
                 dim_q = 1
             else:
                 val_shape = ref_shape
                 dim_q = np.prod(val_shape)

        Qe = basix.ufl.quadrature_element(Vh.mesh.topology.cell_name(), value_shape=val_shape, degree=qdegree, scheme="default")
        Qh = fem.functionspace(Vh.mesh, Qe)
        
        ph = ufl.TrialFunction(Qh)
        qh = ufl.TestFunction(Qh)
        
        Mqh = _assemble_matrix(ufl.inner(ph, qh) * dx_q)
        diag_vec = Mqh.getDiagonal()
        vals = diag_vec.array
        np.sqrt(vals, out=vals)
        np.reciprocal(vals, out=vals)
        Mqh.setDiagonal(diag_vec)
        
        MixedM = _assemble_matrix(ufl.inner(ph, test) * dx_q)
        self.sqrtM = MatMatMult(MixedM, Mqh)

        self.R = _BilaplacianR(self.A, self.Msolver)
        self.Rsolver = _BilaplacianRsolver(self.Asolver, self.M)

        if mean is None:
            self.mean = self.A.createVecRight()
            self.init_vector(self.mean, 0)
        else:
            self.mean = mean

    def init_vector(self, x, dim):
        if dim == "noise":
            template = self.sqrtM.createVecRight()
            try:
                x.destroy()
            except Exception:
                pass
            x.create(comm=template.getComm())
            x.setSizes(template.getSizes())
            x.setUp()
        else:
            template = self.A.createVecRight()
            try:
                sizes = x.getSizes()
            except PETSc.Error:
                sizes = None
            if sizes == template.getSizes():
                return
            try:
                x.destroy()
            except Exception:
                pass
            x.create(comm=template.getComm())
            x.setSizes(template.getSizes())
            x.setUp()

    def sample(self, noise, s, add_mean=True):
        rhs = self.sqrtM.createVecLeft()
        self.sqrtM.mult(noise, rhs)
        self.Asolver.solve(s, rhs)
        if add_mean:
            s.axpy(1.0, self.mean)


def BiLaplacianPrior(Vh, gamma, delta, Theta=None, mean=None, rel_tol=1e-12, max_iter=1000, robin_bc=False, solver_type="krylov", lu_method="default"):
    mesh = Vh.mesh
    if isinstance(gamma, numbers.Number):
        gamma = fem.Constant(mesh, float(gamma))
    if isinstance(delta, numbers.Number):
        delta = fem.Constant(mesh, float(delta))

    def sqrt_precision_varf_handler(trial, test):
        ds = ufl.ds(domain=mesh)
        grad_form = ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
        if Theta is not None:
            grad_form = ufl.inner(Theta * ufl.grad(trial), ufl.grad(test)) * ufl.dx
        mass_form = ufl.inner(trial, test) * ufl.dx
        if robin_bc:
            robin_coeff = ufl.sqrt(gamma * delta) / fem.Constant(mesh, 1.42)
            robin_form = robin_coeff * ufl.inner(trial, test) * ds
        else:
            robin_form = 0.0
        return gamma * grad_form + delta * mass_form + robin_form

    return SqrtPrecisionPDE_Prior(
        Vh,
        sqrt_precision_varf_handler,
        mean,
        rel_tol,
        max_iter,
        solver_type=solver_type,
        lu_method=lu_method,
    )


def MollifiedBiLaplacianPrior(Vh, gamma, delta, locations, m_true, Theta = None, pen = 1e1, order=2, rel_tol=1e-12, max_iter=1000, solver_type="krylov", lu_method="default"):
    if Vh.dofmap.index_map_bs != 1:
        raise ValueError("MollifiedBiLaplacianPrior expects a scalar FunctionSpace.")
    if float(delta) == 0.0:
        raise ValueError("delta must be non-zero for mollifier scaling (gamma/delta).")
    if pen == 0 and float(delta) == 0.0:
        raise ValueError("Intrinsic Gaussian Prior are not supported.")
    locations = np.atleast_2d(np.asarray(locations, dtype=float))
    if locations.shape[1] < 2:
        raise ValueError("locations must have at least two columns (x, y).")
    
    # Create mollifier function
    mfun = fem.Function(Vh) # Or higher order? Using Vh as per hippylibx
    
    mollifier = ExpressionModule.Mollifier()
    # Support Theta as ufl tensor, AnisTensor2D, or (ufl, AnisTensor2D) pair.
    Theta_ufl = Theta
    Theta_mollifier = None
    if Theta is not None:
        if isinstance(Theta, tuple) and len(Theta) == 2:
            Theta_ufl, Theta_mollifier = Theta
        elif hasattr(Theta, "theta0") and hasattr(Theta, "theta1") and hasattr(Theta, "alpha"):
            Theta_mollifier = Theta
            sa = math.sin(Theta.alpha)
            ca = math.cos(Theta.alpha)
            c00 = Theta.theta0 * sa * sa + Theta.theta1 * ca * ca
            c01 = (Theta.theta0 - Theta.theta1) * sa * ca
            c11 = Theta.theta0 * ca * ca + Theta.theta1 * sa * sa
            Theta_ufl = ufl.as_matrix(((c00, c01), (c01, c11)))
    if Theta_mollifier is None:
        anis = ExpressionModule.AnisTensor2D()
        mollifier.set(anis, gamma / delta, order)
    else:
        mollifier.set(Theta_mollifier, gamma / delta, order)
        
    for ii in range(locations.shape[0]):
        mollifier.addLocation(locations[ii, 0], locations[ii, 1])
            
    mfun.interpolate(mollifier)
    
    def sqrt_precision_varf_handler(trial, test): 
        if Theta_ufl is None:
            varfL = ufl.inner(ufl.grad(trial), ufl.grad(test))*ufl.dx
        else:
            varfL = ufl.inner(Theta_ufl*ufl.grad(trial), ufl.grad(test))*ufl.dx
        varfM = ufl.inner(trial,test)*ufl.dx
        varfmo = ufl.inner(mfun*trial, test)*ufl.dx # mfun*inner(trial, test)
        return fem.Constant(Vh.mesh, float(gamma))*varfL + fem.Constant(Vh.mesh, float(delta))*varfM + fem.Constant(Vh.mesh, float(pen))*varfmo
    
    prior = SqrtPrecisionPDE_Prior(Vh, sqrt_precision_varf_handler, None, rel_tol, max_iter, solver_type=solver_type, lu_method=lu_method)  
    
    prior.mean = prior.A.createVecRight()
    prior.init_vector(prior.mean, 0)
    
    test  = ufl.TestFunction(Vh)
    m_true_fun = vector2Function(m_true, Vh)
    
    # Assemble rhs: pen * mfun * m_true * test
    # Need to be careful about forms
    rhs_form = fem.Constant(Vh.mesh, float(pen)) * mfun * ufl.inner(m_true_fun, test) * ufl.dx
    rhs = fem_petsc.assemble_vector(fem.form(rhs_form))
    if hasattr(rhs, "scatter_reverse"):
        try:
            rhs.scatter_reverse(PETSc.InsertMode.ADD)
        except Exception:
            pass
    else:
        try:
            rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        except PETSc.Error:
            pass
    
    # Solve A * mean = rhs
    rhs_vec = rhs.petsc_vec if hasattr(rhs, "petsc_vec") else rhs
    prior.Asolver.solve(prior.mean, rhs_vec)
    
    return prior


def BiLaplacianComputeCoefficients(sigma2, rho, ndim):
    nu = 2. - 0.5*ndim
    kappa = np.sqrt(8*nu)/rho
    
    s = np.sqrt(sigma2)*np.power(kappa,nu)*np.sqrt(np.power(4.*np.pi, 0.5*ndim)/math.gamma(nu) )
    
    gamma = 1./s
    delta = np.power(kappa,2)/s
    
    return gamma, delta


def VectorBiLaplacianPrior(Vh, gamma, delta, mean=None, rel_tol=1e-12, max_iter=1000, robin_bc=False, solver_type="krylov", lu_method="default"):
    if isinstance(gamma[0], numbers.Number):
        gamma = [fem.Constant(Vh.mesh, float(g)) for g in gamma]
        
    if isinstance(delta[0], numbers.Number):
        delta = [fem.Constant(Vh.mesh, float(d)) for d in delta]

    def comp_varf(trial, test, gamma, delta):
        ds = ufl.ds(domain=mesh)
        varfL = gamma*ufl.inner(ufl.grad(trial), ufl.grad(test))*ufl.dx
        varfM = delta*ufl.inner(trial,test)*ufl.dx
        if robin_bc:
            # Need to evaluate constants for the robin coefficient if possible or use ufl algebra
            # gamma*sqrt(delta/gamma) = sqrt(gamma*delta)
            robin_coeff = ufl.sqrt(gamma*delta)/fem.Constant(Vh.mesh, 1.42)
        else:
            robin_coeff = fem.Constant(Vh.mesh, 0.)
            
        varf_robin = robin_coeff*ufl.inner(trial,test)*ds
        return varfL + varfM + varf_robin

    def sqrt_precision_varf_handler(trial, test): 
        varf = fem.Constant(Vh.mesh, 0.)*ufl.inner(trial, test)*ufl.dx
        for trial_i, test_i, gamma_i, delta_i in zip(ufl.split(trial), ufl.split(test), gamma, delta):
             varf = varf + comp_varf(trial_i, test_i, gamma_i, delta_i)
        return varf

    return SqrtPrecisionPDE_Prior(Vh, sqrt_precision_varf_handler, mean, rel_tol, max_iter, solver_type=solver_type, lu_method=lu_method)


class GaussianRealPrior(_Prior):
    """
    Finite-dimensional Gaussian prior N(m, C) with dense covariance.
    """

    def __init__(self, Vh, covariance, mean=None):
        self.Vh = Vh
        self.comm = _space_comm(Vh)
        self.dim = covariance.shape[0]

        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("Covariance must be square")
        if Vh is not None:
            vdim = _space_dim(Vh)
            if vdim != covariance.shape[0]:
                raise ValueError("Covariance incompatible with Finite Element space")
        if not np.issubdtype(covariance.dtype, np.floating):
            raise TypeError("Covariance matrix must be a float array")

        self.covariance = covariance
        self.chol = np.linalg.cholesky(self.covariance)
        self.chol_inv = np.linalg.inv(self.chol)
        self.precision = np.dot(self.chol_inv.T, self.chol_inv)

        self.M = _identity_mat(self.dim, self.comm)
        self.Msolver = Operator2Solver(self.M)

        if mean is None:
            self.mean = PETSc.Vec().createMPI(self.dim, comm=self.comm)
            self.mean.set(0.0)
            self.mean.assemblyBegin()
            self.mean.assemblyEnd()
        else:
            self.mean = mean

        self.R = _dense_mat(self.precision, self.comm)
        self.RSolverOp = _dense_mat(self.covariance, self.comm)
        self.Rsolver = Operator2Solver(self.RSolverOp)
        self.sqrtR = _dense_mat(self.chol_inv.T, self.comm)
        self.sqrtRinv = _dense_mat(self.chol, self.comm)

    def init_vector(self, x, dim):
        try:
            size = x.getSize()
        except PETSc.Error:
            size = None
        if size != self.dim:
            try:
                x.destroy()
            except Exception:
                pass
            x.create(comm=self.comm)
            x.setSizes((PETSc.DECIDE, self.dim))
            x.setUp()
        x.set(0.0)
        x.assemblyBegin()
        x.assemblyEnd()

    def sample(self, noise, s, add_mean=True):
        self.sqrtRinv.mult(noise, s)
        if add_mean:
            s.axpy(1.0, self.mean)


def _space_dim(Vh):
    if hasattr(Vh, "dim"):
        return Vh.dim()
    if hasattr(Vh, "dofmap"):
        return Vh.dofmap.index_map.size_global * Vh.dofmap.index_map_bs
    return None


def _space_comm(Vh):
    if Vh is None:
        return MPI.COMM_WORLD
    if hasattr(Vh, "mesh"):
        return Vh.mesh.comm
    return MPI.COMM_WORLD
