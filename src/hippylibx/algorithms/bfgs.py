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

"""Quasi-Newton BFGS optimization algorithms."""

import numpy as np

from ..modeling.variables import STATE, PARAMETER, ADJOINT
from ..utils.parameterList import ParameterList
from .NewtonCG import LS_ParameterList
from ..utils.petsc import vec_inner, vec_zero

def BFGSoperator_ParameterList():
    """Return parameters for the BFGS operator."""
    parameters = {}
    parameters["BFGS_damping"] = [0.2, "Damping of BFGS"]
    parameters["memory_limit"] = [np.inf, "Number of vectors to store in limited memory BFGS"]
    return ParameterList(parameters)

def BFGS_ParameterList():
    """Return parameters for the BFGS solver."""
    parameters = {}
    parameters["rel_tolerance"]         = [1e-6, "we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
    parameters["abs_tolerance"]         = [1e-12, "we converge when sqrt(g,g) <= abs_tolerance"]
    parameters["gdm_tolerance"]         = [1e-18, "we converge when (g,dm) <= gdm_tolerance"]
    parameters["max_iter"]              = [500, "maximum number of iterations"]
    parameters["globalization"]         = ["LS", "Globalization technique: line search (LS)  or trust region (TR)"]
    parameters["print_level"]           = [0, "Control verbosity of printing screen"]
    ls_list = LS_ParameterList()
    ls_list["max_backtracking_iter"] = 25
    parameters["LS"]                    = [ls_list, "Sublist containing LS globalization parameters"]
    parameters["BFGS_op"]               = [BFGSoperator_ParameterList(), "BFGS operator"]
    return ParameterList(parameters)

class RescaledIdentity(object):
    """
    Default operator for :code:`H0inv`, corresponds to applying :math:`d0 I`
    """
    def __init__(self, init_vector=None):
        """Initialize the instance."""
        self.d0 = 1.0
        self._init_vector = init_vector
        
    def init_vector(self, x, dim):
        """Compute init_vector."""
        if self._init_vector:
            self._init_vector(x,dim)
        else:
            raise

    def solve(self, x, b):
        """Compute solve."""
        vec_zero(x)
        x.axpy(self.d0, b)


class BFGS_operator:
    """Class BFGS_operator."""

    def __init__(self,  parameters=BFGSoperator_ParameterList()):
        """Initialize the instance."""
        self.S, self.Y, self.R = [],[],[]

        self.H0inv = None
        self.help = None
        self.update_scaling = True

        self.parameters = parameters

    def set_H0inv(self, H0inv):
        """
        Set user-defined operator corresponding to :code:`H0inv`

        Input:

            :code:`H0inv`: Fenics operator with method :code:`solve()`
        """
        self.H0inv = H0inv
        


    def solve(self, x, b):
        """
        Solve system:           :math:`H_{bfgs} x = b`
        where :math:`H_{bfgs}` is the approximation to the Hessian build by BFGS. 
        That is, we apply

        .. math::
			x = (H_{bfgs})^{-1} b = H_k b

        where :math:`H_k` matrix is BFGS approximation to the inverse of the Hessian.
        Computation done via double-loop algorithm.
        
        Inputs:

            :code:`x = PETSc.Vec` - `[out]`

            :code:`b = PETSc.Vec` - `[in]`
        """
        A = []
        if self.help is None:
            self.help = b.copy()
        else:
            vec_zero(self.help)
            self.help.axpy(1.0, b)

        for s, y, r in zip(reversed(self.S), reversed(self.Y), reversed(self.R)):
            a = r * vec_inner(s, self.help)
            A.append(a)
            self.help.axpy(-a, y)

        self.H0inv.solve(x, self.help)     # x = H0 * x_copy

        for s, y, r, a in zip(self.S, self.Y, self.R, reversed(A)):
            b = r * vec_inner(y, x)
            x.axpy(a - b, s)


    def update(self, s, y):
        """
        Update BFGS operator with most recent gradient update.
        
        To handle potential break from secant condition, update done via damping
        
        Inputs:

            :code:`s = PETSc.Vec` `[in]` - corresponds to update in medium parameters.

            :code:`y = PETSc.Vec` `[in]` - corresponds to update in gradient.
        """
        damp = self.parameters["BFGS_damping"]
        memlim = self.parameters["memory_limit"]
        if self.help is None:
            self.help = y.copy()
        else:
            vec_zero(self.help)

        sy = vec_inner(s, y)
        self.solve(self.help, y)
        yHy = vec_inner(y, self.help)
        theta = 1.0
        if sy < damp*yHy:
            theta = (1.0-damp)*yHy/(yHy-sy)
            s *= theta
            s.axpy(1-theta, self.help)
            sy = vec_inner(s, y)
        assert(sy > 0.)
        rho = 1./sy
        self.S.append(s.copy())
        self.Y.append(y.copy())
        self.R.append(rho)

        # if L-BFGS
        if len(self.S) > memlim:
            self.S.pop(0)
            self.Y.pop(0)
            self.R.pop(0)
            self.update_scaling = True

        # re-scale H0 based on earliest secant information
        if hasattr(self.H0inv, "d0") and self.update_scaling:
            s0  = self.S[0]
            y0 = self.Y[0]
            d0 = vec_inner(s0, y0) / vec_inner(y0, y0)
            self.H0inv.d0 = d0
            self.update_scaling = False

        return theta



class BFGS:
    """
    Implement BFGS technique with backtracking inexact line search and damped updating
    See `Nocedal & Wright (06), ch.6.2, ch.7.3, ch.18.3`

    The user must provide a model that describes the forward problem, cost functionals, and all the
    derivatives for the gradient and the Hessian.
    
    More specifically the model object should implement following methods:

       - :code:`generate_vector()` -> generate the object containing state, parameter, adjoint
       - :code:`cost(x)` -> evaluate the cost functional, report regularization part and misfit separately
       - :code:`solveFwd(out, x)` -> solve the possibly non-linear forward problem
       - :code:`solveAdj(out, x)` -> solve the linear adjoint problem
       - :code:`evalGradientParameter(x, out)` -> evaluate the gradient of the parameter and compute its norm
       - :code:`applyR(dm, out)`    --> Compute :code:`out` = :math:`R dm`
       - :code:`Rsolver()`          --> A solver for the regularization term
       
    Type :code:`help(Model)` for additional information
    """
    termination_reasons = [
                           "Maximum number of Iteration reached",      #0
                           "Norm of the gradient less than tolerance", #1
                           "Maximum number of backtracking reached",   #2
                           "Norm of (g, da) less than tolerance"       #3
                           ]

    def __init__(self, model, parameters=BFGS_ParameterList()):
        """
        Initialize the BFGS solver.
        Type :code:`BFGS_ParameterList().showMe()` for default parameters and their description
        """
        self.model = model
        
        self.parameters = parameters        
        self.it = 0
        self.converged = False
        self.ncalls = 0
        self.reason = 0
        self.final_grad_norm = 0

        self.BFGSop = BFGS_operator(self.parameters["BFGS_op"])


    def solve(self, x, H0inv, bounds_xPARAM=None):
        """
        Solve the constrained optimization problem with initial guess :code:`x = [u, m0, p]`. 

        .. note:: :code:`u` and :code:`p` may be :code:`None`. 

        :code:`x` will be overwritten.

        :code:`H0inv`: the initial approximated inverse of the Hessian for the BFGS operator. It has an 
        optional method :code:`update(x)` that will update the operator based on :code:`x = [u,m,p]`.

        :code:`bounds_xPARAM`: Bound constraint (list with two entries: min and max). Can be either a scalar value or a 
        :code:`PETSc.Vec`.

        Return the solution :code:`[u,m,p]`
        """
        
        if bounds_xPARAM is not None:
            if hasattr(bounds_xPARAM[0], "array"):
                param_min = np.asarray(bounds_xPARAM[0].array)
            else:
                param_min = bounds_xPARAM[0] * np.ones_like(x[PARAMETER].array)
            if hasattr(bounds_xPARAM[1], "array"):
                param_max = np.asarray(bounds_xPARAM[1].array)
            else:
                param_max = bounds_xPARAM[1] * np.ones_like(x[PARAMETER].array)
        
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        ls_list = self.parameters[self.parameters["globalization"]]
        c_armijo = ls_list["c_armijo"]
        max_backtracking_iter = ls_list["max_backtracking_iter"]
        print_level = self.parameters["print_level"]

        self.BFGSop.parameters["BFGS_damping"] = self.parameters["BFGS_op"]["BFGS_damping"]
        self.BFGSop.parameters["memory_limit"] = self.parameters["BFGS_op"]["memory_limit"]
        self.BFGSop.set_H0inv(H0inv)

        if x[STATE] is None:
            x[STATE] = self.model.generate_vector(STATE)
            
        if x[ADJOINT] is None:
            x[ADJOINT] = self.model.generate_vector(ADJOINT)
            
        self.model.solveFwd(x[STATE], x)
        
        self.it = 0
        self.converged = False
        self.ncalls += 1
        
        mhat = self.model.generate_vector(PARAMETER)    

        mg = self.model.generate_vector(PARAMETER)
        
        x_star = [None, None, None] + x[3::]
        x_star[STATE]     = self.model.generate_vector(STATE)
        x_star[PARAMETER] = self.model.generate_vector(PARAMETER)
        
        cost_old, reg_old, misfit_old = self.model.cost(x)

        if(print_level >= 0):
            print("\n {:3} {:15} {:15} {:15} {:15} {:14} {:14} {:14}".format(
            "It", "cost", "misfit", "reg", "(g,dm)", "||g||L2", "alpha", "theta"))
            print( "{:3d} {:15e} {:15e} {:15e} {:15} {:14} {:14} {:14}".format(
            self.it, cost_old, misfit_old, reg_old, "", "", "", ""))
        
        while (self.it < max_iter) and (self.converged == False):
            
            self.model.solveAdj(x[ADJOINT], x)
            
            if hasattr(self.BFGSop.H0inv, "setPoint"):
                self.BFGSop.H0inv.setPoint(x)
            
            mg_old = mg.copy()
            gradnorm = self.model.evalGradientParameter(x, mg)
            # Update BFGS
            if self.it > 0:
                s = mhat * alpha
                y = mg - mg_old
                theta = self.BFGSop.update(s, y)
            else:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)
                theta = 1.0
                
            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break
            
            self.it += 1

            # compute search direction with BFGS:
            self.BFGSop.solve(mhat, -mg)
            
            # backtracking line-search
            alpha = 1.0
            descent = 0
            n_backtrack = 0
            mg_mhat = vec_inner(mg, mhat)
            while descent == 0 and n_backtrack < max_backtracking_iter:
                # update m and u
                vec_zero(x_star[PARAMETER])
                x_star[PARAMETER].axpy(1., x[PARAMETER])
                x_star[PARAMETER].axpy(alpha, mhat)
                vec_zero(x_star[STATE])
                x_star[STATE].axpy(1., x[STATE])
                if bounds_xPARAM is not None:
                    arr = x_star[PARAMETER].array
                    arr[:] = np.maximum(arr, param_min)
                    arr[:] = np.minimum(arr, param_max)
                    x_star[PARAMETER].assemblyBegin()
                    x_star[PARAMETER].assemblyEnd()
                    
                self.model.solveFwd(x_star[STATE], x_star)
                cost_new, reg_new, misfit_new = self.model.cost(x_star)
                
                # Check if armijo conditions are satisfied
                if (cost_new < cost_old + alpha * c_armijo * mg_mhat) or (-mg_mhat <= self.parameters["gdm_tolerance"]):
                    cost_old = cost_new
                    descent = 1
                    vec_zero(x[PARAMETER])
                    x[PARAMETER].axpy(1., x_star[PARAMETER])
                    vec_zero(x[STATE])
                    x[STATE].axpy(1., x_star[STATE])
                else:
                    n_backtrack += 1
                    alpha *= 0.5

            if print_level >= 0:
                print( "{:3d} {:15e} {:15e} {:15e} {:15e} {:14e} {:14e} {:14e}".format(
                self.it, cost_new, misfit_new, reg_new, mg_mhat, gradnorm, alpha, theta))
                
            if n_backtrack == max_backtracking_iter:
                self.converged = False
                self.reason = 2
                break
            
            if -mg_mhat <= self.parameters["gdm_tolerance"]:
                self.converged = True
                self.reason = 3
                break

                            
        self.final_grad_norm = gradnorm
        self.final_cost      = cost_new
        return x
