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

"""Finite-difference verification utilities for model gradients/Hessians."""

import numpy as np
from petsc4py import PETSc

from .variables import STATE, PARAMETER, ADJOINT
from ..utils.petsc import vec_inner, vec_zero
from .reducedHessian import ReducedHessian
from ..utils.random import parRandom
import os
    
def modelVerify(model,m0, is_quadratic = False, misfit_only=False, verbose = False, eps = None, h = None):
    """
    Verify the reduced Gradient and the Hessian of a model.
    It will produce two loglog plots of the finite difference checks for the gradient and for the Hessian.
    It will also check for symmetry of the Hessian.
    """
    if misfit_only:
        index = 2
    else:
        index = 0

    if h is None:
        h = model.generate_vector(PARAMETER)
        parRandom.normal(1., h)

    def _vprint(*args):
        if verbose:
            print(*args, flush=True)
    
    x = model.generate_vector()
    x[PARAMETER] = m0
    _vprint("modelVerify: solveFwd")
    model.solveFwd(x[STATE], x)
    _vprint("modelVerify: solveAdj")
    model.solveAdj(x[ADJOINT], x)
    _vprint("modelVerify: cost")
    cx = model.cost(x)
    
    grad_x = model.generate_vector(PARAMETER)
    _vprint("modelVerify: evalGradientParameter (base)")
    model.evalGradientParameter(x, grad_x,misfit_only=misfit_only)
        
    grad_xh = vec_inner(grad_x, h)
    
    _vprint("modelVerify: setPointForHessianEvaluations")
    use_gn = bool(os.environ.get("HIPPYLIB_GN_VERIFY"))
    if use_gn:
        _vprint("modelVerify: using Gauss-Newton Hessian approximation")
    model.setPointForHessianEvaluations(x, gauss_newton_approx=use_gn)
    H = ReducedHessian(model, misfit_only=misfit_only)
    Hh = model.generate_vector(PARAMETER)
    _vprint("modelVerify: Hessian mult (base)")
    H.mult(h, Hh)
    Hh_used = Hh
    
    if eps is None:
        n_eps = 32
        eps = np.power(.5, np.arange(n_eps))
        eps = eps[::-1]
    else:
        n_eps = eps.shape[0]
    err_grad = np.zeros(n_eps)
    err_H = np.zeros(n_eps)
    
    x_plus = model.generate_vector()
    grad_xplus = model.generate_vector(PARAMETER)
    for i in range(n_eps):
        my_eps = eps[i]
        
        vec_zero(x_plus[PARAMETER])
        x_plus[PARAMETER].axpy(1., m0 )
        x_plus[PARAMETER].axpy(my_eps, h)
        _vprint(f"modelVerify: iter {i+1}/{n_eps} solveFwd/solveAdj (eps={my_eps:g})")
        model.solveFwd(x_plus[STATE],   x_plus)
        model.solveAdj(x_plus[ADJOINT], x_plus)
        
        dc = model.cost(x_plus)[index] - cx[index]
        err_grad[i] = abs(dc/my_eps - grad_xh)
        
        #Check the Hessian
        model.evalGradientParameter(x_plus, grad_xplus,misfit_only=misfit_only)
        
        err = grad_xplus.copy()
        err.axpy(-1.0, grad_x)
        err.scale(1.0 / my_eps)
        err.axpy(-1.0, Hh_used)
        
        err_H[i] = err.norm(PETSc.NormType.NORM_INFINITY)
    
    if verbose:
        _vprint("modelVerify: plot errors")
        modelVerifyPlotErrors(is_quadratic, eps, err_grad, err_H, verbose=verbose)
        _vprint("modelVerify: plot errors done")

    _vprint("modelVerify: symmetry check")
    xx = model.generate_vector(PARAMETER)
    parRandom.normal(1., xx)
    yy = model.generate_vector(PARAMETER)
    parRandom.normal(1., yy)
    
    ytHx = H.inner(yy,xx)
    xtHy = H.inner(xx,yy)
    _vprint("modelVerify: symmetry check done")
    if np.abs(ytHx + xtHy) > 0.: 
        rel_symm_error = 2*abs(ytHx - xtHy)/(ytHx + xtHy)
    else:
        rel_symm_error = abs(ytHx - xtHy)
    if verbose:
        print( "(yy, H xx) - (xx, H yy) = ", rel_symm_error)
        if rel_symm_error > 1e-10:
            print( "HESSIAN IS NOT SYMMETRIC!!")
            
    return eps, err_grad, err_H

def modelVerifyPlotErrors(is_quadratic, eps, err_grad, err_H, verbose=False):
    """Plot finite-difference gradient/Hessian error curves."""
    try:
        import matplotlib.pyplot as plt
    except:
        if verbose:
            print("Matplotlib is not installed.")
        return
    if is_quadratic:
        fig = plt.figure()
        plt.subplot(121)
        plt.loglog(eps, err_grad, "-ob", eps, eps*(err_grad[0]/eps[0]), "-.k")
        plt.title("FD Gradient Check")
        plt.subplot(122)
        plt.loglog(eps[0], err_H[0], "-ob", [10*eps[0], eps[0], 0.1*eps[0]], [err_H[0],err_H[0],err_H[0]], "-.k")
        plt.title("FD Hessian Check")
    else:  
        fig = plt.figure()
        plt.subplot(121)
        plt.loglog(eps, err_grad, "-ob", eps, eps*(err_grad[0]/eps[0]), "-.k")
        plt.title("FD Gradient Check")
        plt.subplot(122)
        plt.loglog(eps, err_H, "-ob", eps, eps*(err_H[0]/eps[0]), "-.k")
        plt.title("FD Hessian Check")

    out_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "modelVerify.png")
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"Saved modelVerify plot to {out_file}")
