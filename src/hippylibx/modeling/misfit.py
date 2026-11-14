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

"""Misfit models and observation operators for inverse problems."""

import ufl
import numpy as np
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from .pointwiseObservation import assemblePointwiseObservation
from .variables import STATE, PARAMETER
from ..utils.deprecate import deprecated
from ..utils.petsc import vec_inner, vec_zero as _vec_zero


def _as_bcs(bcs):
    """Normalize a boundary condition argument to a list."""
    if bcs is None:
        return []
    if isinstance(bcs, fem.DirichletBC):
        return [bcs]
    return list(bcs)


def _assemble_matrix(form, bcs, kind=None):
    """Assemble a matrix with optional boundary conditions and kind."""
    a = fem.form(form)
    A = fem_petsc.assemble_matrix(a, bcs, kind=kind)
    A.assemble()
    return A

class Misfit(object):
    """
    Abstract class to model the misfit component of the cost functional.
    In the following :code:`x` will denote the variable :code:`[u, m, p]`, denoting respectively 
    the state :code:`u`, the parameter :code:`m`, and the adjoint variable :code:`p`.
    
    The methods in the class misfit will usually access the state u and possibly the
    parameter :code:`m`. The adjoint variables will never be accessed. 
    """
    def cost(self,x):
        """
        Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter m are accessed. """
        raise NotImplementedError("Child class should implement method cost")
        return 0
        
    def grad(self, i, x, out):
        """
        Given the state and the paramter in :code:`x`, compute the partial gradient of the misfit
        functional in with respect to the state (:code:`i == STATE`) or with respect to the parameter (:code:`i == PARAMETER`).
        """
        raise NotImplementedError("Child class should implement method grad")
            
    def setLinearizationPoint(self,x, gauss_newton_approx=False):
        """
        Set the point for linearization.

        Inputs:
        
            :code:`x=[u, m, p]` - linearization point

            :code:`gauss_newton_approx (bool)` - whether to use Gauss Newton approximation 
        """
        raise NotImplementedError("Child class should implement method setLinearizationPoint")
        
    def apply_ij(self,i,j, dir, out):
        r"""
        Apply the second variation :math:`\delta_{ij}` (:code:`i,j = STATE,PARAMETER`) of the cost in direction :code:`dir`.
        """
        raise NotImplementedError("Child class should implement method apply_ij")


@deprecated(version="3.1", msg="Use B = assemblePointwiseObservation(Vh, obs_points) followed by DiscreteStateObservation(B, Mpar)")       
def PointwiseStateObservation(Vh, obs_points):
    """
    This function returns an instance of :code:`DiscreteStateObservation` for pointwise state observations at given locations.
    Inputs:

            :code:`Vh` is the finite element space for the state variable
            
            :code:`obs_points` is a 2D array number of points by geometric dimensions that stores \
            the location of the observations.
    """
    B = assemblePointwiseObservation(Vh, obs_points)
    return DiscreteStateObservation(B)

class DiscreteStateObservation(Misfit):
    """
    This class define a misfit function for a discrete linear observation operator B
    """
    def __init__(self, B, data=None, noise_variance=None):
        """
        Constructor:
            :code:`B` is the observation operator   
            :code:`data` is the data
            :code:`noise_variance` is the variance of the noise
        """
        self.B = B

        if data is None:
            self.d = self.B.createVecLeft()
        else:
            self.d = data
        
        self.Bu = self.B.createVecLeft()
        self.noise_variance = noise_variance
        
    def cost(self,x):
        if self.noise_variance is None: 
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise  ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        self.B.mult(x[STATE], self.Bu)
        self.Bu.axpy(-1., self.d)
        return (.5/self.noise_variance) * vec_inner(self.Bu, self.Bu)
    
    def grad(self, i, x, out):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise  ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        if i == STATE:
            self.B.mult(x[STATE], self.Bu)
            self.Bu.axpy(-1., self.d)
            self.B.multTranspose(self.Bu, out)
            out *= (1./self.noise_variance)
        elif i == PARAMETER:
            _vec_zero(out)
        else:
            raise IndexError()
                
    def setLinearizationPoint(self,x, gauss_newton_approx=False):
        # The cost functional is already quadratic. Nothing to be done here
        return
       
    def apply_ij(self,i,j,dir,out):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise  ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        if i == STATE and j == STATE:
            self.B.mult(dir, self.Bu)
            self.B.multTranspose(self.Bu, out)
            out *= (1./self.noise_variance)
        else:
            _vec_zero(out)

@deprecated(version="3.1", msg="Use B = assemblePointwiseObservation(Vh, obs_points) followed by MultDiscreteStateObservation(B, None, Mpar)")       
def MultPointwiseStateObservation(Vh, obs_points, Mpar):
    """
    This function returns an instance of :code:`DiscreteStateObservation` for pointwise state observations at given locations.
    A multiplicative Gamma(M,M) noise model is assumed
    Inputs:

            :code:`Vh` is the finite element space for the state variable
            :code:`data` is the data
            :code:`obs_points` is a 2D array number of points by geometric dimensions that stores \
            the location of the observations.

            :code:`Mpar` Gamma distribution parameter
    """
    B = assemblePointwiseObservation(Vh, obs_points)
    return MultDiscreteStateObservation(B, None, Mpar)


class MultDiscreteStateObservation(Misfit):
    """
    This class implements discrete state observations defined by the linear operator B.
    A multiplicative Gamma(M,M) noise model is assumed
    """
    def __init__(self, B, data=None, Mpar=1.):
        """
        Constructor:

            :code:`B` is the observation operator
            
            :code:`Mpar` Gamma distribution parameter
        """
        self.B = B
        if data is None:
            self.d = self.B.createVecLeft()
        else:
            self.d = data
        
        self.Bu = self.B.createVecLeft()
        
        self.Bu_lin = self.B.createVecLeft()
        
        self.help = self.B.createVecLeft()

        
        self.Mpar = Mpar
        
        self.ones = self.B.createVecLeft()
        self.ones.set(1.0)
        
    def cost(self,x):
        self.B.mult(x[STATE], self.Bu)
        Bu_local = self.Bu.array
        self.help.array[:] = np.log(Bu_local) + self.d.array / Bu_local
        
        return self.Mpar * vec_inner(self.help, self.ones)
    
    def grad(self, i, x, out):
        _vec_zero(out)
        if i == STATE:
            self.B.mult(x[STATE], self.Bu)
            Bu_local = self.Bu.array
            self.help.array[:] = np.ones_like(Bu_local) / Bu_local - self.d.array / (Bu_local * Bu_local)
            self.B.multTranspose(self.help, out)
            out *= self.Mpar
        elif i == PARAMETER:
            _vec_zero(out)
        else:
            raise IndexError()
                
    def setLinearizationPoint(self,x, gauss_newton_approx=False):
        self.B.mult(x[STATE], self.Bu_lin)
        # The cost functional is already quadratic. Nothing to be done here
        return
       
    def apply_ij(self,i,j,dir,out):
        _vec_zero(out)
        if i == STATE and j == STATE:
            self.B.mult(dir, self.Bu)
            Bdir_local = self.Bu.array
            Bu_lin_local = self.Bu_lin.array
            self.help.array[:] = (
                -Bdir_local * np.power(Bu_lin_local, -2)
                + 2.0 * self.d.array * Bdir_local * np.power(Bu_lin_local, -3)
            )
            self.B.multTranspose(self.help, out)
            out *= self.Mpar
        else:
            _vec_zero(out)
            
class ContinuousStateObservation(Misfit):
    r"""
    This class implements continuous state observations in a
    subdomain :math:`X \subset \Omega` or :math:`X \subset \partial \Omega`.
    """
    def __init__(self, Vh, dX, bcs, data=None, noise_variance=None, form=None):
        r"""
        Constructor:

            :code:`Vh`: the finite element space for the state variable.

            :code:`dX`: the integrator on subdomain `X` where observation are presents. \
            E.g. :code:`dX = ufl.dx` means observation on all :math:`\Omega` and :code:`dX = ufl.ds` means observations on all :math:`\partial \Omega`.
            
            :code:`bcs`: If the forward problem imposes Dirichlet boundary conditions :math:`u = u_D \mbox{ on } \Gamma_D`;  \
            :code:`bcs` is a list of :code:`dolfinx.fem.DirichletBC` objects that prescribes homogeneuos Dirichlet conditions :math:`u = 0 \mbox{ on } \Gamma_D`.
            
            :code:`data` is the data
            
            :code:`noise_variance` is the variance of the noise
            
            :code:`form`: if :code:`form = None` we compute the :math:`L^2(X)` misfit: :math:`\int_X (u - u_d)^2 dX,` \
            otherwise the integrand specified in the given form will be used.
        """
        if form is None:
            u, v = ufl.TrialFunction(Vh), ufl.TestFunction(Vh)
            self.W = _assemble_matrix(ufl.inner(u, v) * dX, _as_bcs(bcs))
        else:
            self.W = _assemble_matrix(form, _as_bcs(bcs))

        if data is None:
            self.d = self.W.createVecRight()
        else:
            self.d = data
                
        self.noise_variance = noise_variance
        
    def cost(self,x):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise  ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        r = self.d.copy()
        r.axpy(-1., x[STATE])
        Wr = self.W.createVecRight()
        self.W.mult(r, Wr)
        return vec_inner(r, Wr) / (2.0 * self.noise_variance)
    
    def grad(self, i, x, out):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise  ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        if i == STATE:
            tmp = x[STATE].copy()
            tmp.axpy(-1.0, self.d)
            self.W.mult(tmp, out)
            out *= (1./self.noise_variance)
        elif i == PARAMETER:
            _vec_zero(out)
        else:
            raise IndexError()
        
    def setLinearizationPoint(self,x, gauss_newton_approx=False):
        # The cost functional is already quadratic. Nothing to be done here
        return
                   
    def apply_ij(self,i,j,dir,out):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise  ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        if i == STATE and j == STATE:
            self.W.mult(dir, out)
            out *= (1./self.noise_variance)
        else:
            _vec_zero(out) 


class MultiStateMisfit(Misfit):
    def __init__(self, misfits):
        self.nmisfits = len(misfits)
        self.misfits = misfits
        
    def append(self, misfit):
        self.nmisfits += 1
        self.misfits.append(misfit)

    def cost(self,x):
        u, m, p = x
        c = 0.
        for i in range(self.nmisfits):
            c += self.misfits[i].cost([ u.data[i], m, None ] )
        return c
    
    def grad(self, i, x, out):
        _vec_zero(out)
        u, m, p = x
        if i == STATE:
            for ii in range(self.nmisfits):
                self.misfits[ii].grad(i, [ u.data[ii], m, None ], out.data[ii] )
        else:
            tmp = out.copy()
            for ii in range(self.nmisfits):
                self.misfits[ii].grad(i, [ u.data[ii], m, None ], tmp )
                out.axpy(1., tmp)
        
    def setLinearizationPoint(self,x, gauss_newton_approx=False):
        u, m, p = x
        for ii in range(self.nmisfits):
            self.misfits[ii].setLinearizationPoint([ u.data[ii], m, None ], gauss_newton_approx)
        
       
    def apply_ij(self,i,j,dir,out):
        _vec_zero(out)
        if i == STATE and j == STATE:
            for s in range(self.nmisfits):
                self.misfits[s].apply_ij(i,j,dir.data[s],out.data[s])
        elif i == STATE and j == PARAMETER:
            for s in range(self.nmisfits):
                self.misfits[s].apply_ij(i,j,dir,out.data[s])
        elif i == PARAMETER and j == STATE:
            tmp = out.copy()
            for s in range(self.nmisfits):
                self.misfits[s].apply_ij(i,j,dir.data[s],tmp)
                out.axpy(1., tmp)
        elif i == PARAMETER and j == PARAMETER:
            tmp = out.copy()
            for s in range(self.nmisfits):
                self.misfits[s].apply_ij(i,j,dir,tmp)
                out.axpy(1., tmp)
        else:
            raise IndexError


class MisfitTD(Misfit):
    r"""This class implements continuous state observations in a
       subdomain X \subset \Omega or X \subset \partial\Omega.
    """
    def __init__(self, misfits, sim_times):
        self.misfits = misfits
        self.sim_times = sim_times

    def cost(self,x):
        c = 0.
        for itime, misfit in enumerate(self.misfits):
            c += misfit.cost([x[STATE].view(self.sim_times[itime]), x[PARAMETER], None])
            
        return c

    def grad(self, i, x, out):
        _vec_zero(out)
        if i == STATE:
            for itime, misfit in enumerate(self.misfits):
                t = self.sim_times[itime]
                misfit.grad(i, [x[STATE].view(t), x[PARAMETER], None], out.view(t))            
        elif i == PARAMETER:
            out_t = out.copy()
            for itime, misfit in enumerate(self.misfits):
                _vec_zero(out_t)
                misfit.grad(i, [x[STATE].view(self.sim_times[itime]), x[PARAMETER], None], out_t)
                out.axpy(1., out_t)
        else:
            raise IndexError()
        
    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        for itime, misfit in enumerate(self.misfits):
            t = self.sim_times[itime]
            misfit.setLinearizationPoint([x[STATE].view(t), x[PARAMETER], None], gauss_newton_approx)   
        
    def _apply_STATE_STATE(self, dir, out):
        for itime, misfit in enumerate(self.misfits):
            t = self.sim_times[itime]
            misfit.apply_ij(STATE, STATE, dir.view(t), out.view(t)) 

    def _apply_STATE_PARAMETER(self, dir, out):
        for itime, misfit in enumerate(self.misfits):
            t = self.sim_times[itime]
            misfit.apply_ij(STATE, PARAMETER, dir, out.view(t)) 
            
    def _apply_PARAMETER_STATE(self, dir, out):
        out_t = out.copy()
        for itime, misfit in enumerate(self.misfits):
            t = self.sim_times[itime]
            misfit.apply_ij(PARAMETER, STATE, dir.view(t), out_t)
            out.axpy(1., out_t)
            
    def _apply_PARAMETER_PARAMETER(self, dir, out):
        out_t = out.copy()
        for misfit in self.misfits:
            misfit.apply_ij(PARAMETER, PARAMETER, dir, out_t)
            out.axpy(1., out_t)

    def apply_ij(self,i,j,dir,out):
        _vec_zero(out)
        if i == STATE and j == STATE:
            self._apply_STATE_STATE(dir, out)
        elif i == STATE and j == PARAMETER:
            self._apply_STATE_PARAMETER(dir, out)
        elif i == PARAMETER and j == STATE:
            self._apply_PARAMETER_STATE(dir, out)
        elif i == PARAMETER and j == PARAMETER:
            self._apply_PARAMETER_PARAMETER(dir, out)
        else:
            raise
