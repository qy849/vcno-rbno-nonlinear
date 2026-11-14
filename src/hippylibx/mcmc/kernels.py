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

"""MCMC proposal kernels and transitions."""

from ..utils.random import parRandom

import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


def _vec_comm(obj):
    """Internal helper."""
    if hasattr(obj, "getComm"):
        return obj.getComm()
    return MPI.COMM_WORLD


def _empty_vec(comm):
    """Internal helper."""
    return PETSc.Vec().createMPI(0, comm=comm)


class MALAKernel:
    """Class MALAKernel."""
    def __init__(self, model):
        """Initialize the instance."""
        self.model = model
        self.pr_mean = model.prior.mean
        self.parameters = {}
        self.parameters["delta_t"]               = 0.25*1e-4
        
        self.noise = _empty_vec(_vec_comm(self.model.prior.R))
        self.model.prior.init_vector(self.noise, "noise")
        
    def name(self):
        """Return the kernel name."""
        return "inf-MALA"
        
    def derivativeInfo(self):
        """Return whether derivatives are required."""
        return 1
    
    def init_sample(self, s):
        """Initialize the chain state."""
        self.model.solveFwd(s.u, [s.u,s.m,s.p])
        s.cost = self.model.cost([s.u,s.m,s.p])[2]
        self.model.solveAdj(s.p, [s.u,s.m,s.p])
        self.model.evalGradientParameter([s.u,s.m,s.p], s.g, misfit_only=True)
        self.model.prior.Rsolver.solve(s.Cg, s.g)
        
    def sample(self, current, proposed): 
        """Generate a new sample."""
        proposed.m = self.proposal(current)
        self.init_sample(proposed)
        rho_mp = self.acceptance_ratio(current, proposed)
        rho_pm = self.acceptance_ratio(proposed, current)
        al = rho_mp - rho_pm
        if(al > math.log(np.random.rand())):
            current.assign(proposed)
            return 1
        else:
            return 0

    def proposal(self, current):
        """Compute the proposal and log ratio."""
        delta_t = self.parameters["delta_t"]
        parRandom.normal(1., self.noise)
        w = _empty_vec(_vec_comm(self.model.prior.R))
        self.model.prior.init_vector(w, 0)
        self.model.prior.sample(self.noise, w, add_mean=False)
        delta_tp2 = 2 + delta_t
        d_gam = self.pr_mean.copy()
        tmp = current.m.copy()
        tmp.axpy(-1.0, self.pr_mean)
        tmp.scale((2.0 - delta_t) / delta_tp2)
        d_gam.axpy(1.0, tmp)
        d_gam.axpy(-(2.0 * delta_t) / delta_tp2, current.Cg)
        w.scale(math.sqrt(8.0 * delta_t) / delta_tp2)
        d_gam.axpy(1.0, w)
        return d_gam

    def acceptance_ratio(self, origin, destination):
        """Return the acceptance ratio."""
        delta_t = self.parameters["delta_t"]
        m_m = destination.m.copy()
        m_m.axpy(-1.0, origin.m)
        p_m = destination.m.copy()
        p_m.axpy(1.0, origin.m)
        p_m.axpy(-2.0, self.pr_mean)
        temp = origin.Cg.dot(origin.g)
        rho_uv = origin.cost + 0.5 * origin.g.dot(m_m) + \
                0.25 * delta_t * origin.g.dot(p_m) + \
                0.25 * delta_t * temp
        return rho_uv
    
    def consume_random(self):
        """Consume random numbers for reproducibility."""
        parRandom.normal(1., self.noise)
        np.random.rand()
        
        

class pCNKernel:
    """Class pCNKernel."""
    def __init__(self, model):
        """Initialize the instance."""
        self.model = model
        self.parameters = {}
        self.parameters["s"]                     = 0.1
        
        self.noise = _empty_vec(_vec_comm(self.model.prior.R))
        self.model.prior.init_vector(self.noise, "noise")
        
    def name(self):
        """Return the kernel name."""
        return "pCN"

    def derivativeInfo(self):
        """Return whether derivatives are required."""
        return 0

    def init_sample(self, current):
        """Initialize the chain state."""
        self.model.solveFwd(current.u, [current.u,current.m,None])
        current.cost = self.model.cost([current.u,current.m,None])[2]
        
    def sample(self, current, proposed): 
        """Generate a new sample."""
        proposed.m = self.proposal(current)
        self.init_sample(proposed)
        al = -proposed.cost + current.cost
        if(al > math.log(np.random.rand())):
            current.assign(proposed)
            return 1
        else:
            return 0

    def proposal(self, current):
        """Compute the proposal and log ratio."""
        #Generate sample from the prior
        parRandom.normal(1., self.noise)
        w = _empty_vec(_vec_comm(self.model.prior.R))
        self.model.prior.init_vector(w, 0)
        self.model.prior.sample(self.noise, w, add_mean=False)
        # do pCN linear combination with current sample
        s = self.parameters["s"]
        w.scale(s)
        w.axpy(1., self.model.prior.mean)
        tmp = current.m.copy()
        tmp.axpy(-1.0, self.model.prior.mean)
        w.axpy(np.sqrt(1. - s * s), tmp)
        
        return w
    
    def consume_random(self):
        """Consume random numbers for reproducibility."""
        parRandom.normal(1., self.noise)
        np.random.rand() 
    
class gpCNKernel:
    """
    Reference:
        `F. J. PINSKI, G. SIMPOSN, A. STUART, H. WEBER,
        Algorithms for Kullback-Leibler Approximation of Probability Measures in Infinite Dimensions,
        http://arxiv.org/pdf/1408.1920v1.pdf, Alg. 5.2`
    """
    def __init__(self, model, nu):
        """Initialize the instance."""
        self.model = model
        self.nu = nu
        self.prior = model.prior
        self.parameters = {}
        self.parameters["s"]                     = 0.1
        
        self.noise = _empty_vec(_vec_comm(self.nu.R if hasattr(self.nu, "R") else self.model.prior.R))
        self.nu.init_vector(self.noise, "noise")
        
    def name(self):
        """Return the kernel name."""
        return "gpCN"

    def derivativeInfo(self):
        """Return whether derivatives are required."""
        return 0

    def init_sample(self, current):
        """Initialize the chain state."""
        self.model.solveFwd(current.u, [current.u,current.m,None])
        current.cost = self.model.cost([current.u,current.m,None])[2]
        
    def sample(self, current, proposed): 
        """Generate a new sample."""
        proposed.m = self.proposal(current)
        self.init_sample(proposed)
        al = self.delta(current) - self.delta(proposed)
        if(al > math.log(np.random.rand())):
            current.assign(proposed)
            return 1
        else:
            return 0
        
    def delta(self,sample):
        """Return the proposal step size."""
        dm_nu = sample.m.copy()
        dm_nu.axpy(-1.0, self.nu.mean)
        return sample.cost + self.prior.cost(sample.m) - .5*self.nu.Hlr.inner(dm_nu, dm_nu)
        

    def proposal(self, current):
        """Compute the proposal and log ratio."""
        #Generate sample from the prior
        parRandom.normal(1., self.noise)
        w_prior = _empty_vec(_vec_comm(self.model.prior.R))
        self.nu.init_vector(w_prior, 0)
        w = _empty_vec(_vec_comm(self.model.prior.R))
        self.nu.init_vector(w, 0)
        self.nu.sample(self.noise, w_prior, w, add_mean=False)
        # do pCN linear combination with current sample
        s = self.parameters["s"]
        w.scale(s)
        w.axpy(1., self.nu.mean)
        tmp = current.m.copy()
        tmp.axpy(-1.0, self.nu.mean)
        w.axpy(np.sqrt(1. - s * s), tmp)
        
        return w
    
    def consume_random(self):
        """Consume random numbers for reproducibility."""
        parRandom.normal(1., self.noise)
        np.random.rand() 
    
    
class ISKernel:
    """Class ISKernel."""
    def __init__(self, model, nu):
        """Initialize the instance."""
        self.model = model
        self.nu = nu
        self.prior = model.prior
        self.parameters = {}
        
        self.noise = _empty_vec(_vec_comm(self.nu.R if hasattr(self.nu, "R") else self.model.prior.R))
        self.nu.init_vector(self.noise, "noise")
        
    def name(self):
        """Return the kernel name."""
        return "IS"

    def derivativeInfo(self):
        """Return whether derivatives are required."""
        return 0

    def init_sample(self, current):
        """Initialize the chain state."""
        self.model.solveFwd(current.u, [current.u,current.m,None])
        current.cost = self.model.cost([current.u,current.m,None])[2]
        
    def sample(self, current, proposed): 
        """Generate a new sample."""
        proposed.m = self.proposal(current)
        self.init_sample(proposed)
        al = self.delta(current) - self.delta(proposed)
        if(al > math.log(np.random.rand())):
            current.assign(proposed)
            return 1
        else:
            return 0
        
    def delta(self,sample):
        """Return the proposal step size."""
        dm_nu = sample.m.copy()
        dm_nu.axpy(-1.0, self.nu.mean)
        return sample.cost + self.prior.cost(sample.m) - .5*self.nu.Hlr.inner(dm_nu, dm_nu)
        

    def proposal(self, current):
        """Compute the proposal and log ratio."""
        #Generate sample from the prior
        parRandom.normal(1., self.noise)
        w_prior = _empty_vec(_vec_comm(self.model.prior.R))
        self.nu.init_vector(w_prior, 0)
        w = _empty_vec(_vec_comm(self.model.prior.R))
        self.nu.init_vector(w, 0)
        self.nu.sample(self.noise, w_prior, w, add_mean=True)
        
        return w
    
    def consume_random(self):
        """Consume random numbers for reproducibility."""
        parRandom.normal(1., self.noise)
        np.random.rand() 
