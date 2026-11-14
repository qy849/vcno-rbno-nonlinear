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

"""Random number helpers for parallel sampling."""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


class Random:
    """
    This class handles parallel generation of random numbers in hippylibx.
    """
    def __init__(self,myid=0, nproc=1, blocksize=1000000, seed=1):
        """
        Create a parallel random number number generator.

        INPUTS:

        - :code:`myid`: id of the calling process.
        - :code:`nproc`: number of processor in the communicator.
        - :code:`blocksize`: number of consecutive random number to be generated before jumping headed in the stream.
        - :code:`seed`: random seed to initialize the random engine.
        """
        self._seed = seed
        self._rng = self._make_rng(myid, nproc)

    def split(self, myid, nproc, blocksize=1000000):
        """Reset the RNG stream for a new MPI rank layout."""
        self._rng = self._make_rng(myid, nproc)

    def _make_rng(self, myid, nproc):
        """Create a numpy Generator for the given rank."""
        try:
            myid = int(myid)
        except (TypeError, ValueError):
            myid = 0
        try:
            nproc = int(nproc)
        except (TypeError, ValueError):
            nproc = 1
        seed_seq = np.random.SeedSequence(self._seed)
        if nproc > 1:
            children = seed_seq.spawn(nproc)
            return np.random.default_rng(children[myid])
        return np.random.default_rng(seed_seq)

    def _apply(self, out, sampler, add=False):
        """Apply sampler to an output container, optionally adding."""
        if out is None:
            return sampler(1).item()
        if hasattr(out, "nvec"):
            for i in range(out.nvec()):
                self._apply(out[i], sampler, add=add)
            return None
        if hasattr(out, "nsteps"):
            for i in range(out.nsteps):
                self._apply(out.data[i], sampler, add=add)
            return None
        if isinstance(out, PETSc.Vec):
            _fill_vec(out, sampler, add)
            return None
        if hasattr(out, "array"):
            _fill_array(out.array, sampler, add)
            if hasattr(out, "scatter_forward"):
                out.scatter_forward()
            elif hasattr(out, "assemblyBegin"):
                out.assemblyBegin()
                out.assemblyEnd()
            return None
        raise TypeError(f"Unsupported output type: {type(out)}")
        
    def uniform(self,a,b, out=None):
        """
        Sample from uniform distribution.
        """
        return self._apply(out, lambda n: self._rng.uniform(a, b, size=n))
    
    def normal(self, sigma, out=None):
        """
        Sample from normal distribution with given variance.
        """
        return self._apply(out, lambda n: self._rng.normal(0.0, sigma, size=n))
            
    def normal_perturb(self,sigma, out):
        """
        Add a normal perturbation to a Vector/MultiVector.
        """
        return self._apply(out, lambda n: self._rng.normal(0.0, sigma, size=n), add=True)
            
    def speckle(self,M, out):
        """
        Add a normal perturbation to a Vector/MultiVector.
        """
        return self._apply(out, lambda n: self._rng.gamma(M, 1.0 / M, size=n), add=True)
            
    def rademacher(self, out=None):
        """
        Sample from Rademacher distribution.
        """
        return self._apply(out, lambda n: self._rng.choice([-1.0, 1.0], size=n))


def _fill_array(arr, sampler, add):
    """Fill or add values in a NumPy array."""
    vals = sampler(arr.size)
    if add:
        arr[:] = arr + vals
    else:
        arr[:] = vals


def _fill_vec(vec, sampler, add):
    """Fill or add values in a PETSc Vec."""
    _fill_array(vec.array, sampler, add)
    vec.assemblyBegin()
    vec.assemblyEnd()
            
try:
    _comm = MPI.COMM_WORLD
    _world_rank = _comm.Get_rank() if callable(getattr(_comm, "Get_rank", None)) else 0
    _world_size = _comm.Get_size() if callable(getattr(_comm, "Get_size", None)) else 1
except Exception:
    _world_rank = 0
    _world_size = 1

parRandom = Random(_world_rank, _world_size)
            
