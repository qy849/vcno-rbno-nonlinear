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

"""Tracer utilities for storing MCMC diagnostics."""

from ..modeling.variables import STATE, PARAMETER

import numpy as np
from dolfinx import fem
from ..utils.vector2function import assign_vec_to_function


def _make_output_functions(Vh, name):
    """Internal helper."""
    mesh = Vh.mesh
    mesh_degree = mesh.geometry.cmap.degree
    bs = Vh.dofmap.index_map_bs
    if bs == 1:
        Vout = fem.functionspace(mesh, ("Lagrange", mesh_degree))
    else:
        Vout = fem.functionspace(mesh, ("Lagrange", mesh_degree, (bs,)))
    return fem.Function(Vh, name=name), fem.Function(Vout, name=name)


class NullTracer(object):
    """Class NullTracer."""
    def __init__(self):
        """Initialize the instance."""
        pass
    def append(self,current, q):
        """Append the sample to the trace."""
        pass
    
class QoiTracer(object):
    """Class QoiTracer."""
    def __init__(self, n):
        """Initialize the instance."""
        self.data = np.zeros(n)
        self.i = 0
        
    def append(self,current, q):
        """Append the sample to the trace."""
        self.data[self.i] = q
        self.i+=1
        
class FullTracer(object):
    """Class FullTracer."""
    def __init__(self, n, Vh, par_fid = None, state_fid = None):
        """Initialize the instance."""
        self.data = np.zeros((n,2))
        self.i = 0
        self.Vh = Vh
        self.par_fid = par_fid
        self.state_fid = state_fid
        self._par_fun, self._par_out = _make_output_functions(Vh[PARAMETER], "parameter")
        self._state_fun, self._state_out = _make_output_functions(Vh[STATE], "state")
        
    def append(self,current, q):
        """Append the sample to the trace."""
        self.data[self.i, 0] = q
        self.data[self.i, 1] = current.cost
        if self.par_fid is not None:
            assign_vec_to_function(self._par_fun, current.m)
            self._par_out.interpolate(self._par_fun)
            if hasattr(self.par_fid, "write_function"):
                self.par_fid.write_function(self._par_out, self.i)
            else:
                self.par_fid.write(self._par_out, self.i)

        if self.state_fid is not None:
            assign_vec_to_function(self._state_fun, current.u)
            self._state_out.interpolate(self._state_fun)
            if hasattr(self.state_fid, "write_function"):
                self.state_fid.write_function(self._state_out, self.i)
            else:
                self.state_fid.write(self._state_out, self.i)

        self.i+=1
