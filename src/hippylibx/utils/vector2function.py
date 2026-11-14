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

"""Helpers to wrap vectors as dolfinx Functions."""

import numpy as np
from dolfinx import fem


def assign_vec_to_function(fun, vec, scale=1.0):
    """Assign a PETSc-like vector into a dolfinx Function."""
    owned = fun.function_space.dofmap.index_map.size_local * fun.function_space.dofmap.index_map_bs
    if hasattr(vec, "array"):
        arr = vec.array
    else:
        arr = np.asarray(vec)

    if arr.size == fun.x.array.size:
        fun.x.array[:] = arr
    elif arr.size == owned:
        fun.x.array[:owned] = arr
    else:
        raise ValueError(
            f"Incompatible vector size {arr.size} for FunctionSpace "
            f"(owned={owned}, local={fun.x.array.size})."
        )
    if scale != 1.0:
        fun.x.array[:] *= scale
    fun.x.scatter_forward()


def vector2Function(x, Vh, **kwargs):
    """
    Wrap a finite element vector :code:`x` into a finite element function in the space :code:`Vh`.
    :code:`kwargs` is optional keywords arguments to be passed to the construction of a dolfinx :code:`Function`.
    """
    fun = fem.Function(Vh, **kwargs)
    assign_vec_to_function(fun, x)
    
    return fun
