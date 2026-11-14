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

"""Linear algebra utilities for PETSc matrices and vectors."""

from petsc4py import PETSc
from mpi4py import MPI

import numpy as np

def _pc_available(pc_type):
    """Internal helper."""
    pc = PETSc.PC().create(MPI.COMM_WORLD)
    try:
        pc.setType(pc_type)
        return True
    except PETSc.Error:
        return False


def amg_method(amg_type="petsc_amg"):
    """
    Determine which AMG preconditioner to use.
    If available use the preconditioner suggested by the user (ML is default).
    If not available use petsc_amg (GAMG) and fall back to "none" when unsupported.
    """
    if amg_type in ("ml_amg", "ml"):
        return "ml_amg" if _pc_available("ml") else "petsc_amg"
    if amg_type in ("hypre_amg", "hypre"):
        return "hypre_amg" if _pc_available("hypre") else "petsc_amg"
    if amg_type in ("petsc_amg", "gamg"):
        return "petsc_amg" if _pc_available("gamg") else "none"
    return "petsc_amg" if _pc_available("gamg") else "none"

def MatMatMult(A,B):
    """
    Compute the matrix-matrix product :math:`AB`.
    """
    Amat = _as_mat(A)
    Bmat = _as_mat(B)
    out = Amat.matMult(Bmat)
    rmap, _ = Amat.getLGMap()
    _, cmap = Bmat.getLGMap()
    if rmap is not None or cmap is not None:
        out.setLGMap(rmap, cmap)
    return out

def MatPtAP(A,P):
    """
    Compute the triple matrix product :math:`P^T A P`.
    """
    Amat = _as_mat(A)
    Pmat = _as_mat(P)
    out = Amat.PtAP(Pmat, fill=1.0)
    _, out_map = Pmat.getLGMap()
    if out_map is not None:
        out.setLGMap(out_map, out_map)
    return out

def MatAtB(A,B):
    """
    Compute the matrix-matrix product :math:`A^T B`.
    """
    Amat = _as_mat(A)
    Bmat = _as_mat(B)
    out = Amat.transposeMatMult(Bmat)
    _, rmap = Amat.getLGMap()
    _, cmap = Bmat.getLGMap()
    if rmap is not None or cmap is not None:
        out.setLGMap(rmap, cmap)
    return out

def Transpose(A):
    """
    Compute the matrix transpose
    """
    Amat = _as_mat(A)
    AT = PETSc.Mat()
    Amat.transpose(AT)
    rmap, cmap = Amat.getLGMap()
    if rmap is not None or cmap is not None:
        AT.setLGMap(cmap, rmap)
    return AT

def SetToOwnedGid(v, gid, val):
    """Map global indices to owned indices."""
    v.setValue(gid, val)

    
def GetFromOwnedGid(v, gid):
    """Gather values from owned indices."""
    return v.getValue(gid)
    

def to_dense(A, mpi_comm = MPI.COMM_WORLD ):
    """
    Convert a sparse matrix A to dense.
    """
    nprocs = mpi_comm.Get_size()
    
    if nprocs > 1:
        raise Exception("to_dense is only serial")
    
    Amat = _as_mat(A)
    n, m = Amat.getSize()
    B = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        cols, vals = Amat.getRow(i)
        if len(cols) > 0:
            B[i, cols] = vals
        Amat.restoreRow(i, cols, vals)
    return B


def trace(A, mpi_comm = MPI.COMM_WORLD ):
    """
    Compute the trace of a sparse matrix :math:`A`.
    """
    nprocs = mpi_comm.Get_size()
    
    if nprocs > 1:
        raise Exception("trace is only serial")
    
    Amat = _as_mat(A)
    n, _ = Amat.getSize()
    tr = 0.0
    for i in range(n):
        cols, vals = Amat.getRow(i)
        if len(cols) > 0:
            tr += vals[cols == i].sum()
        Amat.restoreRow(i, cols, vals)
    return tr

def get_diagonal(A, d):
    """
    Compute the diagonal of the square operator :math:`A`.
    Use :code:`Solver2Operator` if :math:`A^{-1}` is needed.
    """
    ej = d.duplicate()
    xj = d.duplicate()
    d.set(0.0)
    rstart, rend = d.getOwnershipRange()
    for gid in range(rstart, rend):
        ej.set(0.0)
        ej.setValue(gid, 1.0)
        ej.assemblyBegin()
        ej.assemblyEnd()
        A.mult(ej, xj)
        val = xj.getValue(gid)
        d.setValue(gid, val)
    d.assemblyBegin()
    d.assemblyEnd()

    

def estimate_diagonal_inv2(Asolver, k, d):
    r"""
    An unbiased stochastic estimator for the diagonal of :math:`A^{-1}`.
    :math:`d = [ \sum_{j=1}^k v_j .* A^{-1} v_j ] ./ [ \sum_{j=1}^k v_j .* v_j ]`
    where

    - :math:`v_j` are i.i.d. :math:`\mathcal{N}(0, I)`
    - :math:`.*` and :math:`./` represent the element-wise multiplication and division
      of vectors, respectively.
      
    Reference:
        `Costas Bekas, Effrosyni Kokiopoulou, and Yousef Saad, 
        An estimator for the diagonal of a matrix, 
        Applied Numerical Mathematics, 57 (2007), pp. 1214-1229.`
    """
    x = d.duplicate()
    b = d.duplicate()
    d.set(0.0)
    rng = np.random.default_rng(1 + d.getComm().Get_rank())
    for _ in range(k):
        _fill_normal(b, rng)
        Asolver.solve(x, b)
        x.array[:] *= b.array
        d.axpy(1.0 / float(k), x)
        
class DiagonalOperator:
    """Class DiagonalOperator."""
    def __init__(self, d):
        """Initialize the instance."""
        self.d = d
        
    def init_vector(self,x,dim):
        """Compute init_vector."""
        local = self.d.getLocalSize()
        global_size = self.d.getSize()
        x.setSizes((local, global_size))
        x.setUp()
        
    def mult(self,x,y):
        """Compute mult."""
        y.pointwiseMult(self.d, x)
        
    def inner(self,x,y):
        """Compute inner."""
        tmp = y.duplicate()
        tmp.pointwiseMult(self.d, y)
        return x.dot(tmp)
    
class Solver2Operator:
    """Class Solver2Operator."""
    def __init__(self,S,mpi_comm=MPI.COMM_WORLD, init_vector = None):
        """Initialize the instance."""
        self.S = S
        self.my_init_vector = init_vector
        self.tmp = None
        
        if self.my_init_vector is None:
            if hasattr(self.S, "init_vector"):
                self.my_init_vector = self.S.init_vector
            elif hasattr(self.S, "get_operator"):
                self.my_init_vector = self.S.get_operator().init_vector
        
    def init_vector(self, x, dim):
        """Compute init_vector."""
        if self.my_init_vector:
            self.my_init_vector(x,dim)
        else:
            A = self._get_operator()
            tmp = A.createVecRight()
            x.setSizes(tmp.getSizes())
            x.setUp()
        
        
    def mult(self,x,y):
        """Compute mult."""
        self.S.solve(y,x)
        
    def inner(self, x, y):
        """Compute inner."""
        if self.tmp is None:
            self.tmp = y.duplicate()
        self.S.solve(self.tmp, y)
        return self.tmp.dot(x)

    def _get_operator(self):
        """Internal helper."""
        if hasattr(self.S, "get_operator"):
            return self.S.get_operator()
        if hasattr(self.S, "ksp"):
            A, _ = self.S.ksp().getOperators()
            return A
        raise NotImplementedError("Solver2Operator: cannot access operator")
    
class Operator2Solver:
    """Class Operator2Solver."""
    def __init__(self,op, mpi_comm=MPI.COMM_WORLD):
        """Initialize the instance."""
        self.op = op
        self.tmp = None
        
    def init_vector(self, x, dim):
        """Compute init_vector."""
        if hasattr(self.op, "init_vector"):
            self.op.init_vector(x,dim)
        elif hasattr(self.op, "createVecRight"):
            tmp = self.op.createVecLeft() if dim == 0 else self.op.createVecRight()
            x.setSizes(tmp.getSizes())
            x.setUp()
        else:
            raise
        
    def solve(self,y,x):
        """Compute solve."""
        self.op.mult(x,y)
        
    def inner(self, x, y):
        """Compute inner."""
        if self.tmp is None:
            self.tmp = y.duplicate()
        self.op.mult(y, self.tmp)
        return self.tmp.dot(x)


def _as_mat(A):
    """Internal helper."""
    if isinstance(A, PETSc.Mat):
        return A
    if hasattr(A, "mat"):
        return A.mat()
    return A


def _fill_normal(vec, rng):
    """Internal helper."""
    vec.array[:] = rng.normal(0.0, 1.0, size=vec.array.size)
    vec.assemblyBegin()
    vec.assemblyEnd()
