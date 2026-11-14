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

"""Low-rank operator representations and actions."""

from mpi4py import MPI
from petsc4py import PETSc
from .multivector import MultiVector, MatMvMult
from ..utils.petsc import vec_inner, vec_zero
import numpy as np

class LowRankOperator:
    """
    This class model the action of a low rank operator :math:`A = U D U^T`.
    Here :math:`D` is a diagonal matrix, and the columns of are orthonormal
    in some weighted inner-product.
    
    .. note:: This class only works in serial!
    """
    def __init__(self,d,U, my_init_vector = None):
        """
        Construct the low rank operator given :code:`d` and :code:`U`.
        """
        self.d = d
        self.U = U
        self.my_init_vector = my_init_vector
        
    def init_vector(self, x, dim):
        """
        Initialize :code:`x` to be compatible with the range (:code:`dim=0`) or domain (:code:`dim=1`) of :code:`A`.
        """
        assert self.my_init_vector is not None
        self.my_init_vector(x,dim)
        
    def mult(self,x,y):
        """
        Compute :math:`y = Ax = U D U^T x`
        """
        Utx = self.U.dot_v(x)
        dUtx = self.d*Utx   #elementwise mult
        vec_zero(y)
        self.U.reduce(y, dUtx)
        
    def inner(self, x, y):
        """Compute inner."""
        Utx = self.U.dot_v(x)
        Uty = self.U.dot_v(y)
        return np.sum(self.d*Utx*Uty)        
        
    def solve(self, sol, rhs):
        r"""
        Compute :math:`\mbox{sol} = U D^-1 U^T x`
        """
        Utr = self.U.dot_v(rhs)
        dinvUtr = Utr / self.d
        vec_zero(sol)
        self.U.reduce(sol, dinvUtr)

        
    def get_diagonal(self, diag):
        """
        Compute the diagonal of :code:`A`.
        """
        vec_zero(diag)
        tmp = diag.duplicate()
        for i in range(self.U.nvec()):
            ui = self.U[i]
            tmp.set(0.0)
            tmp.array[:] = ui.array * ui.array
            diag.axpy(float(self.d[i]), tmp)

        
    def trace(self,W=None):
        r"""
        Compute the trace of :code:`A`.
        If the weight :code:`W` is given, compute the trace of :math:`W^{1/2} A W^{1/2}`.
        This is equivalent to :math:`\mbox{tr}_W(A) = \sum_i \lambda_i`,
        where :math:`\lambda_i` are the generalized eigenvalues of
        :math:`A x = \lambda W^{-1} x`.
        
        .. note:: If :math:`U` is a :math:`W`-orthogonal matrix then :math:`\mbox{tr}_W(A) = \sum_i D(i,i)`. 
        """
        if W is None:
            tmp = self.U[0].copy()
            vec_zero(tmp)
            self.U.reduce(tmp, np.sqrt(self.d))
            tr = vec_inner(tmp, tmp)
        else:
            WU = MultiVector(self.U[0], self.U.nvec())
            MatMvMult(W,self.U,WU)
            diagWUtU = np.zeros_like(self.d)
            for i in range(self.d.shape[0]):
                diagWUtU[i] = vec_inner(WU[i], self.U[i])
            tr = np.sum(self.d*diagWUtU)
            
        return tr
    
    def trace2(self,W=None):
        r"""
        Compute the trace of :math:`A A` (Note this is the square of Frobenius norm, since :math:`A` is symmetic).
        If the weight :code:`W` is provided, it will compute the trace of :math:`(AW)^2`.
        
        This is equivalent to :math:`\mbox{tr}_W(A) = \sum_i \lambda_i^2`,
        where :math:`\lambda_i` are the generalized eigenvalues of
        :math:`A x = \lambda W^{-1} x`.
        
        .. note:: If :math:`U` is a :math:`W`-orthogonal matrix then :math:`\mbox{tr}_W(A) = \sum_i D(i,i)^2`. 
        """
        if W is None:
            UtU = self.U.dot_mv(self.U)
            dUtU = self.d[:,None] * UtU #diag(d)*UtU.
            tr2 = np.sum(dUtU*dUtU)
        else:
            WU = MultiVector(self.U[0], self.U.nvec())
            MatMvMult(W,self.U,WU)
            UtWU = self.U.dot_mv(WU)
            dUtWU = self.d[:,None] * UtWU #diag(d)*UtU.
            tr2 = np.linalg.norm(dUtWU) ** 2
            
        return tr2
