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

"""Block vector container and BLAS-like operations."""

import numpy as np
from petsc4py import PETSc
from ..utils.vector2function import assign_vec_to_function


class VecWrapper:
    """Class VecWrapper."""
    def __init__(self, vec):
        """Initialize the instance."""
        self.vec = vec

    @property
    def array(self):
        """Compute array."""
        return self.vec.array

    def copy(self):
        """Compute copy."""
        return VecWrapper(_vec_copy(self.vec))

    def zero(self):
        """Compute zero."""
        _vec_zero(self.vec)

    def axpy(self, a, y):
        """Compute axpy."""
        _vec_axpy(self.vec, a, y)

    def inner(self, y):
        """Compute inner."""
        return _vec_inner(self.vec, y)

    def norm(self, norm_type):
        """Compute norm."""
        return _vec_norm(self.vec, norm_type)

    def scale(self, a):
        """Compute scale."""
        _vec_scale(self.vec, a)

    def __imul__(self, a):
        """Special method."""
        self.scale(a)
        return self

    def __mul__(self, other):
        """Special method."""
        if isinstance(other, (VecWrapper, PETSc.Vec)):
            return VecWrapper(_vec_pointwise_mult(self.vec, other))
        out = self.copy()
        out.scale(other)
        return out

    def __rmul__(self, other):
        """Special method."""
        return self.__mul__(other)

    def __getattr__(self, name):
        """Special method."""
        return getattr(self.vec, name)


class MultiVector:
    """Class MultiVector."""
    def __init__(self, v=None, nvec=None):
        """Initialize the instance."""
        self.mv = []
        if v is None:
            return
        if isinstance(v, MultiVector):
            self.mv = [VecWrapper(_vec_copy(vec)) for vec in v.mv]
            return
        if nvec is None:
            raise ValueError("nvec is required when initializing from a vector")
        template = _as_vec(v)
        for _ in range(nvec):
            vec = _vec_duplicate(template)
            _vec_zero(vec)
            self.mv.append(VecWrapper(vec))

    def nvec(self):
        """Compute nvec."""
        return len(self.mv)

    def __len__(self):
        """Special method."""
        return self.nvec()

    def __getitem__(self, i):
        """Special method."""
        return self.mv[i]

    def __setitem__(self, i, v):
        """Special method."""
        self.mv[i] = VecWrapper(_as_vec(v)) if not isinstance(v, VecWrapper) else v

    def setSizeFromVector(self, v, nvec):
        """Compute setSizeFromVector."""
        self.mv = []
        template = _as_vec(v)
        for _ in range(nvec):
            vec = _vec_duplicate(template)
            _vec_zero(vec)
            self.mv.append(VecWrapper(vec))

    def dot(self, v):
        """Compute dot."""
        if isinstance(v, MultiVector):
            if v is self:
                return self._dot_self()
            data = np.zeros((self.nvec(), v.nvec()), dtype=np.float64)
            for i in range(self.nvec()):
                for j in range(v.nvec()):
                    data[i, j] = _vec_inner(self[i], v[j])
            return data.flatten(order="C")
        return np.array([_vec_inner(vi, v) for vi in self.mv], dtype=np.float64)

    def dot_v(self, v):
        """Compute dot_v."""
        return self.dot(v)
    
    def dot_mv(self, mv):
        """Compute dot_mv."""
        shape = (self.nvec(), mv.nvec())
        m = self.dot(mv)
        return m.reshape(shape, order="C")
        
    def reduce(self, v, alpha):
        """Compute reduce."""
        data = np.asarray(alpha, dtype=np.float64).ravel()
        for coeff, vec in zip(data, self.mv):
            _vec_axpy(v, coeff, vec)

    def axpy(self, a, y):
        """Compute axpy."""
        if isinstance(y, MultiVector):
            data = np.asarray(a, dtype=np.float64).ravel()
            assert y.nvec() == self.nvec()
            assert data.size == self.nvec()
            for coeff, vi, yi in zip(data, self.mv, y.mv):
                _vec_axpy(vi, coeff, yi)
        else:
            for vi in self.mv:
                _vec_axpy(vi, a, y)

    def scale(self, k, a=None):
        """Compute scale."""
        if a is None:
            data = np.asarray(k, dtype=np.float64).ravel()
            assert data.size == self.nvec()
            for coeff, vi in zip(data, self.mv):
                _vec_scale(vi, coeff)
        else:
            _vec_scale(self.mv[k], a)

    def zero(self):
        """Compute zero."""
        for vi in self.mv:
            _vec_zero(vi)

    def norm(self, norm_type):
        """Compute norm."""
        return np.array([_vec_norm(vi, norm_type) for vi in self.mv], dtype=np.float64)

    def swap(self, other):
        """Compute swap."""
        self.mv, other.mv = other.mv, self.mv

    def _dot_self(self):
        """Internal helper."""
        size = self.nvec()
        data = np.zeros((size, size), dtype=np.float64)
        for i in range(size):
            data[i, i] = _vec_inner(self[i], self[i])
            for j in range(i):
                val = _vec_inner(self[i], self[j])
                data[i, j] = val
                data[j, i] = val
        return data.flatten(order="C")
    def dot_v(self, v):
        """Compute dot_v."""
        return self.dot(v)
    
    def dot_mv(self,mv):
        """Compute dot_mv."""
        shape = (self.nvec(),mv.nvec())
        m = self.dot(mv)
        return m.reshape(shape, order='C')
        
    def Borthogonalize(self,B):
        """ 
        Returns :math:`QR` decomposition of self. :math:`Q` and :math:`R` satisfy the following relations in exact arithmetic

        .. math::
            R \\,= \\,Z, && (1),

            Q^*BQ\\, = \\, I, && (2),

            Q^*BZ \\, = \\,R, && (3),

            ZR^{-1} \\, = \\, Q, && (4). 
        
        Returns:

            :code:`Bq` of type :code:`MultiVector` -> :code:`B`:math:`^{-1}`-orthogonal vectors
            :code:`r` of type :code:`ndarray` -> The :math:`r` of the QR decomposition.
        
        .. note:: :code:`self` is overwritten by :math:`Q`.    
        """
        return self._mgs_stable(B)
    
    def orthogonalize(self):
        """ 
        Returns :math:`QR` decomposition of self. :math:`Q` and :math:`R` satisfy the following relations in exact arithmetic
        
            .. math::
                QR \\, = \\, Z, && (1),
                
                Q^*Q \\, = \\, I, && (2),
                
                Q^*Z \\, = \\, R, && (3),
                
                ZR^{-1} \\, = \\, Q, && (4).
        
        Returns:

            :code:`r` of type :code:`ndarray` -> The `r` of the QR decomposition
        
        .. note:: :code:`self` is overwritten by :math:`Q`.    
        """
        return self._mgs_reortho()
    
    def _mgs_stable(self, B):
        """ 
        Returns :math:`QR` decomposition of self, which satisfies conditions (1)--(4).
        Uses Modified Gram-Schmidt with re-orthogonalization (Rutishauser variant)
        for computing the :math:`B`-orthogonal :math:`QR` factorization.
        
        References:
            1. `A.K. Saibaba, J. Lee and P.K. Kitanidis, Randomized algorithms for Generalized \
            Hermitian Eigenvalue Problems with application to computing \
            Karhunen-Loe've expansion http://arxiv.org/abs/1307.6885`
            2. `W. Gander, Algorithms for the QR decomposition. Res. Rep, 80(02), 1980`
        
        https://github.com/arvindks/kle
        
        """
        n = self.nvec()
        Bq = MultiVector(self[0], n)
        r  = np.zeros((n,n), dtype = 'd')
        reorth = np.zeros((n,), dtype = 'd')
        eps = np.finfo(np.float64).eps
        
        for k in np.arange(n):
            _mat_mult(B, self[k], Bq[k])
            t = np.sqrt(_vec_inner(Bq[k], self[k]))
            
            nach = 1;    u = 0;
            while nach:
                u += 1
                for i in np.arange(k):
                    s = _vec_inner(Bq[i], self[k])
                    r[i,k] += s
                    self[k].axpy(-s, self[i])
                    
                _mat_mult(B, self[k], Bq[k])
                tt = np.sqrt(_vec_inner(Bq[k], self[k]))
                if tt > t*10.*eps and tt < t/10.:
                    nach = 1;    t = tt;
                else:
                    nach = 0;
                    if tt < 10.*eps*t:
                        tt = 0.
            

            reorth[k] = u
            r[k,k] = tt
            if np.abs(tt*eps) > 0.:
                tt = 1./tt
            else:
                tt = 0.
                
            self.scale(k, tt)
            Bq.scale(k, tt)
            
        return Bq, r 
    
    def _mgs_reortho(self):
        """Internal helper."""
        n = self.nvec()
        r  = np.zeros((n,n), dtype = 'd')
        reorth = np.zeros((n,), dtype = 'd')
        eps = np.finfo(np.float64).eps
        
        for k in np.arange(n):
            t = np.sqrt(_vec_inner(self[k], self[k]))
            
            nach = 1;    u = 0;
            while nach:
                u += 1
                for i in np.arange(k):
                    s = _vec_inner(self[i], self[k])
                    r[i,k] += s
                    self[k].axpy(-s, self[i])
                    
                tt = np.sqrt(_vec_inner(self[k], self[k]))
                if tt > t*10.*eps and tt < t/10.:
                    nach = 1;    t = tt;
                else:
                    nach = 0;
                    if tt < 10.*eps*t:
                        tt = 0.
            

            reorth[k] = u
            r[k,k] = tt
            if np.abs(tt*eps) > 0.:
                tt = 1./tt
            else:
                tt = 0.
                
            self.scale(k, tt)
            
        return r
    
    def export(self, Vh, filename, varname = "mv", normalize=False):
        """
        Export in paraview this multivector.

        Inputs:

        - :code:`Vh`:        the parameter finite element space.
        - :code:`filename`:  the name of the paraview output file.
        - :code:`varname`:   the name of the paraview variable.
        - :code:`normalize`: if :code:`True` the vector is rescaled such that :math:`\\| u \\|_{\\infty} = 1.` 
        """
        if '.xdmf' in filename:
            self._exportXDMF(Vh, filename, varname, normalize)
        else:
            self._exportFile(Vh, filename, varname, normalize)
            
    def _exportXDMF(self, Vh, filename, varname, normalize):
        """
        Specialization of export using XDMF.
        """
        from dolfinx import fem, io

        mesh = Vh.mesh
        fun = fem.Function(Vh, name=varname)
        mesh_degree = mesh.geometry.cmap.degree
        value_size = Vh.dofmap.index_map_bs
        if value_size == 1:
            Vout = fem.functionspace(mesh, ("Lagrange", mesh_degree))
        else:
            Vout = fem.functionspace(mesh, ("Lagrange", mesh_degree, (value_size,)))
        fout = fem.Function(Vout, name=varname)
        with io.XDMFFile(mesh.comm, filename, "w") as fid:
            fid.write_mesh(mesh)
            if not normalize:
                for i in range(self.nvec()):
                    assign_vec_to_function(fun, self[i])
                    fout.interpolate(fun)
                    fid.write_function(fout, i)
            else:
                for i in range(self.nvec()):
                    s = _vec_norm(self[i], "linf")
                    scale = 1.0 if s == 0 else 1.0 / s
                    assign_vec_to_function(fun, self[i], scale=scale)
                    fout.interpolate(fun)
                    fid.write_function(fout, i)
        
    def _exportFile(self, Vh, filename, varname, normalize):
        """
        Fallback export to XDMF when non-XDMF formats are requested.
        """
        if not filename.endswith(".xdmf"):
            filename = f"{filename}.xdmf"
        self._exportXDMF(Vh, filename, varname, normalize)
            
    
def MatMvMult(A, x, y):
    """Multiply a matrix by a MultiVector."""
    assert x.nvec() == y.nvec(), "x and y have non-matching number of vectors"
    if hasattr(A,'matMvMult'):
        A.matMvMult(x,y)
    else:
        for i in range(x.nvec()):
            _mat_mult(A, x[i], y[i])

def MatMvTranspmult(A, x, y):
    """Multiply a matrix transpose by a MultiVector."""
    assert x.nvec() == y.nvec(), "x and y have non-matching number of vectors"
    if hasattr(A,'matMvTranspmult'):
        A.matMvTranspmult(x,y)
    elif hasattr(A, "transpmult"):
        for i in range(x.nvec()):
            A.transpmult(_as_vec(x[i]), _as_vec(y[i]))
    elif hasattr(A, "multTranspose"):
        for i in range(x.nvec()):
            A.multTranspose(_as_vec(x[i]), _as_vec(y[i]))
    else:
        raise AssertionError("A does not have transpmult or multTranspose method implemented")
        
def MvDSmatMult(X, A, Y):
    """Multiply a MultiVector by a dense matrix."""
    assert X.nvec() == A.shape[0], "X Number of vecs incompatible with number of rows in A"
    assert Y.nvec() == A.shape[1], "Y Number of vecs incompatible with number of cols in A"
    for j in range(Y.nvec()):
        _vec_zero(Y[j])
        X.reduce(Y[j], A[:,j].flatten())


def _as_vec(v):
    """Internal helper."""
    return v.vec if isinstance(v, VecWrapper) else v


def _vec_duplicate(v):
    """Internal helper."""
    vec = _as_vec(v)
    if hasattr(vec, "duplicate"):
        return vec.duplicate()
    if hasattr(vec, "copy"):
        return vec.copy()
    raise TypeError("Vector type does not support duplicate or copy")


def _vec_copy(v):
    """Internal helper."""
    vec = _as_vec(v)
    if hasattr(vec, "copy"):
        return vec.copy()
    dup = _vec_duplicate(vec)
    if hasattr(dup, "axpy"):
        dup.axpy(1.0, vec)
    return dup


def _vec_zero(v):
    """Internal helper."""
    vec = _as_vec(v)
    if hasattr(vec, "zeroEntries"):
        vec.zeroEntries()
    elif hasattr(vec, "set"):
        vec.set(0.0)
    else:
        vec.array[:] = 0.0
    if hasattr(vec, "assemblyBegin"):
        vec.assemblyBegin()
        vec.assemblyEnd()


def _vec_axpy(v, a, y):
    """Internal helper."""
    vec = _as_vec(v)
    vec.axpy(a, _as_vec(y))


def _vec_inner(v, y):
    """Internal helper."""
    vec = _as_vec(v)
    if hasattr(vec, "inner"):
        return vec.inner(_as_vec(y))
    return vec.dot(_as_vec(y))


def _vec_scale(v, a):
    """Internal helper."""
    vec = _as_vec(v)
    if hasattr(vec, "scale"):
        vec.scale(a)
    else:
        vec.array[:] *= a
        if hasattr(vec, "assemblyBegin"):
            vec.assemblyBegin()
            vec.assemblyEnd()


def _vec_norm(v, norm_type):
    """Internal helper."""
    vec = _as_vec(v)
    if isinstance(vec, PETSc.Vec):
        mapping = {
            "l2": PETSc.NormType.NORM_2,
            "linf": PETSc.NormType.NORM_INFINITY,
            "l1": PETSc.NormType.NORM_1,
        }
        return vec.norm(mapping.get(norm_type, PETSc.NormType.NORM_2))
    return vec.norm(norm_type)


def _vec_pointwise_mult(a, b):
    """Internal helper."""
    va = _as_vec(a)
    vb = _as_vec(b)
    out = va.duplicate()
    out.pointwiseMult(va, vb)
    return out


def _mat_mult(A, x, y):
    """Internal helper."""
    vx = _as_vec(x)
    vy = _as_vec(y)
    A.mult(vx, vy)
