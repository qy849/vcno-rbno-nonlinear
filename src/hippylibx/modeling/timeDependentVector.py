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

"""Time-indexed storage of PETSc vectors for transient problems."""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem


class TimeDependentVector:
    """
    A class to store time dependent vectors.
    Snapshots are stored/retrieved by specifying
    the time of the snapshot. Times at which the snapshot are taken must be
    specified in the constructor.
    """

    def __init__(self, times, tol=1e-10, mpi_comm=MPI.COMM_WORLD):
        """
        Constructor:

        - :code:`times`: time frame at which snapshots are stored.
        - :code:`tol`  : tolerance to identify the frame of the snapshot.
        """
        self.nsteps = len(times)
        self.data = []

        for _ in range(self.nsteps):
            self.data.append(PETSc.Vec().createMPI(0, comm=mpi_comm))

        self.times = times
        self.tol = tol
        self._mpi_comm = mpi_comm
        self.Vh = None


    def copy(self):
        """
        Return a copy of all the time frames and snapshots
        """
        res = TimeDependentVector(self.times, tol=self.tol, mpi_comm=self._mpi_comm)
        res.Vh = self.Vh
        res.data = []

        for v in self.data:
            res.data.append(v.copy())

        return res

    def initialize(self, Vh):
        """
        Initialize all the snapshots to be compatible
        with the function space :code:`Vh`.
        """
        template_fun = fem.Function(Vh)
        self.data = []

        base_vec = template_fun.x.petsc_vec
        for _ in range(self.nsteps):
            self.data.append(base_vec.duplicate())

        self._mpi_comm = Vh.mesh.comm  # update the communicator
        self.Vh = Vh  # store the function space

    def axpy(self, a, other):
        """
        Compute :math:`x = x + \\mbox{a*other}` snapshot per snapshot.
        """
        for i in range(self.nsteps):
            self.data[i].axpy(a, other.data[i])

    def zero(self):
        """
        Zero out each snapshot.
        """
        for d in self.data:
            d.set(0.0)

    def store(self, u, t):
        """
        Store snapshot :code:`u` relative to time :code:`t`.
        If :code:`t` does not belong to the list of time frame an error is raised.
        """
        i = 0
        while i < self.nsteps - 1 and 2 * t > self.times[i] + self.times[i + 1]:
            i += 1

        assert abs(t - self.times[i]) < self.tol

        self.data[i].set(0.0)
        self.data[i].axpy(1.0, u)

    def retrieve(self, u, t):
        """
        Retrieve snapshot :code:`u` relative to time :code:`t`.
        If :code:`t` does not belong to the list of time frame an error is raised.
        """
        i = 0
        while i < self.nsteps - 1 and 2 * t > self.times[i] + self.times[i + 1]:
            i += 1

        assert abs(t - self.times[i]) < self.tol

        u.set(0.0)
        u.axpy(1.0, self.data[i])

    def view(self, t):
        """
        Return a view of the snapshot at time :code:`t`.
        If :code:`t` does not belong to the list of time frame an error is raised.
        """
        i = 0
        while i < self.nsteps - 1 and 2 * t > self.times[i] + self.times[i + 1]:
            i += 1

        assert abs(t - self.times[i]) < self.tol

        return self.data[i]

    def norm(self, time_norm, space_norm):
        """
        Compute the space-time norm of the snapshot.
        """
        assert time_norm == "linf"
        norm_map = {
            "l2": PETSc.NormType.NORM_2,
            "linf": PETSc.NormType.NORM_INFINITY,
            "l1": PETSc.NormType.NORM_1,
        }
        pnorm = norm_map.get(space_norm, PETSc.NormType.NORM_2)
        s_norm = 0
        for i in range(self.nsteps):
            tmp = self.data[i].norm(pnorm)
            if tmp > s_norm:
                s_norm = tmp

        return s_norm

    def inner(self, other):
        """
        Compute the inner products: :math:`a+= (\\mbox{self[i]},\\mbox{other[i]})` for each snapshot.
        """
        assert (
            other.nsteps == self.nsteps
        ), "vectors do not have the same number of snapshots"
        a = 0.0
        for i in range(self.nsteps):
            a += self.data[i].dot(other.data[i])
        return a

    def element_wise_inner(self, other):
        """
        Compute the element-wise inner products: :math:`a[i] = (\\mbox{self[i]},\\mbox{other[i]})` for each snapshot.
        """
        assert (
            other.nsteps == self.nsteps
        ), "vectors do not have the same number of snapshots"
        out = np.zeros(self.nsteps)
        for i in range(self.nsteps):
            out[i] += self.data[i].dot(other.data[i])

        return out

    def matmul(self, mat, out):
        """
        Compute the matrix-vector product :math:`y = A*x` for each snapshot.
        """
        if hasattr(mat, "getSize"):
            _, ncols = mat.getSize()
        else:
            ncols = mat.size(1)
        vec_size = self.data[0].getSize()
        assert (
            ncols == vec_size
        ), f"Matrix and vector are not compatible. Matrix columns: {ncols} Vector size: {vec_size}."
        assert (
            self.nsteps == out.nsteps
        ), "time dependent vectors do not have the same number of snapshots"
        assert hasattr(mat, "mult"), "Matrix mat has no method: mult"

        for i in range(self.nsteps):
            mat.mult(self.data[i], out.data[i])

    def apply(self, mode):
        """
        Apply the changes to the vector.
        """
        for d in self.data:
            d.assemblyBegin()
            d.assemblyEnd()


    def gather_on_zero(self):
        """
        Gather the vector on process 0.
        """
        comm = self._mpi_comm
        gathered = []
        for d in self.data:
            local = d.array.copy()
            pieces = comm.gather(local, root=0)
            if comm.rank == 0:
                gathered.append(np.concatenate(pieces))
        if comm.rank == 0:
            return np.array(gathered).flatten()
        return np.array([])


    def __add__(self, other):
        if isinstance(other, TimeDependentVector):
            assert (
                other.nsteps == self.nsteps
            ), "vectors do not have the same number of snapshots"

            sum_vec = self.copy()
            sum_vec.axpy(1.0, other)

        elif isinstance(other, (int, float)):
            sum_vec = self.copy() 
            for d in sum_vec.data:
                d.shift(other)
        else:
            raise TypeError(f"Unsupported type {type(other)} for addition with TimeDependentVector")

        return sum_vec

    def __sub__(self, other):
        if isinstance(other, TimeDependentVector):
            assert (
                other.nsteps == self.nsteps
            ), "vectors do not have the same number of snapshots"

            diff_vec = self.copy()
            diff_vec.axpy(-1.0, other)

        elif isinstance(other, (int, float)):
            diff_vec = self.copy() 
            for d in diff_vec.data:
                d.shift(-other)
        else:
            raise TypeError(f"Unsupported type {type(other)} for subtraction with TimeDependentVector")

        return diff_vec


    def __mul__(self, other):
        if isinstance(other, (int, float)):
            prod_vec = self.copy() 
            for d in prod_vec.data:
                d.scale(other)
        else:
            raise TypeError(f"Unsupported type {type(other)} for multiplication with TimeDependentVector")

        return prod_vec


    def __iadd__(self, other):
        if isinstance(other, TimeDependentVector):
            assert (
                other.nsteps == self.nsteps
            ), "vectors do not have the same number of snapshots"
            self.axpy(1.0, other)

        elif isinstance(other, (int, float)):
            for d in self.data:
                d.shift(other)
        else:
            raise TypeError(f"Unsupported type {type(other)} for addition with TimeDependentVector")

        return self


    def __isub__(self, other):
        if isinstance(other, TimeDependentVector):
            assert (
                other.nsteps == self.nsteps
            ), "vectors do not have the same number of snapshots"
            self.axpy(-1.0, other)

        elif isinstance(other, (int, float)):
            for d in self.data:
                d.shift(-other)
        else:
            raise TypeError(f"Unsupported type {type(other)} for subtraction with TimeDependentVector")

        return self


    def __imul__(self, other):
        if isinstance(other, (int, float)):
            for d in self.data:
                d.scale(other)
        else:
            raise TypeError(f"Unsupported type {type(other)} for multiplication with TimeDependentVector")
        return self


    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        neg_vec = self.copy()
        for d in neg_vec.data:
            d.scale(-1.0)
        return neg_vec
