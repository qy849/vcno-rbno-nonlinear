# Copyright (c) 2025-, Georgia Institute of Technology.
"""
Utilities for mapping Cartesian grid data to dolfinx meshes.
"""

import numpy as np
from dolfinx import mesh as dmesh


def numpy2MeshFunction(mesh, h, data, mfun=None, offsets=None):
    """Build MeshTags by sampling Cartesian data at cell midpoints."""
    h = np.asarray(h, dtype=float).ravel()
    data = np.asarray(data)
    gdim = mesh.geometry.dim
    if offsets is None:
        offsets = np.zeros(gdim, dtype=float)
    else:
        offsets = np.asarray(offsets, dtype=float).ravel()
    if h.size != gdim or offsets.size != gdim:
        raise ValueError("h and offsets must match mesh geometry dimension")

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    midpoints = dmesh.compute_midpoints(mesh, tdim, cells)

    idx = []
    for d in range(gdim):
        coord = midpoints[:, d]
        i = np.floor((coord - offsets[d]) / h[d]).astype(np.int64)
        i = np.clip(i, 0, data.shape[d] - 1)
        idx.append(i)

    if gdim == 1:
        values = data[idx[0]]
    elif gdim == 2:
        values = data[idx[0], idx[1]]
    else:
        values = data[idx[0], idx[1], idx[2]]

    values = np.asarray(values, dtype=np.int32)
    tags = dmesh.meshtags(mesh, tdim, cells, values)
    if mfun is not None and hasattr(mfun, "values"):
        try:
            mfun.values[:] = values
        except Exception:
            pass
    return tags


class _NumpyExpressionBase:
    """Base class for numpy-backed expressions on Cartesian grids."""
    def __init__(self, gdim, ncomp=None):
        """Initialize the expression with geometry dimension and components."""
        self._gdim = gdim
        self._ncomp = ncomp
        self._data = None
        self._h = None
        self._offsets = None

    def setData(self, data, *h):
        """Set the grid data and cell spacing."""
        self._data = np.asarray(data)
        self._h = np.asarray(h, dtype=float).ravel()
        if self._h.size != self._gdim:
            raise ValueError("h must have length equal to mesh dimension")
        if self._offsets is None:
            self._offsets = np.zeros(self._gdim, dtype=float)

    def setOffset(self, *offsets):
        """Set offsets for the grid origin."""
        self._offsets = np.asarray(offsets, dtype=float).ravel()
        if self._offsets.size != self._gdim:
            raise ValueError("offsets must have length equal to mesh dimension")

    def __call__(self, x):
        """Evaluate the expression at coordinates x."""
        if self._data is None or self._h is None:
            raise RuntimeError("setData must be called before evaluation")
        coords = np.asarray(x)
        if coords.ndim == 1:
            if self._gdim == 1:
                coords = coords.reshape(1, -1)
            else:
                coords = coords.reshape(self._gdim, -1)
        elif coords.ndim == 2:
            if coords.shape[0] >= self._gdim:
                coords = coords[: self._gdim, :]
            elif coords.shape[1] >= self._gdim:
                coords = coords.T
                coords = coords[: self._gdim, :]
            elif self._gdim == 1:
                coords = coords.reshape(1, -1)
            else:
                raise ValueError("Input points have wrong dimension")
        else:
            if coords.shape[-1] == self._gdim:
                coords = np.moveaxis(coords, -1, 0).reshape(self._gdim, -1)
            else:
                raise ValueError("Input points have wrong dimension")
        if coords.shape[0] < self._gdim:
            raise ValueError("Input points have wrong dimension")
        if coords.shape[0] > self._gdim:
            coords = coords[: self._gdim, :]
        if self._ncomp is None:
            return _interp_scalar(self._data, coords, self._h, self._offsets)
        return _interp_vector(self._data, coords, self._h, self._offsets).T


class NumpyScalarExpression1D(_NumpyExpressionBase):
    """Scalar expression from a 1D numpy grid."""
    def __init__(self):
        """Initialize a 1D scalar expression."""
        super().__init__(1, None)


class NumpyVectorExpression1D(_NumpyExpressionBase):
    """Vector expression from a 1D numpy grid."""
    def __init__(self, ncomp):
        """Initialize a 1D vector expression."""
        super().__init__(1, ncomp)


class NumpyScalarExpression2D(_NumpyExpressionBase):
    """Scalar expression from a 2D numpy grid."""
    def __init__(self):
        """Initialize a 2D scalar expression."""
        super().__init__(2, None)


class NumpyVectorExpression2D(_NumpyExpressionBase):
    """Vector expression from a 2D numpy grid."""
    def __init__(self, ncomp):
        """Initialize a 2D vector expression."""
        super().__init__(2, ncomp)


class NumpyScalarExpression3D(_NumpyExpressionBase):
    """Scalar expression from a 3D numpy grid."""
    def __init__(self):
        """Initialize a 3D scalar expression."""
        super().__init__(3, None)


class NumpyVectorExpression3D(_NumpyExpressionBase):
    """Vector expression from a 3D numpy grid."""
    def __init__(self, ncomp):
        """Initialize a 3D vector expression."""
        super().__init__(3, ncomp)


def _interp_scalar(data, coords, h, offsets):
    """Dispatch scalar interpolation by spatial dimension."""
    gdim = coords.shape[0]
    if gdim == 1:
        return _interp_scalar_1d(data, coords[0], h[0], offsets[0])
    if gdim == 2:
        return _interp_scalar_2d(data, coords[0], coords[1], h, offsets)
    return _interp_scalar_3d(data, coords[0], coords[1], coords[2], h, offsets)


def _interp_vector(data, coords, h, offsets):
    """Dispatch vector interpolation by spatial dimension."""
    gdim = coords.shape[0]
    if gdim == 1:
        return _interp_vector_1d(data, coords[0], h[0], offsets[0])
    if gdim == 2:
        return _interp_vector_2d(data, coords[0], coords[1], h, offsets)
    return _interp_vector_3d(data, coords[0], coords[1], coords[2], h, offsets)


def _interp_scalar_1d(data, x, h, offset):
    """Linearly interpolate scalar 1D data."""
    t = (x - offset) / h
    i0 = np.floor(t).astype(np.int64)
    i1 = i0 + 1
    n0 = data.shape[0]
    valid = (i1 >= 0) & (i0 < n0)
    i0 = np.clip(i0, 0, n0 - 1)
    i1 = np.clip(i1, 0, n0 - 1)
    alpha = t - i0
    vals = (1.0 - alpha) * data[i0] + alpha * data[i1]
    vals[~valid] = 0.0
    return vals


def _interp_vector_1d(data, x, h, offset):
    """Linearly interpolate vector 1D data."""
    t = (x - offset) / h
    i0 = np.floor(t).astype(np.int64)
    i1 = i0 + 1
    n0 = data.shape[0]
    valid = (i1 >= 0) & (i0 < n0)
    i0 = np.clip(i0, 0, n0 - 1)
    i1 = np.clip(i1, 0, n0 - 1)
    alpha = t - i0
    vals = (1.0 - alpha)[:, None] * data[i0] + alpha[:, None] * data[i1]
    vals[~valid] = 0.0
    return vals


def _interp_scalar_2d(data, x, y, h, offsets):
    """Bilinearly interpolate scalar 2D data."""
    tx = (x - offsets[0]) / h[0]
    ty = (y - offsets[1]) / h[1]
    i0 = np.floor(tx).astype(np.int64)
    j0 = np.floor(ty).astype(np.int64)
    i1 = i0 + 1
    j1 = j0 + 1
    n0, n1 = data.shape[:2]
    valid = (i1 >= 0) & (i0 < n0) & (j1 >= 0) & (j0 < n1)
    i0 = np.clip(i0, 0, n0 - 1)
    i1 = np.clip(i1, 0, n0 - 1)
    j0 = np.clip(j0, 0, n1 - 1)
    j1 = np.clip(j1, 0, n1 - 1)
    ax = tx - i0
    ay = ty - j0
    vals = (
        (1.0 - ax) * (1.0 - ay) * data[i0, j0]
        + (1.0 - ax) * ay * data[i0, j1]
        + ax * (1.0 - ay) * data[i1, j0]
        + ax * ay * data[i1, j1]
    )
    vals[~valid] = 0.0
    return vals


def _interp_vector_2d(data, x, y, h, offsets):
    """Bilinearly interpolate vector 2D data."""
    tx = (x - offsets[0]) / h[0]
    ty = (y - offsets[1]) / h[1]
    i0 = np.floor(tx).astype(np.int64)
    j0 = np.floor(ty).astype(np.int64)
    i1 = i0 + 1
    j1 = j0 + 1
    n0, n1 = data.shape[:2]
    valid = (i1 >= 0) & (i0 < n0) & (j1 >= 0) & (j0 < n1)
    i0 = np.clip(i0, 0, n0 - 1)
    i1 = np.clip(i1, 0, n0 - 1)
    j0 = np.clip(j0, 0, n1 - 1)
    j1 = np.clip(j1, 0, n1 - 1)
    ax = tx - i0
    ay = ty - j0
    vals = (
        (1.0 - ax)[:, None] * (1.0 - ay)[:, None] * data[i0, j0]
        + (1.0 - ax)[:, None] * ay[:, None] * data[i0, j1]
        + ax[:, None] * (1.0 - ay)[:, None] * data[i1, j0]
        + ax[:, None] * ay[:, None] * data[i1, j1]
    )
    vals[~valid] = 0.0
    return vals


def _interp_scalar_3d(data, x, y, z, h, offsets):
    """Trilinearly interpolate scalar 3D data."""
    tx = (x - offsets[0]) / h[0]
    ty = (y - offsets[1]) / h[1]
    tz = (z - offsets[2]) / h[2]
    i0 = np.floor(tx).astype(np.int64)
    j0 = np.floor(ty).astype(np.int64)
    k0 = np.floor(tz).astype(np.int64)
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1
    n0, n1, n2 = data.shape[:3]
    valid = (i1 >= 0) & (i0 < n0) & (j1 >= 0) & (j0 < n1) & (k1 >= 0) & (k0 < n2)
    i0 = np.clip(i0, 0, n0 - 1)
    i1 = np.clip(i1, 0, n0 - 1)
    j0 = np.clip(j0, 0, n1 - 1)
    j1 = np.clip(j1, 0, n1 - 1)
    k0 = np.clip(k0, 0, n2 - 1)
    k1 = np.clip(k1, 0, n2 - 1)
    ax = tx - i0
    ay = ty - j0
    az = tz - k0
    vals = (
        (1.0 - ax) * (1.0 - ay) * (1.0 - az) * data[i0, j0, k0]
        + (1.0 - ax) * (1.0 - ay) * az * data[i0, j0, k1]
        + (1.0 - ax) * ay * (1.0 - az) * data[i0, j1, k0]
        + (1.0 - ax) * ay * az * data[i0, j1, k1]
        + ax * (1.0 - ay) * (1.0 - az) * data[i1, j0, k0]
        + ax * (1.0 - ay) * az * data[i1, j0, k1]
        + ax * ay * (1.0 - az) * data[i1, j1, k0]
        + ax * ay * az * data[i1, j1, k1]
    )
    vals[~valid] = 0.0
    return vals


def _interp_vector_3d(data, x, y, z, h, offsets):
    """Trilinearly interpolate vector 3D data."""
    tx = (x - offsets[0]) / h[0]
    ty = (y - offsets[1]) / h[1]
    tz = (z - offsets[2]) / h[2]
    i0 = np.floor(tx).astype(np.int64)
    j0 = np.floor(ty).astype(np.int64)
    k0 = np.floor(tz).astype(np.int64)
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1
    n0, n1, n2 = data.shape[:3]
    valid = (i1 >= 0) & (i0 < n0) & (j1 >= 0) & (j0 < n1) & (k1 >= 0) & (k0 < n2)
    i0 = np.clip(i0, 0, n0 - 1)
    i1 = np.clip(i1, 0, n0 - 1)
    j0 = np.clip(j0, 0, n1 - 1)
    j1 = np.clip(j1, 0, n1 - 1)
    k0 = np.clip(k0, 0, n2 - 1)
    k1 = np.clip(k1, 0, n2 - 1)
    ax = tx - i0
    ay = ty - j0
    az = tz - k0
    vals = (
        (1.0 - ax)[:, None] * (1.0 - ay)[:, None] * (1.0 - az)[:, None] * data[i0, j0, k0]
        + (1.0 - ax)[:, None] * (1.0 - ay)[:, None] * az[:, None] * data[i0, j0, k1]
        + (1.0 - ax)[:, None] * ay[:, None] * (1.0 - az)[:, None] * data[i0, j1, k0]
        + (1.0 - ax)[:, None] * ay[:, None] * az[:, None] * data[i0, j1, k1]
        + ax[:, None] * (1.0 - ay)[:, None] * (1.0 - az)[:, None] * data[i1, j0, k0]
        + ax[:, None] * (1.0 - ay)[:, None] * az[:, None] * data[i1, j0, k1]
        + ax[:, None] * ay[:, None] * (1.0 - az)[:, None] * data[i1, j1, k0]
        + ax[:, None] * ay[:, None] * az[:, None] * data[i1, j1, k1]
    )
    vals[~valid] = 0.0
    return vals
