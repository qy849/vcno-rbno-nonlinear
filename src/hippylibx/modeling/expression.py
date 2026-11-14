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

"""Analytic expression helpers used by priors and test problems."""

import math
import types
import numpy as np


def _normalize_points(x, gdim):
    """Normalize input points to a (gdim, npoints) array."""
    coords = np.asarray(x)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    elif coords.ndim > 2:
        if coords.shape[-1] >= gdim:
            coords = np.moveaxis(coords, -1, 0).reshape(coords.shape[-1], -1)
    if coords.shape[0] < gdim and coords.shape[1] >= gdim:
        coords = coords.T
    if coords.shape[0] > gdim:
        coords = coords[:gdim, :]
    if coords.shape[0] != gdim:
        raise ValueError("Input points have wrong dimension")
    return coords


class AnisTensor2D:
    """
    Symmetric tensor in 2D of the form:

      [  sin(alpha)  cos(alpha) ] [theta0    0   ] [ sin(alpha)  -cos(alpha) ]
      [ -cos(alpha)  sin(alpha) ] [   0   theta1 ] [ cos(alpha)   sin(alpha) ]
    """

    def __init__(self):
        self.theta0 = 1.0
        self.theta1 = 1.0
        self.alpha = 0.0

    def set(self, theta0, theta1, alpha):
        self.theta0 = float(theta0)
        self.theta1 = float(theta1)
        self.alpha = float(alpha)


class Mollifier:
    """
    Mollifier of the form::

        f(x) = sum_i exp(-(e/(l*l))**(o/2))

    with anisotropic quadratic form e built from AnisTensor2D.
    """

    def __init__(self):
        self.l = 1.0
        self.o = 2.0
        self.theta0 = 1.0
        self.theta1 = 1.0
        self.alpha = 0.0
        self._locations = []

    def addLocation(self, x, y):
        self._locations.append((float(x), float(y)))

    def set(self, A, l, o):
        self.theta0 = 1.0 / float(A.theta0)
        self.theta1 = 1.0 / float(A.theta1)
        self.alpha = float(A.alpha)
        self.l = float(l)
        self.o = float(o)

    def __call__(self, x):
        coords = _normalize_points(x, 2)
        if not self._locations:
            return np.zeros(coords.shape[1], dtype=np.float64)

        sa = math.sin(self.alpha)
        ca = math.cos(self.alpha)
        c00 = self.theta0 * sa * sa + self.theta1 * ca * ca
        c01 = (self.theta0 - self.theta1) * sa * ca
        c11 = self.theta0 * ca * ca + self.theta1 * sa * sa

        dx0 = coords[0][None, :] - np.array([p[0] for p in self._locations])[:, None]
        dx1 = coords[1][None, :] - np.array([p[1] for p in self._locations])[:, None]
        e = dx0 * dx0 * c00 + dx1 * dx1 * c11 + 2.0 * dx0 * dx1 * c01
        vals = np.exp(-np.power(e / (self.l * self.l), 0.5 * self.o))
        return np.sum(vals, axis=0)


ExpressionModule = types.SimpleNamespace(AnisTensor2D=AnisTensor2D, Mollifier=Mollifier)
