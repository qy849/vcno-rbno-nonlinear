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

"""Notebook display helpers and small utilities."""

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as cls
import numpy as np
from matplotlib import animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dolfinx import fem, mesh as dmesh
from .vector2function import assign_vec_to_function

"""
Plotting utilities for notebooks (dolfinx).
"""


def _mesh2triang(mesh):
    """Convert a dolfinx mesh to a matplotlib Triangulation."""
    coords = mesh.geometry.x
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    conn = mesh.topology.connectivity(tdim, 0)
    num_cells = mesh.topology.index_map(tdim).size_local
    cells = np.array([conn.links(i) for i in range(num_cells)], dtype=np.int32)
    if cells.shape[1] == 4:
        cells = np.vstack([cells[:, [0, 1, 2]], cells[:, [0, 2, 3]]])
    elif cells.shape[1] != 3:
        raise AttributeError("Only triangular or quadrilateral cells are supported for plotting")
    return tri.Triangulation(coords[:, 0], coords[:, 1], cells)


def _mplot_cellfunction(cellfn):
    """Plot a MeshTags object using matplotlib."""
    mesh = cellfn.mesh
    tri_obj = _mesh2triang(mesh)
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    C = np.zeros(num_cells, dtype=np.int32)
    if hasattr(cellfn, "values") and hasattr(cellfn, "indices"):
        C[cellfn.indices] = cellfn.values
    elif hasattr(cellfn, "array"):
        C = np.asarray(cellfn.array(), dtype=np.int32)
    return plt.tripcolor(tri_obj, facecolors=C)


def _mplot_function(f, vmin, vmax, logscale):
    """Plot a dolfinx Function on a 2D mesh."""
    mesh = f.function_space.mesh
    if mesh.geometry.dim != 2:
        raise AttributeError("Mesh must be 2D")
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    value_size = f.function_space.dofmap.index_map_bs

    if f.x.array.size == num_cells * value_size and value_size == 1:
        C = f.x.array[:num_cells].real
        if logscale:
            return plt.tripcolor(_mesh2triang(mesh), C, vmin=vmin, vmax=vmax, norm=cls.LogNorm())
        return plt.tripcolor(_mesh2triang(mesh), C, vmin=vmin, vmax=vmax)

    coords = mesh.geometry.x[:, :2]
    cells = _vertex_cells(mesh)
    values = _eval_function(f, coords, cells)
    if value_size == 1:
        C = values[:, 0]
        if logscale:
            return plt.tripcolor(_mesh2triang(mesh), C, vmin=vmin, vmax=vmax, norm=cls.LogNorm())
        return plt.tripcolor(_mesh2triang(mesh), C, shading="gouraud", vmin=vmin, vmax=vmax)
    if value_size == 2:
        X = coords[:, 0]
        Y = coords[:, 1]
        U = values[:, 0]
        V = values[:, 1]
        C = np.sqrt(U * U + V * V)
        return plt.quiver(
            X,
            Y,
            U,
            V,
            C,
            units="x",
            headaxislength=7,
            headwidth=7,
            headlength=7,
            scale=4,
            pivot="middle",
        )
    raise AttributeError("Only scalar and 2D vector functions are supported")


def _vertex_cells(mesh):
    """Map each vertex to an incident cell index."""
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(0, tdim)
    v2c = mesh.topology.connectivity(0, tdim)
    num_vertices = mesh.geometry.x.shape[0]
    cells = np.zeros(num_vertices, dtype=np.int32)
    for i in range(num_vertices):
        links = v2c.links(i)
        cells[i] = links[0] if links.size > 0 else -1
    return cells


def _eval_function(f, coords, cells):
    """Evaluate a function at vertex coordinates using cell ids."""
    values = np.zeros((coords.shape[0], f.function_space.dofmap.index_map_bs), dtype=np.float64)
    valid = cells >= 0
    if np.any(valid):
        eval_coords = coords[valid]
        if eval_coords.shape[1] == 2:
            eval_coords = np.hstack((eval_coords, np.zeros((eval_coords.shape[0], 1))))
        vals = f.eval(eval_coords, cells[valid])
        if isinstance(vals, tuple):
            vals = vals[0]
        values[valid, :] = np.asarray(vals)
    return values


def _vec_min_max(vec):
    """Return min/max values for a PETSc-like vector."""
    if hasattr(vec, "array"):
        arr = np.asarray(vec.array)
        if arr.size == 0:
            return 0.0, 0.0
        return float(arr.min()), float(arr.max())
    vmin = vec.min()
    vmax = vec.max()
    if isinstance(vmin, tuple):
        vmin = vmin[0]
    if isinstance(vmax, tuple):
        vmax = vmax[0]
    return float(vmin), float(vmax)


def _add_colorbar(mappable, ax, pad=0.02, size="3%", height="92%"):
    """Attach a colorbar inset to an axis."""
    cax = inset_axes(
        ax,
        width=size,
        height=height,
        loc="center left",
        bbox_to_anchor=(1.0 + pad, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    return plt.colorbar(mappable, cax=cax)


def plot(obj, colorbar=True, subplot_loc=None, mytitle=None, show_axis="off", vmin=None, vmax=None, logscale=False, cmap=None, fontsize=16):
    """
    Plot a generic dolfinx object (if supported)
    """
    if subplot_loc is not None:
        plt.subplot(subplot_loc)
    if isinstance(obj, fem.Function):
        pp = _mplot_function(obj, vmin, vmax, logscale)
    elif isinstance(obj, dmesh.MeshTags):
        pp = _mplot_cellfunction(obj)
    elif isinstance(obj, dmesh.Mesh):
        if obj.geometry.dim != 2:
            raise AttributeError("Mesh must be 2D")
        pp = plt.triplot(_mesh2triang(obj), color="#808080")
        colorbar = False
    else:
        raise AttributeError(f"Failed to plot {type(obj)}")

    ax = plt.gca()
    plt.axis(show_axis)
    ax.set_aspect("equal", adjustable="box")

    if mytitle is not None:
        ax.set_title(mytitle, fontsize=fontsize, pad=2)

    if colorbar:
        _add_colorbar(pp, ax)

    plt.set_cmap(cmap or "viridis")
    return pp


def multi1_plot(objs, titles, same_colorbar=True, show_axis="off", logscale=False, vmin=None, vmax=None, cmap=None, fontsize=16):
    """
    Plot a list of generic dolfinx objects in a single row
    """
    if vmin is None and vmax is None and same_colorbar:
        vmin = 1e30
        vmax = -1e30
        for f in objs:
            if isinstance(f, fem.Function):
                vals = f.x.array
                vmin = min(vmin, vals.min())
                vmax = max(vmax, vals.max())

    nobj = len(objs)
    if nobj == 1:
        plt.figure(figsize=(7.5, 5))
        subplot_loc = 110
    elif nobj == 2:
        plt.figure(figsize=(13, 5))
        subplot_loc = 120
    elif nobj == 3:
        plt.figure(figsize=(18, 4))
        subplot_loc = 130
    else:
        raise AttributeError("Too many figures")

    for i in range(nobj):
        if isinstance(cmap, (list, tuple)):
            cmapi = cmap[i]
        else:
            cmapi = cmap
        plot(
            objs[i],
            colorbar=True,
            subplot_loc=(subplot_loc + i + 1),
            mytitle=titles[i],
            show_axis=show_axis,
            vmin=vmin,
            vmax=vmax,
            logscale=logscale,
            cmap=cmapi,
            fontsize=fontsize,
        )


def plot_pts(points, values, colorbar=True, subplot_loc=None, mytitle=None, show_axis="on", vmin=None, vmax=None, xlim=(0, 1), ylim=(0, 1), cmap=None, fontsize=16):
    """
    Plot a cloud of points
    """
    if subplot_loc is not None:
        plt.subplot(subplot_loc)
    if hasattr(values, "array"):
        vals = values.array
    else:
        vals = np.asarray(values)
    pp = plt.scatter(points[:, 0], points[:, 1], c=vals, marker=",", s=20, vmin=vmin, vmax=vmax)

    ax = plt.gca()
    plt.axis(show_axis)
    ax.set_aspect("equal", adjustable="box")

    if mytitle is not None:
        ax.set_title(mytitle, fontsize=fontsize, pad=2)

    if colorbar:
        _add_colorbar(pp, ax)

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.set_cmap(cmap or "viridis")
    return pp


def show_solution(Vh, ic, state, same_colorbar=True, colorbar=True, mytitle=None, show_axis="off", logscale=False, times=None, cmap=None):
    """
    Plot a TimeDependentVector at specified time steps
    """
    if times is None:
        times = [0, 0.4, 1.0, 2.0, 3.0, 4.0]
    state.store(ic, 0)
    nrows = int(np.ceil(len(times) / 3.0))
    subplot_loc = nrows * 100 + 30
    plt.figure(figsize=(18, 4 * nrows))

    if mytitle is None:
        title_stamp = "Time {0}s"
    else:
        title_stamp = mytitle + " at time {0}s"

    vmin = None
    vmax = None

    if same_colorbar:
        vmin = 1e30
        vmax = -1e30
        for s in state.data:
            smin, smax = _vec_min_max(s)
            if smax > vmax:
                vmax = smax
            if smin < vmin:
                vmin = smin

    counter = 1
    myu = fem.Function(Vh)
    times_arr = None
    if hasattr(state, "times") and len(state.times) > 0:
        times_arr = np.asarray(state.times, dtype=float)
    for i in times:
        t = float(i)
        idx = None
        if times_arr is not None:
            idx = int(np.argmin(np.abs(times_arr - t)))
            t = float(times_arr[idx])
        if idx is not None and hasattr(state, "data") and len(state.data) > idx:
            owned = myu.function_space.dofmap.index_map.size_local * myu.function_space.dofmap.index_map_bs
            myu.x.array[:owned] = state.data[idx].array[:owned]
            myu.x.scatter_forward()
        else:
            try:
                state.retrieve(myu.x, t)
            except Exception:
                continue
        plot(
            myu,
            subplot_loc=(subplot_loc + counter),
            mytitle=title_stamp.format(t),
            colorbar=colorbar,
            logscale=logscale,
            show_axis=show_axis,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        counter = counter + 1


def animate(Vh, state, same_colorbar=True, colorbar=True, subplot_loc=None, mytitle=None, show_axis="off", logscale=False):
    """
    Show animation for a TimeDependentVector
    """
    fig = plt.figure()
    vmin = None
    vmax = None

    if same_colorbar:
        vmin = 1e30
        vmax = -1e30
        for s in state.data:
            smin, smax = _vec_min_max(s)
            if smax > vmax:
                vmax = smax
            if smin < vmin:
                vmin = smin

    def my_animate(i):
        """Render a single animation frame."""
        time_stamp = "Time: {0:f} s"
        obj = fem.Function(Vh)
        assign_vec_to_function(obj, state.data[i])
        t = (mytitle or "") + time_stamp.format(state.times[i])
        plt.clf()
        return plot(obj, colorbar=colorbar, subplot_loc=subplot_loc, mytitle=t, show_axis=show_axis, vmin=vmin, vmax=vmax, logscale=logscale)

    return animation.FuncAnimation(fig, my_animate, np.arange(0, state.nsteps), blit=True)


def coarsen_v(fun, nx=16, ny=16):
    """Placeholder for coarsening a vector field."""
    raise NotImplementedError("coarsen_v is not supported.")


def plot_eigenvalues(d, mytitle=None, subplot_loc=None):
    """
    Plot eigenvalues
    """
    k = d.shape[0]
    if subplot_loc is not None:
        plt.subplot(subplot_loc)
    plt.plot(range(0, k), d, "b*", range(0, k), np.ones(k), "-r")
    plt.yscale("log")
    if mytitle is not None:
        plt.title(mytitle)


def plot_eigenvectors(Vh, U, mytitle, which=None, cmap=None):
    """
    Plot specified vectors in a MultiVector
    """
    if which is None:
        which = [0, 1, 2, 5, 10, 15]
    nrows = int(np.ceil(len(which) / 3.0))
    subplot_loc = nrows * 100 + 30
    plt.figure(figsize=(18, 4 * nrows))

    title_stamp = mytitle + " {0}"
    u = fem.Function(Vh)
    counter = 1
    for i in which:
        assert i < U.nvec()
        s = U[i].norm("linf")
        scale = 1.0 / s if s != 0 else 1.0
        if U[i].array[0] < 0:
            scale *= -1.0
        assign_vec_to_function(u, U[i], scale=scale)
        plot(u, subplot_loc=(subplot_loc + counter), mytitle=title_stamp.format(i), vmin=-1, vmax=1, cmap=cmap)
        counter = counter + 1
