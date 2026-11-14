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

"""Pointwise observation operators for dolfinx function spaces."""

import numpy as np
import basix
from petsc4py import PETSc
from dolfinx import geometry
from dolfinx.cpp.mesh import CellType


def _value_size(Vh):
    """
    Get the value size (number of components) for a FEniCSx function space.
    """
    # Fallback to index_map_bs (block size)
    try:
        bs = Vh.dofmap.index_map_bs
        if bs > 1:
            return bs
    except AttributeError:
        pass

    # Try basix_element first (FEniCSx 0.10+)
    try:
        basix_el = Vh.element.basix_element
        value_shape = basix_el.value_shape
        if len(value_shape) == 0:
            return 1
        size = 1
        for v in value_shape:
            size *= v
        return size
    except AttributeError:
        pass

    # Try element value_shape attribute
    value_shape = getattr(Vh.element, "value_shape", ())
    if value_shape is None:
        return 1
    try:
        shape = tuple(value_shape)
    except TypeError:
        shape = ()
    if len(shape) == 0:
        return 1
    size = 1
    for v in shape:
        size *= v
    return size


def _normalize_targets(targets, gdim):
    """Validate and reshape target coordinates to (n, gdim)."""
    pts = np.asarray(targets, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, gdim)
    if pts.shape[1] != gdim:
        raise ValueError("Targets have wrong dimension")
    return pts


def _find_cells(mesh, points):
    """Find a containing cell index for each point on the local rank."""
    tdim = mesh.topology.dim
    bb_tree = geometry.bb_tree(mesh, tdim)
    if points.shape[1] != 3:
        points3 = np.zeros((points.shape[0], 3), dtype=points.dtype)
        points3[:, : points.shape[1]] = points
    else:
        points3 = points
    candidates = geometry.compute_collisions_points(bb_tree, points3)
    colliding = geometry.compute_colliding_cells(mesh, candidates, points3)
    cells = np.full(points.shape[0], -1, dtype=np.int32)
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if links.size > 0:
            cells[i] = links[0]
    return cells


def _cell_jacobian(cell_type, cell_geom):
    """Return the Jacobian and determinant for a cell geometry."""
    if cell_type == CellType.interval:
        J = (cell_geom[1] - cell_geom[0]).reshape(1, 1)
    elif cell_type == CellType.triangle:
        J = np.column_stack((cell_geom[1] - cell_geom[0], cell_geom[2] - cell_geom[0]))
    elif cell_type == CellType.tetrahedron:
        J = np.column_stack(
            (cell_geom[1] - cell_geom[0], cell_geom[2] - cell_geom[0], cell_geom[3] - cell_geom[0])
        )
    else:
        raise ValueError("Unsupported cell type for Piola mapping")
    detJ = float(np.linalg.det(J)) if J.size > 1 else float(J[0, 0])
    return J, detJ


def _basis_values(Vh, cell_geom, xref, cell_perm):
    """Tabulate basis functions at xref, including DOF transforms and Piola maps."""
    basix_el = Vh.element.basix_element
    phi = basix_el.tabulate(0, xref)[0, 0]
    if not basix_el.dof_transformations_are_identity:
        phi_shape = phi.shape
        phi_flat = np.ascontiguousarray(phi.reshape(-1))
        Vh.element.T_apply(phi_flat, np.asarray([cell_perm], dtype=np.uint32), phi_shape[1])
        phi = phi_flat.reshape(phi_shape)
    if basix_el.map_type == basix.MapType.contravariantPiola:
        J, detJ = _cell_jacobian(Vh.mesh.topology.cell_type, cell_geom)
        phi = (J @ phi.T).T / detJ
    elif basix_el.map_type == basix.MapType.covariantPiola:
        J, _ = _cell_jacobian(Vh.mesh.topology.cell_type, cell_geom)
        K = np.linalg.inv(J)
        phi = (K.T @ phi.T).T
    return phi


def _cell_geometry(mesh, cell):
    """Extract geometry coordinates for a local cell."""
    geom_dofs = mesh.geometry.dofmap[cell]
    return mesh.geometry.x[geom_dofs, : mesh.geometry.dim]


def _as_components(Vh, components):
    """Validate component indices against the function space value size."""
    ncomp = _value_size(Vh)
    if components is None:
        return np.arange(ncomp, dtype=np.int32)
    comps = np.asarray(components, dtype=np.int32).ravel()
    if np.any(comps < 0) or np.any(comps >= ncomp):
        raise ValueError("Component index out of bounds")
    return comps


def assemblePointwiseObservation(Vh, targets, components=None, prune_and_sort=False):
    """
    Assemble the pointwise observation matrix for a dolfinx FunctionSpace.
    """
    mesh = Vh.mesh
    comm = mesh.comm
    gdim = mesh.geometry.dim
    points = _normalize_targets(targets, gdim)

    comps = _as_components(Vh, components)
    ncomp = comps.size
    npts = points.shape[0]
    nrows = npts * ncomp

    bs = Vh.dofmap.index_map_bs
    index_map = Vh.dofmap.index_map
    ndofs_local = index_map.size_local * bs
    ndofs = index_map.size_global * bs

    mat = PETSc.Mat().createAIJ(((PETSc.DECIDE, nrows), (ndofs_local, ndofs)), comm=comm)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    if num_cells > 0:
        nnz = int(Vh.dofmap.cell_dofs(0).size)
    else:
        nnz = int(Vh.element.space_dimension())
    mat.setPreallocationNNZ(max(1, nnz))
    mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    # Use owned cells only to avoid race conditions/redundancy
    t_imap = mesh.topology.index_map(mesh.topology.dim)
    num_owned_cells = t_imap.size_local

    mesh.topology.create_entity_permutations()
    cell_perms = mesh.topology.get_cell_permutation_info()
    cell_indices = _find_cells(mesh, points)

    for i, cell in enumerate(cell_indices):
        if cell < 0 or cell >= num_owned_cells:
            continue

        cell_dofs = Vh.dofmap.cell_dofs(cell)
        global_blocks = Vh.dofmap.index_map.local_to_global(cell_dofs)
        cell_geom = _cell_geometry(mesh, cell)
        xref = mesh.geometry.cmap.pull_back(points[i : i + 1], cell_geom)
        phi = _basis_values(Vh, cell_geom, xref, cell_perms[cell])

        row_base = i * ncomp
        if bs > 1 and phi.shape[1] == 1:
            for local_idx, gblock in enumerate(global_blocks):
                val = phi[local_idx, 0]
                for j, comp in enumerate(comps):
                    row = row_base + j
                    col = gblock * bs + comp
                    
                    mat.setValue(row, col, val, addv=False)
        else:
            for local_idx, gblock in enumerate(global_blocks):
                for j, comp in enumerate(comps):
                    row = row_base + j
                    col = gblock * bs + comp if bs > 1 else gblock
                    
                    mat.setValue(row, col, phi[local_idx, comp], addv=False)

    mat.assemblyBegin()
    mat.assemblyEnd()
    return mat


def assemblePointwiseLOSObservation(Vh, targets, los_coeff, components=None, prune_and_sort=False):
    """
    Assemble the pointwise line-of-sight observation matrix.
    """
    mesh = Vh.mesh
    comm = mesh.comm
    gdim = mesh.geometry.dim
    points = _normalize_targets(targets, gdim)
    los_coeff = np.asarray(los_coeff, dtype=np.float64)
    if los_coeff.shape[0] != points.shape[0]:
        raise ValueError("LOS coefficients must match number of targets")

    comps = _as_components(Vh, components)
    ncomp = comps.size
    npts = points.shape[0]
    nrows = npts

    bs = Vh.dofmap.index_map_bs
    index_map = Vh.dofmap.index_map
    ndofs_local = index_map.size_local * bs
    ndofs = index_map.size_global * bs

    mat = PETSc.Mat().createAIJ(((PETSc.DECIDE, nrows), (ndofs_local, ndofs)), comm=comm)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    if num_cells > 0:
        nnz = int(Vh.dofmap.cell_dofs(0).size)
    else:
        nnz = int(Vh.element.space_dimension())
    mat.setPreallocationNNZ(max(1, nnz))
    mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    t_imap = mesh.topology.index_map(mesh.topology.dim)
    num_owned_cells = t_imap.size_local
    
    mesh.topology.create_entity_permutations()
    cell_perms = mesh.topology.get_cell_permutation_info()
    cell_indices = _find_cells(mesh, points)
    for i, cell in enumerate(cell_indices):
        if cell < 0 or cell >= num_owned_cells:
            continue

        cell_dofs = Vh.dofmap.cell_dofs(cell)
        global_blocks = Vh.dofmap.index_map.local_to_global(cell_dofs)
        cell_geom = _cell_geometry(mesh, cell)
        xref = mesh.geometry.cmap.pull_back(points[i : i + 1], cell_geom)
        phi = _basis_values(Vh, cell_geom, xref, cell_perms[cell])

        for local_idx, gblock in enumerate(global_blocks):
            if bs > 1 and phi.shape[1] == 1:
                val = phi[local_idx, 0]
                for comp in comps:
                    coeff = los_coeff[i, comp]
                    col = gblock * bs + comp
                    mat.setValue(i, col, coeff * val, addv=False)
            else:
                for comp in comps:
                    coeff = los_coeff[i, comp]
                    col = gblock * bs + comp if bs > 1 else gblock
                    mat.setValue(i, col, coeff * phi[local_idx, comp], addv=False)

    mat.assemblyBegin()
    mat.assemblyEnd()
    return mat


def write_vtk(points, data, fname, varname="observation"):
    """
    Write a VTK PolyData file to visualize pointwise data.

    Inputs:
        - points:  locations of the points (numpy array of shape [npoints, 3]).
        - data:    pointwise values (numpy array of shape [npoints]).
        - fname:   filename for the .vtp file to export.
        - varname: name of the variable for the .vtk file.
    """
    ndim = points.shape[1]
    npoints = points.shape[0]

    assert npoints == data.shape[0]
    assert ndim == 3

    with open(fname, "w") as f:
        f.write('<VTKFile version="0.1" byte_order="LittleEndian" type="PolyData">\n')
        f.write("<PolyData>\n")
        f.write(f'<Piece NumberOfPoints="{npoints}" NumberOfVerts="{npoints}">\n')
        f.write("<Points>\n")
        f.write(f'<DataArray NumberOfComponents="{ndim}" format="ascii" type="Float32">\n')
        f.write(np.array_str(points, precision=9, suppress_small=True).replace("[", "").replace("]", ""))
        f.write("\n</DataArray>\n")
        f.write("</Points>\n")
        f.write("<Verts>\n")
        f.write('<DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write(np.array_str(np.arange(0, npoints)).replace("[", "").replace("]", ""))
        f.write("</DataArray>\n")
        f.write('<DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write(np.array_str(np.arange(1, npoints + 1)).replace("[", "").replace("]", ""))
        f.write("</DataArray>\n")
        f.write("</Verts>\n")
        f.write(f'<PointData Scalars="{varname}">\n')
        f.write(f'<DataArray format="ascii" type="Float32" Name="{varname}">\n')
        f.write(np.array_str(data, precision=9, suppress_small=True).replace("[", "").replace("]", ""))
        f.write("\n</DataArray>\n</PointData>\n<CellData>\n</CellData>\n</Piece>\n</PolyData>\n</VTKFile>")


def exportPointwiseObservation(Vh, B, data, fname, varname="observation"):
    """
    Write a VTK PolyData file to visualize pointwise observation data.

    Inputs:
        - Vh:      the finite element space used for observations.
        - B:       observation operator (PETSc matrix).
        - data:    PETSc vector containing the observation data.
        - fname:   filename for the file to export (without extension).
        - varname: name of the variable for the .vtk file.
    """
    from dolfinx import fem

    mesh = Vh.mesh
    comm = mesh.comm
    gdim = mesh.geometry.dim

    # Create coordinate functions
    coord_space = fem.functionspace(mesh, ("Lagrange", 1))
    xyz = []
    for d in range(gdim):
        coord_func = fem.Function(coord_space)

        def coord_expr(x, dim=d):
            return x[dim]

        coord_func.interpolate(coord_expr)
        coord_vec = coord_func.x.petsc_vec.copy()
        result = B.createVecLeft()
        B.mult(coord_vec, result)
        xyz.append(result)

    # Gather data on rank 0
    rank = comm.Get_rank()
    data_vec = data if isinstance(data, PETSc.Vec) else data

    # Get local data
    try:
        data_local = data_vec.getArray()
    except AttributeError:
        data_local = np.asarray(data_vec)

    xyz_local = [v.getArray() for v in xyz]

    # For serial case or single process
    if comm.Get_size() == 1:
        npoints = data_local.shape[0]
        points3 = np.zeros((npoints, 3), dtype=np.float64)
        for d in range(gdim):
            points3[:, d] = xyz_local[d]

        write_vtk(points3, data_local, fname + ".vtp", varname)
    else:
        # Gather on process 0
        data_gathered = comm.gather(data_local, root=0)
        xyz_gathered = [comm.gather(v, root=0) for v in xyz_local]

        if rank == 0:
            data_all = np.concatenate(data_gathered)
            xyz_all = [np.concatenate(v) for v in xyz_gathered]
            npoints = data_all.shape[0]
            points3 = np.zeros((npoints, 3), dtype=np.float64)
            for d in range(gdim):
                points3[:, d] = xyz_all[d]

            write_vtk(points3, data_all, fname + ".vtp", varname)
