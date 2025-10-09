"""
Parse Wannier90 output files.

This module implements several parsing functions for reading Wannier90 output files.
The :py:func:`~pengwann.io.read` function is a convenient wrapper for automatically
parsing all the data required to construct an instance of the
:py:class:`~pengwann.descriptors.DescriptorCalculator` class.
"""

# Copyright (C) 2024-2025 Patrick J. Taylor

# This file is part of pengWann.
#
# pengWann is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# pengWann is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with pengWann.
# If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray

from pengwann._geometry import _build_distance_and_image_matrices
from pengwann.electronic_structure import Basis
from pengwann.geometry import Geometry
from pengwann.type_aliases import Hamiltonian


def read_wannier90_outputs(
    seedname: str, path: str = "."
) -> tuple[Geometry, Basis, NDArray[np.float64]]:
    geometry = read_geometry(seedname, path)
    basis = read_basis(seedname, path)

    num_kpoints, num_bands = basis.u.shape[:-1]
    eigenvalues = read_eigenvalues(f"{path}/{seedname}.eig", num_bands, num_kpoints)

    return geometry, basis, eigenvalues


def read_geometry(seedname: str, path: str = ".") -> Geometry:
    symbols, cart_coords = read_xyz(f"{path}/{seedname}_centres.xyz")
    cell = read_cell(f"{path}/{seedname}.wout")

    if "X" not in symbols:
        raise ValueError(
            f'No Wannier centres ("X" atoms) found in {path}/{seedname}_centres.xyz.'
        )

    frac_coords = np.transpose(np.linalg.inv(cell) @ cart_coords)

    distance_matrix, image_matrix = _build_distance_and_image_matrices(
        frac_coords, cell
    )

    def assign_wannier_indices(
        symbols: tuple[str, ...], distance_matrix: NDArray[np.float64]
    ) -> tuple[tuple[int, ...], ...]:
        wannier_indices: list[int] = []
        atom_indices: list[int] = []
        for idx, symbol in enumerate(symbols):
            if symbol == "X":
                wannier_indices.append(idx)

            else:
                atom_indices.append(idx)

        num_wann = len(wannier_indices)
        wannier_assignments: list[list[int]] = [[] for _ in symbols]
        for i in wannier_indices:
            distances = distance_matrix[i, num_wann:]
            min_idx = distances.argmin() + num_wann

            wannier_assignments[min_idx].append(i)

        return tuple(tuple(indices) for indices in wannier_assignments)

    wannier_assignments = assign_wannier_indices(symbols, distance_matrix)

    return Geometry(
        symbols,
        frac_coords,
        cell,
        distance_matrix,
        image_matrix,
        wannier_assignments,
    )


def read_basis(seedname: str, path: str = ".") -> Basis:
    u, kpoints = read_unitary_matrices(f"{path}/{seedname}_u.mat")
    if os.path.isfile(f"{path}/{seedname}_u_dis.mat"):
        u_dis, _ = read_unitary_matrices(f"{path}/{seedname}_u_dis.mat")
        u = u_dis @ u

    return Basis(u, kpoints)


def read_eigenvalues(
    path: str,
    num_bands: int,
    num_kpoints: int,
) -> NDArray[np.float64]:
    """
    Parse the Kohn-Sham eigenvalues from a Wannier90 .eig file.

    Parameters
    ----------
    path : str
        The filepath to seedname.eig.
    num_bands : int
        The number of bands used in the prior Wannier90 calculation.
    num_kpoints : int
        The number of k-points used in the prior Wanner90 calculation.

    Returns
    -------
    eigenvalues : ndarray of float
        The Kohn-Sham eigenvalues.
    """
    with open(path, "r") as stream:
        lines = stream.readlines()

    eigenvalues = np.zeros((num_bands, num_kpoints))

    n_lines = range(num_bands)
    k_lines = [idx * num_bands for idx in range(num_kpoints)]

    for n, n_line in enumerate(n_lines):
        for k, k_line in enumerate(k_lines):
            eigenvalue = float(lines[n_line + k_line].split()[-1])

            eigenvalues[n, k] = eigenvalue

    return eigenvalues


def read_unitary_matrices(
    path: str,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """
    Parse the unitary matrices U^k from a Wannier90 _u.mat file.

    Parameters
    ----------
    path : str
        The filepath to seedname_u.mat or seedname_u_dis.mat.

    Returns
    -------
    u : ndarray of complex
        The unitary matrices U^k.
    kpoints : ndarray of float
        The k-point mesh used in the prior Wannier90 calculation.
    """
    with open(path, "r") as stream:
        lines = stream.readlines()

    num_kpoints, num_wann, num_bands = [int(string) for string in lines[1].split()]

    u = np.zeros((num_kpoints, num_bands, num_wann), dtype=np.complex128)
    kpoints = np.zeros((num_kpoints, 3))

    k_lines = (idx * (num_wann * num_bands + 2) + 4 for idx in range(num_kpoints))
    n_lines = [idx for idx in range(num_bands)]
    w_lines = [idx * num_bands for idx in range(num_wann)]

    for k, k_line in enumerate(k_lines):
        kpoints[k] = [float(string) for string in lines[k_line - 1].split()]

        for n, n_line in enumerate(n_lines):
            for w, w_line in enumerate(w_lines):
                real, imaginary = [
                    float(string) for string in lines[k_line + n_line + w_line].split()
                ]

                u[k, n, w] = complex(real, imaginary)

    return u, kpoints


def read_hamiltonian(path: str) -> Hamiltonian:
    """
    Parse the Wannier Hamiltonian from a Wannier90 seedname_hr.dat file.

    Parameters
    ----------
    path : str
        The filepath to seedname_hr.dat.

    Returns
    -------
    h : dict of {3-length tuple of int : ndarray of complex} pairs.
        The Wannier Hamiltonian.
    """
    with open(path, "r") as stream:
        lines = stream.readlines()

    num_wann = int(lines[1])
    num_rpoints = int(lines[2])

    start_idx = int(np.ceil(num_rpoints / 15)) + 3

    h: Hamiltonian = {}

    for line in lines[start_idx:]:
        data = line.split()
        bl = tuple([int(string) for string in data[:3]])

        assert len(bl) == 3

        if bl not in h.keys():
            h[bl] = np.zeros((num_wann, num_wann), dtype=np.complex128)

        m, n = [int(string) - 1 for string in data[3:5]]
        real, imaginary = [float(string) for string in data[5:]]

        h[bl][m, n] = complex(real, imaginary)

    return h


def read_cell(path: str) -> NDArray[np.float64]:
    """
    Parse a Wannier90 seedname.wout file to extract the cell vectors.

    Parameters
    ----------
    path : str
        The filepath to seedname.wout.

    Returns
    -------
    cell : ndarray of float
        The cell vectors.
    """
    with open(path, "r") as stream:
        lines = stream.readlines()

    cell_list: list[list[float]] = []
    for idx, line in enumerate(lines):
        if "Lattice Vectors (Ang)" in line:
            for cell_line in lines[idx + 1 : idx + 4]:
                cell_vector = [float(component) for component in cell_line.split()[1:]]

                cell_list.append(cell_vector)

            break

    cell = np.array(cell_list)

    return cell


def read_xyz(path: str) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """
    Parse the symbols and coordinates from a Wannier90 seedname_centres.xyz file.

    Parameters
    ----------
    path : str
        The filepath to seedname_centres.xyz

    Returns
    -------
    symbols : tuple of str
        The elemental symbol for each Wannier centre or atom in the xyz file.

    coords : ndarray of float
        The cartesian coordinates for each Wannier centre or atom in the xyz file.
    """
    with open(path, "r") as stream:
        lines = stream.readlines()

    start_idx = 2

    symbols_list: list[str] = []
    coords_list: list[list[float]] = []
    for line in lines[start_idx:]:
        data = line.split()

        symbol = str(data[0]).capitalize()
        site_coords = [float(coord) for coord in data[1:]]

        symbols_list.append(symbol)
        coords_list.append(site_coords)

    symbols = tuple(symbols_list)
    coords = np.transpose(coords_list)

    return symbols, coords
