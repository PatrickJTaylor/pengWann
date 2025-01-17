"""
This module implements several parsing functions for reading Wannier90 output files.
The :py:func:`~pengwann.io.read` function is a convenient wrapper for automatically
parsing all the data required to construct an instance of the
:py:class:`~pengwann.dos.DOS` class.
"""

import os
import numpy as np
from numpy.typing import NDArray


def read(seedname: str, path: str = ".") -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.complex128],
    dict[tuple[int, ...], NDArray[np.complex128]],
]:
    """
    Wrapper function for reading in the main Wannier90 output files.

    Args:
        seedname (str): Wannier90 seedname (prefix for all output files).
        path: (str): Filepath to main Wannier90 output files. Defaults to '.' i.e. the
            current working directory.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128], dict[tuple[int, ...], NDArray[np.complex128]]]:

        NDArray[np.float64]: The k-points used in the prior DFT calculation.

        NDArray[np.float64]: The Kohn-Sham eigenvalues.

        NDArray[np.float64]: The unitary matrices :math:`U^{k}`.

        dict[tuple[int, ...], NDArray[np.complex128]]: The Hamiltonian in the Wannier basis.
    """
    u, kpoints = read_u(f"{path}/{seedname}_u.mat")
    if os.path.isfile(f"{path}/{seedname}_u_dis.mat"):
        u_dis, _ = read_u(f"{path}/{seedname}_u_dis.mat")
        u = (u_dis @ u).astype(np.complex128)

    h = read_hamiltonian(f"{path}/{seedname}_hr.dat")
    eigenvalues = read_eigenvalues(f"{path}/{seedname}.eig", u.shape[1], u.shape[0])

    return kpoints, eigenvalues, u, h


def read_eigenvalues(
    path: str,
    num_bands: int,
    num_kpoints: int,
) -> NDArray[np.float64]:
    """
    Read in the Kohn-Sham eigenvalues.

    Args:
        path (str): The filepath to seedname.eig.
        num_bands (int): The number of bands.
        num_kpoints (int): The number of k-points.

    Returns:
        NDArray[np.float64]: The Kohn-Sham eigenvalues.

    Notes:
        The output array is a num_bands x num_kpoints matrix.
    """
    eigenvalues_list = []

    with open(path, "r") as stream:
        lines = stream.readlines()

    block_indices = [idx * num_bands for idx in range(num_kpoints)]

    for column_idx in range(num_bands):
        row = []

        for block_idx in block_indices:
            eigenvalue = float(lines[column_idx + block_idx].split()[-1])

            row.append(eigenvalue)

        eigenvalues_list.append(row)

    eigenvalues = np.array(eigenvalues_list)

    return eigenvalues


def read_u(path: str) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    r"""
    Read in the unitary matrices :math:`U^{k}` that define the Wannier functions
    :math:`\ket{w_{nR}}` from the Kohn-Sham states :math:`\ket{\psi_{mk}}`.

    Args:
        path (str): The filepath to seedname_u.mat or seedname_u_dis.mat.

    Returns:
        tuple[NDArray[np.complex128], NDArray[np.float64]]:

        NDArray[np.complex128]: The unitary matrices :math:`U^{k}`.

        NDArray[np.float64]: The k-points corresponding to each :math:`U^{k}`.

    Notes:
        The output array is a num_kpoints x num_bands x num_wann array, each
        num_bands x num_wann block is a matrix :math:`U^{k}`.
    """
    u_list, kpoints_list = [], []

    with open(path, "r") as stream:
        lines = stream.readlines()

    num_kpoints, num_wann, num_bands = [int(string) for string in lines[1].split()]

    block_indices = [idx * (num_wann * num_bands + 2) + 4 for idx in range(num_kpoints)]
    column_indices = [idx * num_bands for idx in range(num_wann)]

    for block_idx in block_indices:
        u_k = []

        kpoint = [float(string) for string in lines[block_idx - 1].split()]
        kpoints_list.append(kpoint)

        for row_idx in range(num_bands):
            row = []

            for column_idx in column_indices:
                element_idx = block_idx + row_idx + column_idx
                real, imaginary = [
                    float(string) for string in lines[element_idx].split()
                ]

                row.append(complex(real, imaginary))

            u_k.append(row)

        u_list.append(u_k)

    u = np.array(u_list)
    kpoints = np.array(kpoints_list)

    return u, kpoints


def read_hamiltonian(path: str) -> dict[tuple[int, ...], NDArray[np.complex128]]:
    """
    Read in the Wannier Hamiltonian.

    Args:
        path (str): The filepath to seedname_hr.dat.

    Returns:
        dict[tuple[int, ...], NDArray[np.complex128]]: The Wannier Hamiltonian.

    Notes:
        H is a dictionary with keys corresponding to Bravais lattice vectors (in tuple
        form). Each value is a num_wann x num_wann matrix.
    """
    with open(path, "r") as stream:
        lines = stream.readlines()

    num_wann = int(lines[1])
    num_rpoints = int(lines[2])

    start_idx = int(np.ceil(num_rpoints / 15)) + 3

    h = {}  # type: dict[tuple[int, ...], NDArray[np.complex128]]

    for line in lines[start_idx:]:
        data = line.split()
        bl = tuple([int(string) for string in data[:3]])

        if bl not in h.keys():
            h[bl] = np.zeros((num_wann, num_wann), dtype=np.complex128)

        m, n = [int(string) - 1 for string in data[3:5]]
        real, imaginary = [float(string) for string in data[5:]]

        h[bl][m, n] = complex(real, imaginary)

    return h
