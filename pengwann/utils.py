"""
This module contains some miscellaneous utility functions required elsewhere in the
codebase.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable
from multiprocessing.shared_memory import SharedMemory
from numpy.typing import NDArray
from pengwann.occupation_functions import fixed
from pymatgen.core import Structure
from scipy.integrate import trapezoid  # type: ignore
from typing import Callable


def assign_wannier_centres(geometry: Structure) -> None:
    """
    Assign Wannier centres to atoms based on a closest distance criterion.

    A :code:`"wannier_centres"` site property will be added to the input `geometry`,
    associating each atom in the structure with a sequence of indices. These indices
    refer to the order of atoms in `geometry` and associate each atom with the Wannier
    centres to which it is closer than any other atom.

    Parameters
    ----------
    geometry : Structure
        A Pymatgen Structure object containing the structure itself as well as the
        positions of the Wannier centres (as "X" atoms).

    Returns
    -------
    None
    """
    wannier_indices, atom_indices = [], []
    for idx in range(len(geometry)):
        symbol = geometry[idx].species_string

        if symbol == "X0+":
            wannier_indices.append(idx)

        else:
            atom_indices.append(idx)

    if not wannier_indices:
        raise ValueError(
            'No Wannier centres ("X" atoms) found in the input Structure object.'
        )

    distance_matrix = geometry.distance_matrix

    wannier_centres_list: list[list[int]] = [[] for idx in range(len(geometry))]
    for i in wannier_indices:
        min_distance, min_idx = np.inf, 2 * len(geometry)

        for j in atom_indices:
            distance = distance_matrix[i, j]

            if distance < min_distance:
                min_distance = distance
                min_idx = j

        wannier_centres_list[i].append(min_idx)
        wannier_centres_list[min_idx].append(i)

    wannier_centres = tuple([tuple(indices) for indices in wannier_centres_list])
    geometry.add_site_property("wannier_centres", wannier_centres)


def get_atom_indices(
    geometry: Structure, symbols: tuple[str, ...]
) -> dict[str, tuple[int, ...]]:
    """
    Categorise the site indices of a Pymatgen Structure according to atomic species.

    Parameters
    ----------
    geometry : Structure
        A Pymatgen Structure object.
    symbols : tuple[str, ...]
        The atomic species to associate site indices with.

    Returns
    -------
    atom_indices : dict[str, tuple[int, ...]]
        The site indices categorised by atomic species.
    """
    atom_indices_list: dict[str, list[int]] = {}
    for symbol in symbols:
        atom_indices_list[symbol] = []

    for idx, atom in enumerate(geometry):
        symbol = atom.species_string
        if symbol in symbols:
            atom_indices_list[symbol].append(idx)

    atom_indices = {}
    for symbol, indices in atom_indices_list.items():
        atom_indices[symbol] = tuple(indices)

    return atom_indices


def get_occupation_matrix(
    eigenvalues: NDArray[np.float64],
    mu: float,
    nspin: int,
    occupation_function: Callable = fixed,
    **function_kwargs,
) -> NDArray[np.float64]:
    """
    Compute an occupation matrix.

    Parameters
    ----------
    eigenvalues : ndarray[float]
        The Kohn-Sham eigenvalues.
    mu : float
        The Fermi level.
    nspin : int
        The number of electrons per fully-occupied Kohn-Sham state. For
        non-spin-polarised calculations set to 2, for spin-polarised calculations set
        to 1.
    occupation_function : callable, optional
        The occupation function used to calculate the occupation matrix. Defaults to
        :py:func:`~pengwann.occupation_functions.fixed` (i.e. fixed occupations).
    **function_kwargs
        Additional keyword arguments to be passed to `occupation_function`.

    Returns
    -------
    occupation_matrix : ndarray[float]
        The occupation matrix.

    Notes
    -----
    Ideally the occupation matrix should be read in directly from the ab initio code
    (in which case this function is redundant). Failing that, the occupation matrix can
    be reconstructed so long as the correct occupation function is used.

    Various pre-defined occupation functions (Gaussian, Marzari-Vanderbilt etc) can be
    found in the :py:mod:`~pengwann.occupation_functions` module. If none of these
    match the occupation function used by the ab initio code, a custom occupation
    function can be defined and passed as `occupation_function` (so long as it takes
    `eigenvalues` and `mu` as the first two positional arguments).
    """
    occupation_matrix = occupation_function(eigenvalues, mu, **function_kwargs)

    occupation_matrix *= nspin

    return occupation_matrix.T


def parse_id(identifier: str) -> tuple[str, int]:
    """
    Parse an atom identifier (e.g. "Ga1") and return the symbol and index separately.

    Parameters
    ----------
    identifier : str
        The identifier to be parsed.

    Returns
    -------
    symbol : str
        The symbol from the id.
    index : int
        The index from the id.
    """
    for i, character in enumerate(identifier):
        if character.isdigit():
            symbol = identifier[:i]
            index = int(identifier[i:])
            break

    return symbol, index


def integrate(
    energies: NDArray[np.float64], descriptor: NDArray[np.float64], mu: float
) -> np.float64 | NDArray[np.float64]:
    """
    Integrate a energy-resolved descriptor up to the Fermi level.

    Parameters
    ----------
    energies : ndarray[float]
        The discrete energies at which the descriptor has been evaluated.
    descriptor : ndarray[float]
        The descriptor to be integrated.
    mu : float
        The Fermi level.

    Returns
    -------
    integral : float | ndarray[float]
        The integrated descriptor.
    """
    for idx, energy in enumerate(energies):
        if energy > mu:
            fermi_idx = idx
            break

    integral = trapezoid(descriptor[:fermi_idx], energies[:fermi_idx], axis=0)

    return np.float64(integral)


def allocate_shared_memory(
    keys: Iterable[str], data: Iterable[NDArray]
) -> tuple[dict[str, tuple[tuple[int, ...], np.dtype]], list[SharedMemory]]:
    """
    Allocate one or more blocks of shared memory and populate them with numpy arrays.

    Parameters
    ----------
    keys : iterable[str]
        A sequence of strings identifying each array to be put into shared memory.
    data : iterable[ndarray]
        The arrays to be put into shared memory.

    Returns
    -------
    memory_metadata : dict[str, tuple[tuple[int, ...], np.dtype]]
        A dictionary containing metadata for each allocated block of shared memory. The
        keys are set by `keys` and the values are a tuple containing the shape and dtype
        of the corresponding array.
    memory_handles : list[SharedMemory]
        A sequence of SharedMemory objects (returned to allow easy access to the
        :code:`unlink` method.
    """
    memory_metadata = {}
    memory_handles = []
    for memory_key, to_share in zip(keys, data):
        memory_metadata[memory_key] = (to_share.shape, to_share.dtype)
        flattened_array = to_share.flatten()

        shared_memory = SharedMemory(
            name=memory_key, create=True, size=flattened_array.nbytes
        )
        buffered_array = np.ndarray(
            flattened_array.shape, dtype=flattened_array.dtype, buffer=shared_memory.buf
        )  # type: NDArray
        buffered_array[:] = flattened_array[:]

        memory_handles.append(shared_memory)

    return memory_metadata, memory_handles
