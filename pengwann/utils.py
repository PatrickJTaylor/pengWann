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
from typing import Any, Callable, Optional


def assign_wannier_centres(geometry: Structure) -> None:
    """
    Assign Wannier centres to atoms based on a closest distance criterion.

    Args:
        geometry (Structure): A Pymatgen Structure object containing the structure
            itself as well as the positions of the Wannier centres (as "X" atoms).

    Returns:
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
    Categorise all site indices of a Pymatgen Structure object according to the atomic
    species.

    Args:
        geometry (Structure): The Pymatgen Structure object.
        symbols (tuple[str, ...]): The atomic species to associate indices with.

    Returns:
        dict[str, tuple[int, ...]]: The site indices categorised by atomic species (as
        dictionary keys).
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
    occupation_function: Optional[Callable] = None,
    **function_kwargs,
) -> NDArray[np.float64]:
    """
    Calculate the occupation matrix.

    Args:
        eigenvalues (NDArray[np.float64]): The Kohn-Sham eigenvalues.
        mu (float): The Fermi level.
        nspin (int): The number of electrons per fully-occupied Kohn-Sham state.
        occupation_function (Optional[Callable]): The occupation function to be used to
            calculate the occupation matrix. Defaults to None (which means fixed
            occupations will be assumed).
        **function_kwargs: Additional keyword arguments to be passed to the occupation
            function in addition to the eigenvalues and the Fermi level.

    Returns:
        NDArray[np.float64]: The occupation matrix.

    Notes:
        Several pre-defined occupation functions may be imported from the
        :py:mod:`~pengwann.occupation_functions` module (Gaussian, Marzari-Vanderbilt
        etc).

        Alternatively, one may choose to use a custom occupation function, in which case
        it must take the eigenvalues and the Fermi level as the first two positional arguments.
    """
    if occupation_function is not None:
        occupation_matrix = occupation_function(eigenvalues, mu, **function_kwargs)

    else:
        occupation_matrix = fixed(eigenvalues, mu)

    occupation_matrix *= nspin

    return occupation_matrix.T


def parse_id(identifier: str) -> tuple[str, int]:
    """
    Parse an atom identifer (e.g. "Ga1") and return individually the elemental symbol
    and the index.

    Args:
        identifier (str): The atom indentifier to be parsed.

    Returns:
        tuple[str, int]:

        str: The elemental symbol for the atom.

        int: The identifying index for the atom.
    """
    for i, character in enumerate(identifier):
        if character.isdigit():
            symbol = identifier[:i]
            idx = int(identifier[i:])
            break

    return symbol, idx


def integrate(
    energies: NDArray[np.float64], descriptor: NDArray[np.float64], mu: float
) -> np.float64:
    """
    Integrate a given descriptor up to the Fermi level.

    Args:
        energies (NDArray[np.float64]): The energies at which the descriptor has been evaluated.
        descriptor (NDArray[np.float64]): The descriptor to be integrated.
        mu (float): The Fermi level.

    Returns:
        np.float64: The resulting integral.
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
