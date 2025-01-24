"""
Various utility functions.

This module contains some miscellaneous utility functions required elsewhere in the
codebase. For the most part, this module is unlikely to be useful to end users, but
there are some niche use cases (hence why it is still documented).
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

import numpy as np
from collections.abc import Iterable
from multiprocessing.shared_memory import SharedMemory
from numpy.typing import NDArray
from pymatgen.core import Structure
from scipy.integrate import trapezoid


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


def integrate_descriptor(
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
        )
        buffered_array[:] = flattened_array[:]

        memory_handles.append(shared_memory)

    return memory_metadata, memory_handles
