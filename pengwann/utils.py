"""
Various utility functions.

This module contains some miscellaneous utility functions required elsewhere in the
codebase. For the most part, this module is unlikely to be useful to end users, but
there are some niche use cases (hence why it is still documented as part of the public
API).
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
from collections.abc import Sequence
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
    symbols : tuple of str
        The atomic species to associate site indices with.

    Returns
    -------
    atom_indices : dict of {str : tuple of int} pairs.
        The site indices categorised by atomic species.
    """
    atom_indices_list = {}
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


def get_spilling_factor(u: NDArray[np.complex128], num_wann: int) -> np.float64:
    r"""
    Compute the spilling factor for a set of Wannier functions.

    Parameters
    ----------
    u : ndarray of complex
        The U matrices that define the Wannier functions in terms of the canonical
        Bloch states.
    num_wann : int
        The total number of Wannier functions.

    Returns
    -------
    spilling_factor : float
        The spilling factor.

    See Also
    --------
    pengwann.io.read_u

    Notes
    -----
    The spilling factor is calculated as :footcite:p:`spilling, WOHP`

    .. math::

        S = \frac{1}{N_{k}}\frac{1}{N_{w}}\sum_{nk} 1 - \sum_{\alpha}
        |\braket{\psi_{nk}|w_{\alpha}}|^{2},

    where :math:`N_{k}` is the total number of k-points, :math:`N_{w}` is the total
    number of Wannier functions, :math:`n` labels bands, :math:`k` labels k-points and
    :math:`\alpha` labels Wannier functions :math:`\ket{w_{\alpha}}`.

        References
        ----------
    .. footbibliography::
    """
    u_star = np.conj(u)
    overlaps = (u_star * u).real

    num_kpoints = u.shape[0]

    return 1 - np.sum(overlaps) / num_kpoints / num_wann


def integrate_descriptor(
    energies: NDArray[np.float64], descriptor: NDArray[np.float64], mu: float
) -> np.float64 | NDArray[np.float64]:
    """
    Integrate a energy-resolved descriptor up to the Fermi level.

    Parameters
    ----------
    energies : ndarray of float
        The discrete energies at which the descriptor has been evaluated.
    descriptor : ndarray of float
        The descriptor to be integrated.
    mu : float
        The Fermi level.

    Returns
    -------
    integral : float or ndarray of float
        The integrated descriptor.
    """
    energies_to_mu = energies[energies <= mu]
    descriptor_to_mu = descriptor[: len(energies_to_mu)]

    integral = trapezoid(descriptor_to_mu, energies_to_mu, axis=0)

    return np.float64(integral)


def allocate_shared_memory(
    keys: Sequence[str], data: Sequence[NDArray]
) -> tuple[dict[str, tuple[tuple[int, ...], np.dtype]], list[SharedMemory]]:
    """
    Allocate one or more blocks of shared memory and populate them with numpy arrays.

    Parameters
    ----------
    keys : iterable of str
        A sequence of strings identifying each array to be put into shared memory.
    data : iterable of ndarray
        The arrays to be put into shared memory.

    Returns
    -------
    memory_metadata : dict of {str : 2-length tuple of tuple of int and data-type} pairs.
        A dictionary containing metadata for each allocated block of shared memory. The
        keys are set by `keys` and the values are a tuple containing the shape and dtype
        of the corresponding array.
    memory_handles : list of SharedMemory
        A sequence of SharedMemory objects (returned to allow easy access to the
        :code:`unlink` method).
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
