"""
Parse periodic structures, assign Wannier centres and identify interatomic interactions.

This module contains the functions necessary to parse the geometry of the target system
and from this identify relevant interatomic interactions from which to compute bonding
descriptors.
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
from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
from pengwann.utils import get_atom_indices
from pymatgen.core import Lattice, Molecule, Structure
from typing import Optional


@dataclass
class AtomicInteraction:
    """
    Represents an interatomic interaction in terms of Wannier functions.

    It is generally expected that an AtomicInteraction instance will be initialised with
    only the `pair_id` and `wannier_interactions` attributes being set, with the rest
    defaulting to None until they are set by methods of the
    :py:class:`~pengwann.descriptors.DescriptorCalculator` class.

    Parameters
    ----------
    pair_id : tuple[str, str]
        A pair of strings labelling atoms i and j.
    wannier_interactions : tuple[WannierInteraction, ...]
        The individual WannierInteraction objects that together comprise the total
        interaction between atoms i and j.
    dos_matrix : ndarray[float] | None, optional
        The total DOS matrix. Defaults to None.
    wohp : ndarray[float] | None, optional
        The total WOHP. Defaults to None.
    wobi : ndarray[float] | None, optional
        The total WOBI. Defaults to None.
    iwohp : float | ndarray[float] | None, optional
        The integrated total WOHP. Defaults to None.
    iwobi : float | ndarray[float] | None, optional
        The integrated total WOBI. Defaults to None.
    population : float | ndarray[float] | None, optional
        The population (integrated DOS matrix). Defaults to None.
    charge : float | ndarray[float] | None, optional
        The charge (valence - population). Defaults to None.

    Attributes
    ----------
    pair_id : tuple[str, str]
        A pair of strings labelling atoms i and j.
    wannier_interactions : tuple[WannierInteraction, ...]
        The individual WannierInteraction objects that together comprise the total
        interaction between atoms i and j.
    dos_matrix : ndarray[float] | None
        The total DOS matrix.
    wohp : ndarray[float] | None
        The total WOHP.
    wobi : ndarray[float] | None
        The total WOBI.
    iwohp : float | ndarray[float] | None
        The integrated total WOHP.
    iwobi : float | ndarray[float] | None
        The integrated total WOBI.
    population : float | ndarray[float] | None
        The population (integrated DOS matrix).
    charge : float | ndarray[float] | None
        The charge (valence - population).

    Returns
    -------
    None
    """

    pair_id: tuple[str, str]
    wannier_interactions: tuple[WannierInteraction, ...]

    dos_matrix: Optional[NDArray[np.float64]] = None
    wohp: Optional[NDArray[np.float64]] = None
    wobi: Optional[NDArray[np.float64]] = None
    iwohp: Optional[np.float64 | NDArray[np.float64]] = None
    iwobi: Optional[np.float64 | NDArray[np.float64]] = None
    population: Optional[np.float64 | NDArray[np.float64]] = None
    charge: Optional[np.float64 | NDArray[np.float64]] = None


@dataclass
class WannierInteraction:
    """
    Represents the interaction between two Wannier functions.

    It is generally expected that a WannierInteraction instance will be initialised with
    only the `i`, `j`, `bl_1` and `bl_2` attributes being set, with the rest defaulting
    to None until they are set by methods of the
    :py:class:`~pengwann.descriptors.DescriptorCalculator` class.

    Parameters
    ----------
    i : int
        An index identifying Wannier function i.
    j : int
        An index identifying Wannier function j.
    bl_1 : ndarray[np.int_]
        The Bravais lattice vector specifying the translation of Wannier function i
        with respect to its home cell.
    bl_2 : ndarray[np.int_]
        The Bravais lattice vector specifying the translation of Wannier function j
        with respect to its home cell.
    dos_matrix : ndarray[float] | None, optional
        The DOS matrix. Defaults to None.
    h_ij : float | None, optional
        Element of the Wannier Hamiltonian required to compute the WOHP. Defaults to
        None.
    p_ij : float | None, optional
        Element of the Wannier density matrix required to compute the WOBI. Defaults
        to None.
    iwohp : float | ndarray[float] | None, optional
        The integrated WOHP. Defaults to None.
    iwobi : float | ndarray[float] | None, optional
        The integrated WOBI. Defaults to None.
    population : float | ndarray[float] | None, optional
        The population (integrated DOS matrix). Defaults to None.

    Attributes
    ----------
    i : int
        An index identifying Wannier function i.
    j : int
        An index identifying Wannier function j.
    bl_1 : ndarray[np.int_]
        The Bravais lattice vector specifying the translation of Wannier function i
        with respect to its home cell.
    bl_2 : ndarray[np.int_]
        The Bravais lattice vector specifying the translation of Wannier function j
        with respect to its home cell.
    dos_matrix : ndarray[float] | None
        The DOS matrix.
    h_ij : float | None
        Element of the Wannier Hamiltonian required to compute the WOHP.
    p_ij : float | None
        Element of the Wannier density matrix required to compute the WOBI.
    iwohp : float | ndarray[float] | None
        The integrated WOHP.
    iwobi : float | ndarray[float] | None
        The integrated WOBI.
    population : float | ndarray[float] | None
        The population (integrated DOS matrix).

    Returns
    -------
    None
    """

    i: int
    j: int
    bl_1: NDArray[np.int_]
    bl_2: NDArray[np.int_]

    dos_matrix: Optional[NDArray[np.float64]] = None
    h_ij: Optional[np.float64] = None
    p_ij: Optional[np.float64] = None
    iwohp: Optional[np.float64 | NDArray[np.float64]] = None
    iwobi: Optional[np.float64 | NDArray[np.float64]] = None
    population: Optional[np.float64 | NDArray[np.float64]] = None

    @property
    def wohp(self):
        """
        The Wannier orbital Hamilton population.

        This will be evaluated lazily upon request, owing to the fact that separately
        storing the WOHP and WOBI for a single interaction is redundant (they are just
        different weightings of the same DOS matrix).

        Returns
        -------
        wohp : ndarray[float] | None
            The WOHP.

        Notes
        -----
        If either the DOS matrix or the relevant element of the Wannier Hamiltonian are
        not available (i.e. the dos_matrix or h_ij attributes are set to None), then
        this property will just return None itself.
        """
        if self.h_ij is None or self.dos_matrix is None:
            return None

        else:
            return -self.h_ij * self.dos_matrix

    @property
    def wobi(self):
        """
        The Wannier orbital bond index.

        This will be evaluated lazily upon request, owing to the fact that separately
        storing the WOHP and WOBI for a single interaction is redundant (they are just
        different weightings of the same DOS matrix).

        Returns
        -------
        wobi : ndarray[float] | None
            The WOBI.

        Notes
        -----
        If either the DOS matrix or the relevant element of the Wannier density matrix
        are not available (i.e. the dos_matrix or p_ij attributes are set to None),
        then this property will just return None itself.
        """
        if self.p_ij is None or self.dos_matrix is None:
            return None

        else:
            return self.p_ij * self.dos_matrix


def build_geometry(path: str, cell: ArrayLike) -> Structure:
    """
    Return a Pymatgen Structure with a :code:`"wannier_centres"` site property.

    The "wannier_centres" site property associates each atom in the structure with a
    sequence of indices. These indices indicate the Wannier centres that have been
    assigned to each atom.

    Parameters
    ----------
    path : str
        Filepath to the xyz file output by Wannier90.
    cell : array_like
        The cell vectors associated with the structure.

    Returns
    -------
    geometry : Structure
        The Pymatgen Structure with a :code:`"wannier_centres"` site property.

    Notes
    -----
    The `geometry` returned by this function can be passed as the `geometry` argument
    to several methods of the :py:class:`~pengwann.descriptors.DescriptorCalculator`
    class.
    """
    lattice = Lattice(cell)

    xyz = Molecule.from_file(path)
    species, coords = [], []
    for site in xyz:  # type: ignore[union-attr]
        symbol = site.species_string.capitalize()
        species.append(symbol)
        coords.append(site.coords)

    geometry = Structure(lattice, species, coords, coords_are_cartesian=True)

    assign_wannier_centres(geometry)

    return geometry


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


def find_interactions(
    geometry: Structure, radial_cutoffs: dict[tuple[str, str], float]
) -> tuple[AtomicInteraction, ...]:
    """
    Identify interatomic interactions according to a set of radial distance cutoffs.

    Parameters
    ----------
    geometry : Structure
        A Pymatgen Structure object with a :code:`"wannier_centres"` site property that
        associates each atom with the indices of its Wannier centres.
    radial_cutoffs : dict[tuple[str, str], float]
        A dictionary defining radial cutoffs for pairs of atomic species e.g
        :code:`{("C", "C"): 1.6, ("C", "O"): 1.5}`.

    Returns
    -------
    interactions : tuple[AtomicInteraction, ...]
        The interactions identified according to the `radial_cutoffs`.

    See Also
    --------
    build_geometry
    pengwann.descriptors.DescriptorCalculator.assign_descriptors :
        Compute bonding descriptors for a set of interatomic interactions.
    """
    if "wannier_centres" not in geometry.site_properties.keys():
        raise ValueError('Input geometry is missing a "wannier_centres" site property.')

    num_wann = len([site for site in geometry if site.species_string == "X0+"])

    if num_wann == 0:
        raise ValueError(
            'Input geometry contains no Wannier centres (i.e. no "X" atoms).'
        )

    symbols_list: list[str] = []
    for pair in radial_cutoffs.keys():
        for symbol in pair:
            if symbol not in symbols_list:
                symbols_list.append(symbol)

    symbols = tuple(symbols_list)

    atom_indices = get_atom_indices(geometry, symbols)

    wannier_centres = geometry.site_properties["wannier_centres"]
    interactions = []
    for pair, cutoff in radial_cutoffs.items():
        symbol_i, symbol_j = pair

        possible_interactions = []
        if symbol_i != symbol_j:
            for i in atom_indices[symbol_i]:
                for j in atom_indices[symbol_j]:
                    possible_interactions.append((i, j))

        # Exclude self-interactions
        else:
            for idx, i in enumerate(atom_indices[symbol_i]):
                for j in atom_indices[symbol_j][idx + 1 :]:
                    possible_interactions.append((i, j))

        for i, j in possible_interactions:
            distance = geometry.get_distance(i, j)

            if distance < cutoff:
                pair_id = (
                    symbol_i + str(i - num_wann + 1),
                    symbol_j + str(j - num_wann + 1),
                )
                wannier_interactions_list = []
                for m in wannier_centres[i]:
                    for n in wannier_centres[j]:
                        _, bl_1 = geometry[i].distance_and_image(geometry[m])
                        _, bl_2 = geometry[j].distance_and_image(geometry[n])

                        wannier_interaction = WannierInteraction(m, n, bl_1, bl_2)
                        wannier_interactions_list.append(wannier_interaction)

                wannier_interactions = tuple(wannier_interactions_list)
                interaction = AtomicInteraction(pair_id, wannier_interactions)
                interactions.append(interaction)

    return tuple(interactions)
