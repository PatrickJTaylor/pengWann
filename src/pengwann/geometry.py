"""
Parse periodic structures, assign Wannier centres and identify interactions.

This module contains the functions necessary to parse the geometry of the target system
and from this identify relevant interatomic/on-site interactions from which to compute
descriptors of bonding and local electronic structure.
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
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from numpy.typing import ArrayLike, NDArray
from pengwann.interactions import (
    AtomicInteractionContainer,
    AtomicInteraction,
    WannierInteraction,
)
from pengwann.io import read_cell, read_xyz


@dataclass(frozen=True)
class Geometry:
    """
    Data structure representing a periodic structure of atoms and Wannier centres.

    Attributes
    ----------
    sites : tuple of Site
        The individual Site objects representing each atom/Wannier centre in the
        structure.
    cell : ndarray of float
        The cell vectors associated with the structure.
    wannier_assignments : tuple of tuple of int
    distance_and_image_matrices : tuple of ndarray of float

    See Also
    --------
    Site

    Notes
    -----
    This class should not normally be initialised using the base constructor. See
    instead the :py:meth:`~pengwann.geometry.Geometry.from_xyz` classmethod.
    """

    sites: tuple[Site, ...]
    cell: NDArray[np.float64]

    def __iter__(self) -> Iterator[Site]:
        return iter(self.sites)

    def __getitem__(self, key: int) -> Site:
        return self.sites[key]

    def __len__(self) -> int:
        return len(self.sites)

    def __str__(self) -> str:
        to_print = [
            "Geometry",
            "========",
            "Cell",
            "----",
            f"{self.cell}",
            "",
            "Sites",
            "-----",
        ]

        for site in self.sites:
            to_print.append(f"{site.symbol}{site.index} {site.coords}")

        return "\n".join(to_print) + "\n"

    @cached_property
    def wannier_assignments(self) -> tuple[tuple[int, ...], ...]:
        """
        Assign Wannier centres to atoms based on a closest distance criterion.

        The indices returned by this property refer to the order of sites in the
        calling Geometry object. Each Wannier centre is associated with a single index
        identifying its closest atom whilst each atom is associated with a sequence of
        indices identifying all of its assigned Wannier centres.

        Returns
        -------
        wannier_assignments : tuple of tuple of int
            The indices assigning Wannier centres to atoms.

        Notes
        -----
        This property is cached once it has been evaluated once.
        """
        wannier_indices, atom_indices = [], []
        for site in self.sites:
            if site.symbol == "X":
                wannier_indices.append(site.index)

            else:
                atom_indices.append(site.index)

        num_wann = len(wannier_indices)

        if num_wann == 0:
            raise ValueError('No Wannier centres ("X" atoms) found in geometry.')

        distance_matrix, image_matrix = self.distance_and_image_matrices

        assignments_list = [[] for site in self.sites]
        for i in wannier_indices:
            distances = distance_matrix[i, num_wann:]
            min_idx = int(distances.argmin()) + num_wann

            assignments_list[i].append(min_idx)
            assignments_list[min_idx].append(i)

        return tuple(tuple(indices) for indices in assignments_list)

    @cached_property
    def distance_and_image_matrices(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
        """
        Compute the distance and image matrices.

        The image matrix mirrors the shape of the distance matrix, with each element
        referring to a Bravais lattice vector that specifies the periodic image of a
        given site that is closest to another site.

        Returns
        -------
        distance_matrix : ndarray of float
            The distance matrix.

        image_matrix : ndarray of np.int_
            The image matrix.

        Notes
        -----
        This property is cached once it has been evaluated once.
        """
        num_sites = len(self)
        num_dim = len(self.cell)
        distance_matrix = np.zeros((num_sites, num_sites))
        image_matrix = np.zeros((num_sites, num_sites, num_dim), dtype=np.int_)

        base_vectors = [[-1, 0, 1] for _ in range(num_dim)]
        image_vectors = np.zeros((3**num_dim, num_dim))
        for i, image_vector in enumerate(product(*base_vectors)):
            image_vectors[i] = image_vector

        for i in range(num_sites):
            for j in range(i + 1, num_sites):
                i_coords, j_coords = self.sites[i].coords, self.sites[j].coords

                v_0 = np.round(i_coords - j_coords)
                trans_j_coords = j_coords + v_0

                cart_vectors = self.cell @ (trans_j_coords + image_vectors - i_coords).T
                distances = np.linalg.norm(cart_vectors, axis=0)
                min_idx = distances.argmin()

                distance_matrix[i, j] = distance_matrix[j, i] = distances[min_idx]

                image = image_vectors[min_idx] + v_0
                image_matrix[i, j] = image
                image_matrix[j, i] = -image

        return distance_matrix, image_matrix

    @classmethod
    def from_xyz(
        cls, seedname: str, path: str = ".", cell: ArrayLike | None = None
    ) -> Geometry:
        """
        Initialise a Geometry object from a seedname_centres.xyz file.

        Parameters
        ----------
        seedname : str
            The seedname (prefix for all input files) chosen in the prior Wannier90
            calculation.
        path : str
            Filepath to the directory containing seedname_centres.xyz and (optionally)
            seedname.win. Defaults to '.' i.e. the current working directory.
        cell : array_like or None, optional
            The cell vectors associated with the structure. Defaults to None, in which
            case the cell vectors will automatically be extracted from a seedname.win
            file in the same directory as the xyz file.

        Returns
        -------
        geometry : Geometry
            The initialised Geometry object.

        Notes
        -----
        If you intend to use the instance returned by this method as an input to the
        :py:func:`~pengwann.geometry.identify_interatomic_interactions` function, please
        ensure that the seedname_centres.xyz file was generated by Wannier90 with
        translate_home_cell = false. If the seedname_centres.xyz file was generated with
        translate_home_cell = true, then the interactions identified by the
        :py:func:`~pengwann.geometry.identify_interatomic_interactions` function may not
        be accurate (more specifically, you may end up calculating interactions between
        the wrong periodic images).
        """
        symbols, cart_coords = read_xyz(f"{path}/{seedname}_centres.xyz")

        if cell is None:
            cell = read_cell(f"{path}/{seedname}.win")

        else:
            cell = np.asarray(cell)

        frac_coords = np.linalg.inv(cell) @ cart_coords
        sites = tuple(
            Site(symbol, idx, coords)
            for idx, (symbol, coords) in enumerate(zip(symbols, frac_coords.T))
        )

        return cls(sites, cell)


@dataclass(frozen=True)
class Site:
    """
    Data structure representing a site in a periodic structure.

    Attributes
    ----------
    symbol : str
        An elemental symbol for atoms or "X" for Wannier centres.
    index : int
        An index identifying this particular site.
    coords : ndarray of float
        The fractional coordinates of this site in the periodic structure.
    """

    symbol: str
    index: int
    coords: NDArray[np.float64]


def identify_onsite_interactions(
    geometry: Geometry, symbols: tuple[str, ...]
) -> AtomicInteractionContainer:
    """
    Identify all on-site interactions for a set of atomic species.

    Parameters
    ----------
    geometry : Geometry
            A Pymatgen Structure object with a :code:`"wannier_centres"` site property
            that associates each atom with the indices of its Wannier centres.
    symbols : tuple of str
            The atomic species to return interactions for. These should match one or
            more of the species present in `geometry`.

    Returns
    -------
    interactions : AtomicInteractionContainer
            The on-site/diagonal AtomicInteraction objects associated with each symbol
            in `symbols`.

    Notes
    -----
    In the context of pengwann, an on-site/diagonal interaction is simply a 2-body
    interaction between atoms or individual Wannier functions in which
    atom i == atom j or Wannier function i == Wannier function j.
    """
    zero_vector = np.array([0, 0, 0])
    assignments = geometry.wannier_assignments

    interactions = []
    for site in geometry:
        if site.symbol in symbols:
            wannier_interactions = []
            for i in assignments[site.index]:
                wannier_interaction = WannierInteraction(i, i, zero_vector, zero_vector)

                wannier_interactions.append(wannier_interaction)

            interaction = AtomicInteraction(
                site.index,
                site.index,
                site.symbol,
                site.symbol,
                tuple(wannier_interactions),
            )

            interactions.append(interaction)

    if not interactions:
        raise ValueError(f"No atoms matching symbols in {symbols} found.")

    return AtomicInteractionContainer(sub_interactions=tuple(interactions))


def identify_interatomic_interactions(
    geometry: Geometry, radial_cutoffs: dict[tuple[str, str], float]
) -> AtomicInteractionContainer:
    """
    Identify interatomic interactions according to a set of radial distance cutoffs.

    Parameters
    ----------
    geometry : Geometry
        A Pymatgen Structure object with a :code:`"wannier_centres"` site property that
        associates each atom with the indices of its Wannier centres.
    radial_cutoffs : dict of {2-length tuple of str : float} pairs
        A dictionary defining radial cutoffs for pairs of atomic species.

    Returns
    -------
    interactions : AtomicInteractionContainer
        The interactions identified according to the `radial_cutoffs`.

    See Also
    --------
    build_geometry
    pengwann.descriptors.DescriptorCalculator.assign_descriptors :
        Compute bonding descriptors for a set of interatomic interactions.

    Examples
    --------
    >>> cutoffs = {("Sr", "O"): 2.8,
    ...            ("V", "O"): 2.0}
    >>> interactions = identify_interatomic_interactions(geometry, cutoffs)
    """
    symbols = tuple({symbol for pair in radial_cutoffs for symbol in pair})

    atom_indices = _get_atom_indices(geometry, symbols)

    distance_matrix, image_matrix = geometry.distance_and_image_matrices
    assignments = geometry.wannier_assignments
    interactions = []
    for pair, cutoff in radial_cutoffs.items():
        symbol_i, symbol_j = pair

        # Exclude self-interactions
        offset = 1 if symbol_i == symbol_j else 0

        for idx, i in enumerate(atom_indices[symbol_i]):
            for j in atom_indices[symbol_j][idx + offset :]:
                distance = distance_matrix[i, j]

                if distance < cutoff:
                    wannier_interactions_list = []
                    for m in assignments[i]:
                        for n in assignments[j]:
                            bl_i = image_matrix[i, m]
                            bl_j = image_matrix[j, n]

                            wannier_interaction = WannierInteraction(m, n, bl_i, bl_j)
                            wannier_interactions_list.append(wannier_interaction)

                    wannier_interactions = tuple(wannier_interactions_list)
                    interaction = AtomicInteraction(
                        i, j, symbol_i, symbol_j, wannier_interactions
                    )
                    interactions.append(interaction)

    return AtomicInteractionContainer(sub_interactions=tuple(interactions))


def _get_atom_indices(
    geometry: Geometry, symbols: tuple[str, ...]
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

    for idx, site in enumerate(geometry):
        if site.symbol in symbols:
            atom_indices_list[site.symbol].append(idx)

    atom_indices = {}
    for symbol, indices in atom_indices_list.items():
        atom_indices[symbol] = tuple(indices)

    return atom_indices
