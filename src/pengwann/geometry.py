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
    sites: tuple[Site, ...]
    cell: NDArray[np.float64]

    def __iter__(self) -> Iterator[Site]:
        return iter(self.sites)

    def __len__(self) -> int:
        return len(self.sites)

    @cached_property
    def wannier_assignments(self) -> tuple[tuple[int, ...], ...]:
        distance_matrix, image_matrix = self.distance_and_image_matrices

        wannier_indices, atom_indices = [], []
        for site in self.sites:
            if site.symbol == "X":
                wannier_indices.append(site.index)

            else:
                atom_indices.append(site.index)

        num_wann = len(wannier_indices)

        if num_wann == 0:
            raise ValueError('No Wannier centres ("X" atoms) found in geometry.')

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
    bl_0 = np.array([0, 0, 0])
    assignments = geometry.wannier_assignments

    interactions = []
    for site in geometry:
        if site.symbol in symbols:
            wannier_interactions = []
            for i in assignments[site.index]:
                wannier_interaction = WannierInteraction(i, i, bl_0, bl_0)

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
    num_wann = len([site for site in geometry if site.symbol == "X"])

    if num_wann == 0:
        raise ValueError(
            'Input geometry contains no Wannier centres (i.e. no "X" atoms).'
        )

    symbols_list: list[str] = []
    for pair in radial_cutoffs:
        for symbol in pair:
            if symbol not in symbols_list:
                symbols_list.append(symbol)

    symbols = tuple(symbols_list)

    atom_indices = _get_atom_indices(geometry, symbols)

    distance_matrix, image_matrix = geometry.distance_and_image_matrices
    assignments = geometry.wannier_assignments
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
