"""
Parse periodic structures, assign Wannier centres and identify interactions.

This module contains the classes and functions necessary to parse the geometry of the
target system and from this identify relevant interatomic/on-site interactions from
which to compute descriptors of bonding and local electronic structure.
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

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self, override

from pengwann._geometry import (
    _build_distance_and_image_matrices,  # pyright:ignore[reportPrivateUsage]
)
from pengwann.interactions import (
    AtomicInteraction,
    AtomicInteractionContainer,
    WannierInteraction,
)
from pengwann.io import read_cell, read_xyz

if TYPE_CHECKING:
    from pymatgen.core import Structure


@dataclass(frozen=True)
class Geometry:
    """
    Data structure representing a periodic system of atoms and Wannier centres.

    The primary function of this class is to store the geometric information required to
    identify interatomic and on-site interactions in terms of individual Wannier
    functions. In addition, for ease of integration into common materials modelling
    workflows, Geometry objects can be easily converted into Pymatgen Structure objects
    with the :py:meth:`~pengwann.geometry.Geometry.as_structure` method.

    Attributes
    ----------
    sites : tuple of Site
        The individual Site objects representing each atom/Wannier centre in the
        structure.
    cell : ndarray of float
        The cell vectors associated with the system.
    wannier_assignments : tuple of tuple of int
    distance_and_image_matrices : tuple of ndarray

    See Also
    --------
    Site

    Notes
    -----
    This class should not normally be initialised using the base constructor. See
    instead the :py:meth:`~pengwann.geometry.Geometry.from_xyz` classmethod.

    Examples
    --------
    To access individual Site objects:

    >>> site_0 = geometry[0]

    To iterate over all sites:

    >>> for site in geometry:
    ...     # Do something for each Site object.
    """

    sites: tuple[Site, ...]
    cell: NDArray[np.float64]

    def __iter__(self) -> Iterator[Site]:
        return iter(self.sites)

    def __getitem__(self, key: int) -> Site:
        return self.sites[key]

    def __len__(self) -> int:
        return len(self.sites)

    @override
    def __str__(self) -> str:
        to_print = [
            "Geometry",
            "========",
            "Cell",
            "----",
            f"{self.cell}",
            "",
            "Assignments",
            "-----------",
        ]

        for site, assignments in zip(self.sites, self.wannier_assignments):
            arrow = "=>" if site.symbol == "X" else "<="
            to_print.append(f"{site.symbol}{site.index} {arrow} {assignments}")

        return "\n".join(to_print) + "\n"

    @property
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
        This property is cached after it has been evaluated once.
        """
        wannier_indices: list[int] = []
        atom_indices: list[int] = []
        for site in self.sites:
            if site.symbol == "X":
                wannier_indices.append(site.index)

            else:
                atom_indices.append(site.index)

        num_wann = len(wannier_indices)

        if num_wann == 0:
            raise ValueError('No Wannier centres ("X" atoms) found in geometry.')

        distance_matrix, _ = self.distance_and_image_matrices

        assignments_list: list[list[int]] = [[] for _ in self.sites]
        for i in wannier_indices:
            distances = distance_matrix[i, num_wann:]
            min_idx = int(distances.argmin()) + num_wann

            assignments_list[i].append(min_idx)
            assignments_list[min_idx].append(i)

        return tuple(tuple(indices) for indices in assignments_list)

    @property
    def distance_and_image_matrices(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """
        Compute the distance and image matrices.

        The image matrix mirrors the shape of the distance matrix, with each element
        referring to a Bravais lattice vector that specifies the periodic image of a
        given site that is the closest to another site.

        Returns
        -------
        distance_matrix : ndarray of float
            The distance matrix.

        image_matrix : ndarray of np.int32
            The image matrix.

        Notes
        -----
        This property is cached after it has been evaluated once.
        """
        coords = np.vstack([site.coords for site in self.sites])

        return _build_distance_and_image_matrices(coords, self.cell)

    def as_structure(self) -> Structure:
        """
        Build a Pymatgen Structure object from the calling Geometry object.

        Returns
        -------
        structure : Structure
            The Pymatgen Structure object.

        Notes
        -----
        The :py:meth:`~pengwann.geometry.Geometry.wannier_assignments` of the Geometry
        object are preserved in the output Pymatgen Structure as a
        :code:`site_property`.

        Examples
        --------
        To access the Wannier assignments:

        >>> assignments = structure.site_properties["wannier_assignments"]
        """
        try:
            from pymatgen.core import (
                Structure,
            )

        except ImportError as base_error:
            raise ImportError(
                """The as_structure method requires the Pymatgen package, which cannot
                be found."""
            ) from base_error

        symbols: list[str] = []
        coords: list[NDArray[np.float64]] = []
        for site in self.sites:
            symbols.append(site.symbol)
            coords.append(site.coords)

        return Structure(
            lattice=self.cell,
            species=symbols,
            coords=coords,
            site_properties={"wannier_assignments": self.wannier_assignments},
        )

    @classmethod
    def from_xyz(
        cls, seedname: str, path: str = ".", cell: ArrayLike | None = None
    ) -> Self:
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
        If you intend to use the Geometry object returned by this method as an input to
        the :py:func:`~pengwann.geometry.identify_interatomic_interactions` function,
        please ensure that the seedname_centres.xyz file was generated by Wannier90 with
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
        The structure in which to identify on-site interactions.
    symbols : tuple of str
            The atomic species to return interactions for. These should match one or
            more of the atomic species present in `geometry`.

    Returns
    -------
    interactions : AtomicInteractionContainer
        The on-site/diagonal AtomicInteraction objects associated with each symbol
        in `symbols`.

    See Also
    --------
    Geometry.from_xyz
    pengwann.descriptors.DescriptorCalculator.assign_descriptors :
        Compute descriptors for a set of on-site interactions.

    Notes
    -----
    In the context of pengwann, an on-site/diagonal interaction is simply a 2-body
    interaction between atoms or individual Wannier functions in which
    atom i == atom j or Wannier function i == Wannier function j.
    """
    zero_vector = np.array([0, 0, 0])
    assignments = geometry.wannier_assignments

    interactions: list[AtomicInteraction] = []
    for site in geometry:
        if site.symbol in symbols:
            wannier_interactions: list[WannierInteraction] = []
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
        The structure in which to identify interatomic interactions.
    radial_cutoffs : dict of {2-length tuple of str : float} pairs
        A dictionary defining radial cutoffs for pairs of atomic species.

    Returns
    -------
    interactions : AtomicInteractionContainer
        The interactions identified according to the `radial_cutoffs`.

    See Also
    --------
    Geometry.from_xyz
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
    interactions: list[AtomicInteraction] = []
    for pair, cutoff in radial_cutoffs.items():
        symbol_i, symbol_j = pair

        # Exclude self-interactions
        offset = 1 if symbol_i == symbol_j else 0

        for idx, i in enumerate(atom_indices[symbol_i]):
            for j in atom_indices[symbol_j][idx + offset :]:
                distance = distance_matrix[i, j]

                if distance < cutoff:
                    wannier_interactions_list: list[WannierInteraction] = []
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
    Categorise the site indices of a Geometry object according to atomic species.

    Parameters
    ----------
    geometry : Geometry
        The structure from which to extract indices.
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

    for idx, site in enumerate(geometry):
        if site.symbol in symbols:
            atom_indices_list[site.symbol].append(idx)

    atom_indices: dict[str, tuple[int, ...]] = {}
    for symbol, indices in atom_indices_list.items():
        atom_indices[symbol] = tuple(indices)

    return atom_indices
