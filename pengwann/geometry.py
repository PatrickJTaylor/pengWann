"""
This module contains the :py:class:`~pengwann.geometry.InteractionFinder`
class, which allows for the identification of bonds between pairs of atoms
(and their associated Wannier functions) according to a distance criterion.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from pengwann.utils import assign_wannier_centres, get_atom_indices
from pymatgen.core import Lattice, Molecule, Structure
from typing import NamedTuple


class AtomicInteraction(NamedTuple):
    """
    A class representing the interaction between atoms i and j in terms
    of their respective Wannier functions.

    Attributes:
        pair_id (tuple[str, str]): A pair of strings identifying atoms
            i and j.
        wannier_interactions (tuple[WannierInteraction, ...]): The individual
            WannierInteractions that together comprise the total interaction
            between atoms i and j.
    """

    pair_id: tuple[str, str]
    wannier_interactions: tuple[WannierInteraction, ...]


class WannierInteraction(NamedTuple):
    """
    A class representing the interaction between Wannier function i and the
        closest image of Wannier function j.

    Attributes:
        i (int): The index for Wannier function i.
        j (int): The index for Wannier function j.
        R_2 (np.ndarray): The Bravais lattice vector specifying the translation
            of Wannier function j that brings it as close as possible to
            Wannier function i.
    """

    i: int
    j: int
    R_2: np.ndarray


class InteractionFinder:
    """
    A class for identifying interatomic interactions according to a
    chosen criterion, providing the indices necessary to calculate the
    relevant WOHPS and WOBIs.

    Args:
        geometry (Structure): A Pymatgen Structure object with a
            'wannier_centres' site_property.

    Returns:
        None

    Notes:
        This class is not really designed to be initialised
        manually. If you do wish to do this, the 'wannier_centres'
        site_property should associate each atom with a list containing
        the indices of its Wannier centres.
    """

    def __init__(self, geometry: Structure) -> None:
        self._geometry = geometry
        self._num_wann = len(
            [
                idx
                for idx in range(len(self._geometry))
                if self._geometry[idx].species_string == "X0+"
            ]
        )

    def get_interactions(
        self, radial_cutoffs: dict[tuple[str, str], float]
    ) -> tuple[AtomicInteraction, ...]:
        """
        Identify interatomic interactions according to a chosen
        criterion.

        Args:
            radial_cutoffs (dict[tuple[str, str], float]): A dictionary
                defining a radial cutoff for pairs of atomic species.

                For example:

                {('Fe', 'O') : 1.8, ('Si', 'O') : 2.0}

        Returns:
            tuple[AtomicInteraction, ...]: The interactions identified by the
            chosen criteria.
        """
        symbols_list: list[str] = []
        for pair in radial_cutoffs.keys():
            for symbol in pair:
                if symbol not in symbols_list:
                    symbols_list.append(symbol)

        symbols = tuple(symbols_list)

        atom_indices = get_atom_indices(self._geometry, symbols)

        wannier_centres = self._geometry.site_properties["wannier_centres"]
        interactions = []
        for pair, cutoff in radial_cutoffs.items():
            symbol_i, symbol_j = pair

            possible_interactions = []
            if symbol_i != symbol_j:
                for i in atom_indices[symbol_i]:
                    for j in atom_indices[symbol_j]:
                        possible_interactions.append((i, j))

            else:
                for idx, i in enumerate(atom_indices[symbol_i]):
                    for j in atom_indices[symbol_j][idx + 1 :]:
                        possible_interactions.append((i, j))

            for i, j in possible_interactions:
                distance = self._geometry.get_distance(i, j)

                if distance < cutoff:
                    pair_id = (
                        symbol_i + str(i - self._num_wann + 1),
                        symbol_j + str(j - self._num_wann + 1),
                    )
                    wannier_interactions_list = []
                    for m in wannier_centres[i]:
                        for n in wannier_centres[j]:
                            dist, image = self._geometry[m].distance_and_image(
                                self._geometry[n]
                            )

                            wannier_interaction = WannierInteraction(m, n, image)
                            wannier_interactions_list.append(wannier_interaction)

                    wannier_interactions = tuple(wannier_interactions_list)
                    interaction = AtomicInteraction(pair_id, wannier_interactions)
                    interactions.append(interaction)

        return tuple(interactions)

    @property
    def geometry(self) -> Structure:
        """
        A Pymatgen Structure object with a 'wannier_indices' site property
        that allows it to be used in conjunction with the project method of
        the DOS class.
        """
        return self._geometry

    @classmethod
    def from_xyz(cls, path: str, cell: ArrayLike) -> InteractionFinder:
        """
        Initialise an InteractionFinder object from an xyz file output
        by Wannier90.

        Args:
            path (str): Filepath to the xyz file containing the
                coordinates of the Wannier centres.
            cell (ArrayLike): The cell vectors associated with the
                structure.

        Returns:
            InteractionFinder: The initialised InteractionFinder object.
        """
        lattice = Lattice(cell)

        xyz = Molecule.from_file(path)
        species = [site.species_string for site in xyz]  # type: ignore[union-attr]
        coords = [site.coords for site in xyz]  # type: ignore[union-attr]

        geometry = Structure(lattice, species, coords, coords_are_cartesian=True)

        assign_wannier_centres(geometry)

        return InteractionFinder(geometry)
