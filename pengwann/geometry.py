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
        R_1 (np.ndarray): The Bravais lattice vector specifying the translation
            of Wannier function i.
        R_2 (np.ndarray): The Bravais lattice vector specifying the translation
            of Wannier function j.
    """

    i: int
    j: int
    R_1: np.ndarray
    R_2: np.ndarray


def build_geometry(path: str, cell: ArrayLike) -> Structure:
    """
    Construct a Pymatgen Structure containing all of the necessary information to
    identify interatomic interactions and the Wannier functions involved.

    More specifically, the final Structure object has a "wannier_centres" site property
    which associates each atom with the indices of its Wannier functions and each
    Wannier centre with the index of its associated atom.

    Args:
        path (str): Filepath to the xyz file containing the coordinates of the Wannier
            centres.
        cell (ArrayLike): The cell vectors associated with the structure.

    Returns:
        Structure: The Pymatgen Structure containing the relevant atoms and Wannier
        centres.
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


def find_interactions(
    geometry: Structure, radial_cutoffs: dict[tuple[str, str], float]
) -> tuple[AtomicInteraction, ...]:
    """
    Identify interatomic interactions according to a radial distance cutoff.

    Args:
        radial_cutoffs (dict[tuple[str, str], float]): A dictionary defining a radial
            cutoff for pairs of atomic species.

            For example:

            {('Fe', 'O') : 1.8, ('Si', 'O') : 2.0}

    Returns:
        tuple[AtomicInteraction, ...]: The interactions identified by the radial
        cutoffs.
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
                        _, R_1 = geometry[i].distance_and_image(geometry[m])
                        _, R_2 = geometry[j].distance_and_image(geometry[n])

                        wannier_interaction = WannierInteraction(m, n, R_1, R_2)
                        wannier_interactions_list.append(wannier_interaction)

                wannier_interactions = tuple(wannier_interactions_list)
                interaction = AtomicInteraction(pair_id, wannier_interactions)
                interactions.append(interaction)

    return tuple(interactions)
