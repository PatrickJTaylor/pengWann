"""
This module contains the functions necessary to parse the geometry of the target system
and from this identify relevant interatomic interactions from which to compute bonding
descriptors.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
from pengwann.utils import assign_wannier_centres, get_atom_indices
from pymatgen.core import Lattice, Molecule, Structure
from typing import Optional


@dataclass
class AtomicInteraction:
    """
    A class representing the interaction between atoms i and j in terms of their
    respective Wannier functions.

    Attributes:
        pair_id (tuple[str, str]): A pair of strings identifying atoms i and j.
        wannier_interactions (tuple[WannierInteraction, ...]): The individual
            :py:class:`~pengwann.geometry.WannierInteraction` objects that together
            comprise the total interaction between atoms i and j.
    """

    pair_id: tuple[str, str]
    wannier_interactions: tuple[WannierInteraction, ...]

    dos_matrix: Optional[NDArray[np.float64]] = None
    wohp: Optional[NDArray[np.float64]] = None
    wobi: Optional[NDArray[np.float64]] = None
    iwohp: Optional[np.float64] = None
    iwobi: Optional[np.float64] = None
    population: Optional[np.float64] = None
    charge: Optional[np.float64] = None


@dataclass
class WannierInteraction:
    """
    A class representing the interaction between the two Wannier functions iR_1 and
    jR_2.

    Attributes:
        i (int): The index for Wannier function i.
        j (int): The index for Wannier function j.
        bl_1 (NDArray[np.int\\_]): The Bravais lattice vector specifying the translation of
            Wannier function i with respect to its home cell.
        bl_2 (NDArray[np.int\\_]): The Bravais lattice vector specifying the translation of
            Wannier function j with respect to its home cell.
    """

    i: int
    j: int
    bl_1: NDArray[np.int_]
    bl_2: NDArray[np.int_]

    dos_matrix: Optional[NDArray[np.float64]] = None
    h_ij: Optional[np.float64] = None
    p_ij: Optional[np.float64] = None
    iwohp: Optional[np.float64] = None
    iwobi: Optional[np.float64] = None
    population: Optional[np.float64] = None

    @property
    def wohp(self):
        if self.h_ij is None:
            return None

        else:
            return -self.h_ij * self.dos_matrix

    @property
    def wobi(self):
        if self.p_ij is None:
            return None

        else:
            return self.p_ij * self.dos_matrix


def build_geometry(path: str, cell: ArrayLike) -> Structure:
    """
    Construct a Pymatgen Structure containing all of the necessary information to
    identify interatomic interactions and the Wannier functions involved.

    More specifically, the final Structure object has a :code:`"wannier_centres"` site
    property which associates each atom with the indices of its Wannier functions and
    each Wannier centre with the index of its associated atom.

    Args:
        path (str): Filepath to the xyz file containing the coordinates of the Wannier
            centres.
        cell (ArrayLike): The cell vectors associated with the structure.

    Returns:
        Structure: The Pymatgen Structure containing the relevant atoms and Wannier
        centres.

    Notes:
        The Pymatgen Structure returned by this function can be used as the
        :code:`geometry` argument to several methods of the
        :py:class:`~pengwann.dos.DOS` class.
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
    Identify interatomic interactions according to a set of radial distance cutoffs.

    Args:
        radial_cutoffs (dict[tuple[str, str], float]): A dictionary defining radial
            cutoffs for pairs of atomic species.

            For example:

            {('Fe', 'O') : 1.8, ('Si', 'O') : 2.0}

    Returns:
        tuple[AtomicInteraction, ...]: The interactions identified by the radial
        cutoffs.

    Notes:
        The :py:class:`~pengwann.geometry.AtomicInteraction` objects returned by this
        function can be supplied to the :py:meth:`pengwann.dos.DOS.get_descriptors`
        method to automate the computation of desirable bonding descriptors.
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
