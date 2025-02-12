"""
Data structures representing interactions between atoms and Wannier functions.

This module contains several dataclasses/namedtuples that serve to store data relating
to interactions between atoms and Wannier functions. It is generally expected that each
of these data structures will be initialised with solely the data required to specify
which atoms or Wannier functions are interacting, the remaining fields will usually be
set by functions and methods in the :py:mod:`~pengwann.descriptors` module.
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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from numpy.typing import NDArray
from pengwann.utils import integrate_descriptor
from typing import NamedTuple


@dataclass(frozen=True)
class AtomicInteractionContainer:
    """
    Data structure for storing multiple AtomicInteraction objects.

    This class exists simply to faciliate convenient indexing and filtering over a
    sequence of AtomicInteraction objects. AtomicInteractionContainer objects can be
    indexed in the same manner as numpy arrays e.g. :code:`interaction_container[i, j]`
    will return all interactions between atoms i and j.

    Attributes
    ----------
    sub_interactions : sequence of AtomicInteraction
        The sequence of AtomicInteraction objects to be stored.

    See Also
    --------
    AtomicInteraction

    Examples
    --------
    To obtain the AtomicInteraction between atoms 0 and 1:

    >>> atomic_interaction_01 = interaction_container[0, 1]

    To iterate over all AtomicInteraction objects:

    >>> for atomic_interaction in interaction_container:
    >>>     # Do something for each interaction.

    To iterate over all the interactions between atom 0 and any other atom:

    >>> for atomic_interaction in interaction_container[0]:
    >>>     # Do something for each interaction.

    To iterate over all interactions including certain atomic species/elemental symbols:

    >>> symbols = ("Fe", "O")
    >>> for atomic_interaction in interaction_container.filter_by_species(symbols):
    >>>     # Do something for each Fe-O interaction.
    """

    sub_interactions: Sequence[AtomicInteraction]

    def __iter__(self) -> Iterator[AtomicInteraction]:
        return iter(self.sub_interactions)

    def __getitem__(
        self, key: int | tuple[int, int]
    ) -> AtomicInteraction | tuple[AtomicInteraction, ...]:
        indices = _slice_interaction_matrix(key, self._interaction_matrix)
        interactions = tuple(self.sub_interactions[idx] for idx in indices)

        if len(indices) > 1:
            return interactions

        else:
            return interactions[0]

    def __len__(self) -> int:
        return len(self.sub_interactions)

    def __str__(self) -> str:
        to_print = ["Atomic interactions"]

        underline = ["=" for character in to_print[-1]]
        to_print.append("".join(underline))

        for interaction in self.sub_interactions:
            to_print.append(interaction.tag)

        return "\n".join(to_print) + "\n"

    @cached_property
    def _interaction_matrix(self) -> list[list[list[int]]]:
        return _build_interaction_matrix(self.sub_interactions)

    def filter_by_species(
        self, symbols: tuple[str, ...]
    ) -> tuple[AtomicInteraction, ...]:
        """
        Return a subset of AtomicInteraction objects filtered by atomic species.

        For a given interaction to be returned by this function, its associated
        elemental symbols must be a subset of `symbols`.

        Parameters
        ----------
        symbols : tuple of str
            The elemental symbols used to filter the AtomicInteraction objects.

        Returns
        -------
        filtered_interactions : tuple of AtomicInteraction
            The subset of AtomicInteraction objects selected according to the input
            `symbols`.

        Notes
        -----
        This function will always return a tuple of AtomicInteraction objects even if
        it has :py:func:`len` = 1.
        """
        symbol_set = set(symbols)
        interactions = tuple(
            interaction
            for interaction in self.sub_interactions
            if set([interaction.symbol_i, interaction.symbol_j]) <= symbol_set
        )

        if not interactions:
            raise ValueError("No interactions involving {symbols} found.")

        return interactions

    def with_integrals(
        self,
        energies: NDArray[np.float64],
        mu: float,
        resolve_orbitals: bool = False,
        valence_counts: dict[str, int] | None = None,
    ) -> AtomicInteractionContainer:
        """
        Return an updated container with integrated descriptors for all interactions.

        The `valence_counts` argument may be provided in order to compute atomic charges
        for all on-site interactions.

        Parameters
        ----------
        energies : ndarray of float
            The discrete energies at which the density of states and all derived
            descriptors have been evaluated.
        mu : float
            The Fermi level
        resolve_orbitals : bool, optional
            If True, integrate descriptors for individual WannierInteraction objects
            as well as each overall AtomicInteraction. Defaults to False.
        valence_counts : dict of {str : int} pairs or None, optional
            The number of valence electrons associated with each atomic species
            according to the pseudopotentials employed in the prior ab initio
            calculation. This is used to compute atomic charges and pertains solely
            to on-site interactions. Defaults to None.

        Returns
        -------
        container_with_integrals : AtomicInteractionConatiner
            A new AtomicInteractionContainer with each AtomicInteraction now being
            associated with its integrated descriptors.
        """
        sub_interactions = []
        for interaction in self.sub_interactions:
            symbol_i, symbol_j = interaction.symbol_i, interaction.symbol_j

            if valence_counts is not None and symbol_i == symbol_j:
                valence_count = valence_counts[symbol_i]

            else:
                valence_count = None

            updated_interaction = interaction.with_integrals(
                energies,
                mu,
                resolve_orbitals=resolve_orbitals,
                valence_count=valence_count,
            )
            sub_interactions.append(updated_interaction)

        return replace(self, sub_interactions=sub_interactions)


@dataclass(frozen=True)
class AtomicInteraction:
    """
    Data structure regarding the interaction between two atoms.

    Within :code:`pengwann`, the interaction between two atoms is comprised of the
    interactions between their respective Wannier functions, hence each
    AtomicInteraction object is associated with a set of WannierInteraction objects.

    AtomicInteraction instances can be indexed much like numpy arrays e.g.
    :code:`atomic_interaction[i, j]` will return all of the WannierInteraction objects
    arising from interactions between Wannier functions :code:`i` and :code:`j`. In
    addition, iterating through an AtomicInteraction object yields the underlying
    WannierInteraction objects stored in the `sub_interactions` field.

    Attributes
    ----------
    i : int
        The index identifying atom i.
    j : int
        The index identifying atom j.
    symbol_i : str
        The elemental symbol for atom i.
    symbol_j : str
        The elemental symbol for atom j.
    tag : str
    sub_interactions: sequence of WannierInteraction
        The WannierInteraction objects concerning the interactions between the atoms'
        respective Wannier functions.
    dos_matrix : ndarray of float or None, optional
        The DOS matrix associated with the interacting atoms. Defaults to None.
    wohp : ndarray of float or None, optional
        The WOHP associated with the interacting atoms. Defaults to None.
    wobi : ndarray of float or None, optional
        The WOBI associated with the interacting atoms. Defaults to None.
    iwohp : float or ndarray of float or None, optional
        The IWOHP (integrated WOHP) associated with the interacting atoms. Defaults to
        None.
    iwobi : float or ndarray of float or None, optional
        The IWOBI (integrated WOBI) associated with the interacting atoms. Defaults to
        None.
    population : float or ndarray of float or None, optional
        The population (integrated DOS matrix) associated with the interacting atoms.
        Defaults to None.
    charge : float or ndarray of float or None, optional
        The charge associated with the interacting atoms. This yields atomic charges
        in the conventional sense only when the interacting atoms are identical (i.e.
        for on-site interactions).

    See Also
    --------
    WannierInteraction
    pengwann.geometry.identify_interatomic_interactions
    pengwann.geometry.identify_onsite_interactions

    Notes
    -----
    It is generally expected that this class will be initialised with solely the
    information required to specify the interacting atoms, namely the `i`, `j`,
    `symbol_i`, `symbol_j` and the `sub_interactions`. The remaining fields will usually
    only be set by methods of the :py:class:`~pengwann.descriptors.DescriptorCalculator`
    class.

    Examples
    --------
    To obtain the WannierInteraction between atoms 0 and 1:

    >>> wannier_interaction_01 = atomic_interaction[0, 1]

    To iterate over all WannierInteraction objects:

    >>> for wannier_interaction in atomic_interaction:
    >>>     # Do something for each interaction.

    To iterate over all the interactions between Wannier function 0 and any other
    Wannier function:

    >>> for wannier_interaction in atomic_interaction[0]:
    >>>     # Do something for each interaction.
    """

    i: int
    j: int
    symbol_i: str
    symbol_j: str
    sub_interactions: Sequence[WannierInteraction]

    dos_matrix: NDArray[np.float64] | None = None
    wohp: NDArray[np.float64] | None = None
    wobi: NDArray[np.float64] | None = None
    iwohp: np.float64 | NDArray[np.float64] | None = None
    iwobi: np.float64 | NDArray[np.float64] | None = None
    population: np.float64 | NDArray[np.float64] | None = None
    charge: np.float64 | NDArray[np.float64] | None = None

    def __iter__(self) -> Iterator[WannierInteraction]:
        return iter(self.sub_interactions)

    def __getitem__(
        self, key: int | tuple[int, int]
    ) -> WannierInteraction | tuple[WannierInteraction, ...]:
        indices = _slice_interaction_matrix(key, self._interaction_matrix)
        interactions = tuple(self.sub_interactions[idx] for idx in indices)

        if len(indices) > 1:
            return interactions

        else:
            return interactions[0]

    def __len__(self) -> int:
        return len(self.sub_interactions)

    def __str__(self) -> str:
        to_print = [f"Atomic interaction {self.tag}"]

        underline = ["=" for character in to_print[-1]]
        to_print.append("".join(underline))

        print_names = (
            ("dos_matrix", "DOS matrix"),
            ("wohp", "WOHP"),
            ("wobi", "WOBI"),
            ("iwohp", "IWOHP"),
            ("iwobi", "IWOBI"),
            ("population", "Population"),
            ("charge", "Charge"),
        )
        for attribute_name, print_name in print_names:
            value = getattr(self, attribute_name)

            print_value = "Not calculated" if value is None else "Calculated"

            line = f"{print_name} => {print_value}"

            to_print.append(line)

        to_print.append("\n")

        subtitle = "Associated Wannier interactions"
        subtitle_underline = ["-" for character in subtitle]
        to_print.extend((subtitle, "".join(subtitle_underline)))

        for w_interaction in self.sub_interactions:
            to_print.append(w_interaction.tag)

        return "\n".join(to_print) + "\n"

    @cached_property
    def _interaction_matrix(self) -> list[list[list[int]]]:
        return _build_interaction_matrix(self.sub_interactions)

    @property
    def tag(self) -> str:
        """
        Generate a simple string that identifies this particular AtomicInteraction.

        The tags generated by this property have the general form "Ai<=>Bj", where A and
        B are elemental symbols whilst i and j are indices.

        Returns
        -------
        generated_tag : str
            The generated tag.
        """
        return f"{self.symbol_i}{self.i} <=> {self.symbol_j}{self.j}"

    def with_summed_descriptors(self) -> AtomicInteraction:
        """
        Return a new AtomicInteraction object with summed DOS matrices, WOHPs and WOBIs.

        Only descriptors that have been calculated for all WannierInteraction objects
        will be summed over and included in the returned AtomicInteraction object.

        Returns
        -------
        interaction_post_sum : AtomicInteraction
            A new AtomicInteraction objects with the DOS matrix, WOHP and WOBI (where
            applicable) having been summed over all WannierInteraction objects.
        """
        new_values = {}

        descriptor_keys = ("dos_matrix", "wohp", "wobi")
        for descriptor_key in descriptor_keys:
            calculated = True

            for w_interaction in self.sub_interactions:
                if w_interaction.dos_matrix is None:
                    raise TypeError(
                        f"""The DOS matrix has not been calculated for interaction
                        {w_interaction.tag}"""
                    )

                if descriptor_key == "wohp":
                    if w_interaction.h_ij is None:
                        calculated = False
                        break

                if descriptor_key == "wobi":
                    if w_interaction.p_ij is None:
                        calculated = False
                        break

            if calculated:
                new_values[descriptor_key] = sum(
                    [
                        getattr(w_interaction, descriptor_key)
                        for w_interaction in self.sub_interactions
                    ]
                )

        return replace(self, **new_values)

    def with_integrals(
        self,
        energies: NDArray[np.float64],
        mu: float,
        resolve_orbitals: bool = False,
        valence_count: int | None = None,
    ) -> AtomicInteraction:
        """
        Return a new AtomicInteraction object with integrated descriptors.

        The `valence_count` argument may be provided in order to compute atomic charges
        for on-site interactions.

        Parameters
        ----------
        energies : ndarray of float
            The discrete energies at which the density of states and all derived
            descriptors have been evaluated.
        mu : float
            The Fermi level
        resolve_orbitals : bool, optional
            If True, integrate descriptors for individual WannierInteraction objects
            as well as the total AtomicInteraction. Defaults to False.
        valence_count : int or None, optional
            The number of valence electrons associated with the interaction according to
            the pseudopotentials employed in the prior ab initio calculation. This is
            used to compute atomic charges and only really makes sense for on-site
            interactions. Defaults to None.

        Returns
        -------
        interaction_with_integrals : AtomicInteraction
            A new AtomicInteraction object with the integrated descriptors passed to the
            relevant attributes.
        """
        new_values = {}

        if self.dos_matrix is not None:
            new_values["population"] = integrate_descriptor(
                energies, self.dos_matrix, mu
            )

            if valence_count is not None:
                new_values["charge"] = valence_count - new_values["population"]

        if self.wohp is not None:
            new_values["iwohp"] = integrate_descriptor(energies, self.wohp, mu)

        if self.wobi is not None:
            new_values["iwobi"] = integrate_descriptor(energies, self.wobi, mu)

        if resolve_orbitals:
            new_values["sub_interactions"] = []
            for w_interaction in self.sub_interactions:
                updated_wannier_interaction = w_interaction.with_integrals(energies, mu)

                new_values["sub_interactions"].append(updated_wannier_interaction)

        return replace(self, **new_values)


class WannierInteraction(NamedTuple):
    """
    Stores data regarding the interaction between two Wannier functions.

    Attributes
    ----------
    i : int
        The index identifying Wannier function i.
    j : int
        The index identifying Wannier function j.
    bl_i : ndarray of np.int_
        The Bravais lattice vector specifying the translation of Wannier function i
        relative to its home cell.
    bl_j : ndarray of np.int_
        The Bravais lattice vector specifying the translation of Wannier function j
        relative to its home cell.
    tag : str
    dos_matrix : ndarray of float or None, optional
        The DOS matrix associated with the interaction. Defaults to None.
    wohp : ndarray of float or None
    wobi : ndarray of float or None
    h_ij : float or None, optional
        The element of the Wannier Hamiltonian associated with the interaction. Defaults
        to None.
    p_ij : float or None, optional
        The element of the Wannier density matrix associated with the interaction.
        Defaults to None.
    iwohp : float or ndarray of float or None, optional
        The IWOHP (integrated WOHP) associated with the interaction. Defaults to None.
    iwobi : float or ndarray of float or None, optional
        The IWOBI (integrated WOBI) associated with the interaction. Defaults to None.
    population : float or ndarray of float or None, optional
        The population (integrated DOS matrix) associated with the interaction. Defaults
        to None.

    Notes
    -----
    It is expected that this class will normally be initialised with solely the data
    required to specify the interacting Wannier functions: the indices `i` and `j`
    alongside the Bravais lattice vectors `bl_i` and `bl_j`. The remaining fields will
    usually only be set by methods of the
    :py:class:`~pengwann.descriptors.DescriptorCalculator` class.
    """

    i: int
    j: int
    bl_i: NDArray[np.int_]
    bl_j: NDArray[np.int_]

    dos_matrix: NDArray[np.float64] | None = None
    h_ij: np.float64 | None = None
    p_ij: np.float64 | None = None
    iwohp: np.float64 | NDArray[np.float64] | None = None
    iwobi: np.float64 | NDArray[np.float64] | None = None
    population: np.float64 | NDArray[np.float64] | None = None

    def __str__(self) -> str:
        to_print = [f"Wannier interaction {self.tag}"]

        underline = ["=" for character in to_print[-1]]
        to_print.append("".join(underline))

        print_names = (
            ("dos_matrix", "DOS matrix"),
            ("h_ij", "H_ij"),
            ("p_ij", "P_ij"),
            ("iwohp", "IWOHP"),
            ("iwobi", "IWOBI"),
            ("population", "Population"),
        )
        for attribute_name, print_name in print_names:
            value = getattr(self, attribute_name)

            if attribute_name in ("h_ij", "p_ij"):
                print_value = "Not calculated" if value is None else value

            else:
                print_value = "Not calculated" if value is None else "Calculated"

            line = f"{print_name} => {print_value}"

            to_print.append(line)

        return "\n".join(to_print) + "\n"

    @property
    def tag(self) -> str:
        """
        Generate a simple string that identifies this particular WannierInteraction.

        The tags generated by this property have the general form "iR_1<=>jR_2", where i and
        j are indices whilst R_1 and R_2 are Bravais lattice vectors.

        Returns
        -------
        generated_tag : str
            The generated tag.
        """
        return f"{self.i}{self.bl_i.tolist()} <=> {self.j}{self.bl_j.tolist()}"

    @property
    def wohp(self) -> NDArray[np.float64] | None:
        """
        The WOHP associated with the interaction.

        Returns
        -------
        wohp : ndarray of float or None
            The WOHP or None (in the case that the DOS matrix or the relevant element
            of the Wannier Hamiltonian are not available).

        Notes
        -----
        The WOHP will be recalculated from the relevant element of the Wannier
        Hamiltonian and the DOS matrix every time this property is accessed (it is not
        cached). This is intended to reduce memory usage and has minimal impact on
        computational cost, owing to the fact that computing the DOS matrix itself is
        by far the most demanding step and this is only done once.
        """
        if self.h_ij is None or self.dos_matrix is None:
            return None

        return -self.h_ij * self.dos_matrix

    @property
    def wobi(self) -> NDArray[np.float64] | None:
        """
        The WOBI associated with the interaction.

        Returns
        -------
        wohp : ndarray of float or None
            The WOBI or None (in the case that the DOS matrix or the relevant element
            of the Wannier density matrix are not available).

        Notes
        -----
        The WOBI will be recalculated from the relevant element of the Wannier
        density matrix and the DOS matrix every time this property is accessed (it is
        not cached). This is intended to reduce memory usage and has minimal impact on
        computational cost, owing to the fact that computing the DOS matrix itself is
        by far the most demanding step and this is only done once.
        """
        if self.p_ij is None or self.dos_matrix is None:
            return None

        return self.p_ij * self.dos_matrix

    def with_integrals(
        self, energies: NDArray[np.float64], mu: float
    ) -> WannierInteraction:
        """
        Return a new WannierInteraction object with integrated descriptors.

        Parameters
        ----------
        energies : ndarray of float
            The discrete energies at which the density of states and all derived
            descriptors have been evaluated.
        mu : float
            The Fermi level

        Returns
        -------
        interaction_with_integrals : WannierInteraction
            A new WannierInteraction object with the integrated descriptors passed to the
            relevant attributes.
        """
        if self.dos_matrix is None:
            raise TypeError(
                """The DOS matrix must be calculated first to derive WOHPs, WOBIs and
                integrated descriptors."""
            )

        new_values = {}

        new_values["population"] = integrate_descriptor(energies, self.dos_matrix, mu)

        wohp = self.wohp
        if wohp is not None:
            new_values["iwohp"] = integrate_descriptor(energies, wohp, mu)

        wobi = self.wobi
        if wobi is not None:
            new_values["iwobi"] = integrate_descriptor(energies, wobi, mu)

        return self._replace(**new_values)


def _slice_interaction_matrix(
    key: int | tuple[int, int], interaction_matrix: list[list[list[int]]]
) -> list[int]:
    """
    Generate the indices needed to access particular interactions via np-style indexing.

    Parameters
    ----------
        key : int | tuple[int, int]
            One or two indices referring to the interactions to be accessed.
        interaction_matrix : list[list[list[int]]]
            A square matrix containing the indices needed to do np-style indexing.

    Returns
    -------
    indices : list[int]
        The indices needed to fetch the desired interactions.
    """
    if isinstance(key, int):
        indices = [
            idx for col_indices in interaction_matrix[key] for idx in col_indices
        ]

    else:
        i, j = key
        indices = interaction_matrix[i][j]

    if not indices:
        raise ValueError(f"No interactions found for indices {key}.")

    return indices


def _build_interaction_matrix(
    interactions: Sequence[AtomicInteraction] | Sequence[WannierInteraction],
) -> list[list[list[int]]]:
    """
    Create a square matrix containing the indices needed to do np-style indexing.

    Parameters
    ----------
    interactions : Sequence[AtomicInteraction] | Sequence[WannierInteraction]
        The AtomicInteraction or WannierInteraction objects which are to be accessed
        via np-style indexing.

    Returns
    -------
    interaction_matrix : list[list[list[int]]]
        The matrix of indices generated from the input `interactions`.
    """
    max_idx = max(max(interaction.i, interaction.j) for interaction in interactions)
    interaction_matrix = [
        [[] for idx in range(max_idx + 1)] for idx in range(max_idx + 1)
    ]
    for idx, interaction in enumerate(interactions):
        i, j = interaction.i, interaction.j

        interaction_matrix[i][j].append(idx)

        if i != j:
            interaction_matrix[j][i].append(idx)

    return interaction_matrix
