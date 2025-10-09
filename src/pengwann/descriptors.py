"""
Compute chemical bonding descriptors from Wannier functions.

This module contains a single class, the
:py:class:`~pengwann.descriptors.DescriptorCalculator`, which contains the core
functionality of :code:`pengwann`: computing various descriptors of chemical bonding
from Wannier functions as output by Wannier90.
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

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import reduce, singledispatchmethod
from textwrap import dedent

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from tqdm.auto import tqdm

from pengwann.electronic_structure import Basis
from pengwann.interactions import (
    AtomicInteraction,
    AtomicInteractions,
    Interactions,
    WannierInteraction,
)


@dataclass(frozen=True, slots=True)
class DescriptorPipeline:
    _pipeline: tuple = ()

    @singledispatchmethod
    def pipe(
        self, interactions: Interactions, show_progress: bool = True
    ) -> Interactions:
        raise NotImplementedError

    @pipe.register
    def _(
        self, interactions: AtomicInteractions, show_progress: bool = True
    ) -> AtomicInteractions:
        preprocessed_interactions = (
            tqdm(interactions.atomic_interactions)
            if show_progress
            else interactions.atomic_interactions
        )

        processed_interactions = tuple(
            self.pipe(atomic_interaction, show_progress=False)
            for atomic_interaction in preprocessed_interactions
        )

        return replace(interactions, atomic_interactions=processed_interactions)

    @pipe.register
    def _(
        self, interactions: AtomicInteraction, show_progress: bool = True
    ) -> AtomicInteraction:
        processed_interactions = self.pipe(
            interactions.wannier_interactions, show_progress
        )

        return replace(interactions, wannier_interactions=processed_interactions)

    @pipe.register
    def _(
        self, interactions: Sequence, show_progress: bool = True
    ) -> WannierInteraction:
        if not all(
            isinstance(interaction, WannierInteraction) for interaction in interactions
        ):
            raise TypeError

        preprocessed_interactions = (
            tqdm(interactions) if show_progress else interactions
        )

        processed_interactions = tuple(
            reduce(lambda i, p: p(i), self._pipeline, interaction)
            for interaction in preprocessed_interactions
        )

        return processed_interactions

    def with_coefficients(self, basis: Basis) -> DescriptorProcessor:
        def process(interaction: WannierInteraction) -> WannierInteraction:
            coefficients = compute_coefficients(
                interaction.i, interaction.j, interaction.bl_i, interaction.bl_j, basis
            )

            return replace(interaction, coefficients=coefficients)

        pipeline = self._pipeline + (process,)

        return replace(self, _pipeline=pipeline)

    def without_coefficients(self) -> DescriptorProcessor:
        def process(interaction: WannierInteraction) -> WannierInteraction:
            return replace(interaction, coefficients=None)

        pipeline = self._pipeline + (process,)

        return replace(self, _pipeline=pipeline)

    def with_pdos(
        self, total_dos: NDArray[np.float64], resolve_k: bool = False
    ) -> DescriptorProcessor:
        def process(interaction: WannierInteraction) -> WannierInteraction:
            pdos = compute_pdos(interaction.coefficients, total_dos, resolve_k)

            return replace(interaction, pdos=pdos)

        pipeline = self._pipeline + (process,)

        return replace(self, _pipeline=pipeline)

    def with_h_ij(
        self, hamiltonian: dict[tuple[int, int, int], NDArray[np.complex128]]
    ) -> DescriptorProcessor:
        def process(interaction: WannierInteraction) -> WannierInteraction:
            h_ij = get_h_ij(
                interaction.i,
                interaction.j,
                interaction.bl_i,
                interaction.bl_j,
                hamiltonian,
            )

            return replace(interaction, h_ij=h_ij)

        pipeline = self._pipeline + (process,)

        return replace(self, _pipeline=pipeline)

    def with_p_ij(self, occupation_matrix: NDArray[np.float64]) -> DescriptorProcessor:
        def process(interaction: WannierInteraction) -> WannierInteraction:
            p_ij = compute_p_ij(interaction.coefficients, occupation_matrix)

            return replace(interaction, p_ij=p_ij)

        pipeline = self._pipeline + (process,)

        return replace(self, _pipeline=pipeline)

    def with_integrals(
        self, energies: NDArray[np.float64], mu: float
    ) -> DescriptorProcessor:
        def process(interaction: WannierInteraction) -> WannierInteraction:
            ipdos = compute_ipdos(interaction.pdos, energies, mu)

            return replace(interaction, ipdos=ipdos)

        pipeline = self._pipeline + (process,)

        return replace(self, _pipeline=pipeline)


def compute_coefficients(
    i: int, j: int, bl_i: NDArray[np.int32], bl_j: NDArray[np.int32], basis: Basis
) -> NDArray[np.float64]:
    c_star = (np.exp(-1j * 2 * np.pi * basis.kpoints @ bl_i))[:, np.newaxis] * basis.u[
        :, :, i
    ]
    c = (np.exp(1j * 2 * np.pi * basis.kpoints @ bl_j))[:, np.newaxis] * np.conj(
        basis.u[:, :, j]
    )

    coefficients = (c_star * c).real.T

    return coefficients


def compute_pdos(
    coefficients: NDArray[np.float64],
    total_dos: NDArray[np.float64],
    resolve_k: bool = False,
) -> NDArray[np.float64]:
    pdos_nk = coefficients[np.newaxis, :, :] * total_dos

    if resolve_k:
        pdos = np.sum(pdos_nk, axis=1)

    else:
        pdos = np.sum(pdos_nk, axis=(1, 2))

    return pdos


def get_h_ij(
    i: int,
    j: int,
    bl_i: NDArray[np.int32],
    bl_j: NDArray[np.int32],
    hamiltonian: dict[tuple[int, int, int], NDArray[np.complex128]],
) -> np.float64:
    bl_vector = tuple(int(component) for component in bl_j - bl_i)

    if bl_vector in hamiltonian:
        h_ij = hamiltonian[bl_vector][i, j].real

    else:
        raise KeyError(
            dedent(f"""
        Matrix elements for Bravais lattice vector {bl_vector} are required for the
        interaction between Wannier functions {i} and {j}, but were not found in the
        Wannier Hamiltonian provided.
        """)
        )

    return h_ij


def compute_p_ij(
    coefficients: NDArray[np.float64], occupation_matrix: NDArray[np.float64]
) -> np.float64:
    p_nk = occupation_matrix * coefficients

    num_kpoints = p_nk.shape[1]
    p_ij = np.sum(p_nk, axis=(0, 1)) / num_kpoints

    return p_ij


def compute_ipdos(
    pdos: NDArray[np.float64], energies: NDArray[np.float64], mu: float
) -> np.float64 | NDArray[np.float64]:
    energies_to_mu = energies[energies <= mu]
    pdos_to_mu = pdos[: len(energies_to_mu)]

    ipdos = trapezoid(pdos_to_mu, energies_to_mu, axis=0)

    return ipdos
