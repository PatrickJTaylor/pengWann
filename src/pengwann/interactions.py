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

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid


@dataclass(frozen=True, slots=True)
class AtomicInteractions:
    atomic_interactions: Sequence[AtomicInteraction]


@dataclass(frozen=True, slots=True)
class AtomicInteraction:
    i: int
    j: int
    symbol_i: str
    symbol_j: str

    wannier_interactions: Sequence[WannierInteraction]

    @property
    def tag(self) -> str:
        return f"{self.symbol_i}{self.i} <=> {self.symbol_j}{self.j}"

    @property
    def pdos(self) -> NDArray[np.float64]:
        for interaction in self.wannier_interactions:
            if interaction.pdos is None:
                tbc = f"projected density of states for atomic interaction {self.tag}"

                raise NotComputedError(interaction.tag, tbc, "PDOS")

        return sum(interaction.pdos for interaction in self.wannier_interactions)

    @property
    def wohp(self) -> NDArray[np.float64]:
        return sum(interaction.wohp for interaction in self.wannier_interactions)

    @property
    def wobi(self) -> NDArray[np.float64]:
        return sum(interaction.wobi for interaction in self.wannier_interactions)

    @property
    def ipdos(self) -> NDArray[np.float64]:
        for interaction in self.wannier_interactions:
            if interaction.ipdos is None:
                tbc = f"""integrated projected density of states for atomic interaction
                {self.tag}"""

                raise NotComputedError(interaction.tag, tbc, "IPDOS")

        return sum(interaction.ipdos for interaction in self.wannier_interactions)

    @property
    def iwohp(self) -> NDArray[np.float64]:
        return sum(interaction.iwohp for interaction in self.wannier_interactions)

    @property
    def iwobi(self) -> NDArray[np.float64]:
        return sum(interaction.iwobi for interaction in self.wannier_interactions)


@dataclass(frozen=True, slots=True)
class WannierInteraction:
    i: int
    j: int
    bl_i: NDArray[np.int_]
    bl_j: NDArray[np.int_]

    pdos: NDArray[np.float64] | None = None
    h_ij: np.float64 | None = None
    p_ij: np.float64 | None = None
    ipdos: np.float64 | NDArray[np.float64] | None = None

    coefficients: NDArray[np.float64] | None = None

    @property
    def tag(self) -> str:
        return f"{self.i}{self.bl_i.tolist()} <=> {self.j}{self.bl_j.tolist()}"

    @property
    def wohp(self) -> NDArray[np.float64]:
        if self.h_ij is None or self.pdos is None:
            dep = "H_ij" if self.h_ij is None else "PDOS"

            raise NotComputedError(self.tag, "WOHP", dep)

        return -self.h_ij * self.pdos

    @property
    def wobi(self) -> NDArray[np.float64]:
        if self.p_ij is None or self.pdos is None:
            dep = "P_ij" if self.p_ij is None else "PDOS"

            raise NotComputedError(self.tag, "WOBI", dep)

        return self.p_ij * self.pdos

    @property
    def iwohp(self) -> np.float64 | NDArray[np.float64]:
        if self.h_ij is None or self.ipdos is None:
            dep = "H_ij" if self.h_ij is None else "IPDOS"

            raise NotComputedError(self.tag, "IWOHP", dep)

        return -self.h_ij * self.ipdos

    @property
    def iwobi(self) -> np.float64 | NDArray[np.float64]:
        if self.p_ij is None or self.ipdos is None:
            dep = "P_ij" if self.p_ij is None else "IPDOS"

            raise NotComputedError(self.tag, "IWOBI", dep)

        return self.p_ij * self.ipdos

    def with_coefficients(self, basis: Basis) -> WannierInteraction:
        c_star = (np.exp(-1j * 2 * np.pi * basis.kpoints @ self.bl_i))[
            :, np.newaxis
        ] * basis.u[:, :, self.i]
        c = (np.exp(1j * 2 * np.pi * basis.kpoints @ self.bl_j))[
            :, np.newaxis
        ] * np.conj(basis.u[:, :, self.j])

        coefficients = (c_star * c).real

        return replace(self, coefficients=coefficients)

    def without_coefficients(self) -> WannierInteraction:
        return replace(self, coefficients=None)

    def with_pdos(
        self, total_dos: NDArray[np.float64], resolve_k: bool = False
    ) -> WannierInteraction:
        if self.coefficients is None:
            raise NotComputedError(self.tag, "PDOS", "C")

        pdos_nk = self.coefficients[np.newaxis, :, :] * total_dos

        if resolve_k:
            pdos = np.sum(pdos_nk, axis=2)

        else:
            pdos = np.sum(pdos_nk, axis=(1, 2))

        return replace(self, pdos=pdos)

    def with_h_ij(
        self, hamiltonian: dict[tuple[int, int, int], NDArray[np.complex128]]
    ) -> WannierInteraction:
        bl_vector = tuple(int(component) for component in self.bl_j - self.bl_i)

        if bl_vector in hamiltonian:
            h_ij = hamiltonian[bl_vector][self.i, self.j].real

        else:
            raise KeyError(
                f"""Matrix elements for Bravais lattice vector {bl_vector} are required
                for interaction {self.tag} but were not found in the Wannier Hamiltonian
                provided."""
            )

        return replace(self, h_ij=h_ij)

    def with_p_ij(self, occupation_matrix: NDArray[np.float64]) -> WannierInteraction:
        if self.coefficients is None:
            raise NotComputedError(self.tag, "P_ij", "C")

        p_nk = occupation_matrix * self.coefficients

        p_ij = np.sum(p_nk, axis=(0, 1)) / p_nk.shape[0]

        return replace(self, p_ij=p_ij)

    def with_integrals(
        self, energies: NDArray[np.float64], mu: float
    ) -> WannierInteraction:
        if self.pdos is None:
            raise NotComputedError(self.tag, "IPDOS", "PDOS")

        energies_to_mu = energies[energies <= mu]
        pdos_to_mu = self.pdos[: len(energies_to_mu)]

        ipdos = trapezoid(pdos_to_mu, energies_to_mu, axis=0)

        return replace(self, ipdos=ipdos)


class NotComputedError(Exception):
    abbreviations = {
        "C": "coefficient matrices",
        "PDOS": "projected density of states",
        "H_ij": "element of the Hamiltonian",
        "P_ij": "element of the density matrix",
        "WOHP": "Wannier orbital Hamilton ipdos",
        "WOBI": "Wannier orbital bond index",
        "IPDOS": "integrated projected density of states",
        "IWOHP": "integrated Wannier orbital Hamilton population",
        "IWOBI": "integrated Wannier orbital bond index",
    }

    def __init__(self, tag: str, to_be_computed: str, dependency: str) -> None:
        if to_be_computed in self.abbreviations:
            tbc = self.abbreviations[to_be_computed]

        else:
            tbc = to_be_computed

        dep = self.abbreviations[dependency]

        message = f"""The {dep} must be computed for interaction {tag} before
        the {tbc} can be calculated."""

        super().__init__(message)
