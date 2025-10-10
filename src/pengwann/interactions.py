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

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import cast, TypeVar

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AtomicInteractions:
    atomic_interactions: tuple[AtomicInteraction, ...]

    def __iter__(self) -> Iterator[AtomicInteraction]:
        return iter(self.atomic_interactions)

    def __getitem__(
        self, key: int | tuple[int, int]
    ) -> AtomicInteraction | tuple[AtomicInteraction, ...]:
        indices = _slice_index_matrix(key, self._index_matrix)
        interactions = tuple(self.atomic_interactions[idx] for idx in indices)

        if isinstance(key, int):
            return interactions

        else:
            return interactions[0]

    def __len__(self) -> int:
        return len(self.atomic_interactions)

    def __str__(self) -> str:
        to_print = ["Atomic interactions"]

        underline = ["=" for _ in to_print[-1]]
        to_print.append("".join(underline))

        for interaction in self:
            to_print.append(interaction.tag)

        return "\n".join(to_print) + "\n"

    @cached_property
    def _index_matrix(self) -> list[list[list[int]]]:
        return _build_index_matrix(self.atomic_interactions)

    def filter_by_species(
        self, symbols: Sequence[str]
    ) -> tuple[AtomicInteraction, ...]:
        symbol_set = set(symbols)
        interactions = tuple(
            interaction
            for interaction in self
            if set((interaction.symbol_i, interaction.symbol_j)) <= symbol_set
        )

        if not interactions:
            raise ValueError(f"No interactions involving {symbols} found.")

        return interactions


@dataclass(frozen=True)
class AtomicInteraction:
    i: int
    j: int
    symbol_i: str
    symbol_j: str

    wannier_interactions: tuple[WannierInteraction, ...]

    def __iter__(self) -> Iterator[WannierInteraction]:
        return iter(self.wannier_interactions)

    def __getitem__(
        self, key: int | tuple[int, int]
    ) -> WannierInteraction | tuple[WannierInteraction, ...]:
        indices = _slice_index_matrix(key, self._index_matrix)
        interactions = tuple(self.wannier_interactions[idx] for idx in indices)

        if isinstance(key, int):
            return interactions

        else:
            return interactions[0]

    def __len__(self) -> int:
        return len(self.wannier_interactions)

    def __str__(self) -> str:
        to_print = [f"Atomic Interaction {self.tag}"]

        underline = ["=" for _ in to_print[-1]]
        to_print.append("".join(underline))

        print_names = (
            ("pdos", "PDOS"),
            ("wohp", "WOHP"),
            ("wobi", "WOBI"),
            ("ipdos", "IPDOS"),
            ("iwohp", "IWOHP"),
            ("iwobi", "IWOBI"),
        )
        for attribute_name, print_name in print_names:
            value = getattr(self, attribute_name)

            if isinstance(value, np.ndarray):
                print_value = "Computed"

            else:
                print_value = value

            line = f"{print_name} => {print_value}"

            to_print.append(line)

        to_print.append("\n")

        subtitle = "Associated Wannier interactions"
        subtitle_underline = ["-" for _ in subtitle]
        to_print.extend((subtitle, "".join(subtitle_underline)))

        for interaction in self:
            to_print.append(interaction.tag)

        return "\n".join(to_print) + "\n"

    @cached_property
    def _index_matrix(self) -> list[list[list[int]]]:
        return _build_index_matrix(self.wannier_interactions)

    @property
    def tag(self) -> str:
        return f"{self.symbol_i}{self.i} <=> {self.symbol_j}{self.j}"

    @property
    def pdos(self) -> NDArray[np.float64] | None:
        return _sum_or_none(
            [interaction.pdos for interaction in self.wannier_interactions]
        )

    @property
    def wohp(self) -> NDArray[np.float64] | None:
        return _sum_or_none(
            [interaction.wohp for interaction in self.wannier_interactions]
        )

    @property
    def wobi(self) -> NDArray[np.float64] | None:
        return _sum_or_none(
            [interaction.wobi for interaction in self.wannier_interactions]
        )

    @property
    def ipdos(self) -> np.float64 | NDArray[np.float64] | None:
        return _sum_or_none(
            [interaction.ipdos for interaction in self.wannier_interactions]
        )

    @property
    def iwohp(self) -> np.float64 | NDArray[np.float64] | None:
        return _sum_or_none(
            [interaction.iwohp for interaction in self.wannier_interactions]
        )

    @property
    def iwobi(self) -> np.float64 | NDArray[np.float64] | None:
        return _sum_or_none(
            [interaction.iwobi for interaction in self.wannier_interactions]
        )


@dataclass(frozen=True, slots=True)
class WannierInteraction:
    i: int
    j: int
    bl_i: NDArray[np.int32]
    bl_j: NDArray[np.int32]

    pdos: NDArray[np.float64] | None = None
    h_ij: np.float64 | None = None
    p_ij: np.float64 | None = None
    ipdos: np.float64 | NDArray[np.float64] | None = None

    coefficients: NDArray[np.float64] | None = None

    def __str__(self) -> str:
        to_print = [f"Wannier Interaction {self.tag}"]

        underline = ["=" for _ in to_print[-1]]
        to_print.append("".join(underline))

        print_names = (
            ("pdos", "PDOS"),
            ("h_ij", "H_ij"),
            ("p_ij", "P_ij"),
            ("ipdos", "IPDOS"),
            ("iwohp", "IWOHP"),
            ("iwobi", "IWOBI"),
        )
        for attribute_name, print_name in print_names:
            value = getattr(self, attribute_name)

            if isinstance(value, np.ndarray):
                print_value = "Computed"

            else:
                print_value = value

            line = f"{print_name} => {print_value}"

            to_print.append(line)

        return "\n".join(to_print) + "\n"

    @property
    def tag(self) -> str:
        return f"{self.i}{self.bl_i.tolist()} <=> {self.j}{self.bl_j.tolist()}"

    @property
    def wohp(self) -> NDArray[np.float64] | None:
        if self.h_ij is None or self.pdos is None:
            return None

        return -1 * self.h_ij * self.pdos

    @property
    def wobi(self) -> NDArray[np.float64] | None:
        if self.p_ij is None or self.pdos is None:
            return None

        return self.p_ij * self.pdos

    @property
    def iwohp(self) -> np.float64 | NDArray[np.float64] | None:
        if self.h_ij is None or self.ipdos is None:
            return None

        return -1 * self.h_ij * self.ipdos

    @property
    def iwobi(self) -> np.float64 | NDArray[np.float64] | None:
        if self.p_ij is None or self.ipdos is None:
            return None

        return self.p_ij * self.ipdos


T = TypeVar("T")


def _sum_or_none(
    descriptors: list[T | None],
) -> T | None:
    if any(descriptor is None for descriptor in descriptors):
        return None

    return cast(T, np.sum(descriptors, axis=0))  # pyright: ignore[reportArgumentType, reportCallIssue]


def _slice_index_matrix(
    key: int | tuple[int, int], index_matrix: list[list[list[int]]]
) -> list[int]:
    match key:
        case (i, j):
            indices = index_matrix[i][j]

        case i:
            indices = [idx for col_indices in index_matrix[i] for idx in col_indices]

    if not indices:
        raise IndexError(f"No interactions found for provided indices: {key}")

    return indices


def _build_index_matrix(
    interactions: tuple[AtomicInteraction, ...] | tuple[WannierInteraction, ...],
) -> list[list[list[int]]]:
    max_idx = max(max(interaction.i, interaction.j) for interaction in interactions)
    index_matrix: list[list[list[int]]] = [
        [[] for _ in range(max_idx + 1)] for _ in range(max_idx + 1)
    ]
    for idx, interaction in enumerate(interactions):
        i, j = interaction.i, interaction.j

        index_matrix[i][j].append(idx)

        if i != j:
            index_matrix[j][i].append(idx)

    return index_matrix
