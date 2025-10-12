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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from pengwann.interactions import (
    AtomicInteraction,
    AtomicInteractions,
    WannierInteraction,
)

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure


@dataclass(frozen=True, slots=True)
class Geometry:
    symbols: tuple[str, ...]
    coords: NDArray[np.float64]
    cell: NDArray[np.float64]
    distance_matrix: NDArray[np.float64]
    image_matrix: NDArray[np.int32]
    wannier_assignments: tuple[tuple[int, ...], ...]

    @override
    def __str__(self) -> str:
        to_print = [
            "Geometry",
            "========",
            "Cell",
            "----",
            f"{self.cell}",
            "",
            "Wannier assignments",
            "-------------------",
        ]

        assignment_lines: list[str] = []
        site_lines: list[str] = []
        for idx in range(len(self.symbols)):
            symbol = self.symbols[idx]
            coords = self.coords[idx]

            if symbol != "X":
                wannier_indices = self.wannier_assignments[idx]

                assignment_lines.append(f"{symbol}{idx} <= {wannier_indices}")

            site_lines.append(f"{symbol}{idx} {coords}")

        to_print += assignment_lines + ["", "Sites", "-----"] + site_lines

        return "\n".join(to_print) + "\n"

    def as_atoms(self) -> Atoms:
        try:
            from ase import Atoms

        except ImportError as base_error:
            raise ImportError(
                """The as_atoms method requires the ase package, which does not seem to
                be available in the current Python environment."""
            ) from base_error

        atoms = Atoms(
            symbols=self.symbols,
            scaled_positions=self.coords,
            cell=self.cell,
            pbc=True,
            info={"wannier_assignments": self.wannier_assignments},
        )

        return atoms

    def as_structure(self) -> Structure:
        try:
            from pymatgen.core import Structure

        except ImportError as base_error:
            raise ImportError(
                """The as_structure method requires the Pymatgen package, which does
                not seem to be available in the current Python environment."""
            ) from base_error

        return Structure(
            lattice=self.cell,
            species=self.symbols,
            coords=self.coords,
            site_properties={"wannier_assignments": self.wannier_assignments},
        )


def find_onsite_interactions(
    geometry: Geometry, symbols: tuple[str, ...]
) -> AtomicInteractions:
    _validate_symbols(geometry, symbols)

    zero_vector = np.array([0, 0, 0], dtype=np.int32)

    interactions: list[AtomicInteraction] = []
    for idx, symbol in enumerate(geometry.symbols):
        if symbol in symbols:
            wannier_interactions: list[WannierInteraction] = []
            for i in geometry.wannier_assignments[idx]:
                wannier_interaction = WannierInteraction(i, i, zero_vector, zero_vector)

                wannier_interactions.append(wannier_interaction)

            interaction = AtomicInteraction(
                idx, idx, symbol, symbol, tuple(wannier_interactions)
            )
            interactions.append(interaction)

    return AtomicInteractions(tuple(interactions))


def find_interatomic_interactions(
    geometry: Geometry,
    radial_cutoffs: dict[tuple[str, str], float],
) -> AtomicInteractions:
    symbols = tuple({symbol for pair in radial_cutoffs for symbol in pair})
    _validate_symbols(geometry, symbols)

    atom_indices = _label_atom_indices(geometry, symbols)

    interactions: list[AtomicInteraction] = []
    for pair, cutoff in radial_cutoffs.items():
        symbol_i, symbol_j = pair

        offset = 1 if symbol_i == symbol_j else 0

        for idx, i in enumerate(atom_indices[symbol_i]):
            for j in atom_indices[symbol_j][idx + offset :]:
                distance = geometry.distance_matrix[i, j]

                if distance < cutoff:
                    wannier_interactions: list[WannierInteraction] = []
                    for m in geometry.wannier_assignments[i]:
                        for n in geometry.wannier_assignments[j]:
                            bl_i = geometry.image_matrix[i, m]
                            bl_j = geometry.image_matrix[j, n]

                            wannier_interaction = WannierInteraction(m, n, bl_i, bl_j)
                            wannier_interactions.append(wannier_interaction)

                    interaction = AtomicInteraction(
                        i, j, symbol_i, symbol_j, tuple(wannier_interactions)
                    )
                    interactions.append(interaction)

    return AtomicInteractions(tuple(interactions))


def _label_atom_indices(
    geometry: Geometry, symbols: tuple[str, ...]
) -> dict[str, list[int]]:
    atom_indices: dict[str, list[int]] = {}
    for symbol in symbols:
        atom_indices[symbol] = []

    for idx, symbol in enumerate(geometry.symbols):
        if symbol in symbols:
            atom_indices[symbol].append(idx)

    return atom_indices


def _validate_symbols(geometry: Geometry, symbols: tuple[str, ...]) -> None:
    for symbol in symbols:
        if symbol not in geometry.symbols:
            raise ValueError(f"No atoms with label {symbol} found in input geometry.")
