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

from typing import NamedTuple, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pengwann.interactions import (
    AtomicInteraction,
    AtomicInteractions,
    WannierInteraction,
)

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure


class Geometry(NamedTuple):
    symbols: tuple[str, ...]
    coords: NDArray[np.float64]
    cell: NDArray[np.float64]
    distance_matrix: NDArray[np.float64]
    image_matrix: NDArray[np.int32]
    wannier_assignments: tuple[tuple[int, ...], ...]

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

        assignment_lines, site_lines = [], []
        for idx in range(len(self.symbols)):
            symbol = self.symbols[idx]
            coords = self.coords[idx]

            if symbol != "X":
                wannier_indices = self.wannier_assignments[idx]

                assignment_lines.append(f"{symbol}{idx} <= {wannier_indices}")

            site_lines.append(f"{symbol}{idx} {coords}")

        to_print += assignment_lines + ["", "Sites", "-----"] + site_lines

        return "\n".join(to_print) + "\n"

    def find_onsite_interactions(self, symbols: tuple[str, ...]) -> AtomicInteractions:
        zero_vector = np.array([0, 0, 0])

        interactions = []
        for idx, symbol in enumerate(self.symbols):
            if symbol in symbols:
                wannier_interactions = []
                for i in self.wannier_assignments[idx]:
                    wannier_interaction = WannierInteraction(
                        i, i, zero_vector, zero_vector
                    )

                    wannier_interactions.append(wannier_interaction)

                interaction = AtomicInteraction(
                    idx, idx, symbol, symbol, tuple(wannier_interactions)
                )
                interactions.append(interaction)

        if not interactions:
            raise ValueError(f"No atoms matching symbols in {symbols} found.")

        return AtomicInteractions(tuple(interactions))

    def find_interatomic_interactions(
        self,
        radial_cutoffs: dict[tuple[str, str], float],
    ) -> AtomicInteractions:
        symbols = tuple({symbol for pair in radial_cutoffs for symbol in pair})
        atom_indices = self._atom_labelled_indices(symbols)

        interactions = []
        for pair, cutoff in radial_cutoffs.items():
            symbol_i, symbol_j = pair

            offset = 1 if symbol_i == symbol_j else 0

            for idx, i in enumerate(atom_indices[symbol_i]):
                for j in atom_indices[symbol_j][idx + offset :]:
                    distance = self.distance_matrix[i, j]

                    if distance < cutoff:
                        wannier_interactions = []
                        for m in self.wannier_assignments[i]:
                            for n in self.wannier_assignments[j]:
                                bl_i = self.image_matrix[i, m]
                                bl_j = self.image_matrix[j, n]

                                wannier_interaction = WannierInteraction(
                                    m, n, bl_i, bl_j
                                )
                                wannier_interactions.append(wannier_interaction)

                        wannier_interactions = tuple(wannier_interactions)
                        interaction = AtomicInteraction(
                            i, j, symbol_i, symbol_j, wannier_interactions
                        )
                        interactions.append(interaction)

        return AtomicInteractions(tuple(interactions))

    def _atom_labelled_indices(self, symbols: tuple[str, ...]) -> dict[str, list[int]]:
        atom_indices = {}
        for symbol in symbols:
            atom_indices[symbol] = []

        for idx, symbol in enumerate(self.symbols):
            if symbol in symbols:
                atom_indices[symbol].append(idx)

        return atom_indices

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
