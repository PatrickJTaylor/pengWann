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
