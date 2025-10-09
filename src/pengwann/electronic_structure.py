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

import warnings
from dataclasses import dataclass
from textwrap import dedent

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Basis:
    u: NDArray[np.complex128]
    kpoints: NDArray[np.float64]

    def __post_init__(self) -> None:
        spilling_factor = compute_spilling_factor(self.u)
        rounded_spilling_factor = abs(round(spilling_factor, ndigits=8))

        if rounded_spilling_factor > 0:
            warnings.warn(
                dedent(f"""
            The spilling factor = {rounded_spilling_factor}.

            It is advisable to verify that the spilling factor is sufficiently low. For
            Wannier functions derived from energetically isolated bands, it should be
            (within machine precision) strictly 0. For Wannier functions derived using
            disentanglement, the spilling factor should still be very close to 0.

            If the spilling factor is significantly > 0, this implies that there are
            parts of the Bloch subspace that the Wannier basis does not span and thus
            any results derived from the Wannier basis should be analysed with caution.
            """)
            )


def compute_total_density_of_states(
    energies: NDArray[np.float64],
    eigenvalues: NDArray[np.float64],
    sigma: float,
    nspin: int,
) -> NDArray[np.float64]:
    x_mu = energies[:, np.newaxis, np.newaxis] - eigenvalues
    dos = 1 / np.sqrt(np.pi * sigma) * np.exp(-(x_mu**2) / sigma) / eigenvalues.shape[1]

    dos *= nspin

    return dos


def compute_spilling_factor(u: NDArray[np.complex128]) -> np.float64:
    u_star = np.conj(u)
    overlaps = (u_star * u).real

    num_kpoints, num_wann = u.shape[0], u.shape[-1]

    spilling_factor = 1 - np.sum(overlaps) / num_kpoints / num_wann

    return spilling_factor
