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

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class Basis(NamedTuple):
    u: NDArray[np.complex128]
    kpoints: NDArray[np.float64]


def compute_total_density_of_states(
    eigenvalues: NDArray[np.float64],
    energy_range: tuple[float, float],
    resolution: float,
    sigma: float,
) -> NDArray[np.float64]:
    emin, emax = energy_range
    energies = np.arange(emin, emax + resolution, resolution, dtype=np.float64)

    x_mu = energies[:, np.newaxis, np.newaxis] - eigenvalues
    dos = 1 / np.sqrt(np.pi * sigma) * np.exp(-(x_mu**2) / sigma) / eigenvalues.shape[1]
    dos = np.swapaxes(dos, 1, 2)

    return dos
