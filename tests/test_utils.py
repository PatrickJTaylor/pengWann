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

import numpy as np

from pengwann.io.wannier90 import read_u
from pengwann.utils import (
    get_spilling_factor,
    integrate_descriptor,
)


def test_get_spilling_factor(shared_datadir, ndarrays_regression, tol) -> None:
    u, _ = read_u(f"{shared_datadir}/wannier90_u.mat")

    num_wann = 8

    spilling_factor = get_spilling_factor(u, num_wann)

    ndarrays_regression.check(
        {"spilling_factor": spilling_factor}, default_tolerance=tol
    )


def test_integrate_descriptor(ndarrays_regression, tol) -> None:
    x = np.linspace(-5, 5, 1000, dtype=np.float64)
    y = x**2
    mu = 0

    integral = integrate_descriptor(x, y, mu)

    ndarrays_regression.check({"integral": integral}, default_tolerance=tol)
