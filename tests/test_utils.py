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
from multiprocessing.shared_memory import SharedMemory
from pengwann.utils import (
    allocate_shared_memory,
    get_atom_indices,
    integrate_descriptor,
)
from pymatgen.core import Structure


def test_get_atom_indices(data_regression) -> None:
    cell = np.diag([5, 5, 5])
    species = ["X0+", "C", "O"]
    coords = [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
    geometry = Structure(cell, species, coords)

    indices = get_atom_indices(geometry, ("C", "O", "X0+"))

    data_regression.check(indices)


def test_integrate_descriptor(ndarrays_regression, tol) -> None:
    x = np.linspace(-5, 5, 1000, dtype=np.float64)
    y = x**2
    mu = 0

    integral = integrate_descriptor(x, y, mu)

    ndarrays_regression.check({"integral": integral}, default_tolerance=tol)
