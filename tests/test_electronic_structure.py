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

import pytest
import numpy as np
from numpy.typing import NDArray

from pengwann.electronic_structure import (
    Basis,
    compute_spilling_factor,
    compute_total_density_of_states,
)


@pytest.fixture
def unitary_matrices(shared_datadir) -> NDArray[np.complex128]:
    return np.load(f"{shared_datadir}/U.npy")


@pytest.fixture
def kpoints(shared_datadir) -> NDArray[np.float64]:
    return np.load(f"{shared_datadir}/kpoints.npy")


@pytest.fixture
def energies() -> NDArray[np.float64]:
    return np.arange(-5, 5.01, 0.01)


@pytest.fixture
def eigenvalues() -> NDArray[np.float64]:
    return np.array(
        [
            [-1.00, -0.75, -0.50, -0.25, 0.25, 0.50, 0.75, 1.00],
            [-1.20, -0.66, -0.47, -0.30, 0.34, 0.44, 0.67, 0.98],
        ]
    )


def test_Basis_spilling_warning(unitary_matrices, kpoints) -> None:
    unitary_matrices *= 10

    with pytest.warns(UserWarning):
        basis = Basis(unitary_matrices, kpoints)


def test_compute_total_density_of_states(
    energies, eigenvalues, ndarrays_regression, tol
) -> None:
    sigma = 0.05
    nspin = 2

    total_dos = compute_total_density_of_states(energies, eigenvalues, sigma, nspin)

    ndarrays_regression.check({"total_dos": total_dos}, default_tolerance=tol)


def test_compute_total_density_of_states_invalid_sigma(energies, eigenvalues) -> None:
    sigma = -0.05
    nspin = 2

    with pytest.raises(ValueError):
        compute_total_density_of_states(energies, eigenvalues, sigma, nspin)


def test_compute_total_density_of_states_invalid_nspin(energies, eigenvalues) -> None:
    sigma = 0.05
    nspin = 3

    with pytest.raises(ValueError):
        compute_total_density_of_states(energies, eigenvalues, sigma, nspin)


def test_compute_spilling_factor(unitary_matrices, ndarrays_regression) -> None:
    spilling_factor = compute_spilling_factor(unitary_matrices)

    ndarrays_regression.check({"spilling_factor": spilling_factor})
