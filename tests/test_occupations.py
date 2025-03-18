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
import pytest
from numpy.typing import NDArray

from pengwann.occupations import (
    cold,
    fermi_dirac,
    fixed,
    gaussian,
    get_occupation_matrix,
)


@pytest.fixture
def eigenvalues() -> NDArray[np.float64]:
    return np.array(
        [
            [-1.00, -0.75, -0.50, -0.25, 0.25, 0.50, 0.75, 1.00],
            [-1.20, -0.66, -0.47, -0.30, 0.34, 0.44, 0.67, 0.98],
        ]
    )


@pytest.fixture
def mu() -> int:
    return 0


@pytest.fixture
def sigma() -> float:
    return 0.2


@pytest.fixture
def nspin() -> int:
    return 2


def test_fixed_occupation_function(eigenvalues, mu, ndarrays_regression, tol) -> None:
    occupations = fixed(eigenvalues, mu)

    ndarrays_regression.check({"occupations": occupations}, default_tolerance=tol)


def test_get_occupation_matrix_default(
    eigenvalues, mu, nspin, ndarrays_regression, tol
) -> None:
    occupations = get_occupation_matrix(eigenvalues, mu, nspin)

    ndarrays_regression.check({"occupations": occupations}, default_tolerance=tol)


@pytest.mark.parametrize(
    "occupation_function",
    (fermi_dirac, gaussian, cold),
    ids=("fermi_dirac", "gaussian", "cold"),
)
class TestOccupationFunctions:
    def test_occupation_function(
        self, occupation_function, eigenvalues, mu, sigma, ndarrays_regression, tol
    ) -> None:
        occupations = occupation_function(eigenvalues, mu, sigma)

        ndarrays_regression.check({"occupations": occupations}, default_tolerance=tol)

    def test_occupation_function_invalid_sigma(
        self, occupation_function, eigenvalues, mu
    ) -> None:
        sigma = -0.2

        with pytest.raises(ValueError):
            occupation_function(eigenvalues, mu, sigma)

    def test_get_occupation_matrix_custom(
        self,
        occupation_function,
        eigenvalues,
        mu,
        nspin,
        sigma,
        ndarrays_regression,
        tol,
    ) -> None:
        occupations = get_occupation_matrix(
            eigenvalues, mu, nspin, occupation_function=occupation_function, sigma=sigma
        )

        ndarrays_regression.check({"occupations": occupations}, default_tolerance=tol)

    def test_get_occupation_matrix_invalid_nspin(
        self, occupation_function, eigenvalues, mu, sigma
    ) -> None:
        nspin = 0.5

        with pytest.raises(ValueError):
            get_occupation_matrix(
                eigenvalues,
                mu,
                nspin,
                occupation_function=occupation_function,
                sigma=sigma,
            )
