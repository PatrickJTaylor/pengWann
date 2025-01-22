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
from pengwann.occupation_functions import (
    cold,
    fermi_dirac,
    fixed,
    gaussian,
    get_occupation_matrix,
)


def test_fixed_occupation_function(ndarrays_regression) -> None:
    eigenvalues = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    mu = 0

    occupations = fixed(eigenvalues, mu)

    ndarrays_regression.check(
        {"occupations": occupations}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


@pytest.mark.parametrize(
    "occupation_function",
    (fermi_dirac, gaussian, cold),
    ids=("fermi_dirac", "gaussian", "cold"),
)
def test_occupation_function(occupation_function, ndarrays_regression) -> None:
    eigenvalues = np.array([-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1])
    mu = 0
    sigma = 0.2

    occupations = occupation_function(eigenvalues, mu, sigma)

    ndarrays_regression.check(
        {"occupations": occupations}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


@pytest.mark.parametrize(
    "occupation_function",
    (fermi_dirac, gaussian, cold),
    ids=("fermi_dirac", "gaussian", "cold"),
)
def test_occupation_function_invalid_sigma(occupation_function) -> None:
    eigenvalues = np.array([-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1])
    mu = 0
    sigma = -0.2

    with pytest.raises(ValueError):
        occupation_function(eigenvalues, mu, sigma)


def test_get_occupation_matrix_default(ndarrays_regression) -> None:
    eigenvalues = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    mu = 0
    nspin = 2

    occupations = get_occupation_matrix(eigenvalues, mu, nspin)

    ndarrays_regression.check(
        {"occupations": occupations}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_get_occupation_matrix_custom(ndarrays_regression) -> None:
    eigenvalues = np.array([-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1])
    mu = 0
    nspin = 2
    sigma = 0.2

    occupations = get_occupation_matrix(
        eigenvalues, mu, nspin, occupation_function=fermi_dirac, sigma=sigma
    )

    ndarrays_regression.check(
        {"occupations": occupations}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )
