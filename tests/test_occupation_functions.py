import pytest
import numpy as np
from pengwann.occupation_functions import cold, fermi_dirac, fixed, gaussian


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
