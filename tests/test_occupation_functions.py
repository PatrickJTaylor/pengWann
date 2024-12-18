import pytest
import numpy as np
from pengwann.occupation_functions import cold, fermi_dirac, fixed, gaussian


def test_fixed_occupation_function() -> None:
    eigenvalues = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    mu = 0

    ref_occupations = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    test_occupations = fixed(eigenvalues, mu)

    np.testing.assert_array_equal(test_occupations, ref_occupations, strict=True)


@pytest.mark.parametrize("occupation_function", (fermi_dirac, cold, gaussian))
def test_occupation_function(datadir, occupation_function) -> None:
    eigenvalues = np.array([-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1])
    mu = 0
    sigma = 0.2

    ref_occupations = np.load(f"{datadir}/{occupation_function.__name__}.npy")
    test_occupations = occupation_function(eigenvalues, mu, sigma)

    np.testing.assert_allclose(test_occupations, ref_occupations)


@pytest.mark.parametrize("occupation_function", (fermi_dirac, cold, gaussian))
def test_occupation_function_invalid_sigma(datadir, occupation_function) -> None:
    eigenvalues = np.array([-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1])
    mu = 0
    sigma = -0.2

    with pytest.raises(ValueError):
        occupation_function(eigenvalues, mu, sigma)
