"""
This module contains a set of simple functions for calculating orbital occupations
from a set of Kohn-Sham eigenvalues. Any of these functions can be used together with
the :py:func:`~pengwann.utils.get_occupation_matrix` function to build the occupation
matrix needed to calculate WOBIs with the :py:class:`~pengwann.dos.DOS` class.
"""

import numpy as np
from math import factorial
from numpy.typing import NDArray
from scipy.special import erf


def fixed(eigenvalues: NDArray[np.float64], mu: float) -> NDArray[np.float64]:
    r"""
    A simple heaviside occupation function.

    .. math::
        f_{nk} = \begin{cases}
        1 \: \mathrm{if} \: \epsilon_{nk} \le \mu \\
        0 \: \mathrm{if} \: \epsilon_{nk} > \mu
        \end{cases}

    Args:
        eigenvalues (NDArray[np.float64]): The Kohn-Sham eigenvalues.
        mu (float): The Fermi level.

    Returns:
        NDArray[np.float64]: The occupation numbers.
    """
    return np.heaviside(-1 * (eigenvalues - mu), 1)


def fermi_dirac(
    eigenvalues: NDArray[np.float64], mu: float, sigma: float
) -> NDArray[np.float64]:
    r"""
    The Fermi-Dirac occupation function.

    .. math::
        f_{nk} = \frac{1}{\exp[\frac{\epsilon_{nk} - \mu}{\sigma}] + 1}

    Args:
        eigenvalues (NDArray[np.float64]): The Kohn-Sham eigenvalues.
        mu (float): The Fermi level.
        sigma (float): The smearing width in eV (in this case = kT for some electronic
            temperature T).

    Returns:
        NDArray[np.float64]: The occupation numbers.
    """
    if sigma <= 0:
        raise ValueError("The smearing width must > 0, {sigma} is <= 0")

    x = (eigenvalues - mu) / sigma

    return 1 / (np.exp(x) + 1)


def gaussian(eigenvalues: np.ndarray, mu: float, sigma: float) -> NDArray[np.float64]:
    r"""
    A Gaussian occupation function.

    .. math::
        f_{nk} = \frac{1}{2}\left[1 -
        \mathrm{erf}\left(\frac{\epsilon_{nk} - \mu}{\sigma}\right)\right]

    Args:
        eigenvalues (NDArray[np.float64]): The Kohn-Sham eigenvalues.
        mu (float): The Fermi level.
        sigma (float): The smearing width in eV.

    Returns:
        NDArray[np.float64]: The occupation numbers.
    """
    if sigma <= 0:
        raise ValueError("The smearing width must > 0, {sigma} is <= 0")

    x = (eigenvalues - mu) / sigma

    return (0.5 * (1 - erf(x))).astype(np.float64)


def cold(
    eigenvalues: NDArray[np.float64], mu: float, sigma: float
) -> NDArray[np.float64]:
    r"""
    The Marzari-Vanderbilt occupation function.

    .. math::
        f_{nk} = \frac{1}{2}\left[\sqrt{\frac{2}{\pi}}\exp\left[-x^{2} -
        \sqrt{2}x - 1/2\right] + 1 - \mathrm{erf}\left(x + \frac{1}{\sqrt{2}}
        \right)\right]

    Where :math:`x = \frac{\epsilon_{nk} - \mu}{\sigma}`.

    Args:
        eigenvalues (NDArray[np.float64]): The Kohn-Sham eigenvalues.
        mu (float): The Fermi level.
        sigma (float): The smearing width in eV.

    Returns:
        NDArray[np.float64]: The occupation numbers.
    """
    if sigma <= 0:
        raise ValueError("The smearing width must > 0, {sigma} is <= 0")

    x = (eigenvalues - mu) / sigma

    return 0.5 * (
        np.sqrt(2 / np.pi) * np.exp(-(x**2) - np.sqrt(2) * x - 0.5) + 1 - erf(x + 0.5)
    )
