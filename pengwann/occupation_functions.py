"""
This module contains a set of simple functions for calculating orbital
occupations from a set of Kohn-Sham eigenvalues.
"""

import numpy as np
from math import factorial
from scipy.special import erf


def fixed(eigenvalues: np.ndarray, mu: float) -> np.ndarray:
    r"""
    A simple heaviside occupation function.

    .. math::
        f_{nk} = \begin{cases}
        1 \: \mathrm{if} \: \epsilon_{nk} \le \mu \\
        0 \: \mathrm{if} \: \epsilon_{nk} > \mu
        \end{cases}

    Args:
        eigenvalues (np.ndarray): The Kohn-Sham eigenvalues.
        mu (float): The Fermi level.

    Returns:
        np.ndarray: The occupation numbers.
    """
    return np.heaviside(-1 * (eigenvalues - mu), 1)


def fermi_dirac(eigenvalues: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    r"""
    The Fermi-Dirac occupation function.

    .. math::
        f_{nk} = \frac{1}{\exp[\frac{\epsilon_{nk} - \mu}{\sigma}] + 1}

    Args:
        eigenvalues (np.ndarray): The Kohn-Sham eigenvalues.
        mu (float): The Fermi level.
        sigma (float): The smearing width in eV (in this case = kT for some
            electronic temperature T).

    Returns:
        np.ndarray: The occupation numbers.
    """
    x = (eigenvalues - mu) / sigma

    return 1 / (np.exp(x) + 1)


def gaussian(eigenvalues: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    r"""
    A Gaussian occupation function.

    .. math::
        f_{nk} = \frac{1}{2}\left[1 -
        \mathrm{erf}\left(\frac{\epsilon_{nk} - \mu}{\sigma}\right)\right]

    Args:
        eigenvalues (np.ndarray): The Kohn-Sham eigenvalues.
        mu (float): The Fermi level.
        sigma (float): The smearing width in eV.

    Returns:
        np.ndarray: The occupation numbers.
    """
    x = (eigenvalues - mu) / sigma

    return 0.5 * (1 - erf(x))


def cold(eigenvalues: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    r"""
    The Marzari-Vanderbilt occupation function.

    .. math::
        f_{nk} = \frac{1}{2}\left[\sqrt{\frac{2}{\pi}}\exp\left[-x^{2} -
        \sqrt{2}x - 1/2\right] + 1 - \mathrm{erf}\left(x + \frac{1}{\sqrt{2}}
        \right)\right]

    Where :math:`x = \frac{\epsilon_{nk} - \mu}{\sigma}`.

    Args:
        eigenvalues (np.ndarray): The Kohn-Sham eigenvalues.
        mu (float): The Fermi level.
        sigma (float): The smearing width in eV.

    Returns:
        np.ndarray: The occupation numbers.
    """
    x = (eigenvalues - mu) / sigma

    return 0.5 * (
        np.sqrt(2 / np.pi) * np.exp(-(x**2) - np.sqrt(2) * x - 0.5) + 1 - erf(x + 0.5)
    )
