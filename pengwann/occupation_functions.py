"""
Occupation functions for reconstructing the ab initio occupation matrix.

This module contains a set of simple functions for calculating orbital occupation
numbers from a set of Kohn-Sham eigenvalues. Any of these functions can be used together
with the :py:func:`~pengwann.occupation_functions.get_occupation_matrix` function to
build the occupation matrix needed to calculated WOBIs with the
:py:class:`~pengwann.descriptors.DescriptorCalculator` class.
"""

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

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf
from typing import Callable


def fixed(eigenvalues: NDArray[np.float64], mu: float) -> NDArray[np.float64]:
    r"""
    A simple heaviside occupation function.

    Parameters
    ----------
    eigenvalues : ndarray[float]
        The Kohn-Sham eigenvalues.
    mu : float
        The Fermi level.

    Returns
    -------
    occupation_matrix : ndarray[float]
        The occupation matrix.

    Notes
    -----
    The definition of this occupation function is simply

    .. math::

        f_{nk} = \begin{cases}
        1\; \mathrm{if}\; \epsilon_{nk} \leq \mu \\
        0\; \mathrm{if}\; \epsilon_{nk} > \mu.
        \end{cases}
    """
    occupation_matrix = np.heaviside(-1 * (eigenvalues - mu), 1)

    return occupation_matrix


def fermi_dirac(
    eigenvalues: NDArray[np.float64], mu: float, sigma: float
) -> NDArray[np.float64]:
    r"""
    The Fermi-Dirac occupation function.

    Parameters
    ----------
    eigenvalues : ndarray[float]
        The Kohn-Sham eigenvalues.
    mu : float
        The Fermi level.
    sigma : float
        The smearing width in eV (= kT for some electronic temperature T).

    Returns
    -------
    occupation_matrix : ndarray[float]
        The occupation matrix.

    Notes
    -----
    The Fermi-Dirac occupation function is defined as

    .. math::

        f_{nk} = \left(\exp\left[\frac{\epsilon_{nk} - \mu}{\sigma}\right] + 1\right)
        ^{-1}.
    """
    if sigma <= 0:
        raise ValueError("The smearing width must > 0, {sigma} is <= 0")

    x = (eigenvalues - mu) / sigma
    occupation_matrix = 1 / (np.exp(x) + 1)

    return occupation_matrix


def gaussian(eigenvalues: np.ndarray, mu: float, sigma: float) -> NDArray[np.float64]:
    r"""
    A Gaussian occupation function.

    Parameters
    ----------
    eigenvalues : ndarray[float]
        The Kohn-Sham eigenvalues.
    mu : float
        The Fermi level.
    sigma : float
        The smearing width in eV.

    Returns
    -------
    occupation_matrix : ndarray[float]
        The occupation matrix.

    Notes
    -----
    The definition of this occupation function is

    .. math::

        f_{nk} = \frac{1}{2}\left[1 - \mathrm{erf}\left(\frac{\epsilon_{nk} -
        \mu}{\sigma}\right)\right]
    """
    if sigma <= 0:
        raise ValueError("The smearing width must > 0, {sigma} is <= 0")

    x = (eigenvalues - mu) / sigma

    return 0.5 * (1 - erf(x))


def cold(
    eigenvalues: NDArray[np.float64], mu: float, sigma: float
) -> NDArray[np.float64]:
    r"""
    The Marzari-Vanderbilt (cold) occupation function.

    Parameters
    ----------
    eigenvalues : ndarray[float]
        The Kohn-Sham eigenvalues.
    mu : float
        The Fermi level.
    sigma : float
        The smearing width in eV.

    Returns
    -------
    occupation_matrix : ndarray[float]
        The occupation matrix.

    Notes
    -----
    The Marzari-Vanderbilt occupation function is defined as :footcite:p:`mv_smearing`

    .. math::

        f_{nk} = \frac{1}{2}\left[\sqrt{\frac{2}{\pi}}\exp\left[-x^{2} - \sqrt{2}x -
        1/2\right] + 1 - \mathrm{erf}\left(x + \frac{1}{\sqrt{2}}\right)\right],

    where :math:`x = \frac{\epsilon_{nk} - \mu}{\sigma}`.

    References
    ----------
    .. footbibliography::
    """
    if sigma <= 0:
        raise ValueError("The smearing width must > 0, {sigma} is <= 0")

    x = (eigenvalues - mu) / sigma

    return 0.5 * (
        np.sqrt(2 / np.pi) * np.exp(-(x**2) - np.sqrt(2) * x - 0.5) + 1 - erf(x + 0.5)
    )


def get_occupation_matrix(
    eigenvalues: NDArray[np.float64],
    mu: float,
    nspin: int,
    occupation_function: Callable = fixed,
    **function_kwargs,
) -> NDArray[np.float64]:
    """
    Compute an occupation matrix.

    Parameters
    ----------
    eigenvalues : ndarray[float]
        The Kohn-Sham eigenvalues.
    mu : float
        The Fermi level.
    nspin : int
        The number of electrons per fully-occupied Kohn-Sham state. For
        non-spin-polarised calculations set to 2, for spin-polarised calculations set
        to 1.
    occupation_function : callable, optional
        The occupation function used to calculate the occupation matrix. Defaults to
        :py:func:`~pengwann.occupation_functions.fixed` (i.e. fixed occupations).
    **function_kwargs
        Additional keyword arguments to be passed to `occupation_function`.

    Returns
    -------
    occupation_matrix : ndarray[float]
        The occupation matrix.

    Notes
    -----
    Ideally the occupation matrix should be read in directly from the ab initio code
    (in which case this function is redundant). Failing that, the occupation matrix can
    be reconstructed so long as the correct occupation function is used.

    Various pre-defined occupation functions (Gaussian, Marzari-Vanderbilt etc) can be
    found in the :py:mod:`~pengwann.occupation_functions` module. If none of these
    match the occupation function used by the ab initio code, a custom occupation
    function can be defined and passed as `occupation_function` (so long as it takes
    `eigenvalues` and `mu` as the first two positional arguments).
    """
    occupation_matrix = occupation_function(eigenvalues, mu, **function_kwargs)

    occupation_matrix *= nspin

    return occupation_matrix.T
