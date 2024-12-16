import numpy as np
from math import factorial
from scipy.special import erf # type: ignore

def fixed(eigenvalues: np.ndarray, mu: float) -> np.ndarray:
    """
    A simple heaviside occupation function.

    Args:
        eigenvalues (np.ndarray): The Kohn-Sham eigenvalues.
        mu (float): The Fermi level.

    Returns:
        (np.ndarray): The occupation numbers.
    """
    return np.heaviside(-1 * (eigenvalues - mu), 1)

def fermi_dirac(eigenvalues: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    The Fermi-Dirac occupation function.

    Args:
        eigenvalues (np.ndarray): The Kohn-Sham eigenvalues.
        mu (float): The Fermi level.
        sigma (float): The smearing width in eV (in this case = kT for some
            electronic temperature T).

    Returns:
        (np.ndarray): The occupation numbers.
    """
    x = (eigenvalues - mu) / sigma

    return 1 / (np.exp(x) + 1)

def gaussian(eigenvalues: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    A Gaussian occupation function.

    Args:
        eigenvalues (np.ndarray): The Kohn-Sham eigenvalues.
        mu (float): The Fermi level.
        sigma (float): The smearing width in eV.

    Returns:
        (np.ndarray): The occupation numbers.
    """
    x = (eigenvalues - mu) / sigma

    return 0.5 * (1 - erf(x))

def cold(eigenvalues: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    The Marzari-Vanderbilt occupation function.

    Args:
        eigenvalues (np.ndarray): The Kohn-Sham eigenvalues.
        mu (float): The Fermi level.
        sigma (float): The smearing width in eV.

    Returns:
        (np.ndarray): The occupation numbers.
    """
    x = (eigenvalues - mu) / sigma

    return 0.5 * (np.sqrt(2 / np.pi) * np.exp(-x ** 2 - np.sqrt(2) * x - 0.5) + 1 - erf(x + 0.5))
