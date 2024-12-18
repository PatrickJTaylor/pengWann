import pytest
import numpy as np
from pengwann.occupation_functions import fermi_dirac
from pengwann.utils import (
    assign_wannier_centres,
    get_atom_indices,
    get_occupation_matrix,
)
from pymatgen.core import Structure


def test_assign_wannier_centres(datadir) -> None:
    geometry = Structure.from_file(f"{datadir}/structure.vasp")

    ref_wannier_centres = [
        (9,),
        (8,),
        (8,),
        (9,),
        (9,),
        (8,),
        (9,),
        (8,),
        (1, 2, 5, 7),
        (0, 3, 4, 6),
    ]

    assign_wannier_centres(geometry)

    assert geometry.site_properties["wannier_centres"] == ref_wannier_centres


def test_assign_wannier_centres_invalid_structure(datadir) -> None:
    geometry = Structure.from_file(f"{datadir}/invalid_structure.vasp")

    with pytest.raises(ValueError):
        assign_wannier_centres(geometry)


def test_get_atom_indices(datadir) -> None:
    geometry = Structure.from_file(f"{datadir}/structure.vasp")

    test_indices = get_atom_indices(geometry, ("C", "X0+"))
    ref_indices = {"C": (8, 9), "X0+": (0, 1, 2, 3, 4, 5, 6, 7)}

    assert test_indices == ref_indices


def test_get_atom_indices_uniqueness(datadir) -> None:
    geometry = Structure.from_file(f"{datadir}/structure.vasp")

    test_indices = get_atom_indices(geometry, ("C", "X0+"))
    total_indices = test_indices["C"] + test_indices["X0+"]
    total_indices_set = set(total_indices)

    assert len(total_indices_set) == len(total_indices)


def test_get_atom_indices_all_assigned(datadir) -> None:
    geometry = Structure.from_file(f"{datadir}/structure.vasp")

    test_indices = get_atom_indices(geometry, ("C", "X0+"))

    assert len(test_indices["C"]) == 2 and len(test_indices["X0+"]) == 8


def test_get_occupation_matrix_default() -> None:
    eigenvalues = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    mu = 0
    nspin = 2

    test_occupations = get_occupation_matrix(eigenvalues, mu, nspin)
    ref_occupations = np.array([2, 2, 2, 2, 0, 0, 0, 0], dtype=float)

    np.testing.assert_array_equal(test_occupations, ref_occupations, strict=True)


def test_get_occupation_matrix_custom(datadir) -> None:
    eigenvalues = np.array([-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1])
    mu = 0
    nspin = 2
    sigma = 0.2

    test_occupations = get_occupation_matrix(
        eigenvalues, mu, nspin, occupation_function=fermi_dirac, sigma=sigma
    )
    ref_occupations = np.load(f"{datadir}/fermi_dirac.npy")

    np.testing.assert_allclose(test_occupations, ref_occupations)
