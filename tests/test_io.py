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

from pengwann.io import (
    read_basis,
    read_cell,
    read_eigenvalues,
    read_geometry,
    read_hamiltonian,
    read_wannier90_outputs,
    read_unitary_matrices,
    read_xyz,
)


def test_read_wannier90_outputs(
    shared_datadir, data_regression, ndarrays_regression, tol
) -> None:
    geometry, basis, eigenvalues = read_wannier90_outputs(
        "wannier90", path=shared_datadir
    )

    data_regression.check(
        {
            "symbols": geometry.symbols,
            "wannier_assignments": geometry.wannier_assignments,
        }
    )

    ndarrays_regression.check(
        {
            "frac_coords": geometry.coords,
            "cell": geometry.cell,
            "distance_matrix": geometry.distance_matrix,
            "image_matrix": geometry.image_matrix,
            "U": basis.u,
            "kpoints": basis.kpoints,
            "eigenvalues": eigenvalues,
        }
    )


def test_read_geometry(
    shared_datadir, data_regression, ndarrays_regression, tol
) -> None:
    geometry = read_geometry("wannier90", path=shared_datadir)

    data_regression.check(
        {
            "symbols": geometry.symbols,
            "wannier_assignments": geometry.wannier_assignments,
        }
    )

    ndarrays_regression.check(
        {
            "frac_coords": geometry.coords,
            "cell": geometry.cell,
            "distance_matrix": geometry.distance_matrix,
            "image_matrix": geometry.image_matrix,
        }
    )


def test_read_geometry_no_X(shared_datadir) -> None:
    with pytest.raises(ValueError):
        read_geometry("wannier90_no_X", path=shared_datadir)


def test_read_basis(shared_datadir, ndarrays_regression, tol) -> None:
    basis = read_basis("wannier90", path=shared_datadir)

    ndarrays_regression.check({"U": basis.u, "kpoints": basis.kpoints})


def test_read_hamiltonian(shared_datadir, ndarrays_regression, tol) -> None:
    test_h = read_hamiltonian("wannier90", path=shared_datadir)

    for R, matrix in test_h.items():
        assert matrix.shape == (8, 8)

    h_000 = test_h[(0, 0, 0)]

    ndarrays_regression.check({"H_000": h_000}, default_tolerance=tol)


def test_read_eigenvalues(shared_datadir, ndarrays_regression, tol) -> None:
    num_bands = 12
    num_kpoints = 4096

    eigenvalues = read_eigenvalues(
        f"{shared_datadir}/wannier90.eig", num_bands, num_kpoints
    )

    ndarrays_regression.check({"eigenvalues": eigenvalues}, default_tolerance=tol)


def test_read_unitary_matrices(shared_datadir, ndarrays_regression, tol) -> None:
    u, kpoints = read_unitary_matrices(f"{shared_datadir}/wannier90_u.mat")

    ndarrays_regression.check({"U": u, "kpoints": kpoints}, default_tolerance=tol)


def test_read_cell(shared_datadir, ndarrays_regression, tol) -> None:
    cell = read_cell(f"{shared_datadir}/wannier90.wout")

    ndarrays_regression.check({"cell": cell}, default_tolerance=tol)


def test_read_xyz(shared_datadir, data_regression, ndarrays_regression, tol) -> None:
    symbols, coords = read_xyz(f"{shared_datadir}/wannier90_centres.xyz")

    data_regression.check(symbols)
    ndarrays_regression.check({"coords": coords}, default_tolerance=tol)
