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

from pengwann.io.wannier90 import (
    read,
    read_cell,
    read_eigenvalues,
    read_hamiltonian,
    read_u,
)


def test_read(shared_datadir, ndarrays_regression, tol) -> None:
    kpoints, eigenvalues, u, h = read("wannier90", path=shared_datadir)

    h_000 = h[(0, 0, 0)]

    ndarrays_regression.check(
        {"k-points": kpoints, "eigenvalues": eigenvalues, "U": u, "H_000": h_000},
        default_tolerance=tol,
    )


def test_read_eigenvalues(shared_datadir, ndarrays_regression, tol) -> None:
    num_bands = 12
    num_kpoints = 4096

    eigenvalues = read_eigenvalues(
        f"{shared_datadir}/wannier90.eig", num_bands, num_kpoints
    )

    ndarrays_regression.check({"eigenvalues": eigenvalues}, default_tolerance=tol)


def test_read_u(shared_datadir, ndarrays_regression, tol) -> None:
    u, kpoints = read_u(f"{shared_datadir}/wannier90_u.mat")

    ndarrays_regression.check({"U": u, "kpoints": kpoints}, default_tolerance=tol)


def test_read_hamiltonian(shared_datadir, ndarrays_regression, tol) -> None:
    test_h = read_hamiltonian(f"{shared_datadir}/wannier90_hr.dat")

    for R, matrix in test_h.items():
        assert matrix.shape == (8, 8)

    h_000 = test_h[(0, 0, 0)]

    ndarrays_regression.check({"H_000": h_000}, default_tolerance=tol)


def test_read_u_dis(shared_datadir, ndarrays_regression, tol) -> None:
    _, _, u, _ = read("wannier90", f"{shared_datadir}")

    ndarrays_regression.check({"U": u}, default_tolerance=tol)


def test_read_cell(shared_datadir, ndarrays_regression, tol) -> None:
    cell = read_cell(f"{shared_datadir}/wannier90.win")

    ndarrays_regression.check({"cell": cell}, default_tolerance=tol)


def test_read_cell_bohr(shared_datadir, ndarrays_regression, tol) -> None:
    cell = read_cell(f"{shared_datadir}/wannier90_bohr.win")

    ndarrays_regression.check({"cell": cell}, default_tolerance=tol)


def test_read_cell_not_enough_vectors(shared_datadir) -> None:
    with pytest.raises(ValueError):
        read_cell(f"{shared_datadir}/wannier90_invalid_cell.win")
