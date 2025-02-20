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
from pengwann.geometry import (
    AtomicInteraction,
    Geometry,
    identify_interatomic_interactions,
    identify_onsite_interactions,
    Site,
)


def build_geometry(symbols: list[str]) -> Geometry:
    cell = np.diag([5, 5, 5])

    coords = np.array(
        [[0.1, 0.1, 0.1], [0.6, 0.6, 0.6], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
    )
    sites = tuple(
        Site(symbol, idx, coords)
        for idx, (symbol, coords) in enumerate(zip(symbols, coords))
    )

    return Geometry(sites, cell)


def serialise_interactions(
    interactions: tuple[AtomicInteraction, ...],
) -> dict[str, int | tuple[str, str] | list[int]]:
    serialised_interactions = {"tags": [], "i": [], "j": [], "bl_i": [], "bl_j": []}
    for interaction in interactions:
        serialised_interactions["tags"].append(interaction.tag)

        for w_interaction in interaction.sub_interactions:
            serialised_interactions["i"].append(w_interaction.i)
            serialised_interactions["j"].append(w_interaction.j)

            serial_bl_i = w_interaction.bl_i.tolist()
            serial_bl_j = w_interaction.bl_j.tolist()

            serialised_interactions["bl_i"].append(serial_bl_i)
            serialised_interactions["bl_j"].append(serial_bl_j)

    return serialised_interactions


@pytest.fixture
def geometry() -> Geometry:
    symbols = ["X", "X", "C", "O"]

    return build_geometry(symbols)


@pytest.fixture
def geometry_elemental() -> Geometry:
    symbols = ["X", "X", "C", "C"]

    return build_geometry(symbols)


@pytest.fixture
def geometry_no_x() -> Geometry:
    symbols = ["C", "C", "C", "O"]

    return build_geometry(symbols)


def test_Geometry_from_xyz(
    shared_datadir, data_regression, ndarrays_regression, tol
) -> None:
    geometry = Geometry.from_xyz("wannier90", f"{shared_datadir}")

    symbols, indices, coords_list = [], [], []
    for site in geometry:
        symbols.append(site.symbol)
        indices.append(site.index)
        coords_list.append(site.coords)

    coords = np.vstack(coords_list)

    data_regression.check({"symbols": symbols, "indices": indices})
    ndarrays_regression.check(
        {"cell": geometry.cell, "coords": coords}, default_tolerance=tol
    )


def test_Geometry_from_xyz_with_cell(
    shared_datadir, data_regression, ndarrays_regression, tol
) -> None:
    cell = (
        (-1.7803725545451619, -1.7803725545451616, 0.0000000000000000),
        (-1.7803725545451616, 0.0000000000000000, -1.7803725545451616),
        (-0.0000000000000003, -1.7803725545451616, -1.7803725545451616),
    )
    geometry = Geometry.from_xyz("wannier90", f"{shared_datadir}", cell)

    symbols, indices, coords_list = [], [], []
    for site in geometry:
        symbols.append(site.symbol)
        indices.append(site.index)
        coords_list.append(site.coords)

    coords = np.vstack(coords_list)

    data_regression.check({"symbols": symbols, "indices": indices})
    ndarrays_regression.check(
        {"cell": geometry.cell, "coords": coords}, default_tolerance=tol
    )


def test_Geometry_length(geometry) -> None:
    assert len(geometry) == 4


def test_Geometry_str(geometry, data_regression) -> None:
    geometry_str = str(geometry)

    data_regression.check({"str": geometry_str})


def test_Geometry_slice(geometry, ndarrays_regression, tol) -> None:
    site = geometry[0]

    assert site.symbol == "X"
    assert site.index == 0

    ndarrays_regression.check({"coords": site.coords}, default_tolerance=tol)


def test_Geometry_distance_and_image_matrices(
    geometry, ndarrays_regression, tol
) -> None:
    distance_matrix, image_matrix = geometry.distance_and_image_matrices

    ndarrays_regression.check(
        {"distance_matrix": distance_matrix, "image_matrix": image_matrix},
        default_tolerance=tol,
    )


def test_Geometry_wannier_assignments(geometry, data_regression) -> None:
    assignments = geometry.wannier_assignments

    data_regression.check({"wannier_assignments": assignments})


def test_Geometry_wannier_assignments_no_wann(geometry_no_x) -> None:
    with pytest.raises(ValueError):
        geometry_no_x.wannier_assignments


def test_identify_interatomic_interactions_elemental(
    geometry_elemental, data_regression
) -> None:
    cutoffs = {("C", "C"): 4.5}

    interactions = identify_interatomic_interactions(geometry_elemental, cutoffs)

    serialised_interactions = serialise_interactions(interactions)

    data_regression.check(serialised_interactions)


def test_identify_interatomic_interactions_binary(geometry, data_regression) -> None:
    cutoffs = {("C", "O"): 4.5}

    interactions = identify_interatomic_interactions(geometry, cutoffs)

    serialised_interactions = serialise_interactions(interactions)

    data_regression.check(serialised_interactions)


def test_identify_onsite_interactions(geometry, data_regression) -> None:
    symbols = ("C", "O")

    interactions = identify_onsite_interactions(geometry, symbols)

    serialised_interactions = serialise_interactions(interactions)

    data_regression.check(serialised_interactions)


def test_identify_onsite_interactions_no_symbols(geometry) -> None:
    symbols = ("B", "N")

    with pytest.raises(ValueError):
        identify_onsite_interactions(geometry, symbols)
