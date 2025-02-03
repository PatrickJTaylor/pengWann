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

import json
import pytest
import numpy as np
from pengwann.geometry import (
    AtomicInteraction,
    assign_wannier_centres,
    build_geometry,
    identify_interatomic_interactions,
    identify_onsite_interactions,
    WannierInteraction,
)
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


def serialise_interactions(
    interactions: tuple[AtomicInteraction, ...],
) -> dict[str, int | tuple[str, str] | list[int]]:
    serialised_interactions = {"tags": [], "i": [], "j": [], "bl_1": [], "bl_2": []}
    for interaction in interactions:
        serialised_interactions["tags"].append(interaction.tag)

        for w_interaction in interaction.sub_interactions:
            serialised_interactions["i"].append(w_interaction.i)
            serialised_interactions["j"].append(w_interaction.j)

            serial_bl_1 = w_interaction.bl_1.tolist()
            serial_bl_2 = w_interaction.bl_2.tolist()

            serialised_interactions["bl_1"].append(serial_bl_1)
            serialised_interactions["bl_2"].append(serial_bl_2)

    return serialised_interactions


@pytest.fixture
def geometry() -> Structure:
    cell = np.diag([5, 5, 5])
    species = ["X0+", "X0+", "C", "O"]
    coords = [[0.1, 0.1, 0.1], [0.6, 0.6, 0.6], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
    geometry = Structure(cell, species, coords)

    wannier_centres = ((2,), (3,), (0,), (1,))
    geometry.add_site_property("wannier_centres", wannier_centres)

    return geometry


@pytest.fixture
def wannier_interaction() -> WannierInteraction:
    i = 0
    j = 1
    bl_1 = np.array([0, 0, 0])
    bl_2 = np.array([0, 0, 0])
    dos_matrix = np.linspace(0, 50, 100)
    h_ij = 2
    p_ij = 0.5

    return WannierInteraction(
        i=i, j=j, bl_1=bl_1, bl_2=bl_2, dos_matrix=dos_matrix, h_ij=h_ij, p_ij=p_ij
    )


@pytest.fixture
def atomic_interaction(wannier_interaction) -> AtomicInteraction:
    i = 2
    j = 3
    bl_1 = np.array([0, 0, 0])
    bl_2 = np.array([0, 0, 0])
    dos_matrix = np.linspace(0, 25, 100)
    h_ij = 2.5
    p_ij = 0.7

    second_interaction = WannierInteraction(
        i=i,
        j=j,
        bl_1=bl_1,
        bl_2=bl_2,
        dos_matrix=dos_matrix,
        h_ij=h_ij,
        p_ij=p_ij,
    )
    wannier_interactions = (wannier_interaction, second_interaction)

    i, j, symbol_i, symbol_j = 1, 2, "Ga", "As"

    return AtomicInteraction(
        i=i,
        j=j,
        symbol_i=symbol_i,
        symbol_j=symbol_j,
        sub_interactions=wannier_interactions,
    )


def test_build_geometry(shared_datadir) -> None:
    cell = (
        (-1.7803725545451619, -1.7803725545451616, 0.0000000000000000),
        (-1.7803725545451616, 0.0000000000000000, -1.7803725545451616),
        (-0.0000000000000003, -1.7803725545451616, -1.7803725545451616),
    )
    test_geometry = build_geometry("wannier90", f"{shared_datadir}", cell)
    num_wann = len([site for site in test_geometry if site.species_string == "X0+"])

    with open(f"{shared_datadir}/geometry.json", "r") as stream:
        serial = json.load(stream)

    ref_geometry = Structure.from_dict(serial)

    sm = StructureMatcher()

    assert num_wann == 8
    assert sm.fit(test_geometry, ref_geometry)


def test_assign_wannier_centres(geometry, data_regression) -> None:
    geometry.remove_site_property("wannier_centres")

    assign_wannier_centres(geometry)

    data_regression.check(
        {"wannier_centres": geometry.site_properties["wannier_centres"]}
    )


def test_assign_wannier_centres_invalid_structure(geometry) -> None:
    geometry.remove_species(["X0+"])

    with pytest.raises(ValueError):
        assign_wannier_centres(geometry)


@pytest.mark.parametrize("elemental", (True, False), ids=("elemental", "binary"))
def test_identify_interatomic_interactions(
    geometry, elemental, data_regression
) -> None:
    if elemental:
        geometry.replace(3, "C")
        wannier_centres = ((2,), (3,), (0,), (1,))
        geometry.add_site_property("wannier_centres", wannier_centres)
        cutoffs = {("C", "C"): 4.5}

    else:
        cutoffs = {("C", "O"): 4.5}

    interactions = identify_interatomic_interactions(geometry, cutoffs)

    serialised_interactions = serialise_interactions(interactions)

    data_regression.check(serialised_interactions)


def test_find_interactions_no_site_property(geometry) -> None:
    geometry.remove_site_property("wannier_centres")

    cutoffs = {("C", "O"): 1.5}

    with pytest.raises(ValueError):
        identify_interatomic_interactions(geometry, cutoffs)


def test_find_interactions_no_wann(geometry) -> None:
    geometry.remove_species(["X0+"])

    cutoffs = {("C", "C"): 1.5}

    with pytest.raises(ValueError):
        identify_interatomic_interactions(geometry, cutoffs)


def test_identify_onsite_interactions(geometry, data_regression) -> None:
    symbols = ("C", "O")

    interactions = identify_onsite_interactions(geometry, symbols)

    serialised_interactions = serialise_interactions(interactions)

    data_regression.check(serialised_interactions)


def test_identify_onsite_interactions_no_symbols(geometry) -> None:
    symbols = ("B", "N")

    with pytest.raises(ValueError):
        identify_onsite_interactions(geometry, symbols)
