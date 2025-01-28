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
from dataclasses import replace
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
    interactions: tuple[AtomicInteraction, ...]
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
    test_geometry = build_geometry(f"{shared_datadir}/centres.xyz", cell)
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


def test_WannierInteraction_wohp(wannier_interaction, ndarrays_regression, tol) -> None:
    ndarrays_regression.check({"WOHP": wannier_interaction.wohp}, default_tolerance=tol)


def test_WannierInteraction_wobi(wannier_interaction, ndarrays_regression, tol) -> None:
    ndarrays_regression.check({"WOBI": wannier_interaction.wobi}, default_tolerance=tol)


@pytest.mark.parametrize(
    "dos_matrix_none, h_ij_none",
    ((True, False), (False, True), (True, True)),
    ids=(
        "dos_matrix_none, h_ij_set",
        "dos_matrix_set, h_ij_none",
        "dos_matrix_none, h_ij_none",
    ),
)
def test_WannierInteraction_wohp_none(wannier_interaction, dos_matrix_none, h_ij_none):
    if dos_matrix_none:
        wannier_interaction = wannier_interaction._replace(dos_matrix=None)

    if h_ij_none:
        wannier_interaction = wannier_interaction._replace(h_ij=None)

    assert wannier_interaction.wohp is None


@pytest.mark.parametrize(
    "dos_matrix_none, p_ij_none",
    ((True, False), (False, True), (True, True)),
    ids=(
        "dos_matrix_none, p_ij_set",
        "dos_matrix_set, p_ij_none",
        "dos_matrix_none, p_ij_none",
    ),
)
def test_WannierInteraction_wobi_none(wannier_interaction, dos_matrix_none, p_ij_none):
    if dos_matrix_none:
        wannier_interaction = wannier_interaction._replace(dos_matrix=None)

    if p_ij_none:
        wannier_interaction = wannier_interaction._replace(p_ij=None)

    assert wannier_interaction.wobi is None


def test_WannierInteraction_with_integrals(
    wannier_interaction, ndarrays_regression, tol
) -> None:
    energies = np.linspace(-20, 10, 100)
    mu = 0

    wannier_interaction = wannier_interaction.with_integrals(energies, mu)

    ndarrays_regression.check(
        {
            "population": wannier_interaction.population,
            "IWOHP": wannier_interaction.iwohp,
            "IWOBI": wannier_interaction.iwobi,
        },
        default_tolerance=tol,
    )


def test_WannierInteraction_with_integrals_no_dos_matrix(wannier_interaction) -> None:
    energies = np.linspace(-20, 10, 100)
    mu = 0

    wannier_interaction = wannier_interaction._replace(dos_matrix=None)

    with pytest.raises(TypeError):
        wannier_interaction.with_integrals(energies, mu)


def test_WannierInteraction_with_integrals_no_elements(wannier_interaction) -> None:
    energies = np.linspace(-20, 10, 100)
    mu = 0

    wannier_interaction = wannier_interaction._replace(h_ij=None, p_ij=None)
    wannier_interaction = wannier_interaction.with_integrals(energies, mu)

    assert wannier_interaction.iwohp is None and wannier_interaction.iwobi is None


def test_AtomicInteraction_with_summed_descriptors(
    atomic_interaction, ndarrays_regression, tol
) -> None:
    atomic_interaction = atomic_interaction.with_summed_descriptors()

    ndarrays_regression.check(
        {"WOHP": atomic_interaction.wohp, "WOBI": atomic_interaction.wobi},
        default_tolerance=tol,
    )


def test_AtomicInteraction_with_summed_descriptors_no_dos_matrix(
    atomic_interaction,
) -> None:
    base_interaction = atomic_interaction.sub_interactions[0]
    new_interaction = (base_interaction._replace(dos_matrix=None),)
    wannier_interactions = atomic_interaction.sub_interactions + new_interaction

    atomic_interaction = replace(
        atomic_interaction, sub_interactions=wannier_interactions
    )

    with pytest.raises(TypeError):
        atomic_interaction.with_summed_descriptors()


def test_AtomicInteraction_with_summed_descriptors_no_wohp(atomic_interaction) -> None:
    base_interaction = atomic_interaction.sub_interactions[0]
    new_interaction = (base_interaction._replace(h_ij=None),)
    wannier_interactions = atomic_interaction.sub_interactions + new_interaction

    atomic_interaction = replace(
        atomic_interaction, sub_interactions=wannier_interactions
    )
    atomic_interaction = atomic_interaction.with_summed_descriptors()

    assert atomic_interaction.wohp is None


def test_AtomicInteraction_with_summed_descriptors_no_wobi(atomic_interaction) -> None:
    base_interaction = atomic_interaction.sub_interactions[0]
    new_interaction = (base_interaction._replace(p_ij=None),)
    wannier_interactions = atomic_interaction.sub_interactions + new_interaction

    atomic_interaction = replace(
        atomic_interaction, sub_interactions=wannier_interactions
    )
    atomic_interaction = atomic_interaction.with_summed_descriptors()

    assert atomic_interaction.wobi is None
