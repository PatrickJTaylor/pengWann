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
from dataclasses import replace
from pengwann.interactions import (
    AtomicInteractionContainer,
    AtomicInteraction,
    WannierInteraction,
)
from typing import Any


def none_to_nan(data: Any) -> Any:
    if data is None:
        return np.nan

    else:
        return data


@pytest.fixture
def wannier_interaction() -> WannierInteraction:
    i = 0
    j = 1
    bl_i = np.array([0, 0, 0])
    bl_j = np.array([0, 0, 0])
    dos_matrix = np.linspace(0, 50, 100)
    h_ij = 2
    p_ij = 0.5

    return WannierInteraction(
        i=i, j=j, bl_i=bl_i, bl_j=bl_j, dos_matrix=dos_matrix, h_ij=h_ij, p_ij=p_ij
    )


@pytest.fixture
def atomic_interaction(wannier_interaction) -> AtomicInteraction:
    i = 2
    j = 3
    bl_i = np.array([1, 0, 0])
    bl_j = np.array([0, 0, 0])
    dos_matrix = np.linspace(0, 25, 100)
    h_ij = 2.5
    p_ij = 0.7

    second_interaction = WannierInteraction(
        i=i,
        j=j,
        bl_i=bl_i,
        bl_j=bl_j,
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


@pytest.fixture
def interaction_container(
    wannier_interaction, atomic_interaction
) -> AtomicInteractionContainer:
    i = 4
    j = 5
    bl_i = np.array([0, 1, 0])
    bl_j = np.array([0, 0, 0])
    dos_matrix = np.linspace(0, 30, 100)
    h_ij = 1.5
    p_ij = 0.2

    second_interaction = WannierInteraction(
        i=i,
        j=j,
        bl_i=bl_i,
        bl_j=bl_j,
        dos_matrix=dos_matrix,
        h_ij=h_ij,
        p_ij=p_ij,
    )
    wannier_interactions = (wannier_interaction, second_interaction)

    i, j, symbol_i, symbol_j = 1, 4, "Ga", "As"
    second_atomic_interaction = AtomicInteraction(
        i=i,
        j=j,
        symbol_i=symbol_i,
        symbol_j=symbol_j,
        sub_interactions=wannier_interactions,
    )

    return AtomicInteractionContainer(
        sub_interactions=(atomic_interaction, second_atomic_interaction)
    )


def test_WannierInteraction_wohp(wannier_interaction, ndarrays_regression, tol) -> None:
    ndarrays_regression.check({"WOHP": wannier_interaction.wohp}, default_tolerance=tol)


def test_WannierInteraction_wobi(wannier_interaction, ndarrays_regression, tol) -> None:
    ndarrays_regression.check({"WOBI": wannier_interaction.wobi}, default_tolerance=tol)


def test_WannierInteraction_wohp_no_dos_matrix(wannier_interaction) -> None:
    wannier_interaction = wannier_interaction._replace(dos_matrix=None)

    assert wannier_interaction.wohp is None


def test_WannierInteraction_wohp_no_h_ij(wannier_interaction) -> None:
    wannier_interaction = wannier_interaction._replace(h_ij=None)

    assert wannier_interaction.wohp is None


def test_WannierInteraction_wobi_no_dos_matrix(wannier_interaction) -> None:
    wannier_interaction = wannier_interaction._replace(dos_matrix=None)

    assert wannier_interaction.wobi is None


def test_WannierInteraction_wobi_no_p_ij(wannier_interaction) -> None:
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

    assert wannier_interaction.iwohp is None
    assert wannier_interaction.iwobi is None


def test_WannierInteraction_str(wannier_interaction, data_regression) -> None:
    wannier_interaction_str = str(wannier_interaction)

    data_regression.check({"str": wannier_interaction_str})


def test_AtomicInteraction_slice_2_indices(atomic_interaction) -> None:
    i = 0
    j = 1

    wannier_interaction = atomic_interaction[i, j]

    assert wannier_interaction.i == i
    assert wannier_interaction.j == j


def test_AtomicInteraction_slice_no_indices(atomic_interaction) -> None:
    i = 1
    j = 3

    with pytest.raises(ValueError):
        atomic_interaction[i, j]


def test_AtomicInteraction_slice_1_index(wannier_interaction) -> None:
    i = 0
    j = 3
    bl_i = np.array([1, 0, 0])
    bl_j = np.array([0, 0, 0])

    second_interaction = WannierInteraction(
        i=i,
        j=j,
        bl_i=bl_i,
        bl_j=bl_j,
    )
    wannier_interactions = (wannier_interaction, second_interaction)

    i, j, symbol_i, symbol_j = 1, 2, "Ga", "As"

    atomic_interaction = AtomicInteraction(
        i=i,
        j=j,
        symbol_i=symbol_i,
        symbol_j=symbol_j,
        sub_interactions=wannier_interactions,
    )

    i = 0

    wannier_interactions = atomic_interaction[i]

    for w_interaction in wannier_interactions:
        assert w_interaction.i == i


def test_AtomicInteraction_length(atomic_interaction) -> None:
    assert len(atomic_interaction) == 2


def test_AtomicInteraction_str(atomic_interaction, data_regression) -> None:
    atomic_interaction_str = str(atomic_interaction)

    data_regression.check({"str": atomic_interaction_str})


def test_AtomicInteraction_with_summed_descriptors(
    atomic_interaction, ndarrays_regression, tol
) -> None:
    atomic_interaction = atomic_interaction.with_summed_descriptors()

    ndarrays_regression.check(
        {
            "DOS matrix": atomic_interaction.dos_matrix,
            "WOHP": atomic_interaction.wohp,
            "WOBI": atomic_interaction.wobi,
        },
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


@pytest.mark.parametrize("valence_count", (2, None), ids=("calc_charge", "no_charge"))
def test_AtomicInteraction_with_integrals(
    atomic_interaction, valence_count, ndarrays_regression, tol
) -> None:
    energies = np.linspace(-20, 10, 100)
    mu = 0

    atomic_interaction = atomic_interaction.with_summed_descriptors()
    atomic_interaction = atomic_interaction.with_integrals(
        energies, mu, resolve_orbitals=True, valence_count=valence_count
    )

    integrals = {
        "population": atomic_interaction.population,
        "charge": none_to_nan(atomic_interaction.charge),
        "IWOHP": atomic_interaction.iwohp,
        "IWOBI": atomic_interaction.iwobi,
    }
    for w_interaction in atomic_interaction:
        tag = w_interaction.tag

        integrals[tag + "_population"] = w_interaction.population
        integrals[tag + "_IWOHP"] = w_interaction.iwohp
        integrals[tag + "_IWOBI"] = w_interaction.iwobi

    ndarrays_regression.check(integrals, default_tolerance=tol)


def test_AtomicInteraction_with_integrals_no_descriptors(atomic_interaction) -> None:
    energies = np.linspace(-20, 10, 100)
    mu = 0

    atomic_interaction = atomic_interaction.with_integrals(energies, mu)

    assert atomic_interaction.population is None
    assert atomic_interaction.charge is None
    assert atomic_interaction.iwohp is None
    assert atomic_interaction.iwobi is None


@pytest.mark.parametrize(
    "valence_counts", ({("Ga", "As"): 2}, None), ids=("calc_charge", "no_charge")
)
def test_AtomicInteractionContainer_with_integrals(
    interaction_container, valence_counts, ndarrays_regression, tol
) -> None:
    energies = np.linspace(-20, 10, 100)
    mu = 0

    atomic_interactions = []
    for atomic_interaction in interaction_container:
        atomic_interactions.append(atomic_interaction.with_summed_descriptors())

    interaction_container = replace(
        interaction_container, sub_interactions=atomic_interactions
    )
    interaction_container = interaction_container.with_integrals(
        energies, mu, resolve_orbitals=True, valence_counts=valence_counts
    )

    integrals = {}
    for interaction in interaction_container:
        tag = interaction.tag

        integrals[tag + "_population"] = interaction.population
        integrals[tag + "_charge"] = none_to_nan(interaction.charge)
        integrals[tag + "_IWOHP"] = interaction.iwohp
        integrals[tag + "_IWOBI"] = interaction.iwobi

        for w_interaction in interaction:
            w_tag = w_interaction.tag

            integrals[w_tag + "_population"] = w_interaction.population
            integrals[w_tag + "_IWOHP"] = w_interaction.iwohp
            integrals[w_tag + "_IWOBI"] = w_interaction.iwobi

    ndarrays_regression.check(integrals, default_tolerance=tol)


def test_AtomicInteractionContainer_with_integrals_no_descriptors(
    interaction_container,
) -> None:
    energies = np.linspace(-20, 10, 100)
    mu = 0

    interaction_container = interaction_container.with_integrals(energies, mu)

    for interaction in interaction_container:
        assert interaction.population is None
        assert interaction.charge is None
        assert interaction.iwohp is None
        assert interaction.iwobi is None


def test_AtomicInteractionContainer_filter_by_species(interaction_container) -> None:
    symbols = ("Ga", "As")
    interactions = interaction_container.filter_by_species(symbols)

    for interaction in interactions:
        assert interaction.symbol_i in symbols
        assert interaction.symbol_j in symbols


def test_AtomicInteractionContainer_filter_by_species_no_matching_symbols(
    interaction_container,
) -> None:
    symbols = ("C", "O")

    with pytest.raises(ValueError):
        interaction_container.filter_by_species(symbols)


def test_AtomicInteractionContainer_slice_2_indices(interaction_container) -> None:
    i = 1
    j = 2

    atomic_interaction = interaction_container[i, j]

    assert atomic_interaction.i == i
    assert atomic_interaction.j == j


def test_AtomicInteractionContainer_slice_no_indices(interaction_container) -> None:
    i = 1
    j = 3

    with pytest.raises(ValueError):
        interaction_container[i, j]


def test_AtomicInteractionContainer_slice_1_index(interaction_container) -> None:
    i = 1

    atomic_interactions = interaction_container[i]

    for interaction in atomic_interactions:
        assert interaction.i == i


def test_AtomicInteractionContainer_length(interaction_container) -> None:
    assert len(interaction_container) == 2


def test_AtomicInteractionContainer_str(interaction_container, data_regression) -> None:
    interaction_container_str = str(interaction_container)

    data_regression.check({"str": interaction_container_str})
