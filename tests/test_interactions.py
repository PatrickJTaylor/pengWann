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

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from pengwann.interactions import (
    AtomicInteraction,
    AtomicInteractions,
    WannierInteraction,
)


@pytest.fixture
def wannier_interaction() -> WannierInteraction:
    i = 0
    j = 1
    bl_i = np.array([0, 0, 0], dtype=np.int32)
    bl_j = np.array([0, 0, 0], dtype=np.int32)
    pdos = np.linspace(0, 50, 100)
    h_ij = np.float64(2)
    p_ij = np.float64(0.5)
    ipdos = np.float64(3.8)

    return WannierInteraction(
        i=i,
        j=j,
        bl_i=bl_i,
        bl_j=bl_j,
        pdos=pdos,
        h_ij=h_ij,
        p_ij=p_ij,
        ipdos=ipdos,
    )


@pytest.fixture
def atomic_interaction(wannier_interaction) -> AtomicInteraction:
    i = 2
    j = 3
    bl_i = np.array([1, 0, 0], dtype=np.int32)
    bl_j = np.array([0, 0, 0], dtype=np.int32)
    pdos = np.linspace(0, 25, 100)
    h_ij = np.float64(2.5)
    p_ij = np.float64(0.7)
    ipdos = np.float64(1.1)

    second_interaction = WannierInteraction(
        i=i,
        j=j,
        bl_i=bl_i,
        bl_j=bl_j,
        pdos=pdos,
        h_ij=h_ij,
        p_ij=p_ij,
        ipdos=ipdos,
    )
    wannier_interactions = (wannier_interaction, second_interaction)

    i, j, symbol_i, symbol_j = 1, 2, "Ga", "As"

    return AtomicInteraction(
        i=i,
        j=j,
        symbol_i=symbol_i,
        symbol_j=symbol_j,
        wannier_interactions=wannier_interactions,
    )


@pytest.fixture
def atomic_interactions(wannier_interaction, atomic_interaction) -> AtomicInteractions:
    i = 4
    j = 5
    bl_i = np.array([0, 1, 0], dtype=np.int32)
    bl_j = np.array([0, 0, 0], dtype=np.int32)
    pdos = np.linspace(0, 30, 100)
    h_ij = np.float64(1.5)
    p_ij = np.float64(0.2)
    ipdos = np.float64(2.3)

    second_interaction = WannierInteraction(
        i=i,
        j=j,
        bl_i=bl_i,
        bl_j=bl_j,
        pdos=pdos,
        h_ij=h_ij,
        p_ij=p_ij,
        ipdos=ipdos,
    )
    wannier_interactions = (wannier_interaction, second_interaction)

    i, j, symbol_i, symbol_j = 1, 4, "Ga", "As"
    second_atomic_interaction = AtomicInteraction(
        i=i,
        j=j,
        symbol_i=symbol_i,
        symbol_j=symbol_j,
        wannier_interactions=wannier_interactions,
    )

    return AtomicInteractions(
        atomic_interactions=(atomic_interaction, second_atomic_interaction)
    )


@pytest.mark.parametrize(
    "property_name",
    ("wohp", "wobi", "iwohp", "iwobi"),
    ids=("wohp", "wobi", "iwohp", "iwobi"),
)
def test_WannierInteraction_properties(
    property_name, wannier_interaction, ndarrays_regression, tol
) -> None:
    descriptor = getattr(wannier_interaction, property_name)

    ndarrays_regression.check({property_name: descriptor})


def test_WannierInteraction_wohp_no_pdos(wannier_interaction) -> None:
    wannier_interaction = replace(wannier_interaction, pdos=None)

    assert wannier_interaction.wohp is None


def test_WannierInteraction_wohp_no_h_ij(wannier_interaction) -> None:
    wannier_interaction = replace(wannier_interaction, h_ij=None)

    assert wannier_interaction.wohp is None


def test_WannierInteraction_wobi_no_pdos(wannier_interaction) -> None:
    wannier_interaction = replace(wannier_interaction, pdos=None)

    assert wannier_interaction.wobi is None


def test_WannierInteraction_wobi_no_p_ij(wannier_interaction) -> None:
    wannier_interaction = replace(wannier_interaction, p_ij=None)

    assert wannier_interaction.wobi is None


def test_WannierInteraction_iwohp_no_ipdos(wannier_interaction) -> None:
    wannier_interaction = replace(wannier_interaction, ipdos=None)

    assert wannier_interaction.iwohp is None


def test_WannierInteraction_iwohp_no_h_ij(wannier_interaction) -> None:
    wannier_interaction = replace(wannier_interaction, h_ij=None)

    assert wannier_interaction.iwohp is None


def test_WannierInteraction_iwobi_no_ipdos(wannier_interaction) -> None:
    wannier_interaction = replace(wannier_interaction, ipdos=None)

    assert wannier_interaction.iwobi is None


def test_WannierInteraction_iwobi_no_p_ij(wannier_interaction) -> None:
    wannier_interaction = replace(wannier_interaction, p_ij=None)

    assert wannier_interaction.iwobi is None


def test_WannierInteraction_str(wannier_interaction, data_regression) -> None:
    wannier_interaction_str = str(wannier_interaction)

    data_regression.check({"str": wannier_interaction_str})


@pytest.mark.parametrize(
    "property_name",
    ("pdos", "wohp", "wobi", "ipdos", "iwohp", "iwobi"),
    ids=("pdos", "wohp", "wobi", "ipdos", "iwohp", "iwobi"),
)
def test_AtomicInteraction_properties(
    property_name, atomic_interaction, ndarrays_regression, tol
) -> None:
    descriptor = getattr(atomic_interaction, property_name)

    ndarrays_regression.check({property_name: descriptor})


def test_AtomicInteraction_pdos_none(
    atomic_interaction, ndarrays_regression, tol
) -> None:
    new_wann = replace(atomic_interaction.wannier_interactions[0], pdos=None)
    wannier_interactions = (new_wann,) + atomic_interaction.wannier_interactions[1:]
    atomic_interaction = replace(
        atomic_interaction, wannier_interactions=wannier_interactions
    )

    assert atomic_interaction.pdos is None


def test_AtomicInteraction_wohp_none(
    atomic_interaction, ndarrays_regression, tol
) -> None:
    new_wann = replace(atomic_interaction.wannier_interactions[0], h_ij=None)
    wannier_interactions = (new_wann,) + atomic_interaction.wannier_interactions[1:]
    atomic_interaction = replace(
        atomic_interaction, wannier_interactions=wannier_interactions
    )

    assert atomic_interaction.wohp is None


def test_AtomicInteraction_wobi_none(
    atomic_interaction, ndarrays_regression, tol
) -> None:
    new_wann = replace(atomic_interaction.wannier_interactions[0], p_ij=None)
    wannier_interactions = (new_wann,) + atomic_interaction.wannier_interactions[1:]
    atomic_interaction = replace(
        atomic_interaction, wannier_interactions=wannier_interactions
    )

    assert atomic_interaction.wobi is None


def test_AtomicInteraction_ipdos_none(
    atomic_interaction, ndarrays_regression, tol
) -> None:
    new_wann = replace(atomic_interaction.wannier_interactions[0], ipdos=None)
    wannier_interactions = (new_wann,) + atomic_interaction.wannier_interactions[1:]
    atomic_interaction = replace(
        atomic_interaction, wannier_interactions=wannier_interactions
    )

    assert atomic_interaction.ipdos is None


def test_AtomicInteraction_iwohp_none(
    atomic_interaction, ndarrays_regression, tol
) -> None:
    new_wann = replace(atomic_interaction.wannier_interactions[0], h_ij=None)
    wannier_interactions = (new_wann,) + atomic_interaction.wannier_interactions[1:]
    atomic_interaction = replace(
        atomic_interaction, wannier_interactions=wannier_interactions
    )

    assert atomic_interaction.iwohp is None


def test_AtomicInteraction_iwobi_none(
    atomic_interaction, ndarrays_regression, tol
) -> None:
    new_wann = replace(atomic_interaction.wannier_interactions[0], p_ij=None)
    wannier_interactions = (new_wann,) + atomic_interaction.wannier_interactions[1:]
    atomic_interaction = replace(
        atomic_interaction, wannier_interactions=wannier_interactions
    )

    assert atomic_interaction.iwobi is None


def test_AtomicInteraction_slice_2_indices(atomic_interaction) -> None:
    i = 0
    j = 1

    wannier_interaction = atomic_interaction[i, j]

    assert wannier_interaction.i == i
    assert wannier_interaction.j == j


def test_AtomicInteraction_slice_no_indices(atomic_interaction) -> None:
    i = 1
    j = 3

    with pytest.raises(IndexError):
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
        wannier_interactions=wannier_interactions,
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


def test_AtomicInteractions_filter_by_species(atomic_interactions) -> None:
    symbols = ("Ga", "As")
    interactions = atomic_interactions.filter_by_species(symbols)

    for interaction in interactions:
        assert interaction.symbol_i in symbols
        assert interaction.symbol_j in symbols


def test_AtomicInteractions_filter_by_species_no_matching_symbols(
    atomic_interactions,
) -> None:
    symbols = ("C", "O")

    with pytest.raises(ValueError):
        atomic_interactions.filter_by_species(symbols)


def test_AtomicInteractions_slice_2_indices(atomic_interactions) -> None:
    i = 1
    j = 2

    atomic_interaction = atomic_interactions[i, j]

    assert atomic_interaction.i == i
    assert atomic_interaction.j == j


def test_AtomicInteractions_slice_no_indices(atomic_interactions) -> None:
    i = 1
    j = 3

    with pytest.raises(IndexError):
        atomic_interactions[i, j]


def test_AtomicInteractions_slice_1_index(atomic_interactions) -> None:
    i = 1

    interactions = atomic_interactions[i]

    for interaction in interactions:
        assert interaction.i == i


def test_AtomicInteractions_length(atomic_interactions) -> None:
    assert len(atomic_interactions) == 2


def test_AtomicInteractions_str(atomic_interactions, data_regression) -> None:
    atomic_interactions_str = str(atomic_interactions)

    data_regression.check({"str": atomic_interactions_str})
