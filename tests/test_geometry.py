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

import numpy as np
import pytest
from ase.io.jsonio import decode
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from pengwann.geometry import (
    Geometry,
    find_interatomic_interactions,
    find_onsite_interactions,
)
from pengwann.interactions import AtomicInteractions


def build_geometry(symbols: list[str]) -> Geometry:
    cell = np.diag([5.0, 5.0, 5.0])

    coords = np.array(
        [[0.1, 0.1, 0.1], [0.6, 0.6, 0.6], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
    )
    distance_matrix = np.array(
        [
            [0.0, 4.33012702, 1.29903811, 3.03108891],
            [4.33012702, 0.0, 3.03108891, 1.29903811],
            [1.29903811, 3.03108891, 0.0, 4.33012702],
            [3.03108891, 1.29903811, 4.33012702, 0.0],
        ]
    )
    image_matrix = np.array(
        [
            [[0, 0, 0], [-1, -1, -1], [0, 0, 0], [-1, -1, -1]],
            [[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [-1, -1, -1]],
            [[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]],
        ],
        dtype=np.int32,
    )
    wannier_assignments = ((), (), (0,), (1,))

    return Geometry(
        symbols, coords, cell, distance_matrix, image_matrix, wannier_assignments
    )


def serialise_interactions(
    interactions: AtomicInteractions,
) -> dict[str, int | tuple[str, str] | list[int]]:
    serialised_interactions = {"tags": [], "i": [], "j": [], "bl_i": [], "bl_j": []}
    for interaction in interactions:
        serialised_interactions["tags"].append(interaction.tag)

        for w_interaction in interaction.wannier_interactions:
            serialised_interactions["i"].append(w_interaction.i)
            serialised_interactions["j"].append(w_interaction.j)

            serial_bl_i = w_interaction.bl_i.tolist()
            serial_bl_j = w_interaction.bl_j.tolist()

            serialised_interactions["bl_i"].append(serial_bl_i)
            serialised_interactions["bl_j"].append(serial_bl_j)

    return serialised_interactions


@pytest.fixture
def geometry() -> Geometry:
    symbols = ("X", "X", "C", "O")

    return build_geometry(symbols)


@pytest.fixture
def geometry_elemental() -> Geometry:
    symbols = ("X", "X", "C", "C")

    return build_geometry(symbols)


def test_Geometry_as_structure(shared_datadir, geometry) -> None:
    structure = geometry.as_structure()

    with open(f"{shared_datadir}/pmg_geometry.json", "r") as stream:
        serial = json.load(stream)

    ref_structure = Structure.from_dict(serial)

    sm = StructureMatcher()

    assert sm.fit(structure, ref_structure)


def test_Geometry_as_atoms(shared_datadir, geometry) -> None:
    atoms = geometry.as_atoms()

    with open(f"{shared_datadir}/ase_geometry.json", "r") as stream:
        serial = json.load(stream)

    ref_atoms = decode(serial)

    sym = SymmetryEquivalenceCheck()

    assert sym.compare(atoms, ref_atoms)


def test_Geometry_str(geometry, data_regression) -> None:
    geometry_str = str(geometry)

    data_regression.check({"str": geometry_str})


def test_find_interatomic_interactions_elemental(
    geometry_elemental, data_regression
) -> None:
    cutoffs = {("C", "C"): 4.5}

    interactions = find_interatomic_interactions(geometry_elemental, cutoffs)

    serialised_interactions = serialise_interactions(interactions)

    data_regression.check(serialised_interactions)


def test_find_interatomic_interactions_binary(geometry, data_regression) -> None:
    cutoffs = {("C", "O"): 4.5}

    interactions = find_interatomic_interactions(geometry, cutoffs)

    serialised_interactions = serialise_interactions(interactions)

    data_regression.check(serialised_interactions)


def test_find_interatomic_interactions_no_symbols(geometry) -> None:
    cutoffs = {("B", "N"): 2.0}

    with pytest.raises(ValueError):
        find_interatomic_interactions(geometry, cutoffs)


def test_find_onsite_interactions(geometry, data_regression) -> None:
    symbols = ("C", "O")

    interactions = find_onsite_interactions(geometry, symbols)

    serialised_interactions = serialise_interactions(interactions)

    data_regression.check(serialised_interactions)


def test_find_onsite_interactions_no_symbols(geometry) -> None:
    symbols = ("B", "N")

    with pytest.raises(ValueError):
        find_onsite_interactions(geometry, symbols)
