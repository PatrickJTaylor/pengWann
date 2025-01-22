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
from pengwann.geometry import assign_wannier_centres, build_geometry, find_interactions
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


@pytest.fixture
def ref_geometry(shared_datadir) -> Structure:
    with open(f"{shared_datadir}/geometry.json", "r") as stream:
        serial = json.load(stream)

    geometry = Structure.from_dict(serial)

    return geometry


def test_build_geometry(ref_geometry, shared_datadir) -> None:
    cell = (
        (-1.7803725545451619, -1.7803725545451616, 0.0000000000000000),
        (-1.7803725545451616, 0.0000000000000000, -1.7803725545451616),
        (-0.0000000000000003, -1.7803725545451616, -1.7803725545451616),
    )
    test_geometry = build_geometry(f"{shared_datadir}/centres.xyz", cell)
    num_wann = len([site for site in test_geometry if site.species_string == "X0+"])

    sm = StructureMatcher()

    assert num_wann == 8
    assert sm.fit(test_geometry, ref_geometry)


def test_assign_wannier_centres(shared_datadir, data_regression) -> None:
    geometry = Structure.from_file(f"{shared_datadir}/structure.vasp")

    assign_wannier_centres(geometry)

    data_regression.check(
        {"wannier_centres": geometry.site_properties["wannier_centres"]}
    )


def test_assign_wannier_centres_invalid_structure(shared_datadir) -> None:
    geometry = Structure.from_file(f"{shared_datadir}/invalid_structure.vasp")

    with pytest.raises(ValueError):
        assign_wannier_centres(geometry)


@pytest.mark.parametrize("binary", (False, True), ids=("element", "binary"))
def test_find_interactions(ref_geometry, binary, data_regression) -> None:
    if binary:
        ref_geometry.replace(9, "O", properties={"wannier_centres": (0, 3, 4, 6)})
        cutoffs = {("C", "O"): 1.6}

    else:
        cutoffs = {("C", "C"): 1.6}

    interactions = find_interactions(ref_geometry, cutoffs)

    serialised_interactions = {
        "pair_ids": [],
        "i": [],
        "j": [],
        "bl_1": [],
        "bl_2": [],
    }  # type: dict[str, list]
    for interaction in interactions:
        serialised_interactions["pair_ids"].append(interaction.pair_id)

        for w_interaction in interaction.wannier_interactions:
            serialised_interactions["i"].append(w_interaction.i)
            serialised_interactions["j"].append(w_interaction.j)

            serial_bl_1 = w_interaction.bl_1.tolist()
            serial_bl_2 = w_interaction.bl_2.tolist()

            serialised_interactions["bl_1"].append(serial_bl_1)
            serialised_interactions["bl_2"].append(serial_bl_2)

    data_regression.check(serialised_interactions)


def test_find_interactions_no_site_property(ref_geometry) -> None:
    ref_geometry.remove_site_property("wannier_centres")
    cutoffs = {("C", "C"): 1.6}

    with pytest.raises(ValueError):
        find_interactions(ref_geometry, cutoffs)


def test_find_interactions_no_wann(ref_geometry) -> None:
    ref_geometry.remove_species(["X0+"])
    cutoffs = {("C", "C"): 1.6}

    with pytest.raises(ValueError):
        find_interactions(ref_geometry, cutoffs)
