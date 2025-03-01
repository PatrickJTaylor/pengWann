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
from pengwann.geometry import Geometry
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


def test_Geometry_as_structure(shared_datadir) -> None:
    geometry = Geometry.from_xyz("wannier90", f"{shared_datadir}")
    structure = geometry.as_structure()

    with open(f"{shared_datadir}/geometry.json", "r") as stream:
        serial = json.load(stream)

    ref_structure = Structure.from_dict(serial)

    sm = StructureMatcher()

    assert sm.fit(structure, ref_structure)
