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

from pengwann.utils import get_atom_indices
from pymatgen.core import Structure


def test_get_atom_indices(shared_datadir, data_regression) -> None:
    geometry = Structure.from_file(f"{shared_datadir}/structure.vasp")

    indices = get_atom_indices(geometry, ("C", "X0+"))

    data_regression.check(indices)


def test_get_atom_indices_uniqueness(shared_datadir) -> None:
    geometry = Structure.from_file(f"{shared_datadir}/structure.vasp")

    test_indices = get_atom_indices(geometry, ("C", "X0+"))
    total_indices = test_indices["C"] + test_indices["X0+"]
    total_indices_set = set(total_indices)

    assert len(total_indices_set) == len(total_indices)


def test_get_atom_indices_all_assigned(shared_datadir, data_regression) -> None:
    geometry = Structure.from_file(f"{shared_datadir}/structure.vasp")

    indices = get_atom_indices(geometry, ("C", "X0+"))

    data_regression.check({"num_C": len(indices["C"]), "num_X": len(indices["X0+"])})
