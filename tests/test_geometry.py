import pytest
import numpy as np
from pengwann.geometry import InteractionFinder
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


def test_InteractionFinder_init_from_xyz(datadir) -> None:
    cell = (
        (-1.7803725545451619, -1.7803725545451616, 0.0000000000000000),
        (-1.7803725545451616, 0.0000000000000000, -1.7803725545451616),
        (-0.0000000000000003, -1.7803725545451616, -1.7803725545451616),
    )
    finder = InteractionFinder.from_xyz(f"{datadir}/centres.xyz", cell)

    ref_geometry = Structure.from_file(f"{datadir}/ref_geometry.vasp")
    sm = StructureMatcher()

    assert finder._num_wann == 8
    assert sm.fit(finder._geometry, ref_geometry)


def test_InteractionFinder_init_no_wann(datadir) -> None:
    geometry = Structure.from_file(f"{datadir}/ref_geometry.vasp")
    geometry.remove_species(["X0+"])

    with pytest.raises(ValueError):
        InteractionFinder(geometry)


def test_InteractionFinder_init_no_site_property(datadir) -> None:
    geometry = Structure.from_file(f"{datadir}/ref_geometry.vasp")

    with pytest.raises(ValueError):
        InteractionFinder(geometry)


def test_InteractionFinder_get_interactions(datadir) -> None:
    cell = (
        (-1.7803725545451619, -1.7803725545451616, 0.0000000000000000),
        (-1.7803725545451616, 0.0000000000000000, -1.7803725545451616),
        (-0.0000000000000003, -1.7803725545451616, -1.7803725545451616),
    )
    finder = InteractionFinder.from_xyz(f"{datadir}/centres.xyz", cell)

    cutoffs = {("C", "C"): 1.6}
    interactions = finder.get_interactions(cutoffs)
    test_atomic_interaction = interactions[0]
    test_wannier_interaction = test_atomic_interaction.wannier_interactions[0]

    ref_R_1 = np.array([0, 1, 0])
    ref_R_2 = np.array([0, 0, 0])

    assert test_atomic_interaction.pair_id == ("C1", "C2")
    assert test_wannier_interaction.i == 1 and test_wannier_interaction.j == 0
    np.testing.assert_array_equal(test_wannier_interaction.R_1, ref_R_1, strict=True)
    np.testing.assert_array_equal(test_wannier_interaction.R_2, ref_R_2, strict=True)
