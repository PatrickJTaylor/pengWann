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
from numpy.typing import NDArray
from pengwann.descriptors import DescriptorCalculator
from pengwann.io import read_hamiltonian
from pengwann.geometry import AtomicInteraction, WannierInteraction
from pymatgen.core import Structure
from pengwann.utils import allocate_shared_memory
from typing import Any


def none_to_nan(data: Any) -> Any:
    if data is None:
        return np.nan

    else:
        return data


@pytest.fixture
def dcalc(shared_datadir) -> DescriptorCalculator:
    dos_array = np.load(f"{shared_datadir}/dos_array.npy")
    kpoints = np.load(f"{shared_datadir}/kpoints.npy")
    u = np.load(f"{shared_datadir}/U.npy")
    occupation_matrix = np.load(f"{shared_datadir}/occupation_matrix.npy")
    h = read_hamiltonian(f"{shared_datadir}/wannier90_hr.dat")

    energies = np.arange(-25, 25 + 0.1, 0.1, dtype=np.float64)
    num_wann = 8
    nspin = 2

    dcalc = DescriptorCalculator(
        dos_array, num_wann, nspin, kpoints, u, h, occupation_matrix, energies
    )

    return dcalc


@pytest.fixture
def ref_geometry(shared_datadir) -> Structure:
    with open(f"{shared_datadir}/geometry.json", "r") as stream:
        serial = json.load(stream)

    geometry = Structure.from_dict(serial)

    return geometry


def test_DescriptorCalculator_from_eigenvalues(ndarrays_regression) -> None:
    eigenvalues = np.array([[-4, -3, -2, -1, 1, 2, 3, 4], [-5, -4, -3, -2, 2, 3, 4, 5]])
    num_wann = 8
    nspin = 2
    energy_range = (-6, 6)
    resolution = 0.01
    sigma = 0.05
    kpoints = np.zeros_like((10, 3))
    u = np.zeros_like((10, 4, 4))

    dcalc = DescriptorCalculator.from_eigenvalues(
        eigenvalues, num_wann, nspin, energy_range, resolution, sigma, kpoints, u
    )

    dos_array = dcalc._dos_array

    ndarrays_regression.check(
        {"dos_array": dos_array}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_DescriptorCalculator_energies(dcalc, ndarrays_regression) -> None:
    ndarrays_regression.check(
        {"energies": dcalc.energies}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_DescriptorCalculator_get_coefficient_matrix(
    dcalc, ndarrays_regression
) -> None:
    i = 0
    bl_vector = np.array([0, 0, 0])

    c = dcalc.get_coefficient_matrix(i, bl_vector)

    ndarrays_regression.check({"C_iR": c}, default_tolerance={"atol": 0, "rtol": 1e-07})


@pytest.mark.parametrize("resolve_k", (False, True), ids=("sum_k", "resolve_k"))
def test_DescriptorCalculator_get_dos_matrix(
    dcalc, resolve_k, ndarrays_regression
) -> None:
    i, j = 1, 0
    bl_1 = np.array([0, 0, 0])
    bl_2 = np.array([-1, 1, 0])

    c_star = np.conj(dcalc.get_coefficient_matrix(i, bl_1))
    c = dcalc.get_coefficient_matrix(j, bl_2)

    dos_matrix = dcalc.get_dos_matrix(c_star, c, resolve_k=resolve_k)

    ndarrays_regression.check(
        {"dos_matrix": dos_matrix}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_DescriptorCalculator_get_density_matrix_element(
    dcalc, ndarrays_regression
) -> None:
    i, j = 1, 4
    bl_1 = np.array([0, 0, 0])
    bl_2 = np.array([-1, 0, 0])

    c_star = np.conj(dcalc.get_coefficient_matrix(i, bl_1))
    c = dcalc.get_coefficient_matrix(j, bl_2)

    p_ij = dcalc.get_density_matrix_element(c_star, c)

    ndarrays_regression.check(
        {"P_ij": p_ij}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_DescriptorCalculator_get_density_matrix_element_no_occupation_matrix(
    dcalc,
) -> None:
    dcalc._occupation_matrix = None

    c_star = np.ones_like((10, 10))
    c = c_star

    with pytest.raises(TypeError):
        dcalc.get_density_matrix_element(c_star, c)


@pytest.mark.parametrize("resolve_k", (False, True), ids=("sum_k", "resolve_k"))
def test_DescriptorCalculator_get_pdos(
    dcalc, resolve_k, ref_geometry, ndarrays_regression
) -> None:
    symbols = ("C",)
    interactions = dcalc.get_pdos(ref_geometry, symbols, resolve_k=resolve_k)

    pdos = {}
    for interaction in interactions:
        id_i, id_j = interaction.pair_id
        label = id_i + id_j

        pdos[label] = interaction.dos_matrix

        for w_interaction in interaction.wannier_interactions:
            w_label = str(w_interaction.i) + str(w_interaction.j)

            pdos[w_label] = w_interaction.dos_matrix

    ndarrays_regression.check(pdos, default_tolerance={"atol": 0, "rtol": 1e-07})


def test_DescriptorCalculator_get_pdos_no_symbols(dcalc, ref_geometry) -> None:
    symbols = ("D",)

    with pytest.raises(ValueError):
        dcalc.get_pdos(ref_geometry, symbols)


@pytest.mark.parametrize(
    "resolve_k, resolve_orbitals",
    ((False, False), (False, True), (True, False), (True, True)),
    ids=(
        "sum_k, sum_orbitals",
        "sum_k, resolve_orbitals",
        "resolve_k, sum_orbitals",
        "resolve_k, resolve_orbitals",
    ),
)
def test_DescriptorCalculator_assign_populations(
    dcalc, resolve_k, resolve_orbitals, ref_geometry, ndarrays_regression
) -> None:
    symbols = ("C",)
    interactions = dcalc.get_pdos(ref_geometry, symbols, resolve_k=resolve_k)

    mu = 9.8675
    dcalc.assign_populations(interactions, mu, resolve_orbitals=resolve_orbitals)

    populations = {}
    for interaction in interactions:
        id_i, id_j = interaction.pair_id
        label = id_i + id_j

        populations[label] = interaction.population

        for w_interaction in interaction.wannier_interactions:
            w_label = str(w_interaction.i) + str(w_interaction.j)

            populations[w_label] = none_to_nan(w_interaction.population)

    ndarrays_regression.check(populations, default_tolerance={"atol": 0, "rtol": 1e-07})


@pytest.mark.parametrize(
    "resolve_orbitals", (False, True), ids=("sum_orbitals", "resolve_orbitals")
)
def test_DescriptorCalculator_assign_populations_no_energies(
    dcalc, resolve_orbitals
) -> None:
    dcalc._energies = None

    wannier_interaction_1 = WannierInteraction(
        i=1, j=0, bl_1=np.array([0, 1, 0]), bl_2=np.array([0, 0, 0])
    )
    wannier_interaction_2 = WannierInteraction(
        i=5, j=6, bl_1=np.array([0, 1, 1]), bl_2=np.array([0, 0, 0])
    )
    interactions = (
        AtomicInteraction(
            pair_id=("C1", "C2"),
            wannier_interactions=(wannier_interaction_1, wannier_interaction_2),
        ),
    )

    mu = 9.8675

    with pytest.raises(TypeError):
        dcalc.assign_populations(interactions, mu, resolve_orbitals=resolve_orbitals)


@pytest.mark.parametrize(
    "resolve_k, resolve_orbitals",
    ((False, False), (False, True), (True, False), (True, True)),
    ids=(
        "sum_k, sum_orbitals",
        "sum_k, resolve_orbitals",
        "resolve_k, sum_orbitals",
        "resolve_k, resolve_orbitals",
    ),
)
def test_DescriptorCalculator_assign_populations_no_atomic_dos_matrix(
    dcalc, ref_geometry, resolve_k, resolve_orbitals
) -> None:
    symbols = ("C",)
    interactions = dcalc.get_pdos(ref_geometry, symbols, resolve_k=resolve_k)

    if resolve_orbitals:
        interactions[0].wannier_interactions[0].dos_matrix = None

    else:
        interactions[0].dos_matrix = None

    mu = 9.8675

    with pytest.raises(TypeError):
        dcalc.assign_populations(interactions, mu, resolve_orbitals=resolve_orbitals)


@pytest.mark.parametrize("resolve_k", (False, True), ids=("sum_k", "resolve_k"))
def test_DescriptorCalculator_assign_populations_charges(
    dcalc, resolve_k, ref_geometry, ndarrays_regression
) -> None:
    symbols = ("C",)
    interactions = dcalc.get_pdos(ref_geometry, symbols, resolve_k=resolve_k)

    mu = 9.8675
    valence = {"C": 4}
    dcalc.assign_populations(interactions, mu, valence=valence)

    charges = {}
    for interaction in interactions:
        id_i, id_j = interaction.pair_id
        label = id_i + id_j

        charges[label] = interaction.charge

    ndarrays_regression.check(charges, default_tolerance={"atol": 0, "rtol": 1e-07})


def test_DescriptorCalculator_assign_populations_charges_wrong_valence(
    dcalc, ref_geometry
) -> None:
    symbols = ("C",)
    interactions = dcalc.get_pdos(ref_geometry, symbols)

    mu = 9.8675
    valence = {"D": 4}

    with pytest.raises(ValueError):
        dcalc.assign_populations(interactions, mu, valence=valence)


@pytest.mark.parametrize(
    "calc_wohp, calc_wobi, resolve_k",
    (
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (True, False, False),
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ),
    ids=(
        "no_wohp, no_wobi, sum_k",
        "no_wohp, no_wobi, resolve_k",
        "no_wohp, calc_wobi, sum_k",
        "calc_wohp, no_wobi, sum_k",
        "no_wohp, calc_wobi, resolve_k",
        "calc_wohp, no_wobi, resolve_k",
        "calc_wohp, calc_wobi, sum_k",
        "calc_wohp, calc_wobi, resolve_k",
    ),
)
def test_DescriptorCalculator_assign_descriptors(
    dcalc, calc_wohp, calc_wobi, resolve_k, ndarrays_regression
) -> None:
    wannier_interaction_1 = WannierInteraction(
        i=1, j=0, bl_1=np.array([0, 1, 0]), bl_2=np.array([0, 0, 0])
    )
    wannier_interaction_2 = WannierInteraction(
        i=5, j=6, bl_1=np.array([0, 1, 1]), bl_2=np.array([0, 0, 0])
    )
    interactions = (
        AtomicInteraction(
            pair_id=("C1", "C2"),
            wannier_interactions=(wannier_interaction_1, wannier_interaction_2),
        ),
    )

    dcalc.assign_descriptors(
        interactions, calc_wohp=calc_wohp, calc_wobi=calc_wobi, resolve_k=resolve_k
    )

    descriptors = {}
    for interaction in interactions:
        id_i, id_j = interaction.pair_id
        label = id_i + id_j

        descriptors[label + "_dos_matrix"] = interaction.dos_matrix
        descriptors[label + "_WOHP"] = none_to_nan(interaction.wohp)
        descriptors[label + "_WOBI"] = none_to_nan(interaction.wobi)

        for w_interaction in interaction.wannier_interactions:
            w_label = str(w_interaction.i) + str(w_interaction.j)

            descriptors[w_label + "_dos_matrix"] = w_interaction.dos_matrix
            descriptors[w_label + "_WOHP"] = none_to_nan(w_interaction.wohp)
            descriptors[w_label + "_WOBI"] = none_to_nan(w_interaction.wobi)

    ndarrays_regression.check(descriptors, default_tolerance={"atol": 0, "rtol": 1e-07})


def test_DescriptorCalculator_assign_descriptors_no_h(dcalc) -> None:
    wannier_interaction_1 = WannierInteraction(
        i=1, j=0, bl_1=np.array([0, 1, 0]), bl_2=np.array([0, 0, 0])
    )
    wannier_interaction_2 = WannierInteraction(
        i=5, j=6, bl_1=np.array([0, 1, 1]), bl_2=np.array([0, 0, 0])
    )
    interactions = (
        AtomicInteraction(
            pair_id=("C1", "C2"),
            wannier_interactions=(wannier_interaction_1, wannier_interaction_2),
        ),
    )

    dcalc._h = None

    with pytest.raises(TypeError):
        dcalc.assign_descriptors(interactions)


def test_DescriptorCalculator_assign_descriptors_no_occupation_matrix(dcalc) -> None:
    wannier_interaction_1 = WannierInteraction(
        i=1, j=0, bl_1=np.array([0, 1, 0]), bl_2=np.array([0, 0, 0])
    )
    wannier_interaction_2 = WannierInteraction(
        i=5, j=6, bl_1=np.array([0, 1, 1]), bl_2=np.array([0, 0, 0])
    )
    interactions = (
        AtomicInteraction(
            pair_id=("C1", "C2"),
            wannier_interactions=(wannier_interaction_1, wannier_interaction_2),
        ),
    )

    dcalc._occupation_matrix = None

    with pytest.raises(TypeError):
        dcalc.assign_descriptors(interactions)


@pytest.mark.parametrize(
    "resolve_k, resolve_orbitals",
    ((False, False), (False, True), (True, False), (True, True)),
    ids=(
        "sum_k, sum_orbitals",
        "sum_k, resolve_orbitals",
        "resolve_k, sum_orbitals",
        "resolve_k, resolve_orbitals",
    ),
)
def test_DescriptorCalculator_integrate_descriptors(
    dcalc, resolve_k, resolve_orbitals, ndarrays_regression
) -> None:
    wannier_interaction_1 = WannierInteraction(
        i=1, j=0, bl_1=np.array([0, 1, 0]), bl_2=np.array([0, 0, 0])
    )
    wannier_interaction_2 = WannierInteraction(
        i=5, j=6, bl_1=np.array([0, 1, 1]), bl_2=np.array([0, 0, 0])
    )
    interactions = (
        AtomicInteraction(
            pair_id=("C1", "C2"),
            wannier_interactions=(wannier_interaction_1, wannier_interaction_2),
        ),
    )

    dcalc.assign_descriptors(interactions, resolve_k=resolve_k)

    mu = 9.8675

    dcalc.integrate_descriptors(interactions, mu, resolve_orbitals=resolve_orbitals)

    integrals = {}  # type: dict[str, float | np.float64 | NDArray[np.float64]]
    for interaction in interactions:
        id_i, id_j = interaction.pair_id
        label = id_i + id_j

        integrals[label + "_IWOHP"] = interaction.iwohp  # type: ignore
        integrals[label + "_IWOBI"] = interaction.iwobi  # type: ignore

        for w_interaction in interaction.wannier_interactions:
            w_label = str(w_interaction.i) + str(w_interaction.j)

            integrals[w_label + "_IWOHP"] = none_to_nan(w_interaction.iwohp)
            integrals[w_label + "_IWOBI"] = none_to_nan(w_interaction.iwobi)

    ndarrays_regression.check(integrals, default_tolerance={"atol": 0, "rtol": 1e-07})


@pytest.mark.parametrize(
    "resolve_orbitals", (False, True), ids=("sum_orbitals", "resolve_orbitals")
)
def test_DescriptorCalculator_integrate_descriptors_no_energies(
    dcalc, resolve_orbitals
) -> None:
    wannier_interaction_1 = WannierInteraction(
        i=1, j=0, bl_1=np.array([0, 1, 0]), bl_2=np.array([0, 0, 0])
    )
    wannier_interaction_2 = WannierInteraction(
        i=5, j=6, bl_1=np.array([0, 1, 1]), bl_2=np.array([0, 0, 0])
    )
    interactions = (
        AtomicInteraction(
            pair_id=("C1", "C2"),
            wannier_interactions=(wannier_interaction_1, wannier_interaction_2),
        ),
    )

    mu = 9.8675
    dcalc._energies = None

    with pytest.raises(TypeError):
        dcalc.integrate_descriptors(interactions, mu, resolve_orbitals=resolve_orbitals)


def test_DescriptorCalculator_get_density_of_energy(dcalc, ndarrays_regression) -> None:
    wannier_interaction_1 = WannierInteraction(
        i=1, j=0, bl_1=np.array([0, 1, 0]), bl_2=np.array([0, 0, 0])
    )
    wannier_interaction_2 = WannierInteraction(
        i=5, j=6, bl_1=np.array([0, 1, 1]), bl_2=np.array([0, 0, 0])
    )
    interactions = (
        AtomicInteraction(
            pair_id=("C1", "C2"),
            wannier_interactions=(wannier_interaction_1, wannier_interaction_2),
        ),
    )

    dcalc.assign_descriptors(interactions)

    doe = dcalc.get_density_of_energy(interactions)

    ndarrays_regression.check(
        {"DOE": doe}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_DescriptorCalculator_get_density_of_energy_no_wohp(dcalc) -> None:
    wannier_interaction_1 = WannierInteraction(
        i=1, j=0, bl_1=np.array([0, 1, 0]), bl_2=np.array([0, 0, 0])
    )
    wannier_interaction_2 = WannierInteraction(
        i=5, j=6, bl_1=np.array([0, 1, 1]), bl_2=np.array([0, 0, 0])
    )
    interactions = (
        AtomicInteraction(
            pair_id=("C1", "C2"),
            wannier_interactions=(wannier_interaction_1, wannier_interaction_2),
        ),
    )

    with pytest.raises(TypeError):
        dcalc.get_density_of_energy(interactions)


def test_DescriptorCalculator_get_bwdf(
    dcalc, ref_geometry, ndarrays_regression
) -> None:
    wannier_interaction_1 = WannierInteraction(
        i=1, j=0, bl_1=np.array([0, 1, 0]), bl_2=np.array([0, 0, 0])
    )
    wannier_interaction_2 = WannierInteraction(
        i=5, j=6, bl_1=np.array([0, 1, 1]), bl_2=np.array([0, 0, 0])
    )
    interactions = (
        AtomicInteraction(
            pair_id=("C1", "C2"),
            wannier_interactions=(wannier_interaction_1, wannier_interaction_2),
        ),
    )

    dcalc.assign_descriptors(interactions)

    mu = 9.8675

    dcalc.integrate_descriptors(interactions, mu)

    r_range = (0, 5)
    nbins = 500

    r, bwdf = dcalc.get_bwdf(interactions, ref_geometry, r_range, nbins)

    ndarrays_regression.check(
        {"r": r, "BWDF": bwdf[("C", "C")]}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


@pytest.mark.parametrize(
    "calc_wobi, resolve_k",
    ((False, False), (False, True), (True, False), (True, True)),
    ids=(
        "no_wobi, sum_k",
        "no_wobi, resolve_k",
        "calc_wobi, sum_k",
        "calc_wobi, resolve_k",
    ),
)
def test_DescriptorCalculator_process_interaction(
    dcalc, calc_wobi, resolve_k, ndarrays_regression
) -> None:
    memory_keys = ["dos_array", "kpoints", "u"]
    shared_data = [dcalc._dos_array, dcalc._kpoints, dcalc._u]
    if calc_wobi:
        memory_keys.append("occupation_matrix")
        shared_data.append(dcalc._occupation_matrix)

    memory_metadata, memory_handles = allocate_shared_memory(memory_keys, shared_data)

    wannier_interaction = WannierInteraction(
        i=1, j=0, bl_1=np.array([0, 1, 0]), bl_2=np.array([0, 0, 0])
    )

    amended_interaction = dcalc.process_interaction(
        wannier_interaction,
        dcalc._num_wann,
        dcalc._nspin,
        calc_wobi,
        resolve_k,
        memory_metadata,
    )

    for memory_handle in memory_handles:
        memory_handle.unlink()

    descriptors = {"dos_matrix": amended_interaction.dos_matrix}
    if calc_wobi:
        descriptors["P_ij"] = wannier_interaction.p_ij

    ndarrays_regression.check(descriptors, default_tolerance={"atol": 0, "rtol": 1e-07})
