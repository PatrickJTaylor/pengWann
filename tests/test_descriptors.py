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
from multiprocessing import set_start_method
from pengwann.descriptors import DescriptorCalculator
from pengwann.geometry import (
    AtomicInteractionContainer,
    AtomicInteraction,
    WannierInteraction,
)
from pymatgen.core import Structure
from typing import Any

# TODO: Figure out why multithreaded fork() occurs only when pytest is run.
set_start_method("spawn", force=True)


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

    h_1 = np.load(f"{shared_datadir}/h_1.npy")
    h_2 = np.load(f"{shared_datadir}/h_2.npy")

    h = {(0, -1, 0): h_1, (0, -1, -1): h_2}

    energies = np.arange(-15, 25 + 0.1, 0.1)
    num_wann = 8
    nspin = 2

    dcalc = DescriptorCalculator(
        dos_array, num_wann, nspin, kpoints, u, h, occupation_matrix, energies
    )

    return dcalc


@pytest.fixture
def interactions() -> AtomicInteractionContainer:
    w_interaction_1 = WannierInteraction(
        i=1, j=0, bl_i=np.array([0, 1, 0]), bl_j=np.array([0, 0, 0])
    )
    w_interaction_2 = WannierInteraction(
        i=5, j=6, bl_i=np.array([0, 1, 1]), bl_j=np.array([0, 0, 0])
    )
    interactions = (
        AtomicInteraction(
            i=1,
            j=2,
            symbol_i="C",
            symbol_j="C",
            sub_interactions=(w_interaction_1, w_interaction_2),
        ),
    )

    return AtomicInteractionContainer(sub_interactions=interactions)


@pytest.fixture
def geometry(shared_datadir) -> Structure:
    with open(f"{shared_datadir}/geometry.json", "r") as stream:
        serial = json.load(stream)

    geometry = Structure.from_dict(serial)

    return geometry


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_DescriptorCalculator_from_eigenvalues(ndarrays_regression, tol) -> None:
    eigenvalues = np.array(
        [
            [-1.00, -0.75, -0.50, -0.25, 0.25, 0.50, 0.75, 1.00],
            [-1.20, -0.66, -0.47, -0.30, 0.34, 0.44, 0.67, 0.98],
        ]
    )
    num_wann = 10
    nspin = 2
    energy_range = (-5, 5)
    resolution = 0.01
    sigma = 0.05
    kpoints = np.zeros((10, 3))
    u = np.zeros((10, 10, 10))

    dcalc = DescriptorCalculator.from_eigenvalues(
        eigenvalues, num_wann, nspin, energy_range, resolution, sigma, kpoints, u
    )

    ndarrays_regression.check({"dos_array": dcalc._dos_array}, default_tolerance=tol)


def test_DescriptorCalculator_energies(dcalc, ndarrays_regression, tol) -> None:
    ndarrays_regression.check({"energies": dcalc.energies}, default_tolerance=tol)


@pytest.mark.parametrize("resolve_k", (False, True), ids=("sum_k", "resolve_k"))
class TestkResolvedMethods:
    def test_DescriptorCalculator_get_dos_matrix(
        self, dcalc, resolve_k, ndarrays_regression, tol
    ) -> None:
        i, j = 1, 0
        bl_i = np.array([0, 0, 0])
        bl_j = np.array([-1, 1, 0])

        c_star = np.conj(dcalc.get_coefficient_matrix(i, bl_i))
        c = dcalc.get_coefficient_matrix(j, bl_j)

        dos_matrix = dcalc.get_dos_matrix(c_star, c, resolve_k=resolve_k)

        ndarrays_regression.check({"dos_matrix": dos_matrix}, default_tolerance=tol)

    @pytest.mark.parametrize(
        "calc_wohp, calc_wobi",
        ((False, False), (True, False), (False, True), (True, True)),
        ids=(
            "no_wohp, no_wobi",
            "calc_wohp, no_wobi",
            "no_wohp, calc_wobi",
            "calc_wohp, calc_wobi",
        ),
    )
    def test_DescriptorCalculator_assign_descriptors(
        self,
        dcalc,
        interactions,
        calc_wohp,
        calc_wobi,
        resolve_k,
        ndarrays_regression,
        tol,
    ) -> None:
        processed_interactions = dcalc.assign_descriptors(
            interactions, calc_wohp=calc_wohp, calc_wobi=calc_wobi, resolve_k=resolve_k
        )
        descriptors = {}
        for interaction in processed_interactions:
            tag = interaction.tag

            descriptors[tag + "_dos_matrix"] = interaction.dos_matrix
            descriptors[tag + "_WOHP"] = none_to_nan(interaction.wohp)
            descriptors[tag + "_WOBI"] = none_to_nan(interaction.wobi)

            for w_interaction in interaction.sub_interactions:
                w_tag = w_interaction.tag

                descriptors[w_tag + "_dos_matrix"] = w_interaction.dos_matrix
                descriptors[w_tag + "_WOHP"] = none_to_nan(w_interaction.wohp)
                descriptors[w_tag + "_WOBI"] = none_to_nan(w_interaction.wobi)

        ndarrays_regression.check(descriptors, default_tolerance=tol)

    @pytest.mark.parametrize("calc_p_ij", (False, True), ids=("no_p_ij", "calc_p_ij"))
    def test_DescriptorCalculator_parallelise(
        self, dcalc, interactions, resolve_k, calc_p_ij, ndarrays_regression, tol
    ) -> None:
        wannier_interactions = interactions[1, 2].sub_interactions

        processed_wannier_interactions = dcalc.parallelise(
            wannier_interactions, calc_p_ij=calc_p_ij, resolve_k=resolve_k, num_proc=4
        )

        descriptors = {}
        for w_interaction in processed_wannier_interactions:
            tag = w_interaction.tag

            descriptors[tag + "_dos_matrix"] = w_interaction.dos_matrix
            descriptors[tag + "_p_ij"] = none_to_nan(w_interaction.p_ij)

        ndarrays_regression.check(descriptors, default_tolerance=tol)


def test_DescriptorCalculator_get_coefficient_matrix(
    dcalc, ndarrays_regression, tol
) -> None:
    i = 0
    bl_vector = np.array([0, 0, 0])

    c = dcalc.get_coefficient_matrix(i, bl_vector)

    ndarrays_regression.check({"C_iR": c}, default_tolerance=tol)


def test_DescriptorCalculator_get_density_matrix_element(
    dcalc, ndarrays_regression, tol
) -> None:
    i, j = 1, 4
    bl_i = np.array([0, 0, 0])
    bl_j = np.array([-1, 0, 0])

    c_star = np.conj(dcalc.get_coefficient_matrix(i, bl_i))
    c = dcalc.get_coefficient_matrix(j, bl_j)

    p_ij = dcalc.get_density_matrix_element(c_star, c)

    ndarrays_regression.check({"P_ij": p_ij}, default_tolerance=tol)


def test_DescriptorCalculator_get_density_matrix_element_no_occupation_matrix(
    dcalc,
) -> None:
    dcalc._occupation_matrix = None

    c_star = np.ones_like((10, 10))
    c = c_star

    with pytest.raises(TypeError):
        dcalc.get_density_matrix_element(c_star, c)


def test_DescriptorCalculator_assign_descriptors_no_h(dcalc, interactions) -> None:
    dcalc._h = None

    with pytest.raises(TypeError):
        dcalc.assign_descriptors(interactions)


def test_DescriptorCalculator_assign_descriptors_no_occupation_matrix(
    dcalc, interactions
) -> None:
    dcalc._occupation_matrix = None

    with pytest.raises(TypeError):
        dcalc.assign_descriptors(interactions)


def test_DescriptorCalculator_assign_h_ij(
    dcalc, interactions, ndarrays_regression, tol
) -> None:
    wannier_interaction = interactions.sub_interactions[0].sub_interactions[0]

    wannier_interaction_with_h = dcalc.assign_h_ij(wannier_interaction)

    ndarrays_regression.check(
        {"H_ij": wannier_interaction_with_h.h_ij}, default_tolerance=tol
    )


def test_DescriptorCalculator_parallelise_no_occupation_matrix(
    dcalc, interactions
) -> None:
    dcalc._occupation_matrix = None
    wannier_interactions = interactions.sub_interactions[0].sub_interactions

    calc_p_ij = True
    resolve_k = False

    with pytest.raises(TypeError):
        dcalc.parallelise(wannier_interactions, calc_p_ij, resolve_k)
