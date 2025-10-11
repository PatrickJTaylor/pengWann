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

import pytest
import numpy as np
from numpy.typing import NDArray

from pengwann.descriptors import (
    compute_coefficients,
    compute_ipdos,
    compute_p_ij,
    compute_pdos,
    DescriptorPipeline,
    get_h_ij,
)
from pengwann.electronic_structure import Basis
from pengwann.interactions import (
    AtomicInteractions,
    AtomicInteraction,
    WannierInteraction,
)
from pengwann.type_aliases import Hamiltonian


@pytest.fixture
def dpl() -> DescriptorPipeline:
    return DescriptorPipeline()


@pytest.fixture
def interactions() -> AtomicInteractions:
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
            wannier_interactions=(w_interaction_1, w_interaction_2),
        ),
    )

    return AtomicInteractions(atomic_interactions=interactions)


@pytest.fixture
def wannier_interaction() -> WannierInteraction:
    i, j = 1, 0
    bl_i = np.array([0, 1, 0])
    bl_j = np.array([0, 0, 0])

    return WannierInteraction(i, j, bl_i, bl_j)


@pytest.fixture
def basis(shared_datadir) -> Basis:
    u = np.load(f"{shared_datadir}/U.npy")
    kpoints = np.load(f"{shared_datadir}/kpoints.npy")

    return Basis(u, kpoints)


@pytest.fixture
def coefficients(shared_datadir) -> NDArray[np.float64]:
    return np.load(f"{shared_datadir}/coefficients.npy")


@pytest.fixture
def total_dos(shared_datadir) -> NDArray[np.float64]:
    return np.load(f"{shared_datadir}/total_dos.npy")


@pytest.fixture
def occupation_matrix(shared_datadir) -> NDArray[np.float64]:
    return np.load(f"{shared_datadir}/occupation_matrix.npy")


@pytest.fixture
def hamiltonian(shared_datadir) -> Hamiltonian:
    h_1 = np.load(f"{shared_datadir}/h_1.npy")
    h_2 = np.load(f"{shared_datadir}/h_2.npy")

    h = {(0, -1, 0): h_1, (0, -1, -1): h_2}

    return h


@pytest.fixture
def pdos(shared_datadir) -> NDArray[np.float64]:
    return np.load(f"{shared_datadir}/pdos.npy")


def test_compute_coefficients(basis, ndarrays_regression, tol) -> None:
    i, j = 1, 0
    bl_i = np.array([0, 0, 0])
    bl_j = np.array([-1, 1, 0])

    coefficients = compute_coefficients(i, j, bl_i, bl_j, basis)

    ndarrays_regression.check({"coefficients": coefficients}, default_tolerance=tol)


@pytest.mark.parametrize("resolve_k", (False, True), ids=("sum_k", "resolve_k"))
def test_compute_pdos(
    coefficients, total_dos, resolve_k, ndarrays_regression, tol
) -> None:
    pdos = compute_pdos(coefficients, total_dos, resolve_k)

    ndarrays_regression.check({"pdos": pdos}, default_tolerance=tol)


def test_compute_p_ij(
    coefficients, occupation_matrix, ndarrays_regression, tol
) -> None:
    p_ij = compute_p_ij(coefficients, occupation_matrix)

    ndarrays_regression.check({"p_ij": p_ij}, default_tolerance=tol)


def test_compute_ipdos_sum_k(pdos, ndarrays_regression, tol) -> None:
    energies = np.arange(-20, 10, 100)
    mu = 0

    pdos = pdos.sum(axis=1)

    ipdos = compute_ipdos(pdos, energies, mu)

    ndarrays_regression.check({"ipdos": ipdos}, default_tolerance=tol)


def test_compute_ipdos_resolve_k(pdos, ndarrays_regression, tol) -> None:
    energies = np.arange(-20, 10, 100)
    mu = 0

    ipdos = compute_ipdos(pdos, energies, mu)

    ndarrays_regression.check({"ipdos": ipdos}, default_tolerance=tol)


def test_get_h_ij(hamiltonian, ndarrays_regression, tol) -> None:
    i, j = 5, 6
    bl_i = np.array([0, 1, 1])
    bl_j = np.array([0, 0, 0])

    h_ij = get_h_ij(i, j, bl_i, bl_j, hamiltonian)

    ndarrays_regression.check({"h_ij": h_ij}, default_tolerance=tol)


def test_get_h_ij_missing_bl_vector(hamiltonian) -> None:
    i, j = 3, 2
    bl_i = np.array([1, 0, 0])
    bl_j = np.array([0, 0, 0])

    with pytest.raises(KeyError):
        get_h_ij(i, j, bl_i, bl_j, hamiltonian)


def test_DescriptorPipeline_with_coefficients(
    dpl, wannier_interaction, basis, ndarrays_regression, tol
) -> None:
    dpl = dpl.with_coefficients(basis)

    processed_interaction = dpl._pipeline[0](wannier_interaction)

    ndarrays_regression.check(
        {"coefficients": processed_interaction.coefficients}, default_tolerance=tol
    )


def test_DescriptorPipeline_without_coefficients(dpl, wannier_interaction) -> None:
    wannier_interaction = replace(
        wannier_interaction, coefficients=np.array([0.2, 0.7])
    )
    dpl = dpl.without_coefficients()

    processed_interaction = dpl._pipeline[0](wannier_interaction)

    assert processed_interaction.coefficients is None


@pytest.mark.parametrize(
    "with_dependency", (True, False), ids=("with_dependency", "without_dependency")
)
class TestOrderDependentProcesses:
    @pytest.mark.parametrize("resolve_k", (False, True), ids=("sum_k", "resolve_k"))
    def test_DescriptorPipeline_with_pdos(
        self,
        dpl,
        wannier_interaction,
        coefficients,
        total_dos,
        resolve_k,
        with_dependency,
        ndarrays_regression,
        tol,
    ) -> None:
        dpl = dpl.with_pdos(total_dos, resolve_k)

        if with_dependency:
            wannier_interaction = replace(
                wannier_interaction, coefficients=coefficients
            )
            processed_interaction = dpl._pipeline[0](wannier_interaction)

            ndarrays_regression.check(
                {"pdos": processed_interaction.pdos}, default_tolerance=tol
            )

        else:
            with pytest.raises(ValueError):
                dpl._pipeline[0](wannier_interaction)

    def test_DescriptorPipeline_with_p_ij(
        self,
        dpl,
        wannier_interaction,
        coefficients,
        occupation_matrix,
        with_dependency,
        ndarrays_regression,
        tol,
    ) -> None:
        dpl = dpl.with_p_ij(occupation_matrix)

        if with_dependency:
            wannier_interaction = replace(
                wannier_interaction, coefficients=coefficients
            )
            processed_interaction = dpl._pipeline[0](wannier_interaction)

            ndarrays_regression.check(
                {"p_ij": processed_interaction.p_ij}, default_tolerance=tol
            )

        else:
            with pytest.raises(ValueError):
                dpl._pipeline[0](wannier_interaction)

    @pytest.mark.parametrize("resolve_k", (False, True), ids=("sum_k", "resolve_k"))
    def test_DescriptorPipeline_with_integrals(
        self,
        dpl,
        wannier_interaction,
        pdos,
        resolve_k,
        with_dependency,
        ndarrays_regression,
        tol,
    ) -> None:
        energies = np.arange(-20, 10, 100)
        mu = 0
        dpl = dpl.with_integrals(energies, mu)

        if not resolve_k:
            pdos = pdos.sum(axis=1)

        if with_dependency:
            wannier_interaction = replace(wannier_interaction, pdos=pdos)
            processed_interaction = dpl._pipeline[0](wannier_interaction)

            ndarrays_regression.check(
                {"ipdos": processed_interaction.ipdos}, default_tolerance=tol
            )

        else:
            with pytest.raises(ValueError):
                dpl._pipeline[0](wannier_interaction)


def test_DescriptorPipeline_with_h_ij(
    dpl, wannier_interaction, hamiltonian, ndarrays_regression, tol
) -> None:
    dpl = dpl.with_h_ij(hamiltonian)

    processed_interaction = dpl._pipeline[0](wannier_interaction)

    ndarrays_regression.check(
        {"h_ij": processed_interaction.h_ij}, default_tolerance=tol
    )


def test_DescriptorPipeline_pipe(
    dpl,
    interactions,
    basis,
    total_dos,
    hamiltonian,
    occupation_matrix,
    ndarrays_regression,
    tol,
) -> None:
    energies = np.arange(-20, 10, 100)
    mu = 0

    processed_interactions = (
        dpl.with_coefficients(basis)
        .with_pdos(total_dos)
        .with_p_ij(occupation_matrix)
        .without_coefficients()
        .with_h_ij(hamiltonian)
        .with_integrals(energies, mu)
        .pipe(interactions)
    )

    descriptors = {}
    for interaction in processed_interactions:
        tag = interaction.tag

        descriptors[tag + "_pdos"] = interaction.pdos
        descriptors[tag + "_wohp"] = interaction.wohp
        descriptors[tag + "_wobi"] = interaction.wobi
        descriptors[tag + "_ipdos"] = interaction.ipdos
        descriptors[tag + "_iwohp"] = interaction.iwohp
        descriptors[tag + "_iwobi"] = interaction.iwobi

        for w_interaction in interaction.wannier_interactions:
            w_tag = w_interaction.tag

            assert w_interaction.coefficients is None

            descriptors[w_tag + "_pdos"] = w_interaction.pdos
            descriptors[w_tag + "_wohp"] = w_interaction.wohp
            descriptors[w_tag + "_wobi"] = w_interaction.wobi
            descriptors[w_tag + "_ipdos"] = w_interaction.ipdos
            descriptors[w_tag + "_iwohp"] = w_interaction.iwohp
            descriptors[w_tag + "_iwobi"] = w_interaction.iwobi

    ndarrays_regression.check(descriptors, default_tolerance=tol)
