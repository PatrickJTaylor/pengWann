import json
import pytest
import numpy as np
from pengwann.descriptors import DescriptorCalculator
from pengwann.io import read_hamiltonian
from pengwann.geometry import AtomicInteraction, WannierInteraction
from pymatgen.core import Structure


@pytest.fixture
def dcalc(shared_datadir) -> DescriptorCalculator:
    dos_array = np.load(f"{shared_datadir}/dos_array.npy")
    kpoints = np.load(f"{shared_datadir}/kpoints.npy")
    u = np.load(f"{shared_datadir}/U.npy")
    occupation_matrix = np.load(f"{shared_datadir}/occupation_matrix.npy")
    h = read_hamiltonian(f"{shared_datadir}/wannier90_hr.dat")

    energies = np.arange(-25, 25 + 0.1, 0.1, dtype=np.float64)
    nspin = 2

    dcalc = DescriptorCalculator(
        dos_array, nspin, kpoints, u, h, occupation_matrix, energies
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
    nspin = 2
    energy_range = (-6, 6)
    resolution = 0.01
    sigma = 0.05
    kpoints = np.zeros_like((10, 3))
    u = np.zeros_like((10, 4, 4))

    dcalc = DescriptorCalculator.from_eigenvalues(
        eigenvalues, nspin, energy_range, resolution, sigma, kpoints, u
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


def test_DescriptorCalculator_get_p_ij(dcalc, ndarrays_regression) -> None:
    i, j = 1, 4
    bl_1 = np.array([0, 0, 0])
    bl_2 = np.array([-1, 0, 0])

    c_star = np.conj(dcalc.get_coefficient_matrix(i, bl_1))
    c = dcalc.get_coefficient_matrix(j, bl_2)

    p_ij = dcalc.get_p_ij(c_star, c)

    ndarrays_regression.check(
        {"P_ij": p_ij}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_DescriptorCalculator_get_p_ij_no_occupation_matrix(dcalc) -> None:
    dcalc._occupation_matrix = None

    c_star = np.ones_like((10, 10))
    c = c_star

    with pytest.raises(TypeError):
        dcalc.get_p_ij(c_star, c)


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

            population = w_interaction.population
            if population is None:
                populations[w_label] = np.nan

            else:
                populations[w_label] = population

    ndarrays_regression.check(populations, default_tolerance={"atol": 0, "rtol": 1e-07})


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


@pytest.mark.parametrize("resolve_k", (False, True), ids=("sum_k", "resolve_k"))
def test_DescriptorCalculator_assign_descriptors(
    dcalc, resolve_k, ndarrays_regression
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

    descriptors = {}
    for interaction in interactions:
        id_i, id_j = interaction.pair_id
        label = id_i + id_j

        descriptors[label + "_WOHP"] = interaction.wohp
        descriptors[label + "_WOBI"] = interaction.wobi

        for w_interaction in interaction.wannier_interactions:
            w_label = str(w_interaction.i) + str(w_interaction.j)

            descriptors[w_label + "_WOHP"] = w_interaction.wohp
            descriptors[w_label + "_WOBI"] = w_interaction.wobi

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

    integrals = {}
    for interaction in interactions:
        id_i, id_j = interaction.pair_id
        label = id_i + id_j

        integrals[label + "_IWOHP"] = interaction.iwohp
        integrals[label + "_IWOBI"] = interaction.iwobi

        for w_interaction in interaction.wannier_interactions:
            w_label = str(w_interaction.i) + str(w_interaction.j)
            iwohp = w_interaction.iwohp
            iwobi = w_interaction.iwobi

            if iwohp is None:
                integrals[w_label + "_IWOHP"] = np.nan

            else:
                integrals[w_label + "_IWOHP"] = iwohp

            if iwobi is None:
                integrals[w_label + "_IWOBI"] = np.nan

            else:
                integrals[w_label + "_IWOBI"] = iwobi

    ndarrays_regression.check(integrals, default_tolerance={"atol": 0, "rtol": 1e-07})


def test_DescriptorCalculator_integrate_descriptors_no_energies(dcalc) -> None:
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
        dcalc.integrate_descriptors(interactions, mu)


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

    num_wann = 7
    doe = dcalc.get_density_of_energy(interactions, num_wann)

    ndarrays_regression.check(
        {"DOE": doe}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )
