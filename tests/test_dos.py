import json
import pytest
import numpy as np
from pengwann.dos import DOS
from pengwann.io import read_Hamiltonian
from pengwann.geometry import AtomicInteraction, WannierInteraction
from pymatgen.core import Structure


@pytest.fixture
def dos(shared_datadir) -> None:
    dos_array = np.load(f"{shared_datadir}/dos_array.npy")
    kpoints = np.load(f"{shared_datadir}/kpoints.npy")
    U = np.load(f"{shared_datadir}/U.npy")
    occupation_matrix = np.load(f"{shared_datadir}/occupation_matrix.npy")
    H = read_Hamiltonian(f"{shared_datadir}/wannier90_hr.dat")

    energies = np.arange(-25, 25 + 0.1, 0.1)
    nspin = 2

    dos = DOS(energies, dos_array, nspin, kpoints, U, H, occupation_matrix)

    return dos


@pytest.fixture
def ref_geometry(shared_datadir) -> Structure:
    with open(f"{shared_datadir}/geometry.json", "r") as stream:
        serial = json.load(stream)

    geometry = Structure.from_dict(serial)

    return geometry


def test_DOS_from_eigenvalues(ndarrays_regression) -> None:
    eigenvalues = np.array([[-4, -3, -2, -1, 1, 2, 3, 4], [-5, -4, -3, -2, 2, 3, 4, 5]])
    nspin = 2
    energy_range = (-6, 6)
    resolution = 0.01
    sigma = 0.05
    kpoints = np.zeros_like((10, 3))
    U = np.zeros_like((10, 4, 4))

    dos = DOS.from_eigenvalues(
        eigenvalues, nspin, energy_range, resolution, sigma, kpoints, U
    )

    dos_array = dos._dos_array

    ndarrays_regression.check(
        {"dos_array": dos_array}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


@pytest.mark.parametrize("sum_matrix", (True, False), ids=("sum_nk", "resolve_nk"))
def test_DOS_get_dos_matrix(dos, sum_matrix, ndarrays_regression) -> None:
    i, j = 1, 0
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([-1, 1, 0])

    dos_matrix = dos.get_dos_matrix(i, j, R_1, R_2, sum_matrix=sum_matrix)

    ndarrays_regression.check(
        {"dos_matrix": dos_matrix}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


@pytest.mark.parametrize(
    "dos_matrix",
    (None, np.ones_like((5))),
    ids=("without_dos_matrix", "with_dos_matrix"),
)
def test_DOS_get_WOHP(dos, dos_matrix, ndarrays_regression) -> None:
    i, j = 1, 0
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([-1, 1, 0])

    WOHP = dos.get_WOHP(i, j, R_1, R_2, dos_matrix=dos_matrix)

    ndarrays_regression.check(
        {"WOHP": WOHP}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_DOS_get_WOHP_no_H(dos) -> None:
    dos._H = None

    i, j = 1, 7
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([-1, 0, 0])

    with pytest.raises(ValueError):
        dos.get_WOHP(i, j, R_1, R_2)


@pytest.mark.parametrize(
    "dos_matrix",
    (None, np.ones_like((5))),
    ids=("without_dos_matrix", "with_dos_matrix"),
)
def test_DOS_get_WOBI(dos, dos_matrix, ndarrays_regression) -> None:
    i, j = 1, 0
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([-1, 1, 0])

    WOBI = dos.get_WOBI(i, j, R_1, R_2, dos_matrix=dos_matrix)

    ndarrays_regression.check(
        {"WOBI": WOBI}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_DOS_get_WOBI_no_occupation_matrix(dos) -> None:
    dos._occupation_matrix = None

    i, j = 2, 0
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([1, -1, -1])

    with pytest.raises(ValueError):
        dos.get_WOBI(i, j, R_1, R_2)


def test_DOS_P_ij(dos, ndarrays_regression) -> None:
    i, j = 1, 4
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([-1, 0, 0])

    P_ij = dos.P_ij(i, j, R_1, R_2)

    ndarrays_regression.check(
        {"P_ij": P_ij}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_DOS_project(dos, ref_geometry, shared_datadir, ndarrays_regression) -> None:
    pdos = dos.project(ref_geometry, ("C",))

    ndarrays_regression.check(pdos, default_tolerance={"atol": 0, "rtol": 1e-07})


def test_DOS_get_populations(dos, ref_geometry, shared_datadir, ndarrays_regression) -> None:
    mu = 9.8675
    pdos = dos.project(ref_geometry, ("C",))
    populations = dos.get_populations(pdos, mu)
    populations = np.array(
        (populations["C1"]["population"], populations["C2"]["population"])
    )

    ndarrays_regression.check(
        {"populations": populations}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_DOS_get_populations_charges(dos, ref_geometry, shared_datadir, ndarrays_regression) -> None:
    mu = 9.8675
    valence = {"C": 4}
    pdos = dos.project(ref_geometry, ("C",))
    populations = dos.get_populations(pdos, mu, valence=valence)
    charges = np.array((populations["C1"]["charge"], populations["C2"]["charge"]))

    ndarrays_regression.check(
        {"charges": charges}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_DOS_get_populations_wrong_valence(dos, ref_geometry, shared_datadir) -> None:
    mu = 9.8675
    valence = {"N": 4}
    pdos = dos.project(ref_geometry, ("C",))

    with pytest.raises(ValueError):
        populations = dos.get_populations(pdos, mu, valence=valence)


@pytest.mark.parametrize("sum_k", (True, False), ids=("sum_k", "resolve_k"))
def test_DOS_get_descriptors(dos, sum_k, ndarrays_regression) -> None:
    wannier_interaction_1 = WannierInteraction(
        i=1, j=0, R_1=np.array([0, 1, 0]), R_2=np.array([0, 0, 0])
    )
    wannier_interaction_2 = WannierInteraction(
        i=5, j=6, R_1=np.array([0, 1, 1]), R_2=np.array([0, 0, 0])
    )
    interactions = (
        AtomicInteraction(
            pair_id=("C1", "C2"),
            wannier_interactions=(wannier_interaction_1, wannier_interaction_2),
        ),
    )

    descriptors = dos.get_descriptors(interactions, sum_k=sum_k)
    WOHP = descriptors[("C1", "C2")]["WOHP"]
    WOBI = descriptors[("C1", "C2")]["WOBI"]

    ndarrays_regression.check(
        descriptors[("C1", "C2")], default_tolerance={"atol": 0, "rtol": 1e-07}
    )


@pytest.mark.parametrize("sum_k", (True, False), ids=("sum_k", "resolve_k"))
def test_DOS_process_interaction(
    dos, sum_k, data_regression, ndarrays_regression
) -> None:
    wannier_interaction_1 = WannierInteraction(
        i=1, j=0, R_1=np.array([0, 1, 0]), R_2=np.array([0, 0, 0])
    )
    wannier_interaction_2 = WannierInteraction(
        i=5, j=6, R_1=np.array([0, 1, 1]), R_2=np.array([0, 0, 0])
    )
    interaction = AtomicInteraction(
        pair_id=("C1", "C2"),
        wannier_interactions=(wannier_interaction_1, wannier_interaction_2),
    )

    if sum_k:
        labels = ("WOHP", "WOBI", "sum_k")

    else:
        labels = ("WOHP", "WOBI")

    interaction_and_labels = (interaction, labels)

    pair_id, descriptors = dos.process_interaction(interaction_and_labels)

    data_regression.check({"pair_id": pair_id})
    ndarrays_regression.check(descriptors, default_tolerance={"atol": 0, "rtol": 1e-07})


def test_DOS_get_BWDF(dos, ndarrays_regression) -> None:
    integrated_descriptors = {("C1", "C2"): {"IWOHP": 5}, ("C3", "C4"): {"IWOHP": 3}}
    geometry = Structure(
        ((3, 0, 0), (0, 3, 0), (0, 0, 3)),
        ("C", "C", "C", "C"),
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 2)),
        coords_are_cartesian=True,
    )

    bwdf = dos.get_BWDF(integrated_descriptors, geometry)
    r, weights = bwdf[("C", "C")]

    ndarrays_regression.check(
        {"r": r, "weights": weights}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_DOS_get_density_of_energy(dos, ndarrays_regression) -> None:
    wannier_interaction_1 = WannierInteraction(
        i=1, j=0, R_1=np.array([0, 1, 0]), R_2=np.array([0, 0, 0])
    )
    wannier_interaction_2 = WannierInteraction(
        i=5, j=6, R_1=np.array([0, 1, 1]), R_2=np.array([0, 0, 0])
    )
    interactions = (
        AtomicInteraction(
            pair_id=("C1", "C2"),
            wannier_interactions=(wannier_interaction_1, wannier_interaction_2),
        ),
    )

    descriptors = dos.get_descriptors(interactions)

    num_wann = 7
    doe = dos.get_density_of_energy(descriptors, num_wann)

    ndarrays_regression.check(
        {"DOE": doe}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


@pytest.mark.parametrize("sum_k", (True, False), ids=("sum_k", "resolve_k"))
def test_DOS_integrate_descriptors(dos, sum_k, ndarrays_regression) -> None:
    wannier_interaction_1 = WannierInteraction(
        i=1, j=0, R_1=np.array([0, 1, 0]), R_2=np.array([0, 0, 0])
    )
    wannier_interaction_2 = WannierInteraction(
        i=5, j=6, R_1=np.array([0, 1, 1]), R_2=np.array([0, 0, 0])
    )
    interactions = (
        AtomicInteraction(
            pair_id=("C1", "C2"),
            wannier_interactions=(wannier_interaction_1, wannier_interaction_2),
        ),
    )

    descriptors = dos.get_descriptors(interactions, sum_k=sum_k)
    mu = 9.8675

    integrals = dos.integrate_descriptors(descriptors, mu)

    ndarrays_regression.check(
        integrals[("C1", "C2")], default_tolerance={"atol": 0, "rtol": 1e-07}
    )
