import pytest
import numpy as np
from pengwann.dos import DOS
from pengwann.io import read_Hamiltonian

def test_dos_init_from_eigenvalues(datadir) -> None:
    eigenvalues = np.array([[-4, -3, -2, -1, 1, 2, 3, 4],
                            [-5, -4, -3, -2, 2, 3, 4, 5]])
    nspin = 2
    energy_range = (-6, 6)
    resolution = 0.01
    sigma = 0.05
    kpoints = np.zeros_like((10, 3))
    U = np.zeros_like((10, 4, 4))

    dos = DOS.from_eigenvalues(eigenvalues, nspin, energy_range, resolution, sigma, kpoints, U)

    test_dos_array = dos._dos_array
    ref_dos_array = np.load(f'{datadir}/dos_array.npy')

    np.testing.assert_allclose(test_dos_array, ref_dos_array)

@pytest.fixture
def instance_dos(datadir) -> None:
    dos_array = np.load(f'{datadir}/reduced_dos_array.npy')
    kpoints = np.load(f'{datadir}/kpoints.npy')
    U = np.load(f'{datadir}/U.npy')
    occupation_matrix = np.load(f'{datadir}/occupation_matrix.npy')
    H = read_Hamiltonian(f'{datadir}/wannier90_hr.dat')

    tbd = []
    for R in H.keys():
        if sum([abs(element) for element in R]) > 3:
            tbd.append(R)

    for R in tbd:
        del H[R]

    energies = np.arange(-25, 25 + 0.1, 0.1)
    nspin = 2

    dos = DOS(energies, dos_array, nspin, kpoints, U, H, occupation_matrix)

    return dos

def test_DOS_get_dos_matrix(instance_dos, datadir) -> None:
    i, j = 1, 0
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([1, -1, 0])

    test_dos_matrix = instance_dos.get_dos_matrix(i, j, R_1, R_2)
    ref_dos_matrix = np.load(f'{datadir}/dos_matrix.npy')

    np.testing.assert_allclose(test_dos_matrix, ref_dos_matrix)

def test_DOS_get_dos_matrix_nk_resolved(instance_dos, datadir) -> None:
    i, j = 1, 0
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([1, -1, 0])

    test_dos_matrix = instance_dos.get_dos_matrix(i, j, R_1, R_2, sum_matrix=False)
    ref_dos_matrix = np.load(f'{datadir}/dos_matrix_nk.npy')

    np.testing.assert_allclose(test_dos_matrix, ref_dos_matrix)

def test_DOS_P_ij(instance_dos, datadir) -> None:
    i, j = 1, 4
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([0, -1, 0])

    test_P_ij = instance_dos.P_ij(i, j, R_1, R_2)
    ref_P_ij = np.load(f'{datadir}/P_ij.npy')

    np.testing.assert_allclose(test_P_ij, ref_P_ij)

def test_DOS_get_WOHP(instance_dos, datadir) -> None:
    i, j = 1, 3
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([0, -1, 0])
    dos_matrix = np.load(f'{datadir}/dos_matrix.npy')

    test_WOHP = instance_dos.get_WOHP(i, j, R_1, R_2, dos_matrix)
    ref_WOHP = np.load(f'{datadir}/WOHP.npy')

    np.testing.assert_allclose(test_WOHP, ref_WOHP)

def test_DOS_get_WOBI(instance_dos, datadir) -> None:
    i, j = 1, 4
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([0, -1, 0])

    dos_matrix = np.load(f'{datadir}/dos_matrix.npy')

    test_WOBI = instance_dos.get_WOBI(i, j, R_1, R_2, dos_matrix)
    ref_WOBI = np.load(f'{datadir}/WOBI.npy')

    np.testing.assert_allclose(test_WOBI, ref_WOBI)
