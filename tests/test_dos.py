import pytest
import numpy as np
from pengwann.dos import DOS
from pengwann.io import read_Hamiltonian
from pengwann.geometry import AtomicInteraction, WannierInteraction
from pymatgen.core import Structure

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
def load_dos(datadir) -> None:
    dos_array = np.load(f'{datadir}/reduced_dos_array.npy')
    kpoints = np.load(f'{datadir}/kpoints.npy')
    U = np.load(f'{datadir}/U.npy')
    occupation_matrix = np.load(f'{datadir}/occupation_matrix.npy')
    H = read_Hamiltonian(f'{datadir}/wannier90_hr.dat')

    energies = np.arange(-25, 25 + 0.1, 0.1)
    nspin = 2

    dos = DOS(energies, dos_array, nspin, kpoints, U, H, occupation_matrix)

    return dos

def test_DOS_get_dos_matrix(load_dos, datadir) -> None:
    i, j = 1, 0
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([-1, 1, 0])

    test_dos_matrix = load_dos.get_dos_matrix(i, j, R_1, R_2)
    ref_dos_matrix = np.load(f'{datadir}/dos_matrix.npy')

    np.testing.assert_allclose(test_dos_matrix, ref_dos_matrix)

def test_DOS_get_dos_matrix_nk_resolved(load_dos, datadir) -> None:
    i, j = 1, 3
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([-1, 1, 0])

    test_dos_matrix = load_dos.get_dos_matrix(i, j, R_1, R_2, sum_matrix=False)
    ref_dos_matrix = np.load(f'{datadir}/dos_matrix_nk.npy')

    np.testing.assert_allclose(test_dos_matrix, ref_dos_matrix)

def test_DOS_P_ij(load_dos, datadir) -> None:
    i, j = 1, 4
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([-1, 0, 0])

    test_P_ij = load_dos.P_ij(i, j, R_1, R_2)
    ref_P_ij = np.load(f'{datadir}/P_ij.npy')

    np.testing.assert_allclose(test_P_ij, ref_P_ij)

def test_DOS_get_WOHP(load_dos, datadir) -> None:
    i, j = 1, 0
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([-1, 1, 0])
    dos_matrix = np.load(f'{datadir}/dos_matrix.npy')

    test_WOHP = load_dos.get_WOHP(i, j, R_1, R_2, dos_matrix)
    ref_WOHP = np.load(f'{datadir}/WOHP.npy')

    np.testing.assert_allclose(test_WOHP, ref_WOHP)

def test_DOS_get_WOHP_no_H(load_dos) -> None:
    load_dos._H = None

    i, j = 1, 7
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([-1, 0, 0])

    with pytest.raises(ValueError):
        load_dos.get_WOHP(i, j, R_1, R_2)

def test_DOS_get_WOBI(load_dos, datadir) -> None:
    i, j = 1, 0
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([-1, 1, 0])

    dos_matrix = np.load(f'{datadir}/dos_matrix.npy')

    test_WOBI = load_dos.get_WOBI(i, j, R_1, R_2, dos_matrix)
    ref_WOBI = np.load(f'{datadir}/WOBI.npy')

    np.testing.assert_allclose(test_WOBI, ref_WOBI)

def test_DOS_get_WOBI_no_occupation_matrix(load_dos) -> None:
    load_dos._occupation_matrix = None

    i, j = 2, 0
    R_1 = np.array([0, 0, 0])
    R_2 = np.array([1, -1, -1])

    with pytest.raises(ValueError):
        load_dos.get_WOBI(i, j, R_1, R_2)

def test_DOS_project(load_dos, datadir) -> None:
    geometry = Structure.from_file(f'{datadir}/structure.vasp')
    wannier_centres = ((9,),
                       (8,), 
                       (8,), 
                       (9,), 
                       (9,), 
                       (8,), 
                       (8,), 
                       (9,), 
                       (1, 2, 5, 6),
                       (0, 3, 4, 7))
    geometry.add_site_property('wannier_centres', wannier_centres)

    pdos = load_dos.project(geometry, ('C',))
    test_pdos_C = pdos['C']
    ref_pdos_C = np.load(f'{datadir}/pdos.npy')

    np.testing.assert_allclose(test_pdos_C, ref_pdos_C)

def test_DOS_get_descriptors(load_dos, datadir) -> None:
    wannier_interaction_1 = WannierInteraction(i=1, j=0, R_2=np.array([-1, 1, 0]))
    wannier_interaction_2 = WannierInteraction(i=1, j=7, R_2=np.array([-1, 0, 0]))
    interactions = (AtomicInteraction(pair_id=('C1', 'C2'), wannier_interactions=(wannier_interaction_1, wannier_interaction_2)),)

    test_descriptors = load_dos.get_descriptors(interactions)
    test_WOHP = test_descriptors[('C1', 'C2')]['WOHP']
    test_WOBI = test_descriptors[('C1', 'C2')]['WOBI']
    ref_WOHP = np.load(f'{datadir}/total_WOHP.npy')
    ref_WOBI = np.load(f'{datadir}/total_WOBI.npy')

    np.testing.assert_allclose(test_WOHP, ref_WOHP)
    np.testing.assert_allclose(test_WOBI, ref_WOBI)

def test_DOS_integrate_descriptors(load_dos, datadir) -> None:
    wohp = np.load(f'{datadir}/total_WOHP.npy')
    wobi = np.load(f'{datadir}/total_WOBI.npy')

    mu = 9.8675
    descriptors = {('C1', 'C2') : {'WOHP' : wohp, 'WOBI' : wobi}}

    test_integrals = load_dos.integrate_descriptors(descriptors, mu)
    ref_IWOHP = np.load(f'{datadir}/IWOHP.npy')
    ref_IWOBI = np.load(f'{datadir}/IWOBI.npy')

    np.testing.assert_allclose(test_integrals[('C1', 'C2')]['IWOHP'], ref_IWOHP)
    np.testing.assert_allclose(test_integrals[('C1', 'C2')]['IWOBI'], ref_IWOBI)
