import numpy as np
from pengwann.io import read, read_eigenvalues, read_Hamiltonian, read_U

def test_read_eigenvalues(datadir) -> None:
    num_bands = 12
    num_kpoints = 4096

    test_eigenvalues = read_eigenvalues(f'{datadir}/wannier90.eig', num_bands, num_kpoints)
    ref_eigenvalues = np.load(f'{datadir}/eigenvalues.npy')

    np.testing.assert_allclose(test_eigenvalues, ref_eigenvalues)

def test_read_U(datadir) -> None:
    test_U, test_kpoints = read_U(f'{datadir}/wannier90_u.mat')
    ref_U, ref_kpoints = np.load(f'{datadir}/U.npy'), np.load(f'{datadir}/kpoints.npy')

    np.testing.assert_allclose(test_U, ref_U)
    np.testing.assert_allclose(test_kpoints, ref_kpoints)

def test_read_Hamiltonian(datadir) -> None:
    test_H = read_Hamiltonian(f'{datadir}/wannier90_hr.dat')

    for R, matrix in test_H.items():
        assert matrix.shape == (8, 8)

    test_H_000 = test_H[(0, 0, 0)]
    ref_H_000 = np.load(f'{datadir}/H.npy')

    np.testing.assert_allclose(test_H_000, ref_H_000)

def test_read_wrapper(datadir) -> None:
    _, _, test_U, _ = read('wannier90', f'{datadir}')
    ref_U = np.load(f'{datadir}/U_with_dis.npy')

    np.testing.assert_allclose(test_U, ref_U)
