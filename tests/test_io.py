from pengwann.io import read, read_eigenvalues, read_Hamiltonian, read_U


def test_read_eigenvalues(datadir, ndarrays_regression) -> None:
    num_bands = 12
    num_kpoints = 4096

    eigenvalues = read_eigenvalues(f"{datadir}/wannier90.eig", num_bands, num_kpoints)

    ndarrays_regression.check(
        {"eigenvalues": eigenvalues}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_read_U(datadir, ndarrays_regression) -> None:
    U, kpoints = read_U(f"{datadir}/wannier90_u.mat")

    ndarrays_regression.check(
        {"U": U, "kpoints": kpoints}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_read_Hamiltonian(datadir, ndarrays_regression) -> None:
    test_H = read_Hamiltonian(f"{datadir}/wannier90_hr.dat")

    for R, matrix in test_H.items():
        assert matrix.shape == (8, 8)

    H_000 = test_H[(0, 0, 0)]

    ndarrays_regression.check(
        {"H_000": H_000}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_read_U_dis(datadir, ndarrays_regression) -> None:
    _, _, U, _ = read("wannier90", f"{datadir}")

    ndarrays_regression.check({"U": U}, default_tolerance={"atol": 0, "rtol": 1e-07})
