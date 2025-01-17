from pengwann.io import read, read_eigenvalues, read_hamiltonian, read_u


def test_read_eigenvalues(shared_datadir, ndarrays_regression) -> None:
    num_bands = 12
    num_kpoints = 4096

    eigenvalues = read_eigenvalues(
        f"{shared_datadir}/wannier90.eig", num_bands, num_kpoints
    )

    ndarrays_regression.check(
        {"eigenvalues": eigenvalues}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_read_u(shared_datadir, ndarrays_regression) -> None:
    u, kpoints = read_u(f"{shared_datadir}/wannier90_u.mat")

    ndarrays_regression.check(
        {"U": u, "kpoints": kpoints}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_read_hamiltonian(shared_datadir, ndarrays_regression) -> None:
    test_h = read_hamiltonian(f"{shared_datadir}/wannier90_hr.dat")

    for R, matrix in test_h.items():
        assert matrix.shape == (8, 8)

    h_000 = test_h[(0, 0, 0)]

    ndarrays_regression.check(
        {"H_000": h_000}, default_tolerance={"atol": 0, "rtol": 1e-07}
    )


def test_read_u_dis(shared_datadir, ndarrays_regression) -> None:
    _, _, u, _ = read("wannier90", f"{shared_datadir}")

    ndarrays_regression.check({"U": u}, default_tolerance={"atol": 0, "rtol": 1e-07})
