import pytest


@pytest.fixture(scope="session")
def tol():
    return {"atol": 1e-14, "rtol": 1e-07}
