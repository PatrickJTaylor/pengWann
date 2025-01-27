import pytest


@pytest.fixture(scope="session")
def tol():
    return {"atol": 0, "rtol": 1e-07}
