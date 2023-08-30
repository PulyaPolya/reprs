import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true")


@pytest.fixture(scope="session")
def plot(pytestconfig):
    return pytestconfig.getoption("plot")


@pytest.fixture()
def n_kern_files(monkeypatch):
    monkeypatch.setenv("N_KERN_FILES", os.environ.get("N_KERN_FILES", "1"))
