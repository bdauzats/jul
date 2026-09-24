"""Shared fixtures. Tests that load an MLX model are marked `slow` and skipped by default.

    pytest tests                 # fast tests only (the default)
    JUL_SLOW=1 pytest tests      # everything, loads MiniCPM5-2B and WeMM-Embedding-4B
    JUL_SLOW=1 pytest tests -m slow   # only the ones that need a model
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "cli"))

SLOW = os.environ.get("JUL_SLOW") == "1"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads a model; run with JUL_SLOW=1")


def pytest_collection_modifyitems(config, items):
    if SLOW:
        return
    skip = pytest.mark.skip(reason="needs a model; set JUL_SLOW=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def client():
    from jul import TypeSafeClient
    c = TypeSafeClient(model="minicpm5-2b")
    yield c
    c.close()


@pytest.fixture
def home(tmp_path):
    """An isolated context store, so tests never touch ~/.jul."""
    return tmp_path / "contexts"
