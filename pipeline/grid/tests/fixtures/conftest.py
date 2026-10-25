"""pytest fixtures for pipeline/grid tests."""

from __future__ import annotations

import pytest

from pipeline.grid.tests.fixtures.gen import cleanup, make_fixture


@pytest.fixture
def grid_fixture():
    root = make_fixture()
    try:
        yield root
    finally:
        cleanup(root)
