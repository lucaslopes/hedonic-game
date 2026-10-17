"""Shared fixtures of the experiment tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _bare_method_registry():
    """Every test starts from the runner's own registry: a test (or the front door it drives) that calls
    ``extra_methods.install()`` has it undone afterwards, so no test depends on the order the suite runs in."""
    from hedonic.experiments.overlapping import extra_methods

    extra_methods.uninstall()
    yield
    extra_methods.uninstall()
