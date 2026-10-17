"""Shared fixtures of the experiment tests."""

from __future__ import annotations

import sys

import pytest

# The experiment command line is POSIX-only (tmux, fork, flock); Windows runs the library tests alone.
if sys.platform == "win32":
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture(autouse=True)
def _bare_method_registry():
    """Every test starts from the runner's own registry: a test (or the front door it drives) that calls
    ``extra_methods.install()`` has it undone afterwards, so no test depends on the order the suite runs in."""
    from hedonic.experiments.overlapping import extra_methods

    extra_methods.uninstall()
    yield
    extra_methods.uninstall()
