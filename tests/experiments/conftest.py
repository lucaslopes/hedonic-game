"""Shared fixtures of the experiment tests."""

from __future__ import annotations

import sys

# The experiment command line is POSIX-only (tmux, fork, flock); Windows runs the library tests alone.
if sys.platform == "win32":
    collect_ignore_glob = ["test_*.py"]

