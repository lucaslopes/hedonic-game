"""Shared fixtures of the experiment tests."""

from __future__ import annotations

import os
import sys
import tempfile

# The behaviour fingerprint's file cache goes to a folder of the test session, never the user's ~/.cache.
os.environ["HEDONIC_FINGERPRINT_CACHE"] = tempfile.mkdtemp(prefix="hedonic-fingerprints-")

# The experiment command line is POSIX-only (tmux, fork, flock); Windows runs the library tests alone.
if sys.platform == "win32":
    collect_ignore_glob = ["test_*.py"]
