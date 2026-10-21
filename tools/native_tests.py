"""Runs the native tests with the bit-level pins on, as the wheel builds do.

    python tools/native_tests.py

The pins (tests/test_native_core.py, tests/test_native_backend.py) hold exact
results recorded on macOS arm64 (the same ones native/rust/tests/pins.rs pins),
skipped unless HIG_PINNED=1. This sets it, so a platform whose rounding differs
from the recorded results fails, and fails on any skipped test. It also runs
the exact-arithmetic equilibrium oracle (tests/test_equilibrium.py) on the
platform's own results, and checks that hedonic.arrays computes Game's bits,
borrowing the edges through the binding (tests/test_arrays.py). It tests the
installed hedonic, not the checkout's src/.
"""

import os
import sys
import unittest
from pathlib import Path

os.environ["HIG_PINNED"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TESTS = ["tests.test_native_core", "tests.test_native_backend", "tests.test_equilibrium",
         "tests.test_arrays"]

if __name__ == "__main__":
    import hedonic

    print(f"hedonic from {Path(hedonic.__file__).parent}", flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromNames(TESTS)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
