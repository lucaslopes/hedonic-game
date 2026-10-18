"""The behaviour fingerprint of the partition detector, and what the disjoint store keys on it."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments import fingerprint
from hedonic.experiments.disjoint import store as store_mod

# The rounding of the core is fixed by its source, so this holds on every platform; it was the same for
# hedonic 1.0.0 and 2.0.0. A change means the partition path computes something else: if that is intended,
# every disjoint store's Hedonic rows are recomputed, and this value is updated in the same commit.
PINNED = "partitions-v1:0668d1011dc78b6fa63e6c9a0c17901e4738d421a270e906858fb415593c9c12"


class TestFingerprint(unittest.TestCase):
    def test_the_loaded_build_has_the_pinned_fingerprint(self):
        self.assertEqual(fingerprint.compute(), PINNED)

    def test_the_grid_covers_what_the_store_calls(self):
        calls = list(fingerprint.partition_calls())
        self.assertEqual(len(calls), 8736)
        options = {key[3] for key, _, _ in calls}
        self.assertIn("noisy_start", options)
        self.assertTrue(all(kwargs["max_memberships"] == 1 for _, _, kwargs in calls))
        self.assertIn(2**63 + 12345, {kwargs["seed"] for _, _, kwargs in calls})
        self.assertIn("density", {key[1] for key, _, _ in calls})

    def test_cached_per_compiled_core(self):
        from hedonic.experiments.runtime import core_sha256

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_FINGERPRINT_CACHE": d}), \
                patch.dict(fingerprint._CACHE, clear=True):
            self.assertEqual(fingerprint.partition_fingerprint(), PINNED)
            path = Path(d) / f"{core_sha256()}.json"
            self.assertEqual(json.loads(path.read_text())["fingerprint"], PINNED)
            fingerprint._CACHE.clear()
            with patch.object(fingerprint, "compute") as compute:
                self.assertEqual(fingerprint.partition_fingerprint(), PINNED)  # read from the file
                compute.assert_not_called()
            fingerprint._CACHE.clear()
            path.write_text(json.dumps({"grid": "partitions-v0", "core_sha256": core_sha256(), "fingerprint": "x"}))
            self.assertEqual(fingerprint.partition_fingerprint(), PINNED)  # another grid's entry is recomputed
        with patch.dict(os.environ, {"HEDONIC_FINGERPRINT_CACHE": ""}), patch.dict(fingerprint._CACHE, clear=True):
            self.assertEqual(fingerprint.partition_fingerprint(), PINNED)  # no file cache at all


class TestStoreIdentity(unittest.TestCase):
    INFO = {"method_call_name": "community_hedonic",
            "parameters": {"resolution": 1, "initial_membership": None, "allow_isolation": False, "max_memberships": 1}}

    def _digest(self, legacy=False, **fields):
        runtime = dict(store_mod.runtime_identity(), **fields)
        return store_mod.spec_hash(store_mod.method_identity("Hedonic", self.INFO, n_runs=2, runtime=runtime,
                                                             legacy=legacy))

    def test_hedonic_is_keyed_on_behaviour_not_source(self):
        base = self._digest()
        self.assertEqual(self._digest(hedonic_core_source_sha256="1" * 64), base)
        self.assertNotEqual(self._digest(hedonic_partition_fingerprint="partitions-v1:" + "0" * 64), base)
        # Under the rule before the fingerprint, the source decided.
        self.assertNotEqual(self._digest(legacy=True, hedonic_core_source_sha256="1" * 64), self._digest(legacy=True))

    def test_every_runtime_records_both(self):
        runtime = store_mod.runtime_identity()
        self.assertEqual(runtime["hedonic_partition_fingerprint"], PINNED)
        self.assertEqual(len(runtime["hedonic_core_source_sha256"]), 64)
        self.assertEqual(store_mod.method_source("Hedonic", runtime), runtime["hedonic_core_source_sha256"])
        self.assertIsNone(store_mod.method_source("Mirror", runtime))


if __name__ == "__main__":
    unittest.main()
