"""The behaviour fingerprints of the core (partitions, covers, audit), and what the disjoint store keys on them."""

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
# The same for the cover detector and the fixed-cover audit (hedonic 2.0.0rc5 to rc12: the C sources ead786fe... and
# c8684664... give all three). A change of either means the overlap studies' records are of another results class
# (runtime.CORE_CLASSES): if intended, add the class there, say so, and update the pin in the same commit.
PINNED_COVERS = "covers-v1:ddc34c763b38292fd19c03e9a89da401336fcddb098ea1c7983dbb964b93966a"
PINNED_AUDIT = "audit-v1:39d5ad0e58d4e6816d33c7209da31937226f19ebff88d0c004e9f1629c0d7e48"


class TestFingerprint(unittest.TestCase):
    def test_the_loaded_build_has_the_pinned_fingerprint(self):
        self.assertEqual(fingerprint.compute(), PINNED)

    def test_the_loaded_build_has_the_pinned_cover_and_audit_fingerprints(self):
        self.assertEqual(fingerprint.compute_covers(), PINNED_COVERS)
        self.assertEqual(fingerprint.compute_audit(), PINNED_AUDIT)
        self.assertEqual(fingerprint.fingerprints(), (PINNED, PINNED_COVERS, PINNED_AUDIT))

    def test_the_cover_grid_covers_caps_starts_and_options(self):
        calls = list(fingerprint.cover_calls())
        self.assertEqual(len(calls), 6912)
        self.assertEqual({kwargs["max_memberships"] for _, _, kwargs in calls}, {2, 3, -1})
        self.assertEqual({key[4] for key, _, _ in calls},
                         {"plain", "edge_weights", "node_weights", "max_total_communities", "max_communities",
                          "cover_start", "wide_start", "flat_start"})
        wide = next(kwargs["initial_membership"] for key, _, kwargs in calls if key[4] == "wide_start")
        self.assertTrue(all(len(row) == 5 for row in wide), "a start with more labels than every finite cap")
        self.assertEqual({kwargs["allow_isolation"] for _, _, kwargs in calls}, {False, True})

    def test_the_audit_grid_varies_cap_tolerance_pruning_and_weights(self):
        calls = list(fingerprint.audit_calls())
        self.assertEqual(len(calls), 1728)
        self.assertEqual({key[2] for key, _, _ in calls}, {"own", "above", "uncapped"})
        self.assertEqual({(kwargs["atol"], kwargs["rtol"]) for _, _, kwargs in calls}, set(fingerprint.AUDIT_TOLERANCES))
        self.assertEqual({kwargs["prune"] for _, _, kwargs in calls}, {True, False})
        self.assertEqual({kwargs["edge_weights"] is None for _, _, kwargs in calls}, {True, False})
        uncapped = [(g, kwargs) for key, g, kwargs in calls if key[2] == "uncapped"]
        self.assertTrue(all(kwargs["max_memberships"] == g.vcount() for g, kwargs in uncapped))
        # Covers with up to five labels per vertex on twelve communities: many cells between score crossings.
        dense = next(kwargs["memberships"] for key, _, kwargs in calls if key[1] == "dense")
        self.assertEqual(max(len(row) for row in dense), 5)

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
            # One file holds every grid of the core; the partition grid stays where rc11 and earlier read it.
            fingerprint._CACHE.clear()
            self.assertEqual(fingerprint.fingerprints(), (PINNED, PINNED_COVERS, PINNED_AUDIT))
            data = json.loads(path.read_text())
            self.assertEqual(data["fingerprints"], {fingerprint.GRID: PINNED, fingerprint.COVER_GRID: PINNED_COVERS,
                                                    fingerprint.AUDIT_GRID: PINNED_AUDIT})
            self.assertEqual((data["grid"], data["fingerprint"]), (fingerprint.GRID, PINNED))
            fingerprint._CACHE.clear()
            with patch.object(fingerprint, "compute") as one, patch.object(fingerprint, "compute_covers") as two, \
                    patch.object(fingerprint, "compute_audit") as three:
                self.assertEqual(fingerprint.fingerprints(), (PINNED, PINNED_COVERS, PINNED_AUDIT))  # all from the file
                for mock in (one, two, three):
                    mock.assert_not_called()
            # A file of the first format (one grid) is read, and the other grids are added to it.
            fingerprint._CACHE.clear()
            path.write_text(json.dumps({"grid": fingerprint.GRID, "core_sha256": core_sha256(), "fingerprint": PINNED}))
            with patch.object(fingerprint, "compute") as compute:
                self.assertEqual(fingerprint.partition_fingerprint(), PINNED)
                compute.assert_not_called()
            self.assertEqual(fingerprint.audit_fingerprint(), PINNED_AUDIT)
            self.assertEqual(json.loads(path.read_text())["fingerprints"],
                             {fingerprint.GRID: PINNED, fingerprint.AUDIT_GRID: PINNED_AUDIT})
        with patch.dict(os.environ, {"HEDONIC_FINGERPRINT_CACHE": ""}), patch.dict(fingerprint._CACHE, clear=True):
            self.assertEqual(fingerprint.partition_fingerprint(), PINNED)  # no file cache at all
        with self.assertRaises(ValueError):
            fingerprint.fingerprint("covers-v0")


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
