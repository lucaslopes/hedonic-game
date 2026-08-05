"""Protocol-lock and evidence-reconciliation regression tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hedonic.experiments import CLI
from hedonic.experiments.overlapping import protocol


class TestOverlappingProtocol(unittest.TestCase):
    def test_tracked_lock_matches_code_and_released_package(self):
        identity = protocol.current_experiment_identity()
        self.assertTrue(identity["tracked_files_match_lock"])
        self.assertTrue(identity["lucas_igraph"]["package_identity_matches_lock"])
        self.assertEqual(identity["lucas_igraph"]["actual_version"], "1.0.0.2")
        self.assertEqual(identity["lucas_igraph"]["actual_igraph_version"], "1.0.0.2")
        self.assertEqual(protocol.load_protocol_lock()["expected_conditions"], 125)
        self.assertIn("overlapping-audit", CLI.COMMANDS)

    def test_reconciliation_is_read_only_and_explains_legacy_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifacts"
            run = artifact / "shards" / "amazon-all" / "runs" / "record.json"
            run.parent.mkdir(parents=True)
            run.write_text(json.dumps({
                "dataset": "amazon", "cover": "all", "method": "cpm", "seed": 0,
                "resolution": 0.5, "protocol_version": 6, "status": "completed",
                "timeout_seconds": 10.0, "memory_limit_bytes": 100,
                "max_memberships": 2,
                "run_options": {"omega": True, "omega_sample_size": 10},
                "dataset_report": {"n": 6, "m": 8},
            }))
            orchestration = artifact / "orchestration"
            orchestration.mkdir()
            (orchestration / "plan.json").write_text(json.dumps({"jobs": [{
                "name": "amazon-all", "max_memberships": 2,
                "memory": {"detector_memory_limit_bytes": 100},
            }]}))
            config = root / "config.toml"
            config.write_text("""
[overlapping_paper]
methods = ["cpm"]
seeds = "0"
resolutions = "auto"
timeout_per_run = 10
omega = true
omega_sample_size = 10
[[overlapping_paper.jobs]]
name = "amazon-all"
dataset = "amazon"
cover = "all"
""")
            lock = root / "protocol.lock.json"
            lock.write_text(json.dumps({
                "schema_version": 2,
                "protocol_name": "test",
                "run_protocol_version": 6,
                "expected_conditions": 1,
                "analysis_graph_policy": "common_undirected_simple_v1",
                "lucas_igraph": {
                    "distribution": "lucas-igraph", "source_path": ".", "revision": None,
                },
                "tracked_files": {},
            }))
            before = run.read_bytes()
            report = protocol.reconcile(artifact, config, lock)
            self.assertEqual(report["expected_conditions"], 1)
            self.assertEqual(report["present_records"], 1)
            self.assertEqual(report["admissible_records"], 0)
            self.assertIn("missing_experiment_identity", report["rows"][0]["rejection_reasons"])
            self.assertIn("missing_dataset_metadata_identity", report["rows"][0]["rejection_reasons"])
            self.assertIn("missing_analysis_graph_identity", report["rows"][0]["rejection_reasons"])
            self.assertEqual(run.read_bytes(), before)

    def test_old_revision_only_identity_is_not_package_admissible(self):
        expected = protocol.current_experiment_identity()
        old = {
            **{key: expected[key] for key in (
                "protocol_lock_sha256", "protocol_name", "run_protocol_version",
                "analysis_graph_policy", "tracked_files",
            )},
            "schema_version": 1,
            "lucas_igraph": {
                "distribution": "lucas-igraph",
                "actual_revision": expected["lucas_igraph"]["released_native_sha"],
            },
        }
        reasons = protocol.identity_rejection_reasons(old, expected)
        self.assertIn("experiment_identity.schema_version_mismatch", reasons)
        self.assertTrue(any(reason.startswith("lucas_igraph_") for reason in reasons))


if __name__ == "__main__":
    unittest.main()
