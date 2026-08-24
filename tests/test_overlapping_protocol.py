"""Protocol-lock and evidence-reconciliation regression tests."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments import CLI
from hedonic.experiments.overlapping import benchmark, protocol
from hedonic.experiments.overlapping.methods import (
    METHODS,
    method_dependency_identity,
)
from hedonic.experiments.overlapping.metrics import (
    evaluate_cover,
    quality_overlapping_cpm,
)
from hedonic.experiments.overlapping.snap import (
    bounded_induced_dataset,
    common_undirected_analysis_dataset,
    smoke_dataset,
)


class TestOverlappingProtocol(unittest.TestCase):
    def test_schema_three_dependency_sections_are_not_vacuously_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "incomplete.lock.json"
            lock.write_text(json.dumps({
                "schema_version": 3,
                "protocol_name": "incomplete-paper-lock",
                "run_protocol_version": benchmark.RUN_PROTOCOL_VERSION,
                "analysis_graph_policy": "common_undirected_simple_v1",
                "lucas_igraph": {},
                "tracked_files": {},
            }))
            identity = protocol.current_experiment_identity(lock)
            self.assertFalse(identity["external_dependency_lock_complete"])
            self.assertFalse(identity["external_dependencies_match_lock"])
            self.assertFalse(identity["scientific_dependency_lock_complete"])
            self.assertFalse(identity["scientific_dependencies_match_lock"])
            self.assertFalse(
                identity["lucas_igraph"]["package_identity_matches_lock"]
            )

    def test_tracked_lock_matches_code_and_released_package(self):
        identity = protocol.current_experiment_identity()
        self.assertTrue(identity["tracked_files_match_lock"])
        self.assertTrue(identity["lucas_igraph"]["package_identity_matches_lock"])
        self.assertEqual(identity["lucas_igraph"]["actual_version"], "1.0.0.3")
        self.assertEqual(identity["lucas_igraph"]["actual_igraph_version"], "1.0.0.3")
        lock = protocol.load_protocol_lock()
        self.assertEqual(lock["expected_conditions"], 125)
        amazon = lock["dataset_content_identities"]["amazon/top5000"]
        self.assertEqual(
            amazon["ground_truth_cover_sha256"],
            "99150b37e302d931426e784d924fa1df79925739fd51075ad36a5e1f7eae0d3d",
        )
        for dataset_identity in lock["dataset_content_identities"].values():
            for field in ("graph_sha256", "ground_truth_cover_sha256"):
                digest = dataset_identity[field]
                self.assertEqual(len(digest), 64)
                self.assertEqual(f"{int(digest, 16):064x}", digest)
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
                "schema_version": 3,
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

    def test_reconcile_rejects_a_current_environment_outside_the_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text("""
[overlapping_paper]
profile = "standard"
methods = ["cpm"]
seeds = "0"
resolutions = "auto"
max_nodes = 0
timeout_per_run = 10
omega = true
omega_sample_size = 10
not_rerun_external_methods = []
[[overlapping_paper.jobs]]
name = "amazon-all"
dataset = "amazon"
cover = "all"
""")
            lock = root / "protocol.lock.json"
            lock.write_text(json.dumps({
                "schema_version": 3,
                "protocol_name": "test-current-identity",
                "run_protocol_version": benchmark.RUN_PROTOCOL_VERSION,
                "expected_conditions": 1,
                "analysis_graph_policy": "common_undirected_simple_v1",
                "protocol_config": {
                    "path": "config.toml",
                    "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                },
                "lucas_igraph": {"distribution": "lucas-igraph"},
                "tracked_files": {},
            }))
            fake_identity = {
                "schema_version": 3,
                "protocol_lock_sha256": "lock-digest",
                "protocol_name": "test-current-identity",
                "run_protocol_version": benchmark.RUN_PROTOCOL_VERSION,
                "analysis_graph_policy": "common_undirected_simple_v1",
                "tracked_files": {},
                "tracked_files_match_lock": False,
                "lucas_igraph": {
                    "distribution": "lucas-igraph",
                    "expected_version": "1",
                    "actual_version": "1",
                    "expected_igraph_version": "1",
                    "actual_igraph_version": "1",
                    "package_identity_matches_lock": False,
                },
                "external_dependencies": {},
                "external_dependencies_match_lock": False,
            }
            artifact = root / "artifacts"
            shard = artifact / "shards" / "amazon-all" / "runs"
            shard.mkdir(parents=True)
            (shard / "record.json").write_text(json.dumps({
                "experiment_identity": fake_identity,
                "dataset": "amazon",
                "cover": "all",
                "method": "cpm",
                "seed": 0,
                "status": "completed",
            }))
            orchestration = artifact / "orchestration"
            orchestration.mkdir()
            (orchestration / "plan.json").write_text(json.dumps({
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "experiment_identity": fake_identity,
                "profile": "standard",
                "max_nodes": 0,
                "methods": ["cpm"],
                "seeds": "0",
                "resolutions": "auto",
                "jobs": [{"name": "amazon-all"}],
            }))
            with patch.object(
                protocol, "current_experiment_identity", return_value=fake_identity
            ):
                report = protocol.reconcile(artifact, config, lock)
            reasons = report["rows"][0]["rejection_reasons"]
            self.assertIn("current_tracked_files_mismatch_lock", reasons)
            self.assertIn("current_lucas_igraph_mismatch_lock", reasons)
            self.assertIn("current_external_dependencies_mismatch_lock", reasons)

    def test_strict_reconcile_accepts_bound_artifacts_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifacts"
            config = root / "config.toml"
            config.write_text("""
[overlapping_paper]
profile = "standard"
methods = ["cpm"]
seeds = "0"
resolutions = "auto"
max_nodes = 0
timeout_per_run = 10
omega = true
omega_sample_size = 10
not_rerun_external_methods = []
[[overlapping_paper.jobs]]
name = "amazon-all"
dataset = "amazon"
cover = "all"
""")
            dataset = common_undirected_analysis_dataset(
                bounded_induced_dataset(
                    smoke_dataset("amazon", cover_variant="all"), 0
                )
            )
            lock = root / "protocol.lock.json"
            production_lock = protocol.load_protocol_lock()
            lock.write_text(json.dumps({
                "schema_version": 3,
                "protocol_name": "test-strict",
                "run_protocol_version": benchmark.RUN_PROTOCOL_VERSION,
                "expected_conditions": 1,
                "analysis_graph_policy": "common_undirected_simple_v1",
                "protocol_config": {
                    "path": "config.toml",
                    "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                },
                "dataset_content_identities": {
                    "amazon/all": dataset.report["content_identity"]
                },
                "lucas_igraph": {
                    **production_lock["lucas_igraph"],
                    "uv_lock_path": str(Path("uv.lock").resolve()),
                },
                "external_dependencies": production_lock[
                    "external_dependencies"
                ],
                "scientific_dependencies": production_lock[
                    "scientific_dependencies"
                ],
                "tracked_files": {},
            }))
            identity = protocol.current_experiment_identity(lock)
            shard = artifact / "shards" / "amazon-all"
            graph_artifact = benchmark._persist_analysis_graph(
                shard, dataset.graph
            )
            gt_artifact = benchmark._persist_ground_truth_cover(
                shard, dataset.cover
            )
            final_artifact = benchmark._persist_final_cover(shard, dataset.cover)
            run = shard / "runs" / "record.json"
            run.parent.mkdir(parents=True)
            memory_limit = 1_000_000
            resolution = dataset.graph.density()
            metrics = evaluate_cover(
                dataset.cover,
                dataset.cover,
                dataset.graph.vcount(),
                compute_omega=True,
                omega_sample_size=10,
                omega_seed=0,
            )
            metrics["runtime_seconds"] = 0.1
            metrics["cpm_overlapping_quality"] = quality_overlapping_cpm(
                dataset.graph, dataset.cover, resolution
            )
            metrics["cpm_overlapping_quality_status"] = "computed"
            record = {
                "schema_version": benchmark.SCHEMA_VERSION,
                "protocol_version": benchmark.RUN_PROTOCOL_VERSION,
                "experiment_identity": identity,
                "profile": "standard",
                "dataset": "amazon",
                "cover": "all",
                "method": "cpm",
                "seed": 0,
                "resolution": resolution,
                "status": "completed",
                "dataset_report": dataset.report,
                "dataset_metadata_identity": protocol.dataset_metadata_identity(
                    dataset.report
                ),
                "max_memberships": 2,
                "ground_truth_max_memberships": 2,
                "n_iterations": None,
                "local_move_only": False,
                "allow_isolation": False,
                "ensure_equilibrium": False,
                "initialization": None,
                "timeout_seconds": 10.0,
                "memory_limit_bytes": memory_limit,
                "detector_memory": {
                    "limit_bytes": memory_limit,
                    "enforcement": "process_tree_rss_polling",
                    "result_transport": "temporary_pickle_file",
                    "observed_peak_rss_bytes": 100,
                },
                "run_options": {
                    "omega": True,
                    "omega_sample_size": 10,
                    "external_baseline_policy": False,
                },
                "method_metadata": {
                    "parameters": METHODS["cpm"].parameters,
                    "dependency": method_dependency_identity("cpm"),
                },
                "runtime_seconds": 0.1,
                "metrics": metrics,
                "metrics_sha256": benchmark._metrics_digest(metrics),
                "analysis_graph_sha256": graph_artifact["content_sha256"],
                "analysis_graph_artifact": graph_artifact["artifact"],
                "analysis_graph_artifact_sha256": graph_artifact[
                    "artifact_sha256"
                ],
                "ground_truth_cover_sha256": gt_artifact["content_sha256"],
                "ground_truth_cover_artifact": gt_artifact["artifact"],
                "ground_truth_cover_artifact_sha256": gt_artifact[
                    "artifact_sha256"
                ],
                "final_cover_artifact": final_artifact["artifact"],
                "final_cover_artifact_sha256": final_artifact["artifact_sha256"],
                "final_cover_sha256": final_artifact["content_sha256"],
                "equilibrium_status": "not_applicable",
                "equilibrium_certificate": {
                    "status": "not_applicable",
                    "auditor": (
                        "hedonic.experiments.overlapping.robustness.audit_cover"
                    ),
                    "auditor_source_sha256": identity["tracked_files"].get(
                        "src/hedonic/experiments/overlapping/robustness.py"
                    ),
                },
            }
            run.write_text(json.dumps(record))
            orchestration = artifact / "orchestration"
            orchestration.mkdir()
            (orchestration / "plan.json").write_text(json.dumps({
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "experiment_identity": identity,
                "profile": "standard",
                "max_nodes": 0,
                "methods": ["cpm"],
                "seeds": "0",
                "resolutions": "auto",
                "jobs": [{
                    "name": "amazon-all",
                    "max_memberships": 2,
                    "dataset_content_identity": dataset.report["content_identity"],
                    "memory": {
                        "detector_memory_limit_bytes": memory_limit,
                        "ground_truth_community_count": dataset.report[
                            "content_identity"
                        ]["ground_truth_community_count"],
                        "ground_truth_max_memberships_per_node": dataset.report[
                            "content_identity"
                        ]["ground_truth_max_memberships_per_node"],
                    },
                }],
            }))
            report = protocol.reconcile(artifact, config, lock)
            self.assertEqual(report["admissible_records"], 1, report["rows"][0])
            self.assertTrue(report["ready_for_publication"])
            original_record_bytes = run.read_bytes()
            original_condition_specs = protocol._condition_specs_from_raw

            def mutate_after_scan(raw):
                run.write_bytes(original_record_bytes + b"\n")
                return original_condition_specs(raw)

            with patch.object(
                protocol, "_condition_specs_from_raw", side_effect=mutate_after_scan
            ):
                single_read_report = protocol.reconcile(artifact, config, lock)
            self.assertEqual(
                single_read_report["rows"][0]["record_sha256"],
                hashlib.sha256(original_record_bytes).hexdigest(),
            )
            self.assertNotEqual(
                single_read_report["rows"][0]["record_sha256"],
                hashlib.sha256(run.read_bytes()).hexdigest(),
            )
            run.write_bytes(original_record_bytes)
            tampered_record = json.loads(original_record_bytes)
            tampered_record["metrics"]["predicted_singleton_count"] = 0
            tampered_record["metrics"]["singleton_count"] = 0
            tampered_record["metrics"]["predicted_singleton_fraction"] = 1.0
            tampered_record["metrics"]["singleton_fraction"] = 1.0
            tampered_record["metrics_sha256"] = benchmark._metrics_digest(
                tampered_record["metrics"]
            )
            run.write_text(json.dumps(tampered_record))
            metric_tamper_report = protocol.reconcile(artifact, config, lock)
            self.assertEqual(metric_tamper_report["admissible_records"], 0)
            self.assertTrue(
                any(
                    reason.endswith("_recomputed_mismatch")
                    for reason in metric_tamper_report["rows"][0][
                        "rejection_reasons"
                    ]
                )
            )
            run.write_bytes(original_record_bytes)
            (shard / final_artifact["artifact"]).write_bytes(b"tampered")
            report = protocol.reconcile(artifact, config, lock)
            self.assertEqual(report["admissible_records"], 0)
            self.assertIn(
                "final_cover_artifact_sha256_mismatch",
                report["rows"][0]["rejection_reasons"],
            )

    def test_metric_diagnostics_reject_impossible_values(self):
        dataset = common_undirected_analysis_dataset(
            bounded_induced_dataset(
                smoke_dataset("amazon", cover_variant="all"), 0
            )
        )
        metrics = evaluate_cover(
            dataset.cover,
            dataset.cover,
            dataset.graph.vcount(),
            compute_omega=True,
            omega_sample_size=10,
            omega_seed=0,
        )
        metrics["runtime_seconds"] = 0.1
        metrics["cpm_overlapping_quality"] = quality_overlapping_cpm(
            dataset.graph, dataset.cover, dataset.graph.density()
        )
        metrics["cpm_overlapping_quality_status"] = "computed"
        record = {
            "method": "cpm",
            "seed": 0,
            "runtime_seconds": 0.1,
            "max_memberships": 2,
            "dataset_report": dataset.report,
            "metrics": metrics,
            "metrics_sha256": benchmark._metrics_digest(metrics),
        }
        self.assertEqual(
            protocol._metrics_rejection_reasons(
                record, omega=True, omega_sample_size=10
            ),
            [],
        )
        metrics.update({
            "predicted_singleton_count": -4,
            "singleton_count": -4,
            "predicted_average_community_size": -2.0,
            "average_community_size": -2.0,
            "predicted_max_memberships_per_vertex": -1,
            "overlap_pair_events": -1,
            "community_count_ratio": -8.0,
            "overlap_pair_events_truncated": "false",
        })
        record["metrics_sha256"] = benchmark._metrics_digest(metrics)
        reasons = protocol._metrics_rejection_reasons(
            record, omega=True, omega_sample_size=10
        )
        self.assertTrue(
            any("predicted_singleton_count" in reason for reason in reasons)
        )
        self.assertTrue(
            any("predicted_average_community_size" in reason for reason in reasons)
        )
        self.assertTrue(
            any("predicted_max_memberships_per_vertex" in reason for reason in reasons)
        )
        self.assertTrue(any("overlap_pair_events" in reason for reason in reasons))
        self.assertTrue(any("community_count_ratio" in reason for reason in reasons))
        self.assertIn("metrics.overlap_pair_events_truncated_not_boolean", reasons)


if __name__ == "__main__":
    unittest.main()
