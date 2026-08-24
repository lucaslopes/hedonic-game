"""Small, archive-free tests for the shared SNAP benchmark surface."""

from __future__ import annotations

import gzip
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import igraph as ig

from hedonic.experiments import CLI
from hedonic.experiments.overlapping import benchmark, reproduce_paper, snap
from hedonic.experiments.overlapping.methods import (
    METHODS,
    effective_resolution,
    method_availability,
    normalize_cover,
    run_method,
)
from hedonic.experiments.overlapping.metrics import structural_overlap_metrics
from hedonic.experiments.overlapping.robustness import audit_cover
from hedonic.experiments.overlapping.snap import (
    ANALYSIS_GRAPH_POLICY,
    bounded_induced_dataset,
    common_undirected_analysis_dataset,
    load_snap_dataset,
    smoke_dataset,
)


def _write_gzip(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(text)


class TestSnapLoader(unittest.TestCase):
    def test_streamed_raw_remaps_original_ids_and_reports_drops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "networks"
            amazon = root / "Amazon"
            _write_gzip(
                amazon / "com-amazon.ungraph.txt.gz",
                "# source target\n10 20\n20 30\n",
            )
            _write_gzip(
                amazon / "top5000.cmty.txt.gz",
                "10 20 999\n20 30\n999\n",
            )
            cache = Path(directory) / "cache"
            dataset = load_snap_dataset(
                "amazon", cover_variant="top5000", data_root=root, cache_dir=cache
            )
            self.assertEqual(dataset.cover, [[0, 1], [1, 2]])
            self.assertEqual(dataset.report["id_mapping_strategy"], "raw_sorted_original_id_to_contiguous_index")
            self.assertEqual(dataset.report["validation"]["missing_member_count"], 2)
            self.assertEqual(dataset.report["validation"]["dropped_communities_lt_2_members"], 1)
            self.assertTrue((cache / dataset.report["normalized_cache_path"]).is_file())

            cached = load_snap_dataset(
                "amazon", cover_variant="top5000", data_root=root, cache_dir=cache
            )
            self.assertEqual(cached.cover, dataset.cover)
            self.assertEqual(cached.report["source_kind"], "validated_normalized_cache")

            cache_path = cache / dataset.report["normalized_cache_path"]
            with cache_path.open("rb") as stream:
                payload = pickle.load(stream)
            payload["cover"] = [[0, 2]]
            with cache_path.open("wb") as stream:
                pickle.dump(payload, stream)
            repaired = load_snap_dataset(
                "amazon", cover_variant="top5000", data_root=root, cache_dir=cache
            )
            self.assertEqual(repaired.cover, dataset.cover)
            self.assertEqual(repaired.report["source_kind"], "streamed_raw_gzip")

            with cache_path.open("rb") as stream:
                payload = pickle.load(stream)
            payload["report"]["overlap_statistics"][
                "max_memberships_per_node"
            ] = 999
            with cache_path.open("wb") as stream:
                pickle.dump(payload, stream)
            stats_repaired = load_snap_dataset(
                "amazon", cover_variant="top5000", data_root=root, cache_dir=cache
            )
            self.assertEqual(
                stats_repaired.report["overlap_statistics"][
                    "max_memberships_per_node"
                ],
                2,
            )
            self.assertEqual(
                stats_repaired.report["source_kind"], "streamed_raw_gzip"
            )
            reviewed = common_undirected_analysis_dataset(
                bounded_induced_dataset(dataset, 0)
            ).report["content_identity"]
            with patch.object(
                reproduce_paper,
                "load_snap_dataset",
                return_value=stats_repaired,
            ):
                accepted = reproduce_paper._job_memory_profile(
                    {
                        "name": "amazon-top5000",
                        "dataset": "amazon",
                        "cover": "top5000",
                    },
                    data_root=root,
                    profile="standard",
                    max_nodes=0,
                    methods=["hedonic_multiphase"],
                    memory_budget_bytes=16 * 1024**3,
                    safety_factor=1.5,
                    configured_membership_cap=None,
                    output_dir=Path(directory) / "accepted-output",
                    locked_dataset_identities={"amazon/top5000": reviewed},
                )
            self.assertEqual(accepted["dataset_content_identity"], reviewed)

            # A pickle cache can self-consistently rewrite its graph, cover,
            # self-digest, and report statistics.  The locked paper coordinator
            # must still reject that content against its independent reviewed
            # graph/GT identity before scheduling any detector.
            with cache_path.open("rb") as stream:
                payload = pickle.load(stream)
            forged_graph = ig.Graph(n=3, edges=[(0, 2)], directed=False)
            forged_cover = [[0, 2]]
            forged_stats = snap.cover_statistics(forged_cover)
            payload["graph"] = forged_graph
            payload["cover"] = forged_cover
            payload["cache_content_identity"] = snap._cache_content_identity(
                forged_graph, forged_cover
            )
            payload["report"].update({
                "n": forged_graph.vcount(),
                "m": forged_graph.ecount(),
                "directed": forged_graph.is_directed(),
                "number_of_communities": forged_stats["n_communities"],
                "number_of_covered_nodes": forged_stats["n_covered_nodes"],
                "community_size_statistics": forged_stats["community_size"],
                "overlap_statistics": forged_stats["overlap"],
            })
            with cache_path.open("wb") as stream:
                pickle.dump(payload, stream)
            poisoned = load_snap_dataset(
                "amazon", cover_variant="top5000", data_root=root, cache_dir=cache
            )
            self.assertEqual(poisoned.report["source_kind"], "validated_normalized_cache")
            with patch.object(
                reproduce_paper, "load_snap_dataset", return_value=poisoned
            ):
                with self.assertRaisesRegex(ValueError, "reviewed protocol lock"):
                    reproduce_paper._job_memory_profile(
                        {
                            "name": "amazon-top5000",
                            "dataset": "amazon",
                            "cover": "top5000",
                        },
                        data_root=root,
                        profile="standard",
                        max_nodes=0,
                        methods=["hedonic_multiphase"],
                        memory_budget_bytes=16 * 1024**3,
                        safety_factor=1.5,
                        configured_membership_cap=None,
                        output_dir=Path(directory) / "output",
                        locked_dataset_identities={"amazon/top5000": reviewed},
                    )

    def test_normalized_cache_is_bound_to_the_requested_data_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            cache = base / "cache"
            roots = [base / "first", base / "second"]
            for root, edge in zip(roots, ("10 20\n", "10 20\n20 30\n")):
                amazon = root / "Amazon"
                _write_gzip(amazon / "com-amazon.ungraph.txt.gz", edge)
                _write_gzip(amazon / "top5000.cmty.txt.gz", "10 20\n")
            first = load_snap_dataset(
                "amazon", cover_variant="top5000", data_root=roots[0], cache_dir=cache
            )
            second = load_snap_dataset(
                "amazon", cover_variant="top5000", data_root=roots[1], cache_dir=cache
            )
            self.assertNotEqual(first.graph.vcount(), second.graph.vcount())
            self.assertNotEqual(
                first.report["normalized_cache_path"],
                second.report["normalized_cache_path"],
            )
            _write_gzip(
                roots[0] / "Amazon" / "com-amazon.ungraph.txt.gz",
                "10 20\n20 40\n",
            )
            changed = load_snap_dataset(
                "amazon", cover_variant="top5000", data_root=roots[0], cache_dir=cache
            )
            self.assertNotEqual(
                changed.report["normalized_cache_path"],
                first.report["normalized_cache_path"],
            )
            self.assertEqual(changed.graph.vcount(), 3)

    def test_dblp_label_attribute_remaps_cached_cover_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "networks"
            pkl = root / "DBLP" / "pkl"
            pkl.mkdir(parents=True)
            graph = ig.Graph(n=3, edges=[(0, 1), (1, 2)], directed=False)
            graph.vs["label"] = [100, 300, 700]
            with (pkl / "com-dblp.ungraph.pkl").open("wb") as f:
                pickle.dump(graph, f)
            with (pkl / "top5000.cmty.pkl").open("wb") as f:
                pickle.dump([[100, 300], [700, 999]], f)

            dataset = load_snap_dataset(
                "dblp",
                cover_variant="top5000",
                data_root=root,
                cache_dir=Path(directory) / "cache",
            )
            self.assertEqual(dataset.cover, [[0, 1]])
            self.assertEqual(dataset.report["id_mapping_strategy"], "cached_vertex_label_attribute")
            self.assertEqual(dataset.report["validation"]["missing_member_count"], 1)

    def test_wikipedia_category_cover_preserves_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "networks"
            wikipedia = root / "Wikipedia"
            _write_gzip(wikipedia / "wiki-topcats.txt.gz", "1 2\n2 3\n")
            _write_gzip(
                wikipedia / "wiki-topcats-categories.txt.gz",
                "Category:Graphs; 1 2 99\nCategory:Other; 2 3\n",
            )
            dataset = load_snap_dataset(
                "wikipedia", cover_variant="all", data_root=root, cache_dir=Path(directory) / "cache"
            )
            self.assertTrue(dataset.graph.is_directed())
            self.assertEqual(dataset.cover, [[0, 1], [1, 2]])

            analysis = common_undirected_analysis_dataset(dataset)
            self.assertFalse(analysis.graph.is_directed())
            self.assertEqual(analysis.graph.get_edgelist(), [(0, 1), (1, 2)])
            self.assertTrue(analysis.report["source_graph"]["directed"])
            self.assertEqual(
                analysis.report["analysis_graph"]["policy"], ANALYSIS_GRAPH_POLICY
            )
            self.assertTrue(
                analysis.report["analysis_graph"]["applied_before_all_methods_and_metrics"]
            )

    def test_bounded_dataset_reindexes_cover(self):
        dataset = smoke_dataset("amazon", cover_variant="all")
        bounded = bounded_induced_dataset(dataset, 4)
        self.assertEqual(bounded.graph.vcount(), 4)
        self.assertTrue(all(0 <= v < 4 for community in bounded.cover for v in community))
        self.assertIn("bounded_subgraph", bounded.report)


class TestBenchmarkHelpers(unittest.TestCase):
    def test_cached_metrics_reject_extra_identity_shadow_keys(self):
        dataset = common_undirected_analysis_dataset(
            bounded_induced_dataset(
                smoke_dataset("amazon", cover_variant="all"), 0
            )
        )
        resolution = dataset.graph.density()
        metrics = benchmark.evaluate_cover(
            dataset.cover,
            dataset.cover,
            dataset.graph.vcount(),
            compute_omega=False,
            omega_sample_size=100,
            omega_seed=0,
        )
        metrics["cpm_overlapping_quality"] = (
            benchmark.quality_overlapping_cpm(
                dataset.graph, dataset.cover, resolution
            )
        )
        metrics["cpm_overlapping_quality_status"] = "computed"
        metrics["runtime_seconds"] = 0.1
        record = {
            "runtime_seconds": 0.1,
            "metrics": metrics,
            "metrics_sha256": benchmark._metrics_digest(metrics),
        }
        expected = {
            "seed": 0,
            "resolution": resolution,
            "dataset_content_identity": dataset.report["content_identity"],
            "run_options": {"omega": False, "omega_sample_size": 100},
            "_analysis_graph_object": dataset.graph,
            "_ground_truth_cover": dataset.cover,
        }
        self.assertTrue(
            benchmark._cached_metrics_match(record, expected, dataset.cover)
        )
        tampered_metrics = {**metrics, "status": "failed", "dataset": "evil"}
        tampered = {
            **record,
            "metrics": tampered_metrics,
            "metrics_sha256": benchmark._metrics_digest(tampered_metrics),
        }
        self.assertFalse(
            benchmark._cached_metrics_match(tampered, expected, dataset.cover)
        )
        flattened = benchmark._flatten_record({
            "dataset": "amazon",
            "status": "completed",
            "metrics": tampered_metrics,
        })
        self.assertEqual(flattened["dataset"], "amazon")
        self.assertEqual(flattened["status"], "completed")

    def test_exact_labeled_equilibrium_is_not_transferred_to_scoring_projection(self):
        graph = ig.Graph(n=4, edges=[(0, 1), (0, 2), (0, 3)])
        exact_memberships = [[2], [0, 1], [2], [2]]
        exact_audit = audit_cover(
            graph,
            exact_memberships,
            max_memberships=3,
            allow_isolation=True,
            gamma=0.5,
            compute_intervals=False,
        )
        projection = benchmark._cover_projection_from_membership_rows(
            exact_memberships, graph.vcount()
        )
        self.assertIsNotNone(projection)
        scoring_cover, metadata = projection
        projected_memberships = [[] for _ in range(graph.vcount())]
        for label, community in enumerate(scoring_cover):
            for vertex in community:
                projected_memberships[vertex].append(label)
        projected_audit = audit_cover(
            graph,
            projected_memberships,
            max_memberships=3,
            allow_isolation=True,
            gamma=0.5,
            compute_intervals=False,
        )
        self.assertTrue(exact_audit["is_local_equilibrium_at_resolution"])
        self.assertFalse(projected_audit["is_local_equilibrium_at_resolution"])
        self.assertEqual(metadata["duplicate_community_bodies_removed"], 1)
        self.assertTrue(metadata["projection_changed"])
        self.assertFalse(
            metadata["canonical_scoring_projection_certified_as_equilibrium"]
        )
        scoring_sha256 = benchmark.cover_sha256(scoring_cover)
        certificate = benchmark._equilibrium_certificate(
            "hedonic_multiphase",
            graph,
            exact_memberships,
            final_membership_sha256=benchmark._raw_membership_digest(
                exact_memberships
            ),
            canonical_scoring_cover_sha256=scoring_sha256,
            max_memberships=3,
            resolution=0.5,
            allow_isolation=True,
        )
        self.assertEqual(certificate["status"], "verified")
        self.assertEqual(
            certificate["certificate_target"],
            "exact_labeled_final_memberships",
        )
        self.assertEqual(
            certificate["canonical_scoring_cover_sha256"], scoring_sha256
        )
        self.assertFalse(
            certificate[
                "canonical_scoring_projection_certified_as_equilibrium"
            ]
        )

    def test_large_detector_cover_does_not_deadlock_result_transport(self):
        if "fork" not in benchmark.mp.get_all_start_methods():
            self.skipTest("large-result transport regression requires fork inheritance")
        dataset = smoke_dataset("amazon", cover_variant="all")
        large_cover = [[vertex] for vertex in range(100_000)]
        method_meta = {"runtime_seconds": 0.01, "parameters": {}}
        with patch.object(
            benchmark, "run_method", return_value=(large_cover, method_meta)
        ):
            outcome = benchmark._run_with_timeout(
                "cpm",
                dataset.graph,
                max_memberships=2,
                resolution=dataset.graph.density(),
                seed=0,
                timeout_seconds=5,
            )
        self.assertEqual(outcome["status"], "ok")
        self.assertEqual(len(outcome["cover"]), len(large_cover))
        self.assertEqual(
            outcome["memory"]["result_transport"], "temporary_pickle_file"
        )

    def test_timeout_worker_and_terminal_json_records_have_one_resource_source(self):
        """Exercise the isolated smoke detector and every terminal JSON shape.

        The timeout packet deliberately includes the legacy duplicate field to
        ensure the coordinator, which owns configured timeout_seconds, cannot
        regress into passing it twice to _record.
        """
        dataset = smoke_dataset("amazon", cover_variant="all")
        outcome = benchmark._run_with_timeout(
            "hedonic_multiphase",
            dataset.graph,
            max_memberships=2,
            resolution=dataset.graph.density(),
            seed=0,
            timeout_seconds=10,
        )
        self.assertEqual(outcome["status"], "ok")
        self.assertIn("memory", outcome)
        self.assertGreaterEqual(outcome["memory"]["monitor_samples"], 1)
        self.assertIsInstance(outcome["memory"]["rss_samples"], list)

        packets = {
            "success": {
                "status": "ok", "cover": outcome["cover"],
                "method_meta": outcome["method_meta"],
                "pre_cleanup_memberships": outcome[
                    "pre_cleanup_memberships"
                ],
                "final_memberships": outcome["final_memberships"],
                "memory": {"observed_peak_rss_bytes": 10}, "runtime_seconds": 0.01,
            },
            "timeout": {
                "status": "timeout", "runtime_seconds": 0.02,
                "timeout_seconds": 999.0, "termination_reason": "timeout",
                "memory": {"observed_peak_rss_bytes": 11},
            },
            "error": {
                "status": "error", "runtime_seconds": 0.03,
                "error": "RuntimeError: smoke failure", "memory": {"observed_peak_rss_bytes": 12},
            },
            "memory_limit": {
                "status": "memory_limit", "runtime_seconds": 0.04,
                "termination_reason": "memory_limit", "memory": {"observed_peak_rss_bytes": 13},
            },
        }
        expected_status = {
            "success": "completed", "timeout": "timeout", "error": "failed",
            "memory_limit": "memory_limit",
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, packet in packets.items():
                output = Path(directory) / name
                with patch.object(benchmark, "_run_with_timeout", return_value=dict(packet)):
                    self.assertEqual(
                        CLI.main([
                            "overlapping-benchmark", "--profile", "smoke", "--datasets", "amazon",
                            "--methods", "hedonic_multiphase", "--output_dir", str(output),
                            "--timeout_per_run", "17", "--memory_limit_gb", "1", "--no-plots",
                        ]),
                        0,
                    )
                record = json.loads(next((output / "runs").rglob("*.json")).read_text())
                self.assertEqual(record["status"], expected_status[name])
                self.assertEqual(record["dataset"], "amazon")
                self.assertEqual(record["method"], "hedonic_multiphase")
                self.assertEqual(record["seed"], 0)
                self.assertEqual(record["timeout_seconds"], 17.0)
                self.assertGreater(record["memory_limit_bytes"], 0)
                self.assertIn("observed_peak_rss_bytes", record["detector_memory"])
                if name != "success":
                    self.assertIn("runtime_seconds", record)
            timeout_record = json.loads(
                next((Path(directory) / "timeout" / "runs").rglob("*.json")).read_text()
            )
            self.assertEqual(timeout_record["timeout_seconds"], 17.0)

    def test_resource_failed_baseline_is_rerun_on_resume(self):
        packet = {
            "status": "memory_limit", "runtime_seconds": 0.02,
            "memory": {"observed_peak_rss_bytes": 100, "rss_samples": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"
            argv = [
                "overlapping-benchmark", "--profile", "smoke", "--datasets", "amazon",
                "--methods", "cpm", "--output_dir", str(output), "--no-plots",
            ]
            with patch.object(benchmark, "_run_with_timeout", return_value=dict(packet)):
                self.assertEqual(CLI.main(argv), 0)
            record_path = next((output / "runs").rglob("*.json"))
            record = json.loads(record_path.read_text())
            self.assertEqual(record["status"], "skipped_not_scalable")
            self.assertEqual(record["resource_status"], "memory_limit")
            with patch.object(
                benchmark, "_run_with_timeout", return_value=dict(packet)
            ) as rerun:
                self.assertEqual(CLI.main(argv + ["--resume"]), 0)
            rerun.assert_called_once()
            record = json.loads(record_path.read_text())
            self.assertEqual(record["execution"], "rerun")

    def test_resource_failed_hedonic_run_is_rerun_on_resume(self):
        packet = {
            "status": "timeout", "runtime_seconds": 0.02,
            "memory": {"observed_peak_rss_bytes": 100, "rss_samples": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"
            argv = [
                "overlapping-benchmark", "--profile", "smoke", "--datasets", "amazon",
                "--methods", "hedonic_multiphase", "--output_dir", str(output), "--no-plots",
            ]
            with patch.object(benchmark, "_run_with_timeout", return_value=dict(packet)):
                self.assertEqual(CLI.main(argv), 0)
            with patch.object(
                benchmark, "_run_with_timeout", return_value=dict(packet)
            ) as rerun:
                self.assertEqual(CLI.main(argv + ["--resume"]), 0)
            rerun.assert_called_once()
            record = json.loads(next((output / "runs").rglob("*.json")).read_text())
            self.assertEqual(record["status"], "timeout")
            self.assertEqual(record["execution"], "rerun")

    def test_external_baseline_policy_writes_explicit_nonresult_without_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"
            argv = [
                "overlapping-benchmark",
                "--profile", "smoke",
                "--datasets", "amazon",
                "--methods", "cpm,demon",
                "--skip-methods", "cpm,demon",
                "--output_dir", str(output),
                "--no-plots",
            ]
            with patch.object(
                benchmark,
                "_run_with_timeout",
                side_effect=AssertionError("external baseline detector must not run"),
            ):
                self.assertEqual(CLI.main(argv), 0)
            records = [
                json.loads(path.read_text())
                for path in (output / "runs").rglob("*.json")
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(
                {record["status"] for record in records},
                {benchmark.SKIPPED_EXTERNAL_STATUS},
            )
            self.assertTrue(
                all(record["execution"] == "external_baseline_policy" for record in records)
            )
            self.assertTrue(
                all(record["failure_kind"] == "external_baseline_not_rerun" for record in records)
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                manifest["run_status_counts"][benchmark.SKIPPED_EXTERNAL_STATUS], 2
            )

    def test_cover_normalization_and_method_registry(self):
        cover, validation = normalize_cover(
            [[1, 0, 0, 9, "bad"], [0, 1], []], 3
        )
        self.assertEqual(cover, [[0, 1]])
        self.assertEqual(validation["invalid_members_dropped"], 2)
        self.assertEqual(validation["duplicate_members_removed"], 1)
        self.assertEqual(validation["duplicate_communities_removed"], 1)
        availability = method_availability()
        self.assertTrue(
            {
                "hedonic_local",
                "hedonic_multiphase",
                "hedonic_multiphase_x10",
                "hedonic_multiphase_x100",
                "cpm",
                "demon",
            }.issubset(METHODS)
        )
        self.assertIn("available", availability["demon"])
        self.assertIn("install_requirement", availability["cpm"])

    def test_multiphase_density_variants_fix_resolution_and_enable_isolation(self):
        dataset = smoke_dataset("amazon", cover_variant="all")
        density = dataset.graph.density()
        expected = {
            "hedonic_multiphase": min(density, 1.0),
            "hedonic_multiphase_x10": min(density * 10, 1.0),
            "hedonic_multiphase_x100": min(density * 100, 1.0),
        }
        for name, resolution in expected.items():
            adapter = METHODS[name]
            self.assertEqual(
                effective_resolution(adapter, dataset.graph, requested_resolution=0.001),
                resolution,
            )
            self.assertTrue(adapter.parameters["allow_isolation"])
            self.assertTrue(adapter.parameters["ensure_equilibrium"])

    def test_structural_metrics_are_overlap_aware(self):
        metrics = structural_overlap_metrics(
            [[0, 1, 2], [2, 3]], [[0, 1, 2], [2, 3]]
        )
        self.assertAlmostEqual(metrics["inclusion_rate"], 1.0)
        self.assertAlmostEqual(metrics["coverage_rate"], 1.0)
        self.assertGreater(metrics["overlapping_rate"], 0.0)
        self.assertGreater(metrics["distribution_rate"], 0.0)

    def test_hedonic_multiphase_applies_the_recorded_igraph_seed(self):
        dataset = smoke_dataset("amazon", cover_variant="all")
        first, first_meta = run_method(
            METHODS["hedonic_multiphase"],
            dataset.graph,
            max_memberships=3,
            resolution=dataset.graph.density(),
            seed=17,
        )
        second, second_meta = run_method(
            METHODS["hedonic_multiphase"],
            dataset.graph,
            max_memberships=3,
            resolution=dataset.graph.density(),
            seed=17,
        )
        self.assertEqual(first, second)
        self.assertEqual(first_meta["parameters"]["igraph_rng"], "random.Random(seed)")
        self.assertEqual(second_meta["seed"], 17)

    def test_cli_list_help_smoke_resume_and_manifest(self):
        self.assertIn("overlapping-benchmark", CLI.COMMANDS)
        self.assertEqual(CLI.main(["overlapping-benchmark", "--help"]), 0)
        self.assertEqual(CLI.main(["overlapping-benchmark", "--list-networks"]), 0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"
            argv = [
                "overlapping-benchmark",
                "--profile",
                "smoke",
                "--datasets",
                "amazon,dblp",
                "--methods",
                "hedonic_multiphase,hedonic_multiphase_x10,hedonic_multiphase_x100",
                "--output_dir",
                str(output),
                "--no-plots",
            ]
            self.assertEqual(CLI.main(argv), 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 4)
            self.assertEqual(manifest["run_status_counts"].get("completed"), 6)
            self.assertTrue(manifest["experiment_identity"]["tracked_files_match_lock"])
            self.assertTrue(
                manifest["experiment_identity"]["lucas_igraph"]["package_identity_matches_lock"]
            )
            self.assertTrue((output / "results.jsonl").is_file())
            self.assertTrue((output / "results.csv.gz").is_file())
            self.assertTrue((output / "summary.csv").is_file())
            hedonic_records = [
                json.loads(path.read_text())
                for path in (output / "runs").rglob("*.json")
            ]
            self.assertTrue(all(record["allow_isolation"] for record in hedonic_records))
            self.assertTrue(all(record["ensure_equilibrium"] for record in hedonic_records))
            self.assertTrue(
                all(record["equilibrium_status"] == "verified_independent_audit" for record in hedonic_records)
            )
            self.assertTrue(
                all(
                    record["equilibrium_certificate"]["status"] == "verified"
                    and record["equilibrium_certificate"][
                        "is_local_equilibrium_at_resolution"
                    ]
                    and record["equilibrium_certificate"][
                        "analysis_graph_sha256"
                    ]
                    == record["dataset_report"]["content_identity"][
                        "graph_sha256"
                    ]
                    and record["equilibrium_certificate"][
                        "max_positive_regret"
                    ]
                    <= record["equilibrium_certificate"][
                        "max_stability_tolerance"
                    ]
                    and record["equilibrium_certificate"][
                        "certificate_target"
                    ]
                    == "exact_labeled_final_memberships"
                    and record["equilibrium_certificate"][
                        "canonical_scoring_cover_sha256"
                    ]
                    == record["final_cover_sha256"]
                    and record["equilibrium_certificate"][
                        "canonical_scoring_projection_certified_as_equilibrium"
                    ]
                    is False
                    for record in hedonic_records
                )
            )
            self.assertTrue(
                all(
                    record["metrics_sha256"]
                    == benchmark._metrics_digest(record["metrics"])
                    for record in hedonic_records
                )
            )
            self.assertTrue(all(record.get("raw_membership_hash") for record in hedonic_records))
            self.assertTrue(
                all(
                    record.get("raw_membership_artifact")
                    and (output / record["raw_membership_artifact"]).is_file()
                    for record in hedonic_records
                )
            )
            self.assertTrue(
                all(
                    record.get("analysis_graph_artifact")
                    and (output / record["analysis_graph_artifact"]).is_file()
                    and record.get("ground_truth_cover_artifact")
                    and (output / record["ground_truth_cover_artifact"]).is_file()
                    and record["analysis_graph_sha256"]
                    == record["dataset_report"]["content_identity"]["graph_sha256"]
                    and record["ground_truth_cover_sha256"]
                    == record["dataset_report"]["content_identity"][
                        "ground_truth_cover_sha256"
                    ]
                    for record in hedonic_records
                )
            )
            self.assertTrue(
                all(
                    record.get("final_cover_artifact")
                    and (output / record["final_cover_artifact"]).is_file()
                    and record.get("final_cover_sha256")
                    for record in hedonic_records
                )
            )
            self.assertTrue(
                all(
                    record.get("pre_cleanup_membership_artifact")
                    and (output / record["pre_cleanup_membership_artifact"]).is_file()
                    and record.get("final_membership_artifact")
                    and (output / record["final_membership_artifact"]).is_file()
                    and record["equilibrium_certificate"][
                        "final_membership_sha256"
                    ]
                    == record["final_membership_sha256"]
                    and record["final_membership_projection"][
                        "exact_final_membership_sha256"
                    ]
                    == record["final_membership_sha256"]
                    and record["final_membership_projection"][
                        "canonical_scoring_cover_sha256"
                    ]
                    == record["final_cover_sha256"]
                    for record in hedonic_records
                )
            )
            self.assertTrue(all(record["experiment_identity"] for record in hedonic_records))
            self.assertTrue(all(record["dataset_metadata_identity"] for record in hedonic_records))
            self.assertTrue(
                all(
                    record["initialization"]["kind"]
                    == "seeded_random_disjoint"
                    for record in hedonic_records
                )
            )
            self.assertTrue(
                all(
                    record["initialization"]["requested_community_count"] == 3
                    for record in hedonic_records
                )
            )
            self.assertTrue(
                all(
                    record["initialization"]["realized_community_count"]
                    <= record["initialization"]["requested_community_count"]
                    and record["initialization"]["membership_sha256"]
                    for record in hedonic_records
                )
            )
            self.assertTrue(
                all(
                    record["method_metadata"]["initial_membership_supplied"]
                    for record in hedonic_records
                )
            )
            self.assertTrue(all(record["max_memberships"] == 2 for record in hedonic_records))
            self.assertTrue(
                all(record["ground_truth_max_memberships"] == 2 for record in hedonic_records)
            )
            self.assertTrue(
                all(
                    record["resolution"] == min(
                        smoke_dataset(record["dataset"], cover_variant="all").graph.density()
                        * {"hedonic_multiphase": 1, "hedonic_multiphase_x10": 10,
                           "hedonic_multiphase_x100": 100}[record["method"]],
                        1.0,
                    )
                    for record in hedonic_records
                )
            )
            run = next((output / "runs").rglob("*.json"))
            before = run.read_text()
            self.assertEqual(CLI.main(argv + ["--resume"]), 0)
            self.assertEqual(run.read_text(), before)
            # A changed detector timeout is resource-incompatible and must be
            # rerun rather than treated as a valid cached result.
            self.assertEqual(CLI.main(argv + ["--resume", "--timeout_per_run", "31"]), 0)
            changed = json.loads(run.read_text())
            self.assertEqual(changed["execution"], "rerun")
            self.assertEqual(changed["timeout_seconds"], 31.0)
            # A detector setting is part of the experimental condition too.
            changed["allow_isolation"] = False
            run.write_text(json.dumps(changed))
            self.assertEqual(CLI.main(argv + ["--resume"]), 0)
            changed = json.loads(run.read_text())
            self.assertEqual(changed["execution"], "rerun")
            self.assertTrue(changed["allow_isolation"])
            changed["equilibrium_certificate"]["status"] = "not_verified"
            run.write_text(json.dumps(changed))
            self.assertEqual(CLI.main(argv + ["--resume"]), 0)
            changed = json.loads(run.read_text())
            self.assertEqual(changed["execution"], "rerun")
            self.assertEqual(
                changed["equilibrium_certificate"]["status"], "verified"
            )
            changed["equilibrium_certificate"]["max_positive_regret"] = 0.0
            changed["equilibrium_certificate"]["max_stability_tolerance"] = 1.0
            run.write_text(json.dumps(changed))
            self.assertEqual(CLI.main(argv + ["--resume"]), 0)
            changed = json.loads(run.read_text())
            self.assertEqual(changed["execution"], "rerun")
            self.assertLess(
                changed["equilibrium_certificate"]["max_stability_tolerance"],
                1.0,
            )
            changed["metrics"]["matching_f1"] = 999.0
            run.write_text(json.dumps(changed))
            self.assertEqual(CLI.main(argv + ["--resume"]), 0)
            changed = json.loads(run.read_text())
            self.assertEqual(changed["execution"], "rerun")
            self.assertLessEqual(changed["metrics"]["matching_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
