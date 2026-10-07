"""Unit tests for overlapping Leiden (max_memberships) and experiments package."""

from __future__ import annotations

import gzip
import json
import inspect
import os
import random
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import igraph as ig
import numpy as np

from hedonic import Game
from tests._version_contract import (
    PUBLIC_CHECKOUT_REASON,
    lock_tracked_files_present,
    assert_live_stack_reported_against_frozen_lock,
    live_lucas_igraph_lock_section,
    locked_lucas_igraph_version,
)

from hedonic.experiments import CLI
from hedonic.experiments.config import (
    DEFAULT_DBLP_DIR,
    DEFAULT_NETWORKS_DIR,
    DEFAULT_SYNTHETIC_DIR,
    DEFAULT_OUTPUT_DIR,
    load_config_file,
    read_toml,
    reload_paths,
    resolve_experiment_paths,
)
from hedonic.experiments.disjoint import data_loader, sbm_sweep
from hedonic.experiments.overlapping import small_graphs
from hedonic.experiments.overlapping import complexity_scale
from hedonic.experiments.overlapping import resolution_f1
from hedonic.experiments.overlapping import execution as gt_execution
from hedonic.experiments.overlapping.ground_truth_data import (
    covered_induced_dataset,
    prepare_dataset,
    singleton_completed_dataset,
)
from hedonic.experiments.overlapping.ground_truth_robustness import (
    _expected_condition_axis_keys,
    _json_hash,
    _load_cover_artifact,
    _load_membership_artifact,
    _normalize_raw_memberships,
    _persist_cover,
    _persist_memberships,
    _publication_evidence_index,
    _raw_cover_hash,
    main as ground_truth_robustness_main,
)
from hedonic.experiments.overlapping.dblp_full import (
    resolve_max_memberships as resolve_mm_full,
)
from hedonic.experiments.overlapping.dblp_subgraph import (
    resolve_max_memberships as resolve_mm_sub,
)
from hedonic.experiments.overlapping.metrics import (
    cover_diagnostics,
    cover_quality,
    evaluate_cover,
    node_membership_multilabel_metrics,
    omega_index,
    one_to_one_community_metrics,
    partition_to_cover_lists,
    size_weighted_community_f1,
    symmetric_best_match_f1,
)
from hedonic.experiments.overlapping.robustness import (
    audit_cover,
    best_response,
    build_fractional_state,
    cover_hash,
    cover_to_vertex_memberships,
    exhaustive_best_response,
    nearest_equilibrium_tiny,
    fractional_phi,
    perturb_cover_incidence,
)
from hedonic.experiments.overlapping.snap import SnapDataset


class TestCommunityHedonic(unittest.TestCase):
    def test_local_move_only_is_the_public_parameter(self):
        parameter_names = inspect.signature(Game.community_hedonic).parameters
        self.assertIn("local_move_only", parameter_names)
        self.assertNotIn("ensure_equilibrium", parameter_names)
        deprecated_keyword = "only" + "_local_moving"
        self.assertNotIn(deprecated_keyword, parameter_names)
        doc = Game.community_hedonic.__doc__ or ""
        # lucas-igraph 1.0.0.4+ propagates interruption through the native
        # error path; the frozen 1.0.0.3 producer identity stays published.
        self.assertIn("Interruption", doc)
        self.assertIn("error path", doc)
        from hedonic.Game import HISTORICAL_ALGORITHM_IDENTITIES

        self.assertEqual(
            HISTORICAL_ALGORITHM_IDENTITIES["lucas-igraph-1.0.0.3"],
            "community_hedonic/lucas-igraph-1.0.0.3/interrupt-unsupported",
        )
        graph = Game(ig.Graph.Famous("Petersen"))
        cover = graph.community_hedonic(
            resolution=graph.density(),
            max_memberships=2,
            local_move_only=True,
            n_iterations=-1,
        )
        self.assertEqual(len(cover.membership), graph.vcount())
        from hedonic.Game import HEDONIC_ALGORITHM_IDENTITY

        self.assertEqual(
            cover._hedonic_algorithm_identity, HEDONIC_ALGORITHM_IDENTITY
        )
        with self.assertRaises(TypeError):
            graph.community_hedonic(**{deprecated_keyword: True})
        with self.assertRaises(TypeError):
            graph.community_hedonic(ensure_equilibrium=True)

    def test_rejects_zero_total_edge_weight(self):
        empty = Game(ig.Graph(n=3))
        with self.assertRaises(ValueError):
            empty.community_hedonic(max_memberships=1, resolution=0.5)

    def test_negative_iterations_attach_raw_memberships_and_preflight(self):
        graph = Game(ig.Graph.Famous("Petersen"))
        cover = graph.community_hedonic(
            resolution=graph.density(),
            max_memberships=2,
            n_iterations=-1,
            local_move_only=True,
        )
        self.assertTrue(hasattr(cover, "_hedonic_raw_memberships"))
        self.assertEqual(len(cover._hedonic_raw_memberships), graph.vcount())
        preflight = cover._hedonic_token_preflight
        self.assertEqual(preflight["max_memberships"], 2)
        self.assertFalse(preflight["integer_overflow"])

    def test_flat_and_nested_overlapping_init_are_equivalent(self):
        graph = Game(ig.Graph.Ring(6))
        nested = [[0], [0], [1], [1], [2], [2]]
        flat = [0, 0, 1, 1, 2, 2]
        self.assertEqual(graph._as_overlapping_init(flat), nested)
        ig.set_random_number_generator(random.Random(7))
        from_nested = graph.community_hedonic(
            initial_membership=nested,
            max_memberships=2,
            n_iterations=1,
            local_move_only=True,
            resolution=0.5,
            seed=7,
        )
        ig.set_random_number_generator(random.Random(7))
        from_flat = graph.community_hedonic(
            initial_membership=flat,
            max_memberships=2,
            n_iterations=1,
            local_move_only=True,
            resolution=0.5,
            seed=7,
        )
        self.assertEqual(
            [[int(x) for x in row] for row in from_nested.membership],
            [[int(x) for x in row] for row in from_flat.membership],
        )

    def test_rejects_directed_and_nonfinite_resolution(self):
        directed = Game(ig.Graph([(0, 1)], directed=True))
        self.assertTrue(directed.is_directed())
        with self.assertRaises(ValueError):
            directed.community_hedonic(max_memberships=1, resolution=0.5)
        isolated = Game(ig.Graph(n=1))
        with self.assertRaises(ValueError):
            isolated.community_hedonic(max_memberships=1)
        with self.assertRaises(ValueError):
            isolated.community_hedonic(max_memberships=1, resolution=0.5)
        graph = Game(ig.Graph.Famous("Petersen"))
        with self.assertRaises(ValueError):
            graph.community_hedonic(resolution=float("nan"))
        empty = Game(ig.Graph(n=0))
        with self.assertRaises(ValueError):
            empty.community_hedonic(max_memberships=1, resolution=0.0)
        edgeless = Game(ig.Graph(n=2))
        with self.assertRaises(ValueError):
            edgeless.community_hedonic(max_memberships=1, resolution=0.5)
        zero_weight = Game(ig.Graph([(0, 1)]))
        with self.assertRaises(ValueError):
            zero_weight.community_hedonic(
                max_memberships=1, resolution=0.5, edge_weights=[0.0]
            )

    def test_token_preflight_overflow_blocks_full_mode(self):
        from hedonic.Game import token_graph_preflight

        huge = token_graph_preflight(2, 3, 2**32)
        self.assertTrue(huge["integer_overflow"])
        graph = Game(ig.Graph([(0, 1), (0, 2)]))
        with self.assertRaises(OverflowError):
            graph.community_hedonic(
                max_memberships=2**32,
                local_move_only=False,
                n_iterations=1,
                resolution=0.5,
            )
        local = graph.community_hedonic(
            max_memberships=2,
            local_move_only=True,
            n_iterations=1,
            resolution=0.5,
        )
        self.assertFalse(local._hedonic_token_preflight["integer_overflow"])

    def test_start_duplicate_labels_are_traced_and_collapsed(self):
        from hedonic.Game import membership_incidence_trace

        graph = Game(ig.Graph.Ring(4))
        initial = [[0, 0], [1], [2], [3]]
        self.assertTrue(membership_incidence_trace(initial)["duplicate_labels"])
        cover = graph.community_hedonic(
            initial_membership=initial,
            max_memberships=2,
            n_iterations=1,
            local_move_only=True,
            resolution=0.5,
        )
        start = cover._hedonic_start_membership_trace
        returned = cover._hedonic_returned_membership_trace
        self.assertTrue(start["duplicate_labels"])
        self.assertEqual(start["incidence_count"], 5)
        self.assertEqual(start["unique_incidence_count"], 4)
        self.assertEqual(
            start["collisions"],
            [{"vertex": 0, "label": 0, "token_count": 2}],
        )
        self.assertFalse(returned["duplicate_labels"])
        self.assertEqual(returned["collisions"], [])
        self.assertEqual(cover._hedonic_multiplicity_drops, [])
        for row in cover.membership:
            labels = [int(label) for label in row]
            self.assertEqual(len(labels), len(set(labels)))
            self.assertGreaterEqual(len(labels), 1)
        self.assertEqual(cover._hedonic_original_edge_weight, 4.0)
        self.assertTrue(hasattr(cover, "_hedonic_native_quality"))

    def test_negative_iterations_native_full_result_is_audited(self):
        edges = [
            (0, 1),
            (0, 9),
            (0, 10),
            (1, 4),
            (1, 10),
            (1, 11),
            (2, 6),
            (2, 8),
            (3, 6),
            (3, 9),
            (4, 6),
            (4, 9),
            (7, 8),
            (8, 9),
        ]
        initial = [
            [0],
            [3],
            [0],
            [2],
            [3],
            [1],
            [0],
            [1],
            [2],
            [2],
            [2],
            [0],
        ]
        graph = Game(ig.Graph(n=12, edges=edges))
        ig.set_random_number_generator(random.Random(10001))
        result = graph.community_hedonic(
            initial_membership=initial,
            max_memberships=3,
            resolution=graph.density(),
            local_move_only=False,
            allow_isolation=True,
            n_iterations=-1,
        )
        audit = audit_cover(
            graph,
            [list(labels) for labels in result.membership],
            max_memberships=3,
            allow_isolation=True,
            gamma=graph.density(),
            dense=True,
        )
        self.assertTrue(audit["is_local_equilibrium_at_resolution"])
        self.assertEqual(audit["profitable_vertex_count_at_resolution"], 0)

    def test_native_equilibrium_supports_isolation_disabled_mode(self):
        graph = Game(ig.Graph(n=6, edges=[(1, 2), (3, 4)]))
        result = graph.community_hedonic(
            initial_membership=[[0], [1], [1], [2], [2], [2]],
            max_memberships=2,
            resolution=0.1,
            local_move_only=True,
            allow_isolation=False,
            n_iterations=-1,
        )
        self.assertEqual(result.membership[5], [0])
        audit = audit_cover(
            graph,
            [list(labels) for labels in result.membership],
            max_memberships=2,
            allow_isolation=False,
            gamma=0.1,
            dense=True,
        )
        self.assertTrue(audit["is_local_equilibrium_at_resolution"])
        self.assertEqual(audit["profitable_vertex_count_at_resolution"], 0)

    def test_negative_iterations_use_one_native_call_and_preserve_labels(self):
        graph = Game(ig.Graph(n=3, edges=[(0, 1), (1, 2)]))
        calls = []

        class Result:
            def __init__(self, membership):
                self.membership = membership

        def fake_leiden(**kwargs):
            calls.append(kwargs)
            return Result([[100], [100, 300], [300]])

        with patch.object(graph, "community_leiden", side_effect=fake_leiden):
            result = graph.community_hedonic(
                initial_membership=[[0], [0, 1], [1]],
                max_memberships=2,
                local_move_only=False,
                allow_isolation=True,
                n_iterations=-1,
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["initial_membership"], [[0], [0, 1], [1]])
        self.assertEqual(calls[0]["n_iterations"], -1)
        self.assertEqual(result.membership, [[100], [100, 300], [300]])
        self.assertEqual(
            result._hedonic_raw_memberships,
            [[100], [100, 300], [300]],
        )

    def test_positive_iterations_are_forwarded_without_equilibrium_snapshot(self):
        graph = Game(ig.Graph(n=3, edges=[(0, 1), (1, 2)]))
        calls = []

        class Result:
            membership = [[100, 400], [200], [300]]

        def fake_leiden(**kwargs):
            calls.append(kwargs)
            return Result()

        with patch.object(graph, "community_leiden", side_effect=fake_leiden):
            result = graph.community_hedonic(
                initial_membership=[[0], [0, 1], [1]],
                max_memberships=2,
                local_move_only=False,
                allow_isolation=False,
                n_iterations=2,
            )
        self.assertEqual(result.membership, [[100, 400], [200], [300]])
        self.assertEqual(calls[0]["n_iterations"], 2)
        self.assertFalse(hasattr(result, "_hedonic_raw_memberships"))

    def test_max_memberships_returns_cover(self):
        g = Game(ig.Graph.Famous("Petersen"))
        cover = g.community_hedonic(
            resolution=g.density(), n_iterations=-1, max_memberships=4
        )
        communities = partition_to_cover_lists(cover)
        self.assertGreaterEqual(len(communities), 1)
        q = cover_quality(cover)
        self.assertIsInstance(q, float)

    def test_disjoint_mode_max_memberships_one(self):
        g = Game(ig.Graph.Famous("Petersen"))
        part = g.community_hedonic(
            resolution=g.density(), n_iterations=-1, max_memberships=1
        )
        self.assertTrue(hasattr(part, "membership"))
        self.assertIsInstance(part.membership[0], int)

    def test_accepts_flat_init_for_overlapping(self):
        g = Game(ig.Graph.Famous("Petersen"))
        seed = g.community_hedonic(resolution=g.density(), max_memberships=1)
        cover = g.community_hedonic(
            resolution=g.density(),
            max_memberships=4,
            initial_membership=list(seed.membership),
            n_iterations=2,
        )
        self.assertGreaterEqual(len(partition_to_cover_lists(cover)), 1)

    def test_evaluate_cover_metrics(self):
        pred = [[0, 1, 2], [3, 4, 5]]
        gt = [[0, 1, 2], [3, 4, 5]]
        metrics = evaluate_cover(pred, gt, 6, compute_omega=True)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["jaccard"], 1.0)
        self.assertAlmostEqual(metrics["omega"], 1.0)
        self.assertEqual(
            metrics["score_definition_version"],
            "canonical_unique_vertex_set_cover_v1",
        )

    def test_one_to_one_matching_uses_canonical_unique_set_cover(self):
        metrics = one_to_one_community_metrics(
            [[0, 1], [0, 1]], [[0, 1]]
        )
        self.assertAlmostEqual(metrics["matching_precision"], 1.0)
        self.assertAlmostEqual(metrics["matching_recall"], 1.0)
        self.assertAlmostEqual(metrics["matching_f1"], 1.0)
        self.assertEqual(metrics["n_unmatched_predicted_comms"], 0)
        # Historical best-match remains deliberately duplicate-insensitive.
        self.assertAlmostEqual(
            symmetric_best_match_f1([[0, 1], [0, 1]], [[0, 1]]), 1.0
        )

    def test_one_to_one_matching_penalizes_unmatched_gt(self):
        metrics = one_to_one_community_metrics(
            [[0, 1]], [[0, 1], [2, 3]], matching_weight="jaccard"
        )
        self.assertAlmostEqual(metrics["matching_precision"], 1.0)
        self.assertAlmostEqual(metrics["matching_recall"], 0.5)
        self.assertAlmostEqual(metrics["matching_f1"], 2.0 / 3.0)
        self.assertEqual(metrics["n_unmatched_gt_comms"], 1)

    def test_node_membership_micro_and_macro_f1(self):
        metrics = node_membership_multilabel_metrics(
            [[0, 1], [2]], [[0, 1], [1, 2]]
        )
        self.assertAlmostEqual(metrics["node_micro_precision"], 1.0)
        self.assertAlmostEqual(metrics["node_micro_recall"], 0.75)
        self.assertAlmostEqual(metrics["node_micro_f1"], 6.0 / 7.0)
        self.assertAlmostEqual(metrics["node_macro_f1"], 5.0 / 6.0)

    def test_singleton_modes_and_diagnostics(self):
        pred = [[0, 1], [2], []]
        gt = [[0, 1], [3]]
        all_diag = cover_diagnostics(pred, gt, 4, singleton_mode="all")
        gt1_diag = cover_diagnostics(pred, gt, 4, singleton_mode="size_ge_2")
        self.assertEqual(all_diag["predicted_community_count"], 2)
        self.assertEqual(all_diag["gt_community_count"], 2)
        self.assertAlmostEqual(all_diag["predicted_singleton_fraction"], 0.5)
        self.assertAlmostEqual(all_diag["predicted_vertices_covered_fraction"], 0.75)
        self.assertAlmostEqual(all_diag["predicted_average_community_size"], 1.5)
        self.assertAlmostEqual(all_diag["predicted_median_community_size"], 1.5)
        self.assertAlmostEqual(
            all_diag["predicted_average_memberships_per_vertex"], 0.75
        )
        self.assertEqual(gt1_diag["predicted_community_count"], 1)
        self.assertEqual(gt1_diag["gt_community_count"], 1)
        self.assertEqual(gt1_diag["predicted_singleton_fraction"], 0.0)
        self.assertAlmostEqual(
            size_weighted_community_f1(pred, gt, singleton_mode="size_ge_2"),
            1.0,
        )

    def test_sampled_omega_never_allocates_dense_vertex_matrix(self):
        original_zeros = np.zeros

        def reject_dense_vertex_matrix(shape, *args, **kwargs):
            if isinstance(shape, tuple) and shape == (10_000, 10_000):
                raise AssertionError("dense vertex-pair allocation")
            return original_zeros(shape, *args, **kwargs)

        with patch("numpy.zeros", side_effect=reject_dense_vertex_matrix):
            value = omega_index(
                [[0, 1, 2], [2, 3]],
                [[0, 1, 2], [2, 3]],
                10_000,
                sample_size=2_000,
                seed=7,
            )
        self.assertAlmostEqual(value, 1.0)

    def test_sbm_methods_use_community_hedonic(self):
        self.assertEqual(
            sbm_sweep.METHODS["Hedonic"]["method_call_name"], "community_hedonic"
        )
        self.assertEqual(
            sbm_sweep.METHODS["Leiden"]["method_call_name"], "community_hedonic"
        )
        self.assertEqual(
            sbm_sweep.METHODS["Hedonic"]["parameters"]["max_memberships"], 1
        )

    def test_max_memberships_defaults_to_gt_count(self):
        self.assertEqual(resolve_mm_sub(None, 7), 7)
        self.assertEqual(resolve_mm_sub(None, 0), 1)
        self.assertEqual(resolve_mm_sub(4, 7), 4)
        self.assertEqual(resolve_mm_full(None, 13477), 13477)
        self.assertEqual(resolve_mm_full(8, 13477), 8)


class TestExperimentsConfig(unittest.TestCase):
    def test_defaults(self):
        from hedonic.experiments.config import ARTIFACTS_DIR, expand_path

        self.assertEqual(
            DEFAULT_DBLP_DIR,
            expand_path("~/Databases/Hedonic/Networks/DBLP"),
        )
        self.assertEqual(
            DEFAULT_NETWORKS_DIR,
            expand_path("~/Databases/Hedonic/Networks"),
        )
        self.assertEqual(
            DEFAULT_SYNTHETIC_DIR,
            expand_path("~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020"),
        )
        self.assertEqual(
            DEFAULT_OUTPUT_DIR,
            ARTIFACTS_DIR,
        )
        self.assertEqual(DEFAULT_OUTPUT_DIR, expand_path("artifacts"))
        self.assertTrue(
            str(DEFAULT_DBLP_DIR).endswith("Databases/Hedonic/Networks/DBLP")
        )

    def test_env_override(self):
        with patch.dict(
            os.environ,
            {
                "HEDONIC_DBLP_DIR": "/tmp/custom_dblp",
                "HEDONIC_NETWORKS_DIR": "/tmp/custom_networks",
                "HEDONIC_SYNTHETIC_DIR": "/tmp/custom_synth",
                "HEDONIC_OUTPUT_DIR": "/tmp/custom_out",
            },
            clear=False,
        ):
            dblp, synth = reload_paths()
            self.assertEqual(str(dblp), "/tmp/custom_dblp")
            self.assertEqual(str(synth), "/tmp/custom_synth")
            from hedonic.experiments import config as cfg

            self.assertEqual(str(cfg.OUTPUT_DIR), "/tmp/custom_out")
            self.assertEqual(str(cfg.NETWORKS_DIR), "/tmp/custom_networks")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HEDONIC_DBLP_DIR", None)
            os.environ.pop("HEDONIC_NETWORKS_DIR", None)
            os.environ.pop("HEDONIC_SYNTHETIC_DIR", None)
            os.environ.pop("HEDONIC_OUTPUT_DIR", None)
            reload_paths()

    def test_toml_paths_and_section(self):
        with tempfile.TemporaryDirectory() as d:
            toml_path = Path(d) / "hedonic.toml"
            toml_path.write_text(
                "\n".join(
                    [
                        "[paths]",
                        'dblp_dir = "/toml/dblp"',
                        'networks_dir = "/toml/networks"',
                        'synthetic_dir = "/toml/synth"',
                        'output_dir = "/toml/out"',
                        "",
                        "[overlapping_resolution]",
                        'output_dir = "/toml/res-f1"',
                        'resolutions = "0,1"',
                        'seeds = "0,1"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            data = read_toml(toml_path)
            self.assertIn("paths", data)
            with patch.dict(os.environ, {}, clear=False):
                for k in (
                    "HEDONIC_DBLP_DIR",
                    "HEDONIC_SYNTHETIC_DIR",
                    "HEDONIC_OUTPUT_DIR",
                    "HEDONIC_CONFIG",
                ):
                    os.environ.pop(k, None)
                loaded = load_config_file(toml_path, search_cwd=False, apply=True)
                self.assertEqual(loaded["paths"]["dblp_dir"], "/toml/dblp")
                from hedonic.experiments import config as cfg

                self.assertEqual(str(cfg.DBLP_DIR), "/toml/dblp")
                self.assertEqual(str(cfg.NETWORKS_DIR), "/toml/networks")
                self.assertEqual(str(cfg.OUTPUT_DIR), "/toml/out")
                resolved = resolve_experiment_paths(
                    config_path=toml_path,
                    data_dir=None,
                    output_dir=None,
                    experiment_section="overlapping_resolution",
                    search_cwd=False,
                )
                self.assertEqual(str(resolved["output_dir"]), "/toml/res-f1")
                self.assertEqual(
                    resolved["section"]["resolutions"], "0,1"
                )
                # CLI explicit output wins over TOML section
                resolved_cli = resolve_experiment_paths(
                    config_path=toml_path,
                    data_dir=None,
                    output_dir="/cli/out",
                    experiment_section="overlapping_resolution",
                    search_cwd=False,
                )
                self.assertEqual(str(resolved_cli["output_dir"]), "/cli/out")
            reload_paths()

    def test_relative_artifact_paths_are_anchored_at_repo_root(self):
        from hedonic.experiments.config import ARTIFACTS_DIR

        with tempfile.TemporaryDirectory() as d:
            toml_path = Path(d) / "hedonic.toml"
            toml_path.write_text(
                "\n".join(
                    [
                        "[paths]",
                        'output_dir = "artifacts"',
                        "",
                        "[overlapping_resolution]",
                        'output_dir = "artifacts/overlapping/resolution_f1"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HEDONIC_OUTPUT_DIR", None)
                resolved = resolve_experiment_paths(
                    config_path=toml_path,
                    data_dir=None,
                    output_dir=None,
                    experiment_section="overlapping_resolution",
                    search_cwd=False,
                )
            self.assertEqual(
                resolved["output_dir"],
                ARTIFACTS_DIR / "overlapping" / "resolution_f1",
            )
        reload_paths()

    def test_example_toml_ships_and_parses(self):
        root = Path(__file__).resolve().parents[1]
        configs_dir = root / "configs"
        example = configs_dir / "hedonic.example.toml"
        default_cfg = configs_dir / "hedonic.toml"
        self.assertTrue(configs_dir.is_dir(), "configs/ must exist")
        self.assertTrue(example.is_file(), "configs/hedonic.example.toml must ship")
        self.assertTrue(default_cfg.is_file(), "configs/hedonic.toml must ship")
        data = read_toml(example)
        self.assertIn("paths", data)
        self.assertIn("dblp_dir", data["paths"])
        # Example uses home-relative paths, not /Users/<name>
        self.assertTrue(
            str(data["paths"]["dblp_dir"]).startswith("~/")
            or str(data["paths"]["dblp_dir"]).startswith("$"),
            msg=data["paths"]["dblp_dir"],
        )
        default_data = read_toml(default_cfg)
        self.assertTrue(
            str(default_data["paths"]["dblp_dir"]).startswith("~/"),
            msg=default_data["paths"]["dblp_dir"],
        )

    def test_find_config_prefers_configs_dir(self):
        from hedonic.experiments.config import DEFAULT_TOML_NAMES, find_config_file

        self.assertEqual(DEFAULT_TOML_NAMES[0], "configs/hedonic.toml")
        root = Path(__file__).resolve().parents[1]
        # When cwd is the repo root, configs/hedonic.toml is discovered
        old = Path.cwd()
        try:
            os.chdir(root)
            found = find_config_file(search_cwd=True)
            self.assertIsNotNone(found)
            self.assertTrue(str(found).endswith("configs/hedonic.toml"))
        finally:
            os.chdir(old)

    @unittest.skipUnless(
        lock_tracked_files_present(Path("configs/overlapping-ground-truth-protocol.lock.json")),
        PUBLIC_CHECKOUT_REASON,
    )
    def test_ground_truth_protocol_config_and_lock_ship(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs" / "overlapping-ground-truth.toml"
        lock_path = root / "configs" / "overlapping-ground-truth-protocol.lock.json"
        self.assertTrue(config_path.is_file())
        self.assertTrue(lock_path.is_file())
        config = read_toml(config_path)
        self.assertIn("overlapping_ground_truth_robustness", config)
        self.assertEqual(
            config["overlapping_ground_truth_robustness"]["uncovered_policy"],
            "covered-induced",
        )
        self.assertEqual(
            [job["dataset"] for job in config["overlapping_ground_truth_robustness"]["jobs"]],
            ["amazon", "dblp", "livejournal", "youtube"],
        )
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        self.assertEqual(lock["schema_version"], 3)
        self.assertEqual(lock["protocol_name"], "overlapping-ground-truth-robustness-v3")
        self.assertNotIn("run_protocol_version", lock)
        self.assertEqual(lock["expected_conditions"], 3840)
        self.assertEqual(lock["canonical_grid"]["expected_detector_conditions"], 3840)
        self.assertEqual(lock["external_dependencies"], {})
        self.assertEqual(
            set(lock["dataset_content_identities"]),
            {
                "amazon/top5000",
                "dblp/top5000",
                "livejournal/top5000",
                "youtube/top5000",
            },
        )
        self.assertIn(
            "src/hedonic/experiments/overlapping/ground_truth_robustness.py",
            lock["tracked_files"],
        )
        for relative, expected in lock["tracked_files"].items():
            self.assertRegex(expected, r"^[0-9a-f]{64}$", relative)
        from hedonic.experiments.overlapping.certificate_reconcile import (
            classify_frozen_protocol_lock,
        )

        lock_report = classify_frozen_protocol_lock(lock_path)
        self.assertTrue(lock_report["lock_bytes_match_freeze"])
        self.assertEqual(lock_report["unexpected_drift"], [])
        self.assertEqual(lock_report["missing"], [])
        self.assertTrue(lock_report["ok"])
        for dataset, content in lock["dataset_content_identities"].items():
            for field in ("graph_sha256", "ground_truth_cover_sha256"):
                self.assertRegex(
                    content[field], r"^[0-9a-f]{64}$", f"{dataset}.{field}"
                )

        from hedonic.experiments.overlapping.protocol import (
            current_experiment_identity,
        )

        identity = current_experiment_identity(lock_path)
        assert_live_stack_reported_against_frozen_lock(self, identity["lucas_igraph"])
        self.assertTrue(identity["external_dependency_lock_complete"])
        self.assertTrue(identity["external_dependencies_match_lock"])
        self.assertTrue(identity["scientific_dependency_lock_complete"])


class TestDataLoaderHelpers(unittest.TestCase):
    def test_sort_files_by_network_seed(self):
        paths = [
            "/x/resultados/2 Communities of 10 nodes/Noise = 0.01/"
            "P_in = 0.10/Difficulty = 0.50/Network (003)/Partition (000)/A.json",
            "/x/resultados/2 Communities of 10 nodes/Noise = 0.25/"
            "P_in = 0.10/Difficulty = 0.50/Network (001)/Partition (000)/A.json",
        ]
        sorted_p = data_loader.sort_files(paths)
        self.assertIn("Network (001)", sorted_p[0])

    def test_load_json_single_dict(self):
        with tempfile.TemporaryDirectory() as d:
            fp = Path(d) / "r.json"
            fp.write_text(
                json.dumps({"method": "Hedonic", "accuracy": 0.9, "partition": [0, 1]})
            )
            recs = data_loader._load_json_records(str(fp))
            self.assertEqual(len(recs), 1)
            self.assertNotIn("partition", recs[0])
            self.assertEqual(recs[0]["accuracy"], 0.9)


class TestSbmSweep(unittest.TestCase):
    def test_generate_graph_and_accuracy(self):
        g = sbm_sweep.generate_graph(2, 12, 0.35, 0.15, seed=3)
        self.assertEqual(g.vcount(), 24)
        gt = sbm_sweep.get_ground_truth(2, 12, g)
        self.assertAlmostEqual(sbm_sweep.accuracy(g, gt, gt), 1.0)

    def test_generate_graph_preserves_isolated_vertices(self):
        # p_in=0 makes every vertex isolated; the size contract must still be
        # preserved instead of dropping trailing isolated vertices.
        g = sbm_sweep.generate_graph(3, 4, 0.0, 0.0, seed=9)
        self.assertEqual(g.vcount(), 12)
        self.assertEqual(g.ecount(), 0)

    def test_preflight_is_no_write_and_validates_selected_grid(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            code = sbm_sweep.main(
                [
                    "--preset",
                    "v1020",
                    "--max_n_nodes",
                    "1020",
                    "--n_communities",
                    "5",
                    "--seeds",
                    "3",
                    "--p_in",
                    "0.01",
                    "--difficulty",
                    "0.30",
                    "--noises",
                    "0.10",
                    "0.25",
                    "0.50",
                    "0.75",
                    "1.00",
                    "--partition_seeds",
                    "10",
                    "--output_root",
                    str(root),
                    "--preflight",
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(list(root.iterdir()), [])

    def test_preflight_rejects_truncated_community_size(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                sbm_sweep.main(
                    [
                        "--smoke",
                        "--max_n_nodes",
                        "25",
                        "--output_root",
                        d,
                        "--preflight",
                    ]
                )

    def test_resume_skips_complete_v1020_cell(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            methods = {"Mirror": sbm_sweep.METHODS["Mirror"]}
            self.assertTrue(
                sbm_sweep.run_experiment(
                    "resultados",
                    2,
                    4,
                    0.25,
                    0.2,
                    methods=methods,
                    noises=[0.1],
                    partition_seeds=[0],
                    seed=5,
                    output_root=root,
                    layout="v1020",
                )
            )
            with patch.object(
                sbm_sweep, "generate_graph", side_effect=AssertionError("reran")
            ):
                self.assertTrue(
                    sbm_sweep.run_experiment(
                        "resultados",
                        2,
                        4,
                        0.25,
                        0.2,
                        methods=methods,
                        noises=[0.1],
                        partition_seeds=[0],
                        seed=5,
                        output_root=root,
                        layout="v1020",
                        resume=True,
                    )
                )

    def test_resume_reruns_corrupt_v1020_cell(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = (
                root
                / "resultados"
                / "2C_4N"
                / "Noise = 0.10"
                / "P_in = 0.25"
                / "Difficulty = 0.20"
                / "Network (005)"
                / "partition_000.json"
            )
            target.parent.mkdir(parents=True)
            target.write_text("{not-json", encoding="utf-8")
            self.assertTrue(
                sbm_sweep.run_experiment(
                    "resultados",
                    2,
                    4,
                    0.25,
                    0.2,
                    methods={"Mirror": sbm_sweep.METHODS["Mirror"]},
                    noises=[0.1],
                    partition_seeds=[0],
                    seed=5,
                    output_root=root,
                    layout="v1020",
                    resume=True,
                )
            )
            with target.open(encoding="utf-8") as file:
                records = json.load(file)
            self.assertEqual(records[0]["method"], "Mirror")

    def test_archive_descendants_are_rejected(self):
        archive = Path(
            "~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020"
        ).expanduser()
        with self.assertRaises(SystemExit):
            sbm_sweep.main(
                [
                    "--preset",
                    "v1020-smoke",
                    "--output_root",
                    str(archive / "recovery"),
                    "--preflight",
                ]
            )

    def test_mini_run_experiment_writes_json(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ok = sbm_sweep.run_experiment(
                "smoke",
                2,
                8,
                0.25,
                0.2,
                methods={
                    "Mirror": sbm_sweep.METHODS["Mirror"],
                    "Hedonic": sbm_sweep.METHODS["Hedonic"],
                },
                noises=[0.01],
                partition_seeds=[0],
                seed=5,
                output_root=root,
            )
            self.assertTrue(ok)
            files = list(root.rglob("*.json"))
            self.assertGreaterEqual(len(files), 2)

    def test_parse_methods_subset(self):
        subset = sbm_sweep._parse_methods("Hedonic,Leiden")
        self.assertEqual(list(subset), ["Hedonic", "Leiden"])

    def test_main_smoke_flag(self):
        with tempfile.TemporaryDirectory() as d:
            ok = sbm_sweep.main(
                ["--smoke", "--output_root", d, "--folder_name", "unit"]
            )
            self.assertTrue(ok)
            self.assertGreaterEqual(len(list(Path(d).rglob("*.json"))), 1)

    def test_v1020_preset_constants(self):
        """Full V1020 grid matches archived PHYSA synthetic experiment."""
        p = sbm_sweep.V1020_PRESET
        self.assertEqual(p["max_n_nodes"], 1020)
        self.assertEqual(p["n_communities"], [2, 3, 4, 5, 6])
        self.assertEqual(p["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(p["noises"], [0.10, 0.25, 0.50, 0.75, 1.00])
        self.assertEqual(p["partition_seeds"], 10)
        self.assertEqual(p["n_runs"], 10)
        self.assertEqual(p["layout"], "v1020")
        self.assertEqual(p["folder_name"], "resultados")
        self.assertEqual(set(p["methods"]), set(sbm_sweep.METHODS))
        # community sizes used in archived V1020
        for n in p["n_communities"]:
            self.assertEqual(1020 // n, int(1020 / n))

    def test_v1020_smoke_layout_and_schema(self):
        """v1020-smoke writes partition_*.json lists with all methods + CSV schema."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ok = sbm_sweep.main(
                [
                    "--preset",
                    "v1020-smoke",
                    "--output_root",
                    str(root),
                ]
            )
            self.assertTrue(ok)
            jsons = list(root.rglob("partition_*.json"))
            self.assertGreaterEqual(len(jsons), 1)
            # Path shape: .../resultados/2C_20N/Noise = .../P_in = .../Difficulty = .../Network (000)/
            sample = jsons[0]
            parts = sample.parts
            self.assertIn("resultados", parts)
            self.assertTrue(any(re.match(r"\d+C_\d+N$", p) for p in parts))
            self.assertTrue(any(p.startswith("Noise =") for p in parts))
            self.assertTrue(any(p.startswith("Network (") for p in parts))
            with open(sample, encoding="utf-8") as f:
                records = json.load(f)
            self.assertIsInstance(records, list)
            self.assertGreaterEqual(len(records), len(sbm_sweep.METHODS))
            methods_seen = {r["method"] for r in records}
            for m in sbm_sweep.METHODS:
                self.assertIn(m, methods_seen)
            required = {
                "method",
                "number_of_communities",
                "community_size",
                "p_in",
                "p_out",
                "multiplier",
                "duration",
                "accuracy",
                "robustness",
                "noise",
                "network_seed",
                "partition_seed",
                "partition",
            }
            self.assertTrue(required.issubset(records[0].keys()))

            # data_loader simple path → CSV columns match archived resultados.csv.gzip
            df = data_loader.load_experiment_data(
                str(root / "resultados"), simple=True
            )
            expected_cols = {
                "method",
                "number_of_communities",
                "community_size",
                "p_in",
                "p_out",
                "multiplier",
                "resolution",
                "duration",
                "accuracy",
                "robustness",
                "noise",
                "network_seed",
                "partition_seed",
            }
            self.assertTrue(expected_cols.issubset(set(df.columns)))
            self.assertGreater(len(df), 0)
            self.assertEqual(
                set(df["method"].unique()) | set(),
                set(sbm_sweep.METHODS.keys()),
            )

    def test_v1020_preset_refuses_archived_root(self):
        code_or_exc = None
        try:
            sbm_sweep.main(
                [
                    "--preset",
                    "v1020",
                    "--output_root",
                    str(
                        Path(
                            "~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020"
                        ).expanduser()
                    ),
                    # Tiny override so we never actually run if guard fails
                    "--max_n_nodes",
                    "4",
                    "--n_communities",
                    "2",
                    "--seeds",
                    "0",
                    "--p_in",
                    "0.1",
                    "--difficulty",
                    "0.1",
                    "--noises",
                    "0.1",
                    "--partition_seeds",
                    "0",
                    "--methods",
                    "Mirror",
                    "--n_runs",
                    "1",
                ]
            )
            code_or_exc = "no_exit"
        except SystemExit as exc:
            code_or_exc = exc.code
        self.assertNotEqual(code_or_exc, "no_exit")
        self.assertNotEqual(code_or_exc, 0)


class TestCLI(unittest.TestCase):
    def test_help_lists_subcommands(self):
        code = CLI.main(["--help"])
        self.assertEqual(code, 0)

    def test_list_commands(self):
        code = CLI.main(["list"])
        self.assertEqual(code, 0)

    def test_version(self):
        code = CLI.main(["--version"])
        self.assertEqual(code, 0)

    def test_unknown_command(self):
        code = CLI.main(["not-a-command"])
        self.assertEqual(code, 2)

    def test_registry_covers_expected_commands(self):
        expected = {
            "smoke",
            "disjoint",
            "disjoint-load",
            "plots",
            "reproduce-disjoint",
            "overlapping-small",
            "overlapping-dnn",
            "overlapping-dnn-rational",
            "overlapping-oracle",
            "overlapping-integrity",
            "overlapping-certificate-reconcile",
            "overlapping-controlled",
            "overlapping-lfr",
            "overlapping-tracking",
            "overlapping-baselines",
            "overlapping-resource-envelope",
            "overlapping-subgraph",
            "overlapping-full",
            "overlapping-scale",
            "overlapping-resolution",
            "overlapping-benchmark",
            "overlapping-codeseg",
            "overlapping-reproduce",
            "codeseg-setup",
            "codeseg-doctor",
            "overlapping-full-snap",
            "overlapping-gt-robustness",
            "overlapping-gt-spectrum",
            "overlapping-audit",
            "reproduce-overlapping-paper",
            "quickstart",
            "info",
        }
        self.assertEqual(set(CLI.COMMANDS), expected)

    def test_hedonic_front_door(self):
        self.assertEqual(CLI.hedonic_main(["--help"]), 0)
        self.assertEqual(CLI.hedonic_main(["run", "exp", "--help"]), 0)
        self.assertEqual(CLI.hedonic_main(["run", "nope"]), 2)
        self.assertEqual(CLI.main(["quickstart", "--help"]), 0)

    def test_quickstart_parsing_and_catalogue(self):
        from hedonic.experiments.overlapping import quickstart as qs

        self.assertEqual(qs.parse_seeds("0-2,7"), [0, 1, 2, 7])
        self.assertEqual(len(qs.NETWORKS), 7)
        self.assertEqual(len(qs.METHODS), 10)
        self.assertEqual(qs.parse_choice("all", list(qs.BY_KEY), qs.ALIASES, "method"), list(qs.BY_KEY))
        self.assertEqual(qs.parse_choice("hoc,lazyfox", list(qs.BY_KEY), qs.ALIASES, "method"), ["hoc_local", "fox"])
        with self.assertRaises(ValueError):
            qs.parse_choice("nope", list(qs.NETWORKS), {}, "network")
        cfg = qs.config_from_args(qs.build_parser().parse_args(["--networks", "dblp", "--nodes", "1000", "--threads", "4"]))
        # community counts tuned on full DBLP are scaled to the subgraph; threads reach threaded methods
        self.assertLess(qs.method_params(qs.BY_KEY["bigclam"], cfg, "dblp")["communities"], 25000)
        self.assertEqual(qs.method_params(qs.BY_KEY["codeseg"], cfg, "dblp")["threads"], 4)
        self.assertEqual(qs.method_params(qs.BY_KEY["hoc_local"], cfg, "dblp")["max_memberships"], 64)
        self.assertIn("--nodes 1000", cfg.command())

    def test_quickstart_table_lists_every_accuracy_metric(self):
        from hedonic.experiments.overlapping import quickstart as qs

        cfg = qs.Config(methods=["hoc_local", "codeseg"], seeds=[0, 1])
        metrics = {key: 0.5 for key, _ in qs.COLUMNS}
        records = [{"network": "dblp", "method_key": "hoc_local", "seed": s, "status": "completed",
                    "metrics": metrics, "detection_seconds": 1.0} for s in (0, 1)]
        records.append({"network": "dblp", "method_key": "codeseg", "status": "unavailable",
                        "reason": "no C++ compiler found"})
        table = qs.render(qs.summarise(records, cfg), cfg)
        for _, label in qs.COLUMNS:
            self.assertIn(label, table)
        self.assertIn("no C++ compiler found", table)
        self.assertIn("mean ± sd over 2 seeds", table)

    def test_paper_config_plan_order(self):
        from hedonic.experiments.overlapping import quickstart as qs

        cfg = qs.load_config("paper")
        plan = qs.build_plan(cfg, {})
        self.assertEqual(len(plan), 2 * 10 * 10)
        # seed -> method -> network: seed 0, method 1 on DBLP then Amazon, then method 2 ...
        self.assertEqual([(n, s, m.key) for n, s, m in plan[:3]],
                         [("dblp", 0, "hoc_local"), ("amazon", 0, "hoc_local"), ("dblp", 0, "hoc_multilevel")])
        self.assertEqual(plan[20][1], 1)
        self.assertEqual(cfg.nodes, 0)
        self.assertEqual(cfg.command(), "hedonic run exp --config paper")
        # flags override a named config; a saved file round-trips
        args = qs.build_parser().parse_args(["--config", "paper", "--seeds", "0-1"])
        self.assertEqual(qs.config_from_args(args).seeds, [0, 1])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            self.assertEqual(qs.main(["--config", "paper", "--save-config", path]), 0)
            self.assertEqual(len(qs.build_plan(qs.load_config(path), {})), 200)
            toml_path = os.path.join(d, "c.toml")  # the extension decides the format, so it must load back
            self.assertEqual(qs.main(["--config", "paper", "--seeds", "0-2", "--save-config", toml_path]), 0)
            self.assertEqual(qs.load_config(toml_path).seeds, [0, 1, 2])
            self.assertEqual(qs.load_config(toml_path).networks, ["dblp", "amazon"])
        self.assertEqual(CLI.hedonic_main(["run", "paper", "--dry-run"]), 0)

    def test_user_config_defaults_and_profiles(self):
        from hedonic.experiments.overlapping import quickstart as qs

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.toml")
            with patch.dict(os.environ, {"HEDONIC_CONFIG": path}):
                self.assertEqual(CLI.hedonic_main(["config", "set", "output_dir", d]), 0)
                self.assertEqual(CLI.hedonic_main(["config", "set", "run", '"paper"']), 0)
                self.assertEqual(CLI.hedonic_main(["config", "set", "profiles.mine.seeds", '"0-2"']), 0)
                self.assertEqual(CLI.hedonic_main(["config", "set", "profiles.mine.config", '"paper"']), 0)
                cfg = qs.config_from_args(qs.build_parser().parse_args(["--config", "paper"]))
                self.assertEqual(cfg.output_dir, d)  # machine default applies to named configs
                cfg = qs.config_from_args(qs.build_parser().parse_args(["--config", "mine"]))
                self.assertEqual((cfg.seeds, cfg.networks), ([0, 1, 2], ["dblp", "amazon"]))
                cfg = qs.config_from_args(qs.build_parser().parse_args(["--config", "paper", "--output-dir", "x"]))
                self.assertEqual(cfg.output_dir, "x")  # flags win
                self.assertEqual(CLI.hedonic_main(["run"]), 0)  # no TTY: shows the plan, does not start

    def test_hedonic_show(self):
        from hedonic.experiments.overlapping import info

        with tempfile.TemporaryDirectory() as d:
            rows = info.method_status(Path(d))
            self.assertEqual(len(rows), 10)
            self.assertEqual({r["key"]: r["status"] for r in rows}["hoc_local"], "ready")
            nets = info.network_status(Path(d), d, remote=False)
            self.assertEqual(len(nets), 7)
            self.assertTrue(all(n["status"] == "not downloaded" for n in nets))
            for what in ("methods", "networks", "metrics"):
                self.assertEqual(CLI.hedonic_main(["show", what, "--cache-dir", d, "--json"] +
                                                  (["--network-root", d] if what == "networks" else [])), 0)

    def test_hedonic_run_verbs_without_runs(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_RUNS_DIR": d}):
            self.assertEqual(CLI.hedonic_main(["run", "list"]), 0)
            with self.assertRaises(SystemExit):
                CLI.hedonic_main(["run", "status"])

    def test_run_manager_never_trusts_a_recycled_pid_or_idle_tmux_pane(self):
        from hedonic.experiments.overlapping import runmanager as rm

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_RUNS_DIR": d}):
            out = os.path.join(d, "out")
            os.makedirs(out)
            entry = {"name": "r1", "config": {}, "output_dir": out, "created": 0, "session": "",
                     "backend": "process", "log": os.path.join(out, "worker.log"), "pid": os.getpid()}
            # this test process is alive but is not a hedonic worker: the run is interrupted, not "running"
            rm.write_json(rm.progress_path(entry), {"status": "running", "pid": os.getpid(), "plan": []})
            self.assertFalse(rm.is_worker(os.getpid(), "r1"))
            self.assertEqual(rm.effective_status(entry, rm.read_progress(entry)), "interrupted")
            self.assertFalse(rm.tmux_session_exists(""))  # an empty target must never match some other session
            rm.write_json(rm.entry_path("r1"), entry)
            self.assertEqual(rm.cmd_stop("r1"), 0)  # must not signal this process
            self.assertEqual(rm.effective_status(entry, rm.read_progress(entry)), "interrupted")

    def test_run_lookup_is_exact_then_unique_fragment(self):
        from hedonic.experiments.overlapping import runmanager as rm

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_RUNS_DIR": d}):
            for name in ("paper-20260101-000000", "paper-20260202-000000", "exp-1"):
                rm.write_json(rm.entry_path(name), {"name": name, "output_dir": d, "created": 0})
            self.assertEqual(rm.load_entry("exp-1")["name"], "exp-1")
            self.assertEqual(rm.load_entry("20260202")["name"], "paper-20260202-000000")
            with self.assertRaises(SystemExit):
                rm.load_entry("paper")  # ambiguous: refuse rather than stop the wrong run
            with self.assertRaises(SystemExit):
                rm.load_entry("missing")

    def test_worker_error_is_recorded_as_failed_and_resumable(self):
        from hedonic.experiments.overlapping import quickstart as qs
        from hedonic.experiments.overlapping import runmanager as rm

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_RUNS_DIR": d}):
            cfg = qs.Config(methods=["hoc_local"], output_dir=d, cache_dir=d)
            cfg.methods = ["not-a-method"]  # fails inside the worker, after it has started
            entry = {"name": "bad", "config": cfg.__dict__, "output_dir": d, "created": 0, "session": "",
                     "backend": "process", "log": os.path.join(d, "worker.log")}
            rm.write_json(rm.entry_path("bad"), entry)
            with patch("sys.stderr"), patch("sys.stdout"):
                self.assertEqual(rm.worker("bad"), 1)
            progress = rm.read_progress(entry)
            self.assertEqual(progress["status"], "failed")
            self.assertIn("not-a-method", progress["reason"])
            self.assertEqual(rm.effective_status(entry, progress), "failed")
            self.assertIn("hedonic run resume bad", rm.snapshot(entry))

    def test_run_launch_records_absolute_paths_and_resets_progress(self):
        from hedonic.experiments.overlapping import quickstart as qs
        from hedonic.experiments.overlapping import runmanager as rm

        class Proc:
            pid = 999_999_999

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_RUNS_DIR": d}):
            cwd = os.getcwd()
            os.chdir(d)
            try:
                stale = Path(d) / "rel" / "job" / "progress.json"
                stale.parent.mkdir(parents=True)
                stale.write_text('{"status": "completed"}')
                with patch.object(rm.shutil, "which", return_value=None), \
                        patch.object(rm.subprocess, "Popen", return_value=Proc()):
                    entry = rm.launch(qs.Config(output_dir="rel", cache_dir=os.path.join(d, "c")), "job")
                    self.assertTrue(os.path.isabs(entry["output_dir"]))  # `hedonic run list` works from any directory
                    self.assertFalse(stale.exists())  # a resumed run must not look finished
                    with self.assertRaises(ValueError):
                        rm.launch(qs.Config(output_dir="rel"), "bad name")
            finally:
                os.chdir(cwd)

    def test_run_option_validation_and_friendly_errors(self):
        from hedonic.experiments.overlapping import quickstart as qs
        from hedonic.experiments.overlapping import userconfig

        for bad in ("3-1", "x", "-2", ""):
            with self.assertRaises(ValueError):
                qs.parse_seeds(bad)
        self.assertEqual(qs.parse_name("paper-1_a"), "paper-1_a")
        for bad in ("a b", "a.b", "../x", ""):
            with self.assertRaises(ValueError):
                qs.parse_name(bad)
        for flags in (["--threads", "0"], ["--timeout", "0"], ["--nodes", "-1"]):
            with self.assertRaises(ValueError):
                qs.config_from_args(qs.build_parser().parse_args(flags))
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_CONFIG": os.path.join(d, "c.toml")}):
            for argv in (["set", "threads", "abc"], ["set", "threads", "0"], ["set", "output_dir", "true"],
                         ["set", "profiles.q.bogus", "1"]):
                with self.assertRaises(SystemExit):
                    userconfig.main(argv)
            self.assertFalse(os.path.exists(userconfig.path()))  # nothing invalid was written
            self.assertEqual(userconfig.main(["set", "threads", "4"]), 0)
            self.assertEqual(userconfig.defaults()["threads"], 4)

    def test_run_exp_list_uses_the_configured_cache_dir(self):
        from hedonic.experiments.overlapping import quickstart as qs

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_CONFIG": os.path.join(d, "c.toml")}):
            self.assertEqual(CLI.hedonic_main(["config", "set", "cache_dir", d]), 0)
            with patch("hedonic.experiments.overlapping.info.method_status") as method_status, \
                    patch("hedonic.experiments.overlapping.info.network_status", return_value=[]), \
                    patch("hedonic.experiments.overlapping.info.render_networks", return_value=""), \
                    patch("hedonic.experiments.overlapping.info.render_methods", return_value=""):
                self.assertEqual(qs.main(["--list"]), 0)
                self.assertEqual(method_status.call_args.args[0], Path(d))  # not the literal path "None"

    def test_tui_fallback_reasks_on_garbage_and_fit_truncates_by_visible_width(self):
        from hedonic.experiments.overlapping import tui

        with patch.object(tui, "interactive", return_value=False), patch("sys.stdout"):
            with patch("builtins.input", side_effect=["x", "9", "2"]):
                self.assertEqual(tui.select("t", ["a", "b"]), 1)
            with patch("builtins.input", side_effect=["0", "1,3"]):
                self.assertEqual(tui.multiselect("t", ["a", "b", "c"]), [0, 2])
            with patch("builtins.input", side_effect=EOFError), self.assertRaises(tui.Cancelled):
                tui.select("t", ["a", "b"])
        colored = "\033[1m" + "x" * 50 + "\033[0m"
        fitted = tui.fit(colored, 20)
        self.assertEqual(tui.visible_len(fitted), 20)
        self.assertTrue(fitted.endswith("\033[0m"))
        self.assertEqual(tui.fit("short", 20), "short")

    def test_overlapping_resolution_help(self):
        code = CLI.main(["overlapping-resolution", "--help"])
        self.assertEqual(code, 0)

    def test_reproduce_overlapping_paper_help(self):
        code = CLI.main(["reproduce-overlapping-paper", "--help"])
        self.assertEqual(code, 0)

    def test_overlapping_ground_truth_robustness_help(self):
        code = CLI.main(["overlapping-gt-robustness", "--help"])
        self.assertEqual(code, 0)

    def test_overlapping_resolution_smoke_via_cli(self):
        with tempfile.TemporaryDirectory() as d:
            code = CLI.main(
                [
                    "overlapping-resolution",
                    "--smoke",
                    "--resolutions",
                    "0,0.5,1",
                    "--seeds",
                    "0,1",
                    "--output_dir",
                    d,
                ]
            )
            self.assertEqual(code, 0)
            results = Path(d) / "resolution_f1.json"
            plot = Path(d) / "resolution_f1.png"
            self.assertTrue(results.is_file())
            self.assertTrue(plot.is_file())
            self.assertGreater(plot.stat().st_size, 0)
            data = json.loads(results.read_text(encoding="utf-8"))
            self.assertIn("runs", data)
            self.assertIn("aggregated", data)
            self.assertIn("meta", data)
            self.assertEqual(data["meta"]["n_iterations"], -1)
            self.assertFalse(data["meta"]["local_move_only"])
            self.assertTrue(data["meta"]["allow_isolation"])
            self.assertEqual(
                data["meta"]["experiment_identity"]["lucas_igraph"]["actual_version"],
                locked_lucas_igraph_version(),
            )
            # The paper protocol lock is frozen at 1.0.0.3 for historical
            # evidence; the active package lock/runtime is 1.0.0.5.
            self.assertFalse(
                data["meta"]["experiment_identity"]["lucas_igraph"][
                    "version_matches_lock"
                ]
            )
            self.assertTrue(data["meta"]["smoke"])
            # 3 resolutions × 2 seeds
            self.assertEqual(len(data["runs"]), 6)
            self.assertEqual(len(data["aggregated"]), 3)
            for row in data["aggregated"]:
                self.assertIn("f1_mean", row)
                self.assertIn("f1_ci_low", row)
                self.assertIn("f1_ci_high", row)
                self.assertIn("f1_samples", row)
                self.assertEqual(len(row["f1_samples"]), 2)
                self.assertEqual(row["n_seeds"], 2)
                self.assertFalse(row["local_move_only"])
                self.assertTrue(row["allow_isolation"])
                self.assertEqual(row["n_iterations"], -1)

    def test_overlapping_scale_help(self):
        code = CLI.main(["overlapping-scale", "--help"])
        self.assertEqual(code, 0)

    def test_overlapping_scale_smoke_via_cli(self):
        with tempfile.TemporaryDirectory() as d:
            code = CLI.main(
                [
                    "overlapping-scale",
                    "--smoke",
                    "--sizes",
                    "20,32",
                    "--timeout",
                    "120",
                    "--no-process",
                    "--output_dir",
                    d,
                ]
            )
            self.assertEqual(code, 0)
            results = Path(d) / "complexity_scale.json"
            plot = Path(d) / "complexity_scale.png"
            self.assertTrue(results.is_file())
            self.assertTrue(plot.is_file())
            self.assertGreater(plot.stat().st_size, 0)
            data = json.loads(results.read_text(encoding="utf-8"))
            self.assertIn("points", data)
            self.assertIn("meta", data)
            self.assertEqual(data["meta"]["n_iterations"], -1)
            self.assertTrue(data["meta"]["allow_isolation"])
            self.assertEqual(
                data["meta"]["experiment_identity"]["lucas_igraph"]["actual_version"],
                locked_lucas_igraph_version(),
            )
            # The paper protocol lock is frozen at 1.0.0.3 for historical
            # evidence; the active package lock/runtime is 1.0.0.5.
            self.assertFalse(
                data["meta"]["experiment_identity"]["lucas_igraph"][
                    "version_matches_lock"
                ]
            )
            self.assertGreaterEqual(len(data["points"]), 2)

    def test_overlapping_small_via_cli(self):
        code = CLI.main(["overlapping-small"])
        self.assertEqual(code, 0)

    def test_overlapping_subgraph_help(self):
        code = CLI.main(["overlapping-subgraph", "--help"])
        self.assertEqual(code, 0)

    def test_overlapping_full_help(self):
        code = CLI.main(["overlapping-full", "--help"])
        self.assertEqual(code, 0)

    def test_overlapping_help_documents_equilibrium_and_gt_k(self):
        """Help should document n_iterations=-1 and GT-based max_memberships."""
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            CLI.main(["overlapping-subgraph", "--help"])
        help_sub = buf.getvalue().lower()
        self.assertIn("-1", help_sub)
        self.assertIn("ground-truth", help_sub)
        self.assertIn("equilibrium", help_sub)

        buf = io.StringIO()
        with redirect_stdout(buf):
            CLI.main(["overlapping-full", "--help"])
        help_full = buf.getvalue().lower()
        self.assertIn("-1", help_full)
        self.assertIn("ground-truth", help_full)
        self.assertIn("equilibrium", help_full)

    def test_disjoint_help(self):
        code = CLI.main(["disjoint", "--help"])
        self.assertEqual(code, 0)

    def test_disjoint_load_help(self):
        code = CLI.main(["disjoint-load", "--help"])
        self.assertEqual(code, 0)

    def test_smoke_help(self):
        code = CLI.main(["smoke", "--help"])
        self.assertEqual(code, 0)

    def test_smoke_isolated_run(self):
        code = CLI.main(["smoke"])
        self.assertEqual(code, 0)

    def test_disjoint_smoke_to_tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            code = CLI.main(
                ["disjoint", "--smoke", "--output_root", d, "--folder_name", "cli"]
            )
            self.assertEqual(code, 0)
            self.assertGreaterEqual(len(list(Path(d).rglob("*.json"))), 1)

    def test_disjoint_v1020_smoke_via_cli(self):
        with tempfile.TemporaryDirectory() as d:
            code = CLI.main(
                ["disjoint", "--preset", "v1020-smoke", "--output_root", d]
            )
            self.assertEqual(code, 0)
            jsons = list(Path(d).rglob("partition_*.json"))
            self.assertGreaterEqual(len(jsons), 1)
            code = CLI.main(
                [
                    "disjoint-load",
                    "--results_folder",
                    str(Path(d) / "resultados"),
                    "--output",
                    str(Path(d) / "out.csv.gzip"),
                    "--simple",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((Path(d) / "out.csv.gzip").is_file())

    def test_plots_help(self):
        code = CLI.main(["plots", "--help"])
        self.assertEqual(code, 0)

    def test_plots_smoke_via_cli(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "figures"
            code = CLI.main(
                [
                    "plots",
                    "--smoke",
                    "--output_dir",
                    str(out),
                    "--format",
                    "png",
                    "--no-persist",
                ]
            )
            self.assertEqual(code, 0)
            from hedonic.experiments.plots.paper_figures import FIGURE_NAMES

            for stem in FIGURE_NAMES:
                matches = list(out.glob(f"{stem}.*"))
                self.assertTrue(matches, f"missing figure stem {stem}")

    def test_reproduce_disjoint_smoke_via_cli(self):
        with tempfile.TemporaryDirectory() as d:
            code = CLI.main(
                [
                    "reproduce-disjoint",
                    "--preset",
                    "v1020-smoke",
                    "--output_root",
                    d,
                    "--format",
                    "png",
                    "--no-persist",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((Path(d) / "resultados.csv.gzip").is_file())
            figs = list((Path(d) / "figures").glob("*.png"))
            self.assertGreaterEqual(len(figs), 4)


class TestSmallGraphsModule(unittest.TestCase):
    def test_run_tests_entrypoint(self):
        small_graphs.main([])


class TestComplexityScale(unittest.TestCase):
    """Shipped overlapping-scale helpers: timing, timeout stop, plot."""

    def test_synthetic_series_increasing_sizes(self):
        series = complexity_scale.build_synthetic_size_series(
            sizes=(24, 48, 72), n_blocks=4, seed=1
        )
        self.assertEqual(len(series), 3)
        sizes = [p.n_nodes for p in series]
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreater(sizes[-1], sizes[0])
        for p in series:
            self.assertGreater(p.n_nodes, 0)
            self.assertGreaterEqual(p.max_memberships, 1)
            self.assertGreaterEqual(p.density, 0.0)

    def test_time_both_variants_records_flags(self):
        series = complexity_scale.build_synthetic_size_series(
            sizes=(20, 36), n_blocks=4, seed=2
        )
        result = complexity_scale.run_scale_experiment(
            series,
            timeout_s=120.0,
            use_process=False,
        )
        self.assertGreaterEqual(len(result.points), 2)
        for rec in result.points:
            self.assertIn(rec["local_move_only"], (True, False))
            self.assertEqual(rec["n_iterations"], -1)
            self.assertTrue(rec["allow_isolation"])
            self.assertGreater(rec["n_nodes"], 0)
            self.assertIsNotNone(rec["wallclock_s"])
            self.assertGreaterEqual(rec["wallclock_s"], 0.0)
            self.assertGreaterEqual(rec["max_memberships"], 1)
            # resolution must be density of that subgraph
            self.assertAlmostEqual(rec["resolution"], rec["density"], places=9)
            self.assertFalse(rec["timed_out"])

        completed_t = result.completed_for(True)
        completed_f = result.completed_for(False)
        self.assertGreaterEqual(len(completed_t), 1)
        self.assertGreaterEqual(len(completed_f), 1)

    def test_soft_timeout_stops_further_growth(self):
        """Extremely tight soft timeout marks timed_out and stops that line."""
        series = complexity_scale.build_synthetic_size_series(
            sizes=(30, 60, 90), n_blocks=3, seed=3
        )
        # Negative wallclock budget is impossible → first finish still exceeds.
        result = complexity_scale.run_scale_experiment(
            series,
            timeout_s=1e-12,
            use_process=False,
        )
        # First size for each variant should be timed_out; no further sizes.
        by_olm: dict[bool, list] = {True: [], False: []}
        for p in result.points:
            by_olm[bool(p["local_move_only"])].append(p)
        for olm, rows in by_olm.items():
            self.assertGreaterEqual(len(rows), 1, msg=olm)
            self.assertTrue(rows[0]["timed_out"])
            # Growth stopped: at most one record per variant under soft timeout.
            self.assertEqual(len(rows), 1, msg=f"expected stop after first timeout ({olm})")

    def test_plot_writes_nonempty_file(self):
        series = complexity_scale.build_synthetic_size_series(
            sizes=(16, 28), n_blocks=4, seed=4
        )
        result = complexity_scale.run_scale_experiment(
            series, timeout_s=120.0, use_process=False
        )
        with tempfile.TemporaryDirectory() as d:
            plot_path = Path(d) / "complexity_scale.png"
            out = complexity_scale.plot_complexity_scale(result, plot_path)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)
            json_path = Path(d) / "out.json"
            complexity_scale.save_results(result, json_path)
            loaded = complexity_scale.load_results(json_path)
            self.assertEqual(len(loaded.points), len(result.points))

    def test_main_smoke_entrypoint(self):
        with tempfile.TemporaryDirectory() as d:
            code = complexity_scale.main(
                [
                    "--smoke",
                    "--sizes",
                    "18,30",
                    "--timeout",
                    "60",
                    "--no-process",
                    "--output_dir",
                    d,
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((Path(d) / "complexity_scale.json").is_file())
            self.assertTrue((Path(d) / "complexity_scale.png").is_file())

    def test_count_gt_communities(self):
        nodes = [0, 1, 2, 5, 6]
        gt = [[0, 1, 2], [5, 6, 7], [10, 11], [1]]
        # first has 3, second has 2, third 0, fourth 1 (<2)
        self.assertEqual(
            complexity_scale.count_gt_communities_in_nodes(nodes, gt), 2
        )

    def test_parse_variants(self):
        self.assertEqual(complexity_scale.parse_variants("both"), (True, False))
        self.assertEqual(complexity_scale.parse_variants("local"), (True,))
        self.assertEqual(complexity_scale.parse_variants("full"), (False,))
        with self.assertRaises(ValueError):
            complexity_scale.parse_variants("nope")

    def test_run_full_variant_only(self):
        series = complexity_scale.build_synthetic_size_series(
            sizes=(16, 24), n_blocks=4, seed=7
        )
        result = complexity_scale.run_scale_experiment(
            series,
            timeout_s=60.0,
            use_process=False,
            variants=complexity_scale.parse_variants("full"),
        )
        self.assertTrue(result.points)
        self.assertTrue(all(p["local_move_only"] is False for p in result.points))
        self.assertFalse(result.completed_for(True))
        self.assertGreaterEqual(len(result.completed_for(False)), 1)


class TestResolutionF1(unittest.TestCase):
    """Shipped overlapping-resolution: γ×seed F1 CIs + plot (no DBLP)."""

    def test_count_gt_gt1_and_resolve_k(self):
        gt = [[0, 1, 2], [3, 4], [5], [6, 7, 8, 9]]
        self.assertEqual(resolution_f1.count_gt_communities_gt1(gt), 3)
        self.assertEqual(resolution_f1.resolve_max_memberships(None, gt), 3)
        self.assertEqual(resolution_f1.resolve_max_memberships(2, gt), 2)
        self.assertEqual(resolution_f1.resolve_max_memberships(None, [[0]]), 1)

    def test_parse_resolutions_and_seeds(self):
        res = resolution_f1.parse_resolutions("0:1:5")
        self.assertEqual(len(res), 5)
        self.assertAlmostEqual(res[0], 0.0)
        self.assertAlmostEqual(res[-1], 1.0)
        self.assertEqual(resolution_f1.parse_resolutions("0,0.5,1"), [0.0, 0.5, 1.0])
        self.assertEqual(resolution_f1.parse_seeds("0-3"), [0, 1, 2, 3])
        self.assertEqual(resolution_f1.parse_seeds("1,7,9"), [1, 7, 9])

    def test_seeded_init_varies_with_seed(self):
        a = resolution_f1.seeded_initial_membership(20, 4, seed=0)
        b = resolution_f1.seeded_initial_membership(20, 4, seed=1)
        self.assertEqual(len(a), 20)
        self.assertNotEqual(a, b)
        # Contiguous labels from 0
        self.assertEqual(set(a), set(range(max(a) + 1)))

    def test_mean_ci_fields(self):
        stats = resolution_f1.mean_ci([0.5, 0.6, 0.7], confidence=0.95)
        self.assertAlmostEqual(stats["mean"], 0.6)
        self.assertLess(stats["ci_low"], stats["mean"])
        self.assertGreater(stats["ci_high"], stats["mean"])
        self.assertEqual(stats["n"], 3)

    def test_run_one_uses_required_flags(self):
        game, gt = resolution_f1.build_smoke_instance(
            n_blocks=3, block_size=6, seed=1
        )
        k = resolution_f1.resolve_max_memberships(None, gt)
        # K must ignore the singleton planted in build_smoke_instance
        self.assertEqual(k, 3)
        rec = resolution_f1.run_one(
            game, resolution_f1.filter_gt_communities_gt1(gt),
            resolution=0.5,
            seed=0,
            max_memberships=k,
        )
        self.assertEqual(rec["n_iterations"], -1)
        self.assertFalse(rec["local_move_only"])
        self.assertTrue(rec["allow_isolation"])
        self.assertEqual(rec["max_memberships"], k)
        self.assertIn("f1", rec)
        self.assertGreaterEqual(rec["f1"], 0.0)
        self.assertLessEqual(rec["f1"], 1.0)
        # Full metadata + cover for cache / later metrics
        for key in (
            "cover",
            "quality",
            "wallclock_s",
            "seed",
            "resolution",
            "initial_membership",
            "metrics",
            "completed_at",
            "status",
            "n_vertices",
            "n_edges",
            "experiment_identity",
        ):
            self.assertIn(key, rec, msg=key)
        self.assertEqual(rec["status"], "complete")
        self.assertIsInstance(rec["cover"], list)
        self.assertGreaterEqual(len(rec["cover"]), 1)
        self.assertIsInstance(rec["wallclock_s"], float)
        self.assertEqual(rec["seed"], 0)
        self.assertIn("f1", rec["metrics"])

    def test_smoke_experiment_writes_plot_and_ci(self):
        game, gt = resolution_f1.build_smoke_instance(
            n_blocks=3, block_size=6, seed=2
        )
        result = resolution_f1.run_resolution_f1_experiment(
            game,
            gt,
            resolutions=[0.0, 0.5, 1.0],
            seeds=[0, 1],
        )
        self.assertEqual(len(result.runs), 6)
        self.assertEqual(len(result.aggregated), 3)
        self.assertEqual(result.meta["n_iterations"], -1)
        self.assertFalse(result.meta["local_move_only"])
        self.assertTrue(result.meta["allow_isolation"])
        self.assertEqual(result.meta["max_memberships"], 3)
        for row in result.aggregated:
            self.assertEqual(len(row["f1_samples"]), 2)
            self.assertIn("f1_ci_low", row)
            self.assertIn("f1_ci_high", row)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            json_path = resolution_f1.save_results(result, out / "r.json")
            plot_path = resolution_f1.plot_resolution_f1(
                result, out / "r.png"
            )
            self.assertTrue(json_path.is_file())
            self.assertTrue(plot_path.is_file())
            self.assertGreater(plot_path.stat().st_size, 0)
            loaded = resolution_f1.load_results(json_path)
            self.assertEqual(len(loaded.runs), len(result.runs))

    def test_main_smoke_entrypoint(self):
        with tempfile.TemporaryDirectory() as d:
            code = resolution_f1.main(
                [
                    "--smoke",
                    "--resolutions",
                    "0,1",
                    "--seeds",
                    "0,1",
                    "--output_dir",
                    d,
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((Path(d) / "resolution_f1.json").is_file())
            self.assertTrue((Path(d) / "resolution_f1.png").is_file())
            data = json.loads(
                (Path(d) / "resolution_f1.json").read_text(encoding="utf-8")
            )
            # Production defaults for max_memberships rule on smoke graph
            self.assertEqual(
                data["meta"]["max_memberships_rule"],
                "n_gt_communities_size_gt_1",
            )
            self.assertEqual(data["meta"]["n_iterations"], -1)
            self.assertFalse(data["meta"]["local_move_only"])
            self.assertTrue(data["meta"]["allow_isolation"])

    def test_help_documents_resolution_span_and_seeds(self):
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                resolution_f1.main(["--help"])
            except SystemExit as exc:
                self.assertIn(exc.code, (0, None))
        help_text = buf.getvalue().lower()
        self.assertIn("resolution", help_text)
        self.assertIn("seed", help_text)
        self.assertIn("smoke", help_text)
        self.assertIn("0:1:11", help_text)
        self.assertIn("dblp", help_text)
        self.assertIn("resume", help_text)
        self.assertIn("rescore", help_text)
        self.assertIn("singleton-mode", help_text)
        self.assertIn("omega-sample-size", help_text)
        self.assertIn("config", help_text)

    def test_cache_resume_skips_completed_runs(self):
        """Interrupted-then-restarted: completed (γ,seed) cells are not re-run."""
        game, gt = resolution_f1.build_smoke_instance(
            n_blocks=3, block_size=6, seed=3
        )
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            first = resolution_f1.run_resolution_f1_experiment(
                game,
                gt,
                resolutions=[0.0, 1.0],
                seeds=[0, 1],
                cache_dir=out,
                resume=True,
            )
            self.assertEqual(first.meta["n_ran"], 4)
            self.assertEqual(first.meta["n_skipped_cache"], 0)
            runs_dir = out / resolution_f1.RUNS_SUBDIR
            self.assertTrue(runs_dir.is_dir())
            run_files = list(runs_dir.glob("res_*.json"))
            self.assertEqual(len(run_files), 4)
            sample = json.loads(run_files[0].read_text(encoding="utf-8"))
            self.assertIn("cover", sample)
            self.assertIn("quality", sample)
            self.assertIn("wallclock_s", sample)
            self.assertEqual(sample["status"], "complete")

            second = resolution_f1.run_resolution_f1_experiment(
                game,
                gt,
                resolutions=[0.0, 1.0],
                seeds=[0, 1],
                cache_dir=out,
                resume=True,
            )
            self.assertEqual(second.meta["n_ran"], 0)
            self.assertEqual(second.meta["n_skipped_cache"], 4)
            self.assertTrue(all(r.get("from_cache") for r in second.runs))

    def test_partial_resume_only_runs_missing(self):
        game, gt = resolution_f1.build_smoke_instance(
            n_blocks=3, block_size=6, seed=4
        )
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            # Complete only γ=0, seed=0
            partial = resolution_f1.run_resolution_f1_experiment(
                game,
                gt,
                resolutions=[0.0],
                seeds=[0],
                cache_dir=out,
            )
            self.assertEqual(partial.meta["n_ran"], 1)
            full = resolution_f1.run_resolution_f1_experiment(
                game,
                gt,
                resolutions=[0.0, 1.0],
                seeds=[0, 1],
                cache_dir=out,
                resume=True,
            )
            # 4 cells total; 1 cached → 3 new
            self.assertEqual(full.meta["n_skipped_cache"], 1)
            self.assertEqual(full.meta["n_ran"], 3)
            self.assertEqual(len(full.runs), 4)

    def test_rescore_from_cover_without_detection(self):
        game, gt = resolution_f1.build_smoke_instance(
            n_blocks=3, block_size=6, seed=5
        )
        gt_eval = resolution_f1.filter_gt_communities_gt1(gt)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            resolution_f1.run_resolution_f1_experiment(
                game,
                gt,
                resolutions=[0.5],
                seeds=[0],
                cache_dir=out,
            )
            # Rescore-only path (no community_hedonic)
            with patch.object(
                Game,
                "community_hedonic",
                side_effect=AssertionError("detection called during rescore"),
            ):
                rescored = resolution_f1.run_resolution_f1_experiment(
                    game,
                    gt,
                    resolutions=[0.5],
                    seeds=[0],
                    cache_dir=out,
                    rescore_only=True,
                    singleton_mode="both",
                )
            self.assertEqual(rescored.meta["n_ran"], 0)
            self.assertEqual(rescored.meta["n_rescored"], 1)
            self.assertIn("f1", rescored.runs[0])
            self.assertIn("matching_f1", rescored.runs[0])
            self.assertEqual(
                set(rescored.runs[0]["metrics_by_singleton_mode"]),
                {"all", "size_ge_2"},
            )
            self.assertIn(
                "matching_f1_mean",
                rescored.aggregated[0]["metrics_by_singleton_mode"]["size_ge_2"],
            )

            # Direct API: metrics_from_cover uses cached cover only
            rec = resolution_f1.load_run_file(
                resolution_f1.run_cache_path(out / "runs", 0.5, 0)
            )
            self.assertIsNotNone(rec)
            metrics = resolution_f1.metrics_from_cover(
                rec["cover"], gt_eval, game.vcount()
            )
            self.assertIn("f1", metrics)
            self.assertAlmostEqual(metrics["f1"], rec["f1"], places=9)

    def test_old_minimal_cached_record_can_be_rescored(self):
        game, gt = resolution_f1.build_smoke_instance(
            n_blocks=2, block_size=4, seed=6
        )
        old_record = {
            "resolution": 0.25,
            "seed": 0,
            "n_vertices": game.vcount(),
            "wallclock_s": 12.5,
            "cover": [list(range(4)), list(range(4, 8))],
            "f1": 0.123,  # Legacy completion marker; no metrics dict/status.
        }
        with tempfile.TemporaryDirectory() as d:
            runs_dir = Path(d) / resolution_f1.RUNS_SUBDIR
            path = resolution_f1.run_cache_path(runs_dir, 0.25, 0)
            resolution_f1.save_run_file(old_record, path)
            with patch.object(
                Game,
                "community_hedonic",
                side_effect=AssertionError("detection called during rescore"),
            ):
                result = resolution_f1.run_resolution_f1_experiment(
                    game,
                    gt,
                    resolutions=[0.25],
                    seeds=[0],
                    cache_dir=d,
                    rescore_only=True,
                    singleton_mode="size_ge_2",
                )
            rescored = result.runs[0]
            self.assertEqual(rescored["cover"], old_record["cover"])
            self.assertEqual(rescored["wallclock_s"], 12.5)
            self.assertAlmostEqual(rescored["f1"], 1.0)
            self.assertIn("symmetric_best_match_f1", rescored)
            self.assertIn("node_micro_f1", rescored)
            self.assertIn("predicted_community_count", rescored)
            self.assertEqual(result.meta["n_ran"], 0)

    def test_main_writes_runs_cache_and_toml_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            toml_path = root / "cfg.toml"
            out = root / "artifacts"
            toml_path.write_text(
                "\n".join(
                    [
                        "[overlapping_resolution]",
                        f'output_dir = "{out.as_posix()}"',
                        'resolutions = "0,1"',
                        'seeds = "0,1"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            code = resolution_f1.main(
                [
                    "--smoke",
                    "--config",
                    str(toml_path),
                    # CLI output still wins if set; omit to use TOML section
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((out / "resolution_f1.json").is_file())
            self.assertTrue((out / "runs").is_dir())
            self.assertGreaterEqual(len(list((out / "runs").glob("res_*.json"))), 2)


class TestGroundTruthRobustness(unittest.TestCase):
    def test_rescore_artifact_loaders_reject_tampered_content(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            cover_digest = _persist_cover(output, [[0, 1], [1, 2]])
            self.assertEqual(
                _load_cover_artifact(output, cover_digest), [[0, 1], [1, 2]]
            )
            with gzip.open(
                output / "covers" / f"{cover_digest}.json.gz",
                "wt",
                encoding="utf-8",
            ) as stream:
                json.dump([[0, 2]], stream)
            self.assertIsNone(_load_cover_artifact(output, cover_digest))

            memberships = [[0], [0, 1], [1]]
            membership_digest = _persist_memberships(output, memberships, 3)
            self.assertEqual(
                _load_membership_artifact(output, membership_digest), memberships
            )
            with gzip.open(
                output / "raw_memberships" / f"{membership_digest}.json.gz",
                "wt",
                encoding="utf-8",
            ) as stream:
                json.dump([[0], [0], [0]], stream)
            self.assertIsNone(
                _load_membership_artifact(output, membership_digest)
            )

    def test_raw_memberships_preserve_duplicate_communities(self):
        raw = _normalize_raw_memberships(
            [[0, 1], [0, 1], [0, 1], [2], [2], [2]],
            n_vertices=6,
            cap=2,
        )
        raw_cover = []
        for label in range(3):
            raw_cover.append(
                [vertex for vertex, labels in enumerate(raw) if label in labels]
            )
        self.assertEqual(raw_cover, [[0, 1, 2], [0, 1, 2], [3, 4, 5]])
        self.assertNotEqual(_raw_cover_hash(raw_cover), cover_hash(raw_cover))

    @staticmethod
    def _fixture():
        graph = ig.Graph(
            n=6,
            edges=[
                (0, 1),
                (1, 2),
                (2, 0),
                (2, 3),
                (3, 4),
                (4, 2),
                (4, 5),
                (5, 0),
            ],
        )
        cover = [[0, 1, 2], [2, 3, 4], [0, 4, 5]]
        return graph, cover

    def test_prefix_best_response_matches_exhaustive_oracle(self):
        graph, cover = self._fixture()
        memberships = cover_to_vertex_memberships(cover, graph.vcount())
        state = build_fractional_state(graph, memberships)
        for gamma in (0.0, 0.2, 0.7, 1.0):
            for allow_isolation in (False, True):
                for vertex in range(graph.vcount()):
                    fast = best_response(
                        state,
                        vertex,
                        gamma,
                        max_memberships=3,
                        allow_isolation=allow_isolation,
                        dense=True,
                    )
                    slow = exhaustive_best_response(
                        state,
                        vertex,
                        gamma,
                        max_memberships=3,
                        allow_isolation=allow_isolation,
                    )
                    self.assertAlmostEqual(fast["best_utility"], slow["best_utility"])
                    self.assertAlmostEqual(fast["regret"], slow["regret"])
                    sparse = best_response(
                        state,
                        vertex,
                        gamma,
                        max_memberships=3,
                        allow_isolation=allow_isolation,
                        dense=False,
                    )
                    self.assertAlmostEqual(sparse["best_utility"], slow["best_utility"])

    def test_endpoint_certificate_and_disjoint_limit(self):
        graph, cover = self._fixture()
        memberships = cover_to_vertex_memberships(cover, graph.vcount())
        audit = audit_cover(
            graph,
            memberships,
            max_memberships=3,
            allow_isolation=False,
            gamma=0.5,
            dense=True,
        )
        direct = 0
        state = build_fractional_state(graph, memberships)
        for vertex in range(graph.vcount()):
            endpoints = [
                best_response(state, vertex, gamma, 3, False, dense=True)
                for gamma in (0.0, 1.0)
            ]
            direct += int(all(item["regret"] <= 1e-9 for item in endpoints))
        self.assertEqual(audit["robust_vertex_count_gamma_0_1"], direct)
        self.assertEqual(
            audit["robust_fraction_gamma_0_1"],
            direct / graph.vcount(),
        )

        # With one membership per vertex the fractional potential reduces to
        # the ordinary CPM expression under the same 2m normalization.
        disjoint = [[0, 1, 2], [3, 4, 5]]
        disjoint_memberships = cover_to_vertex_memberships(disjoint, graph.vcount())
        gamma = 0.25
        internal = sum(
            1
            for first, second in graph.get_edgelist()
            if any(first in community and second in community for community in disjoint)
        )
        expected = (2.0 * internal - gamma * sum(len(c) ** 2 for c in disjoint)) / (
            2.0 * graph.ecount()
        )
        self.assertAlmostEqual(
            fractional_phi(graph, disjoint_memberships, gamma), expected
        )
        disjoint_audit = audit_cover(
            graph,
            disjoint_memberships,
            max_memberships=1,
            allow_isolation=False,
            gamma=graph.density(),
            dense=True,
        )
        self.assertAlmostEqual(
            disjoint_audit["robust_fraction_gamma_0_1"],
            sbm_sweep.robustness(Game(graph), [0, 0, 0, 1, 1, 1]),
        )

    def test_incidence_switch_preserves_both_degree_sequences(self):
        graph, cover = self._fixture()
        perturbed, metadata = perturb_cover_incidence(
            cover,
            graph.vcount(),
            swaps=1,
            seed=4,
        )
        original_memberships = cover_to_vertex_memberships(cover, graph.vcount())
        final_memberships = cover_to_vertex_memberships(perturbed, graph.vcount())
        self.assertEqual(
            list(map(len, original_memberships)), list(map(len, final_memberships))
        )
        self.assertEqual(sorted(map(len, cover)), sorted(map(len, perturbed)))
        self.assertTrue(metadata["vertex_membership_counts_preserved"])
        self.assertTrue(metadata["community_sizes_preserved"])
        self.assertEqual(metadata["successful_swaps"], 1)
        self.assertAlmostEqual(
            metadata["realized_incidence_distance"],
            2.0
            * metadata["successful_swaps"]
            / metadata["initial_incidence_count"],
        )

    def test_tiny_nearest_equilibrium_calibration(self):
        graph = ig.Graph(n=4, edges=[(0, 1), (1, 2), (2, 3), (3, 0)])
        result = nearest_equilibrium_tiny(
            graph,
            [[0, 1], [2, 3]],
            gamma=0.5,
            n_communities=2,
            max_memberships=1,
            allow_isolation=False,
        )
        self.assertEqual(result["status"], "completed")
        self.assertGreater(result["candidate_count"], 0)
        self.assertGreater(result["equilibrium_count"], 0)
        self.assertIsNotNone(result["nearest"])
        self.assertGreaterEqual(result["nearest"]["distance_to_ground_truth"], 0.0)

    def test_partial_cover_policies_are_explicit(self):
        graph = ig.Graph(n=5, edges=[(0, 1), (1, 2), (2, 3), (3, 4)])
        raw = SnapDataset(
            "fixture",
            "all",
            graph,
            [[0, 1], [1, 2]],
            {"id_mapping_strategy": "fixture"},
        )
        induced = covered_induced_dataset(raw)
        self.assertEqual(induced.graph.vcount(), 3)
        self.assertEqual(induced.cover, [[0, 1], [1, 2]])
        self.assertEqual(
            induced.report["ground_truth_completion"]["policy"], "covered-induced"
        )
        completed = singleton_completed_dataset(raw)
        self.assertEqual(completed.graph.vcount(), 5)
        self.assertEqual(completed.cover[-2:], [[3], [4]])
        self.assertEqual(
            completed.report["ground_truth_completion"]["synthetic_memberships_added"],
            2,
        )
        prepared = prepare_dataset(raw, policy="covered-induced")
        self.assertEqual(prepared.policy, "covered-induced")
        self.assertEqual(prepared.dataset.graph.vcount(), 3)
        self.assertTrue(prepared.graph_identity)

    def test_smoke_runner_and_detector_free_rescore(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "artifacts"
            common = [
                "--smoke",
                "--output-dir",
                str(output),
                "--phases",
                "local",
                "--isolation-policies",
                "fixed_labels",
                "--seeds",
                "0",
                "--resolution-multipliers",
                "1",
                "--perturbation-distances",
                "0",
                "--timeout-per-run",
                "30",
            ]
            self.assertEqual(ground_truth_robustness_main(common), 0)
            self.assertTrue((output / "results.csv").is_file())
            header = (output / "results.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("action_policy", header)
            self.assertIn("completion_policy", header)
            with patch.object(
                Game,
                "community_hedonic",
                side_effect=AssertionError("detection called during rescore"),
            ):
                self.assertEqual(
                    ground_truth_robustness_main(common + ["--rescore-only"]), 0
                )
            records = list((output / "runs" / "amazon-top5000").glob("*.json"))
            self.assertEqual(len(records), 1)
            record = json.loads(records[0].read_text(encoding="utf-8"))
            self.assertIn("rescored_at", record)
            self.assertEqual(record["schema_version"], 3)
            self.assertEqual(record["condition"]["protocol"], "canonical_unique_cover_v3")
            self.assertTrue(record["condition_axis_key"])
            self.assertTrue(record["raw_membership_hash"])
            self.assertTrue(
                (output / "raw_memberships" / f"{record['raw_membership_hash']}.json.gz").is_file()
            )
            records[0].unlink()
            self.assertEqual(
                ground_truth_robustness_main(common + ["--rescore-only"]), 0
            )
            coverage = json.loads(
                (output / "coverage_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(coverage["expected_detector_conditions"], 1)
            self.assertEqual(coverage["observed_detector_records"], 0)
            self.assertFalse(coverage["complete"])
            self.assertIn("results.csv", coverage["artifact_sha256"])
            self.assertTrue(coverage["ground_truth_records_sha256"])

    def test_rescore_reconstructs_and_rejects_altered_detector_condition(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "artifacts"
            common = [
                "--smoke", "--output-dir", str(output), "--phases", "local",
                "--isolation-policies", "fixed_labels", "--seeds", "0",
                "--resolution-multipliers", "1", "--perturbation-distances", "0",
                "--timeout-per-run", "30",
            ]
            self.assertEqual(ground_truth_robustness_main(common), 0)
            record_path = next((output / "runs" / "amazon-top5000").glob("*.json"))
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["condition"]["beta"] = 0.2
            record["condition_identity"] = _json_hash(record["condition"])
            record["condition_key"] = record["condition_identity"][:24]
            record_path.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(
                ground_truth_robustness_main(common + ["--rescore-only"]), 0
            )
            coverage = json.loads(
                (output / "coverage_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(coverage["observed_detector_records"], 0)
            self.assertEqual(coverage["missing_condition_count"], 1)

    def test_canonical_grid_has_3840_nonredundant_conditions(self):
        options = {
            "audit_only": False,
            "datasets": ["amazon", "dblp", "livejournal", "youtube"],
            "cover": "top5000",
            "policy": "covered-induced",
            "phases": ["local", "multiphase"],
            "isolation_policies": ["fixed_labels", "open_labels"],
            "multipliers": [1.0, 10.0, 100.0],
            "detector_seeds": [0, 1, 2, 3, 4],
            "perturbation_distances": [0.0, 0.005, 0.02, 0.05],
            "perturbation_seeds": [100, 101, 102, 103, 104],
        }
        keys = _expected_condition_axis_keys(options)
        self.assertEqual(len(keys), 3840)

    def test_ground_truth_runner_rejects_foreign_identity_records(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "artifacts"
            common = [
                "--smoke", "--output-dir", str(output), "--phases", "local",
                "--isolation-policies", "fixed_labels", "--seeds", "0",
                "--resolution-multipliers", "1", "--perturbation-distances", "0",
                "--timeout-per-run", "30",
            ]
            self.assertEqual(ground_truth_robustness_main(common), 0)
            record_path = next((output / "runs" / "amazon-top5000").glob("*.json"))
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["protocol_identity"] = None
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "foreign or legacy"):
                ground_truth_robustness_main(common + ["--rescore-only"])

    def test_corrupt_membership_artifact_is_rerun_and_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "artifacts"
            common = [
                "--smoke", "--output-dir", str(output), "--phases", "local",
                "--isolation-policies", "fixed_labels", "--seeds", "0",
                "--resolution-multipliers", "1", "--perturbation-distances", "0",
                "--timeout-per-run", "30",
            ]
            self.assertEqual(ground_truth_robustness_main(common), 0)
            record_path = next((output / "runs" / "amazon-top5000").glob("*.json"))
            record = json.loads(record_path.read_text(encoding="utf-8"))
            digest = record["final_membership_hash"]
            artifact = output / "raw_memberships" / f"{digest}.json.gz"
            artifact.write_bytes(b"corrupt")
            self.assertEqual(ground_truth_robustness_main(common), 0)
            self.assertIsNotNone(_load_membership_artifact(output, digest))

    def test_publication_index_binds_records_and_exact_state_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            cover = [[0, 1]]
            memberships = [[0], [0]]
            cover_digest = _persist_cover(output, cover)
            membership_digest = _persist_memberships(output, memberships, 2)
            identity = {"protocol": "test"}
            record = {
                "condition_key": "condition",
                "condition_axis_key": "axis",
                "dataset": "amazon",
                "cover": "top5000",
                "status": "completed",
                "protocol_identity": identity,
                "initial_cover_hash": cover_digest,
                "ground_truth_cover_hash": cover_digest,
                "final_cover_hash": cover_digest,
                "final_membership_hash": membership_digest,
                "pre_cleanup_membership_hash": membership_digest,
                "robustness": {
                    "selected_policy": {
                        "is_local_equilibrium_at_resolution": True
                    }
                },
            }
            path = output / "runs" / "amazon-top5000" / "condition.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(record), encoding="utf-8")
            row = {
                key: record[key]
                for key in (
                    "condition_key",
                    "condition_axis_key",
                    "dataset",
                    "cover",
                    "status",
                )
            }
            index = _publication_evidence_index(output, [row], identity)
            self.assertTrue(index["valid"], index["reasons"])
            artifact = output / "raw_memberships" / f"{membership_digest}.json.gz"
            artifact.write_bytes(b"corrupt")
            index = _publication_evidence_index(output, [row], identity)
            self.assertFalse(index["valid"])
            self.assertTrue(
                any("invalid_membership_artifact" in value for value in index["reasons"])
            )

    def test_publication_index_accepts_registered_capacity_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            identity = {
                "protocol": "test",
                "effective_grid": {
                    "terminal_outcome_policy": {
                        "allowed_statuses": ["unsupported_cleanup"],
                        "native_label_capacity_is_explicit": True,
                    }
                },
            }
            record = {
                "condition_key": "terminal",
                "condition_axis_key": "terminal-axis",
                "dataset": "youtube",
                "cover": "top5000",
                "status": "unsupported_cleanup",
                "error_kind": "native_label_capacity",
                "protocol_identity": identity,
            }
            path = output / "runs" / "youtube-top5000" / "terminal.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(record), encoding="utf-8")
            row = {
                key: record[key]
                for key in (
                    "condition_key",
                    "condition_axis_key",
                    "dataset",
                    "cover",
                    "status",
                )
            }
            index = _publication_evidence_index(output, [row], identity)
            self.assertTrue(index["valid"], index["reasons"])

    def test_subprocess_interrupt_terminates_worker_and_removes_packet(self):
        class FakeProcess:
            def __init__(self):
                self.alive = False
                self.terminated = False

            def start(self):
                self.alive = True

            def join(self, _timeout=None):
                if not self.terminated:
                    raise KeyboardInterrupt

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.terminated = True
                self.alive = False

            def kill(self):
                self.alive = False

        process = FakeProcess()

        class FakeContext:
            def Process(self, **_kwargs):
                return process

        with tempfile.TemporaryDirectory() as d, patch.object(
            gt_execution.mp, "get_context", return_value=FakeContext()
        ):
            with self.assertRaises(KeyboardInterrupt):
                gt_execution.run_in_subprocess(
                    int,
                    "1",
                    timeout_seconds=30,
                    packet_dir=d,
                )
            self.assertTrue(process.terminated)
            self.assertEqual(list(Path(d).glob("hedonic-gt-worker-*.pkl")), [])


class TestNoOverlappingModule(unittest.TestCase):
    def test_overlapping_py_removed(self):
        path = Path(__file__).resolve().parents[1] / "src" / "hedonic" / "overlapping.py"
        self.assertFalse(path.exists(), "overlapping.py should be deleted")

    def test_package_exports_only_game(self):
        import hedonic

        self.assertTrue(hasattr(hedonic, "Game"))
        self.assertFalse(hasattr(hedonic, "OverlappingGame"))


if __name__ == "__main__":
    unittest.main()
