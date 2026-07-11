"""Unit tests for overlapping Leiden (max_memberships) and experiments package."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import igraph as ig

from hedonic import Game
from hedonic.experiments import CLI
from hedonic.experiments.config import (
    DEFAULT_DBLP_DIR,
    DEFAULT_SYNTHETIC_DIR,
    reload_paths,
)
from hedonic.experiments.disjoint import data_loader, sbm_sweep
from hedonic.experiments.overlapping import small_graphs
from hedonic.experiments.overlapping import complexity_scale
from hedonic.experiments.overlapping import resolution_f1
from hedonic.experiments.overlapping.dblp_full import (
    resolve_max_memberships as resolve_mm_full,
)
from hedonic.experiments.overlapping.dblp_subgraph import (
    resolve_max_memberships as resolve_mm_sub,
)
from hedonic.experiments.overlapping.metrics import (
    cover_quality,
    evaluate_cover,
    partition_to_cover_lists,
)


class TestCommunityHedonic(unittest.TestCase):
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
        self.assertEqual(
            str(DEFAULT_DBLP_DIR),
            "~/Databases/Hedonic/Networks/DBLP",
        )
        self.assertEqual(
            str(DEFAULT_SYNTHETIC_DIR),
            "~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020",
        )

    def test_env_override(self):
        with patch.dict(
            os.environ,
            {
                "HEDONIC_DBLP_DIR": "/tmp/custom_dblp",
                "HEDONIC_SYNTHETIC_DIR": "/tmp/custom_synth",
            },
            clear=False,
        ):
            dblp, synth = reload_paths()
            self.assertEqual(str(dblp), "/tmp/custom_dblp")
            self.assertEqual(str(synth), "/tmp/custom_synth")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HEDONIC_DBLP_DIR", None)
            os.environ.pop("HEDONIC_SYNTHETIC_DIR", None)
            reload_paths()


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
                    "~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020",
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
            "overlapping-subgraph",
            "overlapping-full",
            "overlapping-scale",
            "overlapping-resolution",
        }
        self.assertEqual(set(CLI.COMMANDS), expected)

    def test_overlapping_resolution_help(self):
        code = CLI.main(["overlapping-resolution", "--help"])
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
            self.assertFalse(data["meta"]["only_local_moving"])
            self.assertTrue(data["meta"]["allow_isolation"])
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
                self.assertFalse(row["only_local_moving"])
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
            self.assertIn(rec["only_local_moving"], (True, False))
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
            by_olm[bool(p["only_local_moving"])].append(p)
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
        self.assertTrue(all(p["only_local_moving"] is False for p in result.points))
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
        self.assertFalse(rec["only_local_moving"])
        self.assertTrue(rec["allow_isolation"])
        self.assertEqual(rec["max_memberships"], k)
        self.assertIn("f1", rec)
        self.assertGreaterEqual(rec["f1"], 0.0)
        self.assertLessEqual(rec["f1"], 1.0)

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
        self.assertFalse(result.meta["only_local_moving"])
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
            self.assertFalse(data["meta"]["only_local_moving"])
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

