"""Unit tests for overlapping Leiden (max_memberships) and experiments package."""

from __future__ import annotations

import json
import os
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
            "overlapping-small",
            "overlapping-subgraph",
            "overlapping-full",
        }
        self.assertEqual(set(CLI.COMMANDS), expected)

    def test_overlapping_small_via_cli(self):
        code = CLI.main(["overlapping-small"])
        self.assertEqual(code, 0)

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


class TestSmallGraphsModule(unittest.TestCase):
    def test_run_tests_entrypoint(self):
        small_graphs.main([])


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
