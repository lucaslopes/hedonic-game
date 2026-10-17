"""The disjoint SBM sweep: seeded graphs, presets, methods and the smoke pipeline."""
from __future__ import annotations

import json
import random
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hedonic.experiments.disjoint import data_loader, sbm_sweep


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

    def test_generate_graph_golden_fingerprints(self):
        # Pinned outputs of SBM_GENERATOR_ID.  A failure means the sampled
        # benchmark graphs changed: bump SBM_GENERATOR_ID instead of
        # editing these values.
        self.assertEqual(sbm_sweep.SBM_GENERATOR_ID, "hedonic-sbm-pcg64-threshold-v1")
        cases = {
            (2, 12, 0.35, 0.15, 3): (
                55,
                "03a1ebe2722826d83c693be349c34d4487f924cf8d2c85405b1335f30bb67e00",
            ),
            (5, 204, 0.1, 0.75, 99): (
                None,
                "a22da13d47f559f3a1398b5bc1600894b5abe88aa9b3d5a4b7c0ba888e364247",
            ),
        }
        for args, (edges, digest) in cases.items():
            g = sbm_sweep.generate_graph(*args)
            if edges is not None:
                self.assertEqual(g.ecount(), edges)
            self.assertEqual(sbm_sweep.graph_fingerprint(g), digest)

    def test_generate_graph_is_a_pure_function_of_parameters_and_seed(self):
        first = sbm_sweep.generate_graph(3, 20, 0.3, 0.4, seed=5)
        again = sbm_sweep.generate_graph(3, 20, 0.3, 0.4, seed=5)
        self.assertEqual(first.vcount(), again.vcount())
        self.assertEqual(first.get_edgelist(), again.get_edgelist())
        # Parameters are exact decimals: float noise and numpy scalars map
        # to the same graph, and p_out = p_in * difficulty is exact.
        self.assertEqual(
            sbm_sweep.graph_fingerprint(first),
            sbm_sweep.graph_fingerprint(
                sbm_sweep.generate_graph(3, 20, np.float64(0.3), 0.4, seed=5)
            ),
        )
        self.assertEqual(
            sbm_sweep.exact_probability(0.06) * sbm_sweep.exact_probability(0.7),
            sbm_sweep.Fraction(42, 1000),
        )
        # Seed and every parameter select a different stream.
        base = sbm_sweep.graph_fingerprint(first)
        for args in [
            (3, 20, 0.3, 0.4, 6),
            (3, 20, 0.31, 0.4, 5),
            (3, 20, 0.3, 0.41, 5),
            (4, 15, 0.3, 0.4, 5),
        ]:
            self.assertNotEqual(
                sbm_sweep.graph_fingerprint(sbm_sweep.generate_graph(*args)), base
            )

    def test_generate_graph_leaves_global_random_state_alone(self):
        np_state = np.random.get_state()
        py_state = random.getstate()
        sbm_sweep.generate_graph(2, 10, 0.5, 0.5, seed=1)
        self.assertEqual(random.getstate(), py_state)
        after = np.random.get_state()
        self.assertTrue(np.array_equal(after[1], np_state[1]))
        self.assertEqual(after[2:], np_state[2:])

    def test_sbm_edges_boundaries_and_ordering(self):
        complete = sbm_sweep.sbm_edges(2, 3, 1, 1, seed=0)
        self.assertEqual(len(complete), 15)
        edges = sbm_sweep.sbm_edges(3, 10, 0.5, 0.5, seed=2)
        self.assertTrue(np.all(edges[:, 0] < edges[:, 1]))
        self.assertEqual(
            [tuple(e) for e in edges], sorted(tuple(e) for e in edges)
        )
        # p_out = 0 keeps every edge inside its planted block.
        planted = sbm_sweep.sbm_edges(3, 10, 0.8, 0.0, seed=4)
        self.assertTrue(np.all(planted[:, 0] // 10 == planted[:, 1] // 10))
        for bad in [(2, 3, 1.5, 0.1, 0), (2, 3, 0.5, 0.1, -1), (2, 3, 0.5, 3.0, 0)]:
            with self.assertRaises(ValueError):
                sbm_sweep.sbm_edges(*bad)

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


