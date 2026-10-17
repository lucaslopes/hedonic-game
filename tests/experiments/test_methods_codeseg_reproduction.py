from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from hedonic.experiments.overlapping import codeseg_reproduction
from hedonic.experiments.overlapping.metrics import (
    overlapping_normalized_mutual_information_lfk,
)


class TestCoDeSEGMetrics(unittest.TestCase):
    def test_lfk_onmi_known_cover_cases(self):
        self.assertAlmostEqual(
            overlapping_normalized_mutual_information_lfk([[0, 1, 2]], [[0, 1, 2]]),
            1.0,
        )
        self.assertAlmostEqual(
            overlapping_normalized_mutual_information_lfk(
                [[0, 1, 2]], [[0, 1], [2, 3]]
            ),
            0.26966380441295534,
        )
        self.assertAlmostEqual(
            overlapping_normalized_mutual_information_lfk(
                [[0, 1], [1, 2]], [[0, 1], [2, 3]]
            ),
            0.5,
        )

    def test_lfk_onmi_is_bounded(self):
        value = overlapping_normalized_mutual_information_lfk(
            [[0, 1], [2, 3]], [[0, 2], [1, 3]]
        )
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


class TestCoDeSEGRunner(unittest.TestCase):
    def test_protocol_registers_the_nine_paper_methods_and_local_comparator(self):
        self.assertEqual(
            codeseg_reproduction.PAPER_METHODS,
            (
                "codeseg",
                "slpa",
                "bigclam",
                "ncgame",
                "fox",
                "louvain",
                "der",
                "leiden",
                "flpa",
            ),
        )
        # hedonic 2.0: community_hedonic and the multi-phase variants are gone.
        self.assertEqual(codeseg_reproduction.METHODS, codeseg_reproduction.PAPER_METHODS)
        self.assertEqual(
            codeseg_reproduction.EXTENDED_METHODS,
            codeseg_reproduction.METHODS
            + (
                "hedonic_local",
                "angel",
                "infomap",
                "demon",
                "clique_percolation",
                "link_clustering",
                "oslom",
                "neo_kmeans",
                "nise",
                "sse",
                "qoce",
                "svi",
                "essc",
            ),
        )
        defaults = codeseg_reproduction._build_parser().parse_args([]).methods.split(",")
        self.assertEqual(defaults, list(codeseg_reproduction.DBLP_METHODS))
        self.assertNotIn("oslom", codeseg_reproduction.DBLP_METHODS)
        self.assertEqual(len(codeseg_reproduction.DBLP_METHODS), 21)
        self.assertFalse(set(codeseg_reproduction.DBLP_METHODS) & codeseg_reproduction.REMOVED_METHODS)
        for method in ("nise", "sse", "qoce", "svi", "essc"):
            self.assertIn(method, codeseg_reproduction.DBLP_METHODS)

    def test_the_opt_in_methods_are_appended_by_install_and_removed_by_uninstall(self):
        from hedonic.experiments.overlapping import extra_methods

        before = (codeseg_reproduction.EXTENDED_METHODS, dict(codeseg_reproduction.METHOD_SPECS),
                  codeseg_reproduction._run_method)
        extra_methods.install()
        extra_methods.install()  # idempotent
        opt_in = (*extra_methods.EXTRA, *extra_methods.CONTROLS)
        self.assertEqual(codeseg_reproduction.EXTENDED_METHODS, before[0] + opt_in)
        self.assertTrue(set(opt_in) <= set(codeseg_reproduction.METHOD_SPECS))
        extra_methods.uninstall()
        self.assertEqual((codeseg_reproduction.EXTENDED_METHODS, dict(codeseg_reproduction.METHOD_SPECS),
                          codeseg_reproduction._run_method), before)

    def test_uninstall_never_restores_a_hook_patched_in_while_installing(self):
        from unittest.mock import patch

        from hedonic.experiments.overlapping import extra_methods

        real = codeseg_reproduction._run_method
        with patch.object(codeseg_reproduction, "_run_method", lambda *a: ([], {})):
            extra_methods.install()  # wraps the patched hook; the patch's exit then puts the real one back
        self.assertIs(codeseg_reproduction._run_method, real)
        extra_methods.uninstall()  # must not bring the patch back
        self.assertIs(codeseg_reproduction._run_method, real)

    def test_randomized_cover_preserves_ground_truth_shape(self):
        initial, metadata = codeseg_reproduction._randomized_cover_initialization(
            [[0, 1], [1, 2]],
            n_vertices=3,
            seed=0,
        )
        self.assertEqual(len(initial), 3)
        self.assertEqual(metadata["initial_community_count"], 2)
        self.assertEqual(metadata["ground_truth_community_count"], 2)
        self.assertEqual(metadata["ground_truth_max_memberships_per_vertex"], 2)
        self.assertTrue(all(initial_row for initial_row in initial))
        self.assertLessEqual(
            max(len(initial_row) for initial_row in initial),
            metadata["ground_truth_max_memberships_per_vertex"],
        )

    def test_hedonic_adapter_cap_never_reads_community_sizes(self):
        # Largest community size is 5; the largest per-vertex multiplicity
        # is 3 (vertex 0).  The protocol-v3 expression returned min(8, 5) = 5.
        reference = [[0, 1, 2, 3, 4], [0, 5], [0, 6]]
        cap, source = codeseg_reproduction._hedonic_adapter_cap({}, reference, 7)
        self.assertEqual(cap, codeseg_reproduction.HEDONIC_REFERENCE_FREE_MAX_MEMBERSHIPS)
        self.assertEqual(source, "reference_free_registered_default")
        # The default must not depend on the reference at all.
        self.assertEqual(
            codeseg_reproduction._hedonic_adapter_cap({}, [], 7),
            (codeseg_reproduction.HEDONIC_REFERENCE_FREE_MAX_MEMBERSHIPS,
             "reference_free_registered_default"),
        )
        cap, source = codeseg_reproduction._hedonic_adapter_cap(
            {"max_memberships": codeseg_reproduction.HEDONIC_REFERENCE_MULTIPLICITY_CAP},
            reference,
            7,
        )
        self.assertEqual((cap, source), (3, "reference_max_multiplicity_ablation"))
        self.assertEqual(
            codeseg_reproduction._hedonic_adapter_cap({"max_memberships": 64}, reference, 7),
            (64, "user_specified"),
        )
        with self.assertRaises(ValueError):
            codeseg_reproduction._hedonic_adapter_cap({"max_memberships": 1}, reference, 7)

    def test_disjoint_control_is_an_explicit_cap_one(self):
        self.assertEqual(
            codeseg_reproduction._hedonic_adapter_cap(
                {"max_memberships": 1, "disjoint_control": True}, [], 7),
            (1, "disjoint_control"),
        )
        self.assertEqual(
            codeseg_reproduction._hedonic_adapter_cap({"disjoint_control": True}, [], 7),
            (1, "disjoint_control"),
        )
        with self.assertRaises(ValueError):
            codeseg_reproduction._hedonic_adapter_cap(
                {"max_memberships": 2, "disjoint_control": True}, [], 7)

    def test_disjoint_control_adapters_return_one_community_per_vertex(self):
        import igraph as ig
        from hedonic.experiments.overlapping.methods import run_method

        graph = ig.Graph.Famous("Zachary")
        parameters = {"absolute_resolution": 0.1, "disjoint_control": True}
        for name in ("hedonic_local",):  # hedonic_multiphase* were removed
            cover, _ = run_method(codeseg_reproduction.OVERLAPPING_METHODS[name], graph, max_memberships=1,
                                  resolution=float(graph.density()), seed=0, parameters=dict(parameters))
            counts = {}
            for community in cover:
                for vertex in community:
                    counts[vertex] = counts.get(vertex, 0) + 1
            self.assertEqual(len(counts), graph.vcount(), name)
            self.assertEqual(max(counts.values()), 1, name)

    def test_hedonic_cap_flag_reaches_every_hedonic_adapter(self):
        parser = codeseg_reproduction._build_parser()
        args = parser.parse_args(["--hedonic-max-memberships", "64"])
        parameters = codeseg_reproduction._method_parameters(args)
        self.assertEqual(parameters["hedonic_local"], {"max_memberships": 64})
        for name in codeseg_reproduction.REMOVED_METHODS:
            self.assertNotIn(name, parameters)
        default = codeseg_reproduction._method_parameters(parser.parse_args([]))
        self.assertEqual(default["hedonic_local"], {})
        with self.assertRaises(ValueError):
            codeseg_reproduction._method_parameters(
                parser.parse_args(["--hedonic-max-memberships", "1"])
            )

    def test_table2_report_marks_best_and_second_best(self):
        records = [
            {"dataset": "dblp", "method": "a", "status": "completed", "metrics": {"onmi": 0.8, "f1": 0.7}},
            {"dataset": "dblp", "method": "b", "status": "completed", "metrics": {"onmi": 0.6, "f1": 0.9}},
            {"dataset": "dblp", "method": "c", "status": "completed", "metrics": {"onmi": 0.4, "f1": 0.5}},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "table2"
            codeseg_reproduction._write_table2(path, records, ["a", "b", "c"])
            markdown = path.with_suffix(".md").read_text()
            self.assertIn("**90.00**", markdown)
            self.assertIn("<u>60.00</u>", markdown)
            self.assertTrue(path.with_suffix(".tex").is_file())
    def test_native_smoke_writes_metrics_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "codeseg"
            code = codeseg_reproduction.main(
                [
                    "--smoke",
                    "--datasets",
                    "dblp",
                    "--methods",
                    "louvain,leiden,flpa",
                    "--output-dir",
                    str(output),
                    "--seed",
                    "0",
                ]
            )
            self.assertEqual(code, 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status_counts"], {"completed": 3})
            self.assertEqual(
                {record["method"] for record in manifest["records"]},
                {"louvain", "leiden", "flpa"},
            )
            for record in manifest["records"]:
                self.assertGreaterEqual(record["metrics"]["f1"], 0.0)
                self.assertLessEqual(record["metrics"]["f1"], 1.0)
                self.assertGreaterEqual(record["metrics"]["onmi"], 0.0)
                self.assertLessEqual(record["metrics"]["onmi"], 1.0)
            with (output / "metrics.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertIn("f1_percent", rows[0])
            self.assertIn("onmi_percent", rows[0])


if __name__ == "__main__":
    unittest.main()
