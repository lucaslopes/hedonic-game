from __future__ import annotations

import csv
import json
import tempfile
import unittest
import tempfile
from pathlib import Path
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
        self.assertEqual(
            codeseg_reproduction.METHODS,
            codeseg_reproduction.PAPER_METHODS + ("community_hedonic",),
        )
        self.assertEqual(
            codeseg_reproduction.EXTENDED_METHODS,
            codeseg_reproduction.METHODS
            + (
                "hedonic_local",
                "hedonic_multiphase",
                "hedonic_multiphase_x10",
                "hedonic_multiphase_x100",
                "angel",
                "infomap",
                "demon",
                "cpm",
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
        self.assertEqual(len(codeseg_reproduction.DBLP_METHODS), 25)
        for method in ("nise", "sse", "qoce", "svi", "essc"):
            self.assertIn(method, codeseg_reproduction.DBLP_METHODS)

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

    def test_hedonic_cap_flag_reaches_every_hedonic_adapter(self):
        parser = codeseg_reproduction._build_parser()
        args = parser.parse_args(["--hedonic-max-memberships", "64"])
        parameters = codeseg_reproduction._method_parameters(args)
        for name in ("hedonic_local", "hedonic_multiphase",
                     "hedonic_multiphase_x10", "hedonic_multiphase_x100"):
            self.assertEqual(parameters[name], {"max_memberships": 64})
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
