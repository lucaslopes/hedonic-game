"""``hedonic detect``: one network x one method x one seed, scored against its ground truth.

The disjoint part is the same file in hedonic 1 and 2; the overlap part (SNAP pairs, LFR-derived overlap, covers,
the baselines) exists only where ``overlapping/detect_overlap.py`` does, so its tests skip elsewhere."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from hedonic.experiments import CLI, detect
from hedonic.experiments.overlapping import tui

OVERLAP = importlib.util.find_spec("hedonic.experiments.overlapping.detect_overlap") is not None


def run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = detect.main(argv)
    return code, out.getvalue(), err.getvalue()


class TestSpecs(unittest.TestCase):
    def test_a_spec_names_a_registry_entry_and_typed_values(self):
        source, given = detect.parse_spec("sbm:k=3,p_in=0.4", detect.SOURCES, "network source")
        values = detect.resolve(source.params, given, "source sbm")
        self.assertEqual((source.key, values["k"], values["p_in"]), ("sbm", 3, 0.4))
        with self.assertRaisesRegex(ValueError, "sbm"):
            detect.resolve(source.params, {"nonsense": "1"}, "source sbm")
        with self.assertRaises(ValueError):
            detect.parse_spec("nosuch", detect.SOURCES, "network source")

    def test_the_same_run_line_rebuilds_the_same_plan(self):
        args = detect.build_parser().parse_args(["--graph", "sbm:k=3", "--method", "leiden:resolution=0.2",
                                                 "--seed", "4", "--no-save"])
        plan = detect.plan_from_args(args)
        again = detect.plan_from_args(detect.build_parser().parse_args(plan.argv()[1:]))
        self.assertEqual((again.source_values, again.method_values, again.seed),
                         (plan.source_values, plan.method_values, plan.seed))

    def test_it_is_a_top_level_command_and_lists_everything(self):
        self.assertIn("detect", CLI.TOP_LEVEL)
        code, out, _ = run(["--list"])
        self.assertEqual(code, 0)
        for key in ("sbm", "lfr", "famous", "file", "hedonic", "leiden", "louvain", "onepass", "mirror"):
            self.assertIn(key, out)

    def test_without_a_terminal_it_needs_a_graph(self):
        with patch.object(tui, "interactive", return_value=False):
            code, _, err = run([])
        self.assertEqual(code, 2)
        self.assertIn("--graph is required", err)


class TestDisjoint(unittest.TestCase):
    def test_hedonic_recovers_an_easy_planted_partition_and_saves_the_record(self):
        with tempfile.TemporaryDirectory() as d:
            code, out, _ = run(["--graph", "sbm:k=3,size=30,p_in=0.5,difficulty=0.05", "--output", d])
            self.assertEqual(code, 0, out)
            record = json.loads(open(os.path.join(d, "result.json")).read())
        self.assertAlmostEqual(record["scores"]["ari"], 1.0)
        self.assertEqual(len(record["answer"]["membership"]), 90)
        self.assertIn("hedonic detect --graph sbm:", out)

    def test_a_file_with_labels_is_read_with_its_truth(self):
        with tempfile.TemporaryDirectory() as d:
            graph, truth = os.path.join(d, "g.txt"), os.path.join(d, "t.txt")
            with open(graph, "w") as f:
                f.write("a b\nb c\na c\nd e\ne f\nd f\nc d\n")
            with open(truth, "w") as f:
                f.write("a 0\nb 0\nc 0\nd 1\ne 1\nf 1\n")
            code, out, _ = run(["--graph", f"file:path={graph},truth={truth}", "--method", "leiden:resolution=0.5",
                                "--no-save", "--json"])
        self.assertEqual(code, 0)
        self.assertAlmostEqual(json.loads(out)["scores"]["ari"], 1.0)

    def test_a_method_that_starts_from_the_truth_refuses_a_network_without_one(self):
        code, _, err = run(["--graph", "famous:name=Petersen", "--method", "mirror", "--no-save"])
        self.assertEqual(code, 2)
        self.assertIn("ground truth", err)

    def test_a_backed_out_wizard_runs_nothing_and_exits_0(self):
        with patch.object(tui, "interactive", return_value=True), patch.object(tui, "select", side_effect=tui.Cancelled):
            code, out, _ = run(["-i"])
        self.assertEqual(code, 0)
        self.assertIn("not run", out)
        self.assertTrue(tui.take_cancelled())


@unittest.skipUnless(OVERLAP, "the overlap half of detect is hedonic 2 only")
class TestOverlap(unittest.TestCase):
    def test_hedonic_with_more_labels_returns_a_cover_scored_as_one(self):
        code, out, _ = run(["--graph", "lfr_overlap:n=120,min_community=15,max_community=30",
                            "--method", "hedonic:max_memberships=2,resolution=0.1", "--no-save", "--json"])
        self.assertEqual(code, 0, out)
        record = json.loads(out)
        self.assertEqual(record["source"]["truth"], "cover")
        for key in ("f1", "onmi", "omega", "overlapping_vertices"):
            self.assertIn(key, record["scores"])

    def test_the_overlap_entries_are_registered(self):
        self.assertIn("snap", detect.SOURCES)
        self.assertIn("lfr_overlap", detect.SOURCES)
        self.assertIn("max_memberships", [p.name for p in detect.METHODS["hedonic"].params])
        self.assertIn("codeseg", detect.METHODS)

    def test_a_snap_pair_that_is_not_on_disk_is_refused_without_download(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_NETWORKS_DIR": d}):
            code, _, err = run(["--graph", "snap:pair=amazon:top5000", "--no-save"])
        self.assertEqual(code, 2)
        self.assertIn("download=yes", err)
        self.assertIn("pair=amazon:top5000", detect.spec_text("snap", detect.SOURCES["snap"].params,
                                                              {"pair": "amazon:top5000"}))


if __name__ == "__main__":
    unittest.main()
