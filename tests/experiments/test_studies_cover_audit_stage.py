"""The robustness stage of a benchmark run (``cover_audit.Stage``): each saved cover is audited in a child process and
recorded in the ledger ``hedonic run audit`` writes; the worker hears the audit's progress while it waits."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments.overlapping import cover_audit as ca
from tests.experiments.test_studies_cover_audit import CLIQUES, two_cliques, write_unit


def build_run(folder: Path, methods=("hedonic", "clique_percolation")) -> None:
    """A benchmark run folder with one finished unit per method on the two-cliques graph (a HOC unit keeps its labels)."""
    graph = two_cliques()
    labelled = [[0], [0], [0], [0], [1], [1], [1], [1], [2]]
    for method in methods:
        write_unit(folder, "dblp", 0, method, graph, cover=CLIQUES,
                   labelled=labelled if method == "hedonic" else None)


class TestStage(unittest.TestCase):
    def test_a_unit_is_audited_in_a_child_process_and_recorded_where_hedonic_run_audit_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            build_run(run)
            stage = ca.Stage(run, ["dblp"], ["hedonic", "clique_percolation"], [0])
            with patch.object(ca, "load_graph", lambda network, variant, root, induce="full": two_cliques()):
                measure = stage.audit("dblp", 0, "hedonic")
                self.assertEqual(set(measure), {"log", "linear"})
                self.assertTrue(0 < measure["log"] <= 1)
                self.assertTrue(stage.audited("dblp", 0, "hedonic"))
                self.assertFalse(stage.audited("dblp", 0, "clique_percolation"))
                stage.audit("dblp", 0, "clique_percolation")
                stage.finish()
            ledger = run / ca.STAGE_DIR / ca.RESULTS_NAME
            keys = [json.loads(line)["key"] for line in ledger.read_text().splitlines()]
            self.assertEqual(keys, ["dblp/seed0/hedonic/cover", "dblp/seed0/hedonic/labelled", "dblp/seed0/clique_percolation/cover"])
            self.assertTrue((run / ca.STAGE_DIR / ca.TABLE_NAME).is_file())
            self.assertTrue((run / ca.STAGE_DIR / ca.SUMMARY_NAME).is_file())
            # the benchmark's own summary.json is not the audit's
            self.assertFalse((run / "summary.json").exists())
            # a second stage reads the ledger and audits nothing
            again = ca.Stage(run, ["dblp"], ["hedonic", "clique_percolation"], [0])
            with patch.object(ca.Stage, "_in_child", side_effect=AssertionError("audited again")):
                self.assertEqual(again.audit("dblp", 0, "clique_percolation"), stage.measure("dblp", 0, "clique_percolation"))
            # `hedonic run audit` and the stage share the ledger: the standalone tool finds nothing left to do
            plan = ca.resolve(["--source", str(run), "--networks", "dblp", "--methods", "hedonic,clique_percolation", "--seeds", "0",
                               "--output-dir", str(run / ca.STAGE_DIR)])
            self.assertEqual(ca.state_of(plan), "complete")

    def test_a_unit_whose_detector_failed_is_a_finished_audit_without_a_measure(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            write_unit(run, "dblp", 0, "clique_percolation", two_cliques(), status="timeout")
            stage = ca.Stage(run, ["dblp"], ["clique_percolation"], [0])
            with patch.object(ca, "load_graph", side_effect=AssertionError("no graph is needed")):
                self.assertIsNone(stage.audit("dblp", 0, "clique_percolation"))
            stage.close()
            self.assertTrue(stage.audited("dblp", 0, "clique_percolation"))


def slow_audit_in_child(plan, todo):
    """Stands in for the audit process: reports twice, then returns a finished record for each condition."""
    import time as _time

    records = []
    for condition in todo:
        sink = ca.progress_sink(plan["output"] / ca.PROGRESS_NAME, condition["key"])
        sink(ca.vertices_event("", "fixed_labels", 1, 2, 40, 100, 1.0))
        _time.sleep(0.4)
        sink(ca.vertices_event("", "open_labels", 2, 2, 100, 100, 5.0))
        _time.sleep(0.3)
        records.append({"key": condition["key"], "status": "no_cover", "reason": "stand-in"})
    return records


class TestAuditReporting(unittest.TestCase):
    def test_the_worker_hears_the_audit_progress_while_it_waits_for_the_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            build_run(run, methods=("clique_percolation",))
            stage = ca.Stage(run, ["dblp"], ["clique_percolation"], [0])
            heard = []
            stage.on_progress = heard.append
            with patch.object(ca, "audit_in_child", slow_audit_in_child), patch.object(ca, "PROGRESS_POLL_SECONDS", 0.05):
                self.assertIsNone(stage.audit("dblp", 0, "clique_percolation"))
            stage.close()
            self.assertTrue(heard, "the worker was never told the audit's progress")
            self.assertEqual({info["key"] for info in heard}, {"dblp/seed0/clique_percolation/cover"})
            self.assertEqual(heard[0]["total"], 100)
            self.assertIn("exact stability", heard[0]["detail"])

    def test_no_hook_means_no_polling_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            build_run(run, methods=("clique_percolation",))
            stage = ca.Stage(run, ["dblp"], ["clique_percolation"], [0])
            with patch.object(ca, "audit_in_child", slow_audit_in_child), patch.object(ca, "PROGRESS_POLL_SECONDS", 0.05), \
                    patch.object(ca, "read_audit_progress", side_effect=AssertionError("read without a listener")):
                stage.audit("dblp", 0, "clique_percolation")
            stage.close()


if __name__ == "__main__":
    unittest.main()
