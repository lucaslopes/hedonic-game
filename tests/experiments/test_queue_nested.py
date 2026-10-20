"""A queue item that is a whole queue (`run = ["queue", FILE|NAME]`): read, judged, started and drawn as one run."""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments.overlapping import runmanager as rm
from hedonic.experiments.overlapping import runqueue


def _coverage(folder: Path, expected: int, recorded: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "coverage_report.json").write_text(json.dumps(
        {"complete": recorded >= expected, "expected_detector_conditions": expected,
         "recorded_detector_conditions": recorded}))


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._env = patch.dict(os.environ, {"HEDONIC_RUNS_DIR": str(self.root / "registry")})
        self._env.start()
        runqueue._DATA_CACHE.clear()
        self.inner = self.root / "inner.toml"
        self.inner.write_text('name = "inner-queue"\n[[items]]\nname = "s1"\n'
                              'run = ["spectrum", "--name", "s1", "--output-dir", "{root}/s1"]\n'
                              '[[items]]\nname = "s2"\nrun = ["spectrum", "--name", "s2", "--output-dir", "{root}/s2"]\n')
        self.outer = self.root / "outer.toml"

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def _outer(self, item: str) -> dict:
        self.outer.write_text('name = "outer-queue"\n' + item)
        return runqueue.load_queue(str(self.outer), root=str(self.root / "outer"))


class TestLoad(_Base):
    def test_the_item_runs_under_the_inner_queues_name_with_its_file_and_root(self):
        queue = self._outer(f'[[items]]\nrun = ["queue", "{self.inner}", "--root", "{{root}}/in"]\n')
        item = queue["items"][0]
        self.assertEqual(item["name"], "inner-queue")
        self.assertEqual(item["run"][:2], ["queue", str(self.inner.resolve())])
        self.assertEqual(runqueue._option(item["run"], "--root"), str((self.root / "outer" / "in").resolve()))
        self.assertEqual([i["name"] for i in item["queue"]["items"]], ["s1", "s2"])

    def test_a_wrong_name_and_a_queue_that_contains_itself_are_refused(self):
        with self.assertRaisesRegex(SystemExit, "must be that name"):
            self._outer(f'[[items]]\nname = "other"\nrun = ["queue", "{self.inner}"]\n')
        self.outer.write_text(f'name = "outer-queue"\n[[items]]\nrun = ["queue", "{self.outer}"]\n')
        with self.assertRaisesRegex(SystemExit, "runs itself"):
            runqueue.load_queue(str(self.outer), root=str(self.root / "outer"))

    def test_the_shipped_all_queue_runs_the_overlap_paper_then_the_disjoint_one(self):
        queue = runqueue.load_queue("all", root=str(self.root / "all"))
        self.assertEqual([i["name"] for i in queue["items"]], ["overlap-queue", "disjoint-queue"])
        pairs, methods = runqueue.requirements(queue)  # the setup check covers what both papers need
        self.assertIn("friendster:top5000", pairs)
        self.assertIn("angel", methods)


class TestState(_Base):
    def test_the_item_is_complete_when_every_run_of_its_queue_is(self):
        run = self._outer(f'[[items]]\nrun = ["queue", "{self.inner}", "--root", "{self.root / "in"}"]\n')["items"][0]["run"]
        self.assertEqual(runqueue.data_state(run, fresh=True)[0], "missing")
        _coverage(self.root / "in" / "s1", 2, 2)
        self.assertEqual(runqueue.data_state(run, fresh=True)[0], "partial")
        self.assertEqual(runqueue.unit_counts(run), (1, 2, "runs"))
        _coverage(self.root / "in" / "s2", 2, 2)
        runqueue._DATA_CACHE.clear()
        self.assertEqual(runqueue.data_state(run, fresh=True)[0], "complete")


class TestWorkerAndTree(_Base):
    def test_the_worker_starts_the_inner_queue_and_waits_for_it_then_goes_on(self):
        queue = self._outer(f'[[items]]\nrun = ["queue", "{self.inner}", "--root", "{self.root / "in"}"]\n'
                            f'[[items]]\nname = "after"\nrun = ["spectrum", "--name", "after", "--output-dir", '
                            f'"{self.root / "after"}"]\n')
        calls = []

        def launch(name, run, resuming):
            calls.append((name, run[0]))
            folders = ["s1", "s2"] if run[0] == "queue" else ["../after"]
            for folder in folders:  # the started run finishes its work
                _coverage((self.root / "in" / folder).resolve(), 1, 1)
            out = self.root / "runs" / name
            out.mkdir(parents=True, exist_ok=True)
            items = runqueue.inner_queue(run)["items"] if run[0] == "queue" else []
            entry = {"name": name, "kind": "queue" if run[0] == "queue" else "spectrum", "items": items, "output_dir": str(out),
                     "created": time.time(), "session": "", "backend": "process", "log": str(out / "w.log"), "argv": []}
            rm.write_json(rm.entry_path(name), entry)
            rm.write_json(rm.progress_path(entry), {"status": "completed", "plan": [], "setup": []})
            runqueue._DATA_CACHE.clear()
            return 0

        out = self.root / "queues" / "outer"
        out.mkdir(parents=True)
        entry = {"name": "outer-queue", "kind": "queue", "items": queue["items"], "file": queue["file"],
                 "output_dir": str(out), "created": time.time(), "session": "", "backend": "process",
                 "log": str(out / "worker.log"), "schedule": "file"}
        with patch.object(runqueue, "POLL_SECONDS", 0), redirect_stdout(io.StringIO()):
            self.assertEqual(runqueue.work("outer-queue", entry, {}, launch_item=launch), 0)
        self.assertEqual(calls, [("inner-queue", "queue"), ("after", "spectrum")])

    def test_the_tree_shows_the_inner_queues_runs_under_its_item(self):
        queue = self._outer(f'[[items]]\nrun = ["queue", "{self.inner}", "--root", "{self.root / "in"}"]\n')
        _coverage(self.root / "in" / "s1", 2, 2)
        root = runqueue.tree({"name": "outer-queue", "items": queue["items"], "file": queue["file"]}, {})
        item = root.children[0]
        self.assertEqual([c.label for c in item.children], ["1. s1", "2. s2"])
        self.assertEqual((item.done, item.total, item.noun), (1, 2, "runs"))
        self.assertTrue(all(c.key[:2] == ("in", "inner-queue") for c in item.children))  # keys unique in the tree
        self.assertEqual([c.depth for c in item.children], [2, 2])


if __name__ == "__main__":
    unittest.main()
