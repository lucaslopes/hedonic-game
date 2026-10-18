"""`hedonic run queue`: what counts as finished on disk, the queue worker's two verifications, and queue files.

The worker and the views know nothing of a run's kind: they ask ``runqueue._data_state`` what the files say. The tests
stand in a small data rule for it (``fake``: a coverage report in the run's folder), so the queue machinery is checked
apart from the SBM sweep; the disjoint rule itself is checked by the shipped disjoint queue below.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments.overlapping import runmanager as rm
from hedonic.experiments.overlapping import runqueue


def _register(root: Path, name: str, status: str) -> None:
    """A registered run with the given status, standing in for the worker that a launch would start."""
    out = root / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    entry = {"name": name, "kind": "disjoint", "output_dir": str(out), "created": time.time(), "session": "",
             "backend": "process", "log": str(out / "worker.log"), "argv": []}
    rm.write_json(rm.entry_path(name), entry)
    rm.write_json(rm.progress_path(entry), {"status": status, "plan": [], "setup": []})


class _Launcher:
    """Stands in for `hedonic run ... --detach` and `hedonic run resume`: the launched run ends with ``outcome``."""

    def __init__(self, root: Path, outcome: str = "completed"):
        self.root, self.outcome, self.calls = root, outcome, []

    def __call__(self, name: str, run: list[str], resuming: bool) -> int:
        self.calls.append((name, "resume" if resuming else "start"))
        _register(self.root, name, self.outcome)
        return 0


def _coverage(folder: Path, expected: int, recorded: int, stale: str = "") -> None:
    """The files of a ``fake`` run: how many of its units are recorded."""
    folder.mkdir(parents=True, exist_ok=True)
    report = {"expected": expected, "recorded": recorded, "stale": stale}
    (folder / "coverage_report.json").write_text(json.dumps(report))


def _fake_rule(out: Path):
    """A data rule for the ``fake`` verb: its folder is ``--output-dir`` (else ``out/--name``) and its count the
    coverage report there; other verbs have no rule (their registry decides), as in ``runqueue._data_state``."""
    real = runqueue._data_state

    def rule(run: list[str]) -> tuple[str, str]:
        if run[0] != "fake":
            return real(run)
        folder = runqueue._option(run, "--output-dir") or str(out / runqueue._option(run, "--name"))
        report = runqueue._read(Path(folder) / "coverage_report.json")
        if report is None:
            runqueue._note(folder, 0, 0, "units")
            return "missing", folder
        done = 0 if report["stale"] else report["recorded"]
        flags = [i < done for i in range(report["expected"])]
        runqueue._note(folder, done, report["expected"], "units", [("all", "units", flags)], stale=report["stale"])
        if done >= report["expected"]:
            return "complete", folder
        return ("partial" if done or report["stale"] else "missing"), folder

    return rule


def _queue(root: Path, items: list[tuple[str, list[str]]]) -> dict:
    out = root / "queues" / "check"
    out.mkdir(parents=True, exist_ok=True)
    return {"name": "check", "kind": "queue", "items": [{"name": n, "run": r} for n, r in items], "file": "",
            "output_dir": str(out), "created": time.time(), "session": "", "backend": "process",
            "log": str(out / "worker.log")}


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.out = self.root / "out"  # the base folder of a fake run named without --output-dir
        self._patches = [
            patch.dict(os.environ, {"HEDONIC_RUNS_DIR": str(self.root / "registry")}),
            patch("hedonic.experiments.overlapping.userconfig.defaults",
                  return_value={"disjoint_output_root": str(self.root / "disjoint")}),
            patch.object(runqueue, "POLL_SECONDS", 0),
            patch.object(runqueue, "_data_state", _fake_rule(self.out)),
        ]
        for patcher in self._patches:
            patcher.start()
        runqueue._DATA_CACHE.clear()

    def tearDown(self):
        for patcher in reversed(self._patches):
            patcher.stop()
        runqueue._DATA_CACHE.clear()
        self._tmp.cleanup()

    def _state(self, run: list[str]) -> str:
        """What the data on disk says about one item, read afresh (the queue reuses a reading for a minute)."""
        runqueue._DATA_CACHE.clear()
        return runqueue.data_state(run)[0]


class TestWhatIsFinished(_Base):
    def test_units_are_counted_from_the_files(self):
        run = ["fake", "--name", "study"]
        self.assertEqual(self._state(run), "missing")
        _coverage(self.out / "study", expected=4, recorded=1)
        self.assertEqual(self._state(run), "partial")
        self.assertEqual(runqueue.unit_counts(run), (1, 4, "units"))
        _coverage(self.out / "study", expected=4, recorded=4)
        self.assertEqual(self._state(run), "complete")
        self.assertIsNone(runqueue.unit_counts(["x"]))  # a kind without a data rule has no count

    def test_the_overview_lists_every_run_with_what_is_done_and_left(self):
        _coverage(self.out / "done", expected=2, recorded=2)
        _coverage(self.out / "half", expected=4, recorded=1)
        queue = _queue(self.root, [("done", ["fake", "--name", "done"]), ("half", ["fake", "--name", "half"]),
                                   ("none", ["fake", "--name", "none"])])
        runqueue._DATA_CACHE.clear()
        text = "\n".join(runqueue.overview(queue["items"], runqueue.tui.glyphs(False)))
        self.assertIn("100% 2/2 units", text)
        self.assertIn("25% 1/4 units", text)
        self.assertIn("1 of 3 runs complete · 2 still to do", text)
        self.assertEqual(sum(line.lstrip()[:2] in ("1.", "2.", "3.") for line in text.splitlines()), 3)

    def test_the_report_opens_each_run_into_its_groups_and_marks_stale_work(self):
        _coverage(self.out / "done", expected=2, recorded=2)
        _coverage(self.out / "old", expected=4, recorded=4, stale="rows written by another build")
        queue = _queue(self.root, [("done", ["fake", "--name", "done"]), ("old", ["fake", "--name", "old"])])
        queue["items"][0].update(title="Chapter 4", about="what this run is for")
        runqueue._DATA_CACHE.clear()
        text = "\n".join(runqueue.report(queue["items"], runqueue.tui.glyphs(False)))
        self.assertIn("1. done · Chapter 4", text)
        self.assertIn("what this run is for", text)
        self.assertIn("2. old", text)
        self.assertIn("rows written by another build", text)  # stale records are named, and do not count as done
        self.assertIn("0/4 units", text)
        self.assertIn("total: 1 of 2 runs complete · 1 still to do", text)

    def test_the_matrix_puts_every_group_against_every_run_in_a_few_lines(self):
        folder = self.out / "curve"
        runqueue._COUNTS[str(folder)] = (1, 3, "units")
        runqueue._PARTS[str(folder)] = [("Hedonic", "K values", [True, False]), ("Leiden", "K values", [False])]
        item = {"name": "disjoint-paper", "run": ["fake", "--name", "curve"], "short": "paper"}
        with patch.object(runqueue, "data_state", return_value=("partial", str(folder))), \
                patch.object(runqueue, "_status_of", return_value=("stopped", None)):
            lines = runqueue.matrix([item], runqueue.tui.glyphs(False))
        text = "\n".join(lines)
        self.assertEqual(len(lines), 4)  # header, two groups, the totals
        self.assertIn("paper", lines[0])
        self.assertIn("50%", text)  # Hedonic: one of two
        self.assertIn("1/3", lines[-1])
        self.assertTrue(all(len(runqueue.tui.strip_ansi(line)) <= runqueue.WIDTH for line in lines))

    def test_the_view_reads_the_words_of_each_run_from_the_queue_file(self):
        file = self.root / "q.toml"
        file.write_text('name = "check"\n[[items]]\nname = "a"\nrun = ["fake"]\ntitle = "T"\nshort = "S"\n')
        items = runqueue.with_labels([{"name": "a", "run": ["fake"]}, {"name": "b", "run": ["fake"]}], str(file))
        self.assertEqual((items[0]["title"], items[0]["short"]), ("T", "S"))
        self.assertNotIn("title", items[1])
        self.assertEqual(runqueue.with_labels([{"name": "a", "run": []}], str(self.root / "gone.toml")), [{"name": "a", "run": []}])

    def test_a_strip_shades_buckets_when_there_are_more_units_than_marks(self):
        glyphs = runqueue.tui.glyphs(False)
        self.assertEqual(runqueue._strip([True, False, True], glyphs), "#-#")
        long = [True] * 24 + [False] * 24
        strip = runqueue._strip(long, glyphs, width=4)
        self.assertEqual(strip, "##--")
        self.assertEqual(len(runqueue._strip([True] * 100, glyphs)), 24)


class TestQueueWork(_Base):
    def test_finished_items_are_skipped_and_the_others_start_with_their_own_arguments(self):
        _coverage(self.root / "done", expected=2, recorded=2)
        _coverage(self.root / "half", expected=3, recorded=1)
        items = [("done", ["fake", "--name", "done", "--output-dir", str(self.root / "done")]),
                 ("half", ["fake", "--name", "half", "--output-dir", str(self.root / "half")]),
                 ("fresh", ["fake", "--name", "fresh", "--output-dir", str(self.root / "fresh")])]
        launcher = _Launcher(self.root)
        prog: dict = {}
        with contextlib.redirect_stdout(io.StringIO()):
            code = runqueue.work("check", _queue(self.root, items), prog, launch_item=launcher)
        self.assertEqual(code, 0)
        # half's data says partial, so it is started with its own arguments, not resumed from a registry entry
        self.assertEqual(launcher.calls, [("half", "start"), ("fresh", "start")])
        self.assertEqual([row["status"] for row in prog["items"]], ["completed", "completed", "completed"])
        self.assertEqual(prog["items"][0]["note"], "finished earlier, skipped")

    def test_a_registered_run_without_a_data_rule_resumes_from_its_registry_entry(self):
        _register(self.root, "flaky", "failed")
        launcher = _Launcher(self.root)
        prog: dict = {}
        with contextlib.redirect_stdout(io.StringIO()):
            code = runqueue.work("check", _queue(self.root, [("flaky", ["x"])]), prog, launch_item=launcher)
        self.assertEqual(code, 0)
        self.assertEqual(launcher.calls, [("flaky", "resume")])

    def test_completed_with_failures_is_finished_and_the_queue_goes_on(self):
        _register(self.root, "noisy", "completed_with_failures")
        launcher = _Launcher(self.root)
        prog: dict = {}
        items = [("noisy", ["x"]), ("after", ["x"])]
        with contextlib.redirect_stdout(io.StringIO()):
            code = runqueue.work("check", _queue(self.root, items), prog, launch_item=launcher)
        self.assertEqual(code, 0)
        self.assertEqual(launcher.calls, [("after", "start")])  # the failures are results: noisy is not rerun

    def test_data_that_appears_after_the_first_look_is_found_at_the_items_turn(self):
        _coverage(self.root / "first", expected=1, recorded=0)
        items = [("first", ["fake", "--name", "first", "--output-dir", str(self.root / "first")]),
                 ("second", ["fake", "--name", "second", "--output-dir", str(self.root / "second")])]
        launcher = _Launcher(self.root)
        inner = launcher.__call__

        def another_agent_finishes_second(name, run, resuming):
            _coverage(self.root / "second", expected=2, recorded=2)  # produced while the first item ran
            return inner(name, run, resuming)

        prog: dict = {}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = runqueue.work("check", _queue(self.root, items), prog, launch_item=another_agent_finishes_second)
        self.assertEqual(code, 0)
        self.assertEqual(launcher.calls, [("first", "start")])  # second was never launched
        second = prog["items"][1]
        self.assertEqual((second["first"], second["second"]), ("pending", "completed"))
        self.assertIn("another run produced it", second["note"])
        self.assertIn("the data has appeared since", out.getvalue())

    def test_data_that_vanishes_after_the_first_look_runs_again(self):
        done = self.root / "done"
        _coverage(done, expected=1, recorded=1)
        items = [("gone", ["fake", "--name", "gone", "--output-dir", str(done)]),
                 ("after", ["fake", "--name", "after", "--output-dir", str(self.root / "after")])]
        # the first look sees "gone" complete; before its turn the report is replaced by an unfinished one
        real = runqueue.verify

        def look_then_spoil(queue_items):
            rows = real(queue_items)
            _coverage(done, expected=3, recorded=1)
            return rows

        launcher = _Launcher(self.root)
        prog: dict = {}
        with patch.object(runqueue, "verify", look_then_spoil), contextlib.redirect_stdout(io.StringIO()):
            code = runqueue.work("check", _queue(self.root, items), prog, launch_item=launcher)
        self.assertEqual(code, 0)
        self.assertEqual(launcher.calls, [("gone", "start"), ("after", "start")])
        self.assertEqual((prog["items"][0]["first"], prog["items"][0]["second"]), ("completed", "stopped"))

    def test_the_second_look_ignores_the_reading_kept_for_a_minute(self):
        _coverage(self.root / "x", expected=2, recorded=0)
        run = ["fake", "--name", "x", "--output-dir", str(self.root / "x")]
        before = runqueue.data_state(run)[0]
        _coverage(self.root / "x", expected=2, recorded=2)
        self.assertEqual(runqueue.data_state(run)[0], before)  # the kept reading
        self.assertEqual(runqueue.data_state(run, fresh=True)[0], "complete")

    def test_the_plan_says_skip_or_start_from_the_data_on_disk(self):
        _coverage(self.root / "done", expected=1, recorded=1)
        items = [("done", ["fake", "--name", "done", "--output-dir", str(self.root / "done")]),
                 ("fresh", ["fake", "--name", "fresh", "--output-dir", str(self.root / "fresh")])]
        actions = {name: action for name, _, action in runqueue.plan(_queue(self.root, items))}
        self.assertEqual(actions, {"done": "skip: already completed", "fresh": "start"})


class TestQueueFile(_Base):
    def _write(self, text: str) -> str:
        path = self.root / "q.toml"
        path.write_text(text)
        return str(path)

    def test_an_item_runs_under_its_own_name(self):
        with self.assertRaises(SystemExit):
            runqueue.load_queue(self._write('name = "q"\n[[items]]\nname = "a"\n'
                                            'run = ["disjoint", "--preset", "smoke", "--name", "b"]\n'))
        with self.assertRaises(SystemExit):  # one name, twice
            runqueue.load_queue(self._write('name = "q"\n[[items]]\nname = "a"\nrun = ["disjoint", "--preset", "smoke"]\n'
                                            '[[items]]\nname = "a"\nrun = ["disjoint", "--preset", "smoke"]\n'))

    def test_arguments_the_front_door_would_refuse_are_refused_before_anything_starts(self):
        for run in ('["list"]', '["exp", "--networks", "dblp"]', '["disjoint", "--preset", "nonsense"]'):
            with self.subTest(run=run), self.assertRaises(SystemExit):
                runqueue.load_queue(self._write(f'name = "q"\n[[items]]\nname = "a"\nrun = {run}\n'))

    def test_pair_by_pair_queues_are_hedonic_2(self):
        with self.assertRaises(SystemExit) as caught:
            runqueue.load_queue(self._write('name = "q"\npairs = ["dblp:all"]\n[[items]]\nname = "a"\n'
                                            'run = ["disjoint", "--preset", "smoke"]\n'))
        self.assertIn("hedonic 2", str(caught.exception))
        with self.assertRaises(SystemExit):
            runqueue.load_queue(self._write('name = "q"\n[[items]]\nname = "a"\nby_pair = true\n'
                                            'run = ["disjoint", "--preset", "smoke"]\n'))

    def test_the_shipped_disjoint_queue_writes_under_the_disjoint_root(self):
        queue = runqueue.load_queue("disjoint")
        self.assertEqual(queue["name"], "disjoint-queue")
        self.assertEqual([item["name"] for item in queue["items"]], ["disjoint-smoke", "disjoint-paper"])
        self.assertEqual(Path(queue["root"]), (self.root / "disjoint" / "disjoint-queue").resolve())
        for item in queue["items"]:
            self.assertEqual(item["run"][0], "disjoint")
            self.assertEqual(runqueue._option(item["run"], "--name"), item["name"])
            self.assertEqual(Path(runqueue._option(item["run"], "--output-dir")), Path(queue["root"]) / item["name"])
        given = runqueue.load_queue("disjoint", root=str(self.root / "elsewhere"))
        self.assertEqual(Path(given["root"]), (self.root / "elsewhere").resolve())

    def test_an_unknown_queue_names_the_shipped_ones(self):
        with self.assertRaises(SystemExit) as caught:
            runqueue.load_queue("paper")
        self.assertIn("disjoint", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
