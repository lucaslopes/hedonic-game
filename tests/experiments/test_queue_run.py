"""`hedonic run queue`: order, skipping finished items, resuming, stopping at the first failure."""

from __future__ import annotations

import io
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments import CLI
from hedonic.experiments.overlapping import runmanager as rm
from hedonic.experiments.overlapping import runqueue
from hedonic.experiments.overlapping import tui


def _register(root: Path, name: str, status: str) -> None:
    """A registered run that already has the given status (stands in for a real worker)."""
    out = root / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    entry = {"name": name, "kind": "disjoint", "output_dir": str(out), "created": time.time(), "session": "",
             "backend": "process", "log": str(out / "worker.log"), "argv": []}
    rm.write_json(rm.entry_path(name), entry)
    rm.write_json(rm.progress_path(entry), {"status": status, "plan": [], "setup": []})


class FakeLauncher:
    """Stands in for `hedonic run NAME --detach` and `hedonic run resume NAME`: the run ends with its outcome."""

    def __init__(self, root: Path, outcomes: dict[str, str]):
        self.root, self.outcomes, self.calls = root, outcomes, []

    def __call__(self, name: str, run: list[str], resuming: bool) -> int:
        self.calls.append((name, "resume" if resuming else "start"))
        _register(self.root, name, self.outcomes.get(name, "completed"))
        return 0


def _queue(root: Path, names: list[str]) -> dict:
    out = root / "queues" / "q"
    out.mkdir(parents=True, exist_ok=True)
    return {"name": "q", "kind": "queue", "items": [{"name": n, "run": ["x"]} for n in names],
            "file": "", "output_dir": str(out), "created": time.time(), "session": "", "backend": "process",
            "log": str(out / "worker.log")}


def _register_live(root: Path, name: str, done: int, graphs: int, stage: str = "sweep", started: float | None = None) -> None:
    """A registered disjoint run in progress, with the counts its worker wrote (the test fakes the worker alive)."""
    out = root / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    started = time.time() - 300 if started is None else started
    entry = {"name": name, "kind": "disjoint", "output_dir": str(out), "created": started, "session": "",
             "backend": "process", "log": str(out / "worker.log")}
    rm.write_json(rm.entry_path(name), entry)
    rm.write_json(rm.progress_path(entry), {"status": "running", "started": started, "updated": started + 300,
                                            "stage": stage, "graphs": graphs, "done": done})


def _disjoint_item(root: Path, name: str) -> str:
    return (f'[[items]]\nname = "{name}"\nrun = ["disjoint", "--preset", "smoke", "--name", "{name}", '
            f'"--output-dir", "{root / "stores" / name}", "--yes"]\n')


class TestRunQueue(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._env = patch.dict(os.environ, {"HEDONIC_RUNS_DIR": str(self.root / "registry")})
        self._env.start()
        self._poll = patch.object(runqueue, "POLL_SECONDS", 0)
        self._poll.start()

    def tearDown(self):
        self._poll.stop()
        self._env.stop()
        self._tmp.cleanup()

    def _work(self, entry: dict, launcher: FakeLauncher) -> tuple[int, dict]:
        prog: dict = {}
        with redirect_stdout(io.StringIO()):
            code = runqueue.work("q", entry, prog, launch_item=launcher)
        return code, prog

    def test_items_run_in_order_and_a_finished_item_is_skipped(self):
        _register(self.root, "a", "completed")
        launcher = FakeLauncher(self.root, {})
        code, prog = self._work(_queue(self.root, ["a", "b", "c"]), launcher)
        self.assertEqual(code, 0)
        self.assertEqual(launcher.calls, [("b", "start"), ("c", "start")])
        self.assertEqual(prog["status"], "completed")
        self.assertEqual([item["status"] for item in prog["items"]], ["completed", "completed", "completed"])

    def test_the_queue_stops_at_the_first_failure(self):
        launcher = FakeLauncher(self.root, {"b": "failed"})
        code, prog = self._work(_queue(self.root, ["a", "b", "c"]), launcher)
        self.assertEqual(code, 1)
        self.assertEqual(launcher.calls, [("a", "start"), ("b", "start")])  # c never starts
        self.assertEqual(prog["status"], "failed")
        self.assertIn("b ended as failed", prog["reason"])

    def test_rerunning_skips_completed_items_and_resumes_the_failed_one(self):
        entry = _queue(self.root, ["a", "b", "c"])
        self._work(entry, FakeLauncher(self.root, {"b": "failed"}))
        launcher = FakeLauncher(self.root, {})
        code, prog = self._work(entry, launcher)
        self.assertEqual(code, 0)
        self.assertEqual(launcher.calls, [("b", "resume"), ("c", "start")])
        self.assertEqual(prog["status"], "completed")

    def test_plan_says_what_is_left(self):
        _register(self.root, "a", "completed")
        _register(self.root, "b", "stopped")
        queue = {"name": "q", "items": [{"name": n, "run": ["x"]} for n in ("a", "b", "c")]}
        actions = {name: action for name, _, action in runqueue.plan(queue)}
        self.assertEqual(actions, {"a": "skip: already completed", "b": "resume from its finished records",
                                   "c": "start"})

    def test_queue_file_is_checked_before_anything_starts(self):
        path = self.root / "q.toml"
        path.write_text('name = "q"\n' + _disjoint_item(self.root, "a") + _disjoint_item(self.root, "a"))
        with self.assertRaises(SystemExit):  # the same run name twice
            runqueue.load_queue(str(path))
        path.write_text('name = "q"\n[[items]]\nname = "a"\n')  # no run
        with self.assertRaises(SystemExit):
            runqueue.load_queue(str(path))
        path.write_text('name = "q"\n[[items]]\nname = "a"\nrun = ["spectrum-full"]\n')  # an overlap run: hedonic 2
        with self.assertRaises(SystemExit):
            runqueue.load_queue(str(path))
        path.write_text('name = "q"\n' + _disjoint_item(self.root, "a"))
        (item,) = runqueue.load_queue(str(path))["items"]
        self.assertEqual((item["name"], item["run"][:4]), ("a", ["disjoint", "--preset", "smoke", "--name"]))

    def test_dry_run_through_the_cli_lists_the_plan_without_starting_anything(self):
        path = self.root / "q.toml"
        path.write_text('name = "q"\n' + _disjoint_item(self.root, "a") + _disjoint_item(self.root, "b"))
        out = io.StringIO()
        with redirect_stdout(out):
            code = CLI.hedonic_main(["run", "queue", str(path), "--dry-run", "--root", str(self.root / "root")])
        self.assertEqual(code, 0)
        text = out.getvalue()
        for part in ("queue q · 2 runs", "1. a", "2. b", "→ start", "overview", "total: 0 of 2 runs complete"):
            self.assertIn(part, text)
        self.assertFalse(rm.entry_path("q").is_file())  # nothing was registered
        self.assertFalse((self.root / "stores").exists())  # and no store was written
        self.assertFalse((self.root / "root").exists())

    def test_report_prints_the_sections_without_starting_anything(self):
        path = self.root / "q.toml"
        path.write_text('name = "q"\n' + _disjoint_item(self.root, "a"))
        out = io.StringIO()
        with redirect_stdout(out):
            code = CLI.hedonic_main(["run", "queue", str(path), "--report", "--root", str(self.root / "root")])
        self.assertEqual(code, 0)
        self.assertIn("0/16 graphs", out.getvalue())  # the smoke grid, counted from its (empty) store
        self.assertFalse(rm.entry_path("q").is_file())

    def test_snapshot_lists_every_item_with_its_status(self):
        _register(self.root, "alpha-run", "completed")
        text = runqueue.snapshot(_queue(self.root, ["alpha-run", "beta-run"]), {})
        self.assertIn("alpha-run", text)
        self.assertIn("completed", text)
        self.assertIn("beta-run", text)
        self.assertIn("pending", text)

    def test_the_running_item_shows_its_own_progress_inside_its_section(self):
        _register(self.root, "done-run", "completed")
        _register_live(self.root, "live-run", done=4, graphs=10, stage="sweep")
        with patch.object(rm, "worker_alive", return_value=True):
            text = runqueue.snapshot(_queue(self.root, ["done-run", "live-run"]), {"status": "running"})
        lines = text.splitlines()
        self.assertTrue(text.startswith("queue q · 2 runs"))
        for part in ("overview", "1. done-run", "2. live-run", "total: ", "4/10 · 40%", "now      sweep"):
            self.assertIn(part, text)
        self.assertNotIn("running now", text)
        self.assertLess(text.index("2. live-run"), text.index("4/10 · 40%"))  # after its heading
        self.assertNotIn("%", next(line for line in lines if "done-run" in line))  # a finished item stays one line
        for line in lines:
            self.assertLessEqual(tui.visible_len(line), 80)

    def test_a_running_item_without_units_yet_says_starting(self):
        _register_live(self.root, "fresh-run", done=0, graphs=0)
        with patch.object(rm, "worker_alive", return_value=True):
            text = runqueue.snapshot(_queue(self.root, ["fresh-run"]), {"status": "running"})
        self.assertIn("starting", text)
        self.assertNotIn("ETA", text)


NOW = 1_800_000_000.0  # the clock of the progress tests; a live run in them started 300 s before it


class _FrozenClock:
    """Stands in for the time module inside runmanager: time() is fixed, every other attribute is the real module."""

    def __init__(self, now: float):
        self._now = now

    def time(self) -> float:
        return self._now

    def __getattr__(self, name: str):
        return getattr(time, name)


class TestProgressDisplay(unittest.TestCase):
    """What `hedonic run status` and `attach` show for a queue: the bar, the time, the stage, the log and the records."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        # HOME is the temp root, so record folders show as ~/runs/NAME as they do for a real user (short enough to fit)
        self._env = patch.dict(os.environ, {"HEDONIC_RUNS_DIR": str(self.root / "registry"), "NO_COLOR": "1",
                                            "HOME": str(self.root)})
        self._env.start()
        self._poll = patch.object(runqueue, "POLL_SECONDS", 0)
        self._poll.start()

    def tearDown(self):
        self._poll.stop()
        self._env.stop()
        self._tmp.cleanup()

    def _live_run(self, name: str, done: int, graphs: int, log: str = "") -> None:
        """A disjoint run in progress: its registry entry, its progress counts, and its worker log."""
        _register_live(self.root, name, done, graphs, started=NOW - 300)
        (self.root / "runs" / name / "worker.log").write_text(log)

    def _show(self, entry: dict, prog: dict, *, unicode_glyphs: bool, frame: str | None = None) -> str:
        """The queue as `hedonic run status` prints it: its progress is read back from disk, and the clock is fixed."""
        rm.write_json(rm.progress_path(entry), prog)
        with patch.object(rm, "time", _FrozenClock(NOW)), patch.object(rm, "worker_alive", return_value=True):
            return rm.snapshot(entry, frame=frame, unicode_glyphs=unicode_glyphs)

    def _line(self, text: str, start: str) -> str:
        for line in text.splitlines():
            if line.strip().startswith(start):
                return line
        self.fail(f"no line starts with {start!r} in:\n{text}")

    def test_a_queue_shows_its_bar_time_stage_log_and_records(self):
        _register(self.root, "done-run", "completed")
        self._live_run("live-run", 3, 20, log="setup done\n\x1b[36m[3/20] 4C_255N noise 0.1\x1b[0m\n")
        entry = _queue(self.root, ["done-run", "live-run", "later-run"])
        text = self._show(entry, {"status": "running", "current": 1}, unicode_glyphs=True)

        self.assertIn("● running  · 1/3 runs completed", text)
        self.assertIn("completed", self._line(text, "1. done-run"))
        self.assertIn("running", self._line(text, "2. live-run"))
        self.assertIn("pending", self._line(text, "3. later-run"))
        self.assertIn("█" * 4 + "░" * 26 + "  3/20 · 15%", text)  # the bar: 3 of 20 graphs
        self.assertIn("elapsed 5:00", text)
        self.assertNotIn("ETA ~", text)  # a disjoint run publishes no estimate
        self.assertEqual(self._line(text, "now").strip(), "now      sweep")
        # the last line of the child's worker log (colour codes removed) and the record directory
        self.assertIn("[3/20] 4C_255N noise 0.1", self._line(text, "log"))
        self.assertEqual(self._line(text, "records").strip(), "records  ~/runs/live-run")
        for line in text.splitlines():
            self.assertLessEqual(tui.visible_len(line), 80, line)

    def test_without_a_utf8_terminal_the_glyphs_are_plain_ascii(self):
        self._live_run("live-run", 3, 20)
        text = self._show(_queue(self.root, ["live-run"]), {"status": "running", "current": 0}, unicode_glyphs=False)
        self.assertIn("#" * 4 + "-" * 26 + "  3/20 · 15%", text)
        for glyph in "✓◉■█░●✗±" + tui.SPINNER:
            self.assertNotIn(glyph, text)

    def test_a_stopped_queue_names_the_item_the_reason_and_the_resume_command(self):
        _register(self.root, "done-run", "completed")
        entry = _queue(self.root, ["done-run", "stop-run", "later-run"])
        entry["name"] = "stop-q"
        prog: dict = {}
        with redirect_stdout(io.StringIO()):  # the real worker writes the stop reason
            code = runqueue.work("stop-q", entry, prog, launch_item=FakeLauncher(self.root, {"stop-run": "stopped"}))
        self.assertEqual(code, 1)
        text = self._show(entry, prog, unicode_glyphs=True, frame=tui.SPINNER[0])
        self.assertIn("■ stopped  · 1/3 runs completed", text)
        self.assertIn("stopped at stop-run (item 2 of 3)", text)
        self.assertIn("reason: stop-run ended as stopped", text)
        self.assertIn("continue with: hedonic run resume stop-q", text)
        for glyph in tui.SPINNER:  # a stopped queue has no live spinner
            self.assertNotIn(glyph, text)

    def test_a_queue_stopped_by_hand_says_that_no_reason_was_recorded(self):
        _register(self.root, "stop-run", "stopped")
        entry = _queue(self.root, ["stop-run", "later-run"])
        entry["name"] = "stop-q"
        text = self._show(entry, {"status": "stopped", "current": 0}, unicode_glyphs=False)
        self.assertIn("# stopped  · 0/2 runs completed", text)
        self.assertIn("stopped at stop-run (item 1 of 2)", text)
        self.assertIn("stopped with hedonic run stop; the queue recorded no reason", text)
        self.assertIn("continue with: hedonic run resume stop-q", text)


if __name__ == "__main__":
    unittest.main()
