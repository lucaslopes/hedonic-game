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
    entry = {"name": name, "kind": "spectrum", "output_dir": str(out), "created": time.time(), "session": "",
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
    return {"name": "q", "kind": "queue", "items": [{"name": n, "run": ["spectrum-full"]} for n in names],
            "file": "", "output_dir": str(out), "created": time.time(), "session": "", "backend": "process",
            "log": str(out / "worker.log")}


def _register_benchmark(root: Path, name: str, plan: list[dict]) -> None:
    """A registered benchmark run in progress, with the plan its worker wrote (the test fakes the worker alive)."""
    out = root / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    entry = {"name": name, "config": {}, "output_dir": str(out), "created": time.time() - 300, "session": "",
             "backend": "process", "log": str(out / "worker.log")}
    rm.write_json(rm.entry_path(name), entry)
    rm.write_json(rm.progress_path(entry), {"status": "running", "started": time.time() - 300,
                                            "updated": time.time(), "setup": [], "plan": plan})


def _plan_in_progress() -> list[dict]:
    """Ten DBLP units: four finished, the fifth (seed 3, CoDeSEG) running for 30 s, five still pending."""
    finished = [("HOC-local", "hedonic", 0), ("CoDeSEG", "codeseg", 0), ("CoDeSEG", "codeseg", 1),
                ("CoDeSEG", "codeseg", 2)]
    plan = [{"network": "dblp", "seed": seed, "method": method, "label": label, "expected": 60.0,
             "status": "completed", "seconds": 50.0} for label, method, seed in finished]
    plan.append({"network": "dblp", "seed": 3, "method": "codeseg", "label": "CoDeSEG", "expected": 60.0,
                 "status": "running", "started": time.time() - 30})
    plan += [{"network": "dblp", "seed": seed, "method": "codeseg", "label": "CoDeSEG", "expected": 60.0,
              "status": "pending"} for seed in range(4, 9)]
    return plan


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
        path.write_text('name = "q"\n[[items]]\nname = "a"\nrun = ["spectrum-full"]\n'
                        '[[items]]\nname = "a"\nrun = ["spectrum-full"]\n')
        with self.assertRaises(SystemExit):  # the same run name twice
            runqueue.load_queue(str(path))
        path.write_text('name = "q"\n[[items]]\nname = "a"\n')  # no run
        with self.assertRaises(SystemExit):
            runqueue.load_queue(str(path))
        path.write_text('name = "q"\n[[items]]\nname = "a"\nrun = ["spectrum-full"]\n')
        self.assertEqual(runqueue.load_queue(str(path))["items"], [{"name": "a", "run": ["spectrum-full"]}])

    def test_dry_run_through_the_cli_lists_the_plan_without_starting_anything(self):
        path = self.root / "q.toml"
        path.write_text('name = "q"\n[[items]]\nname = "a"\nrun = ["spectrum-full"]\n'
                        '[[items]]\nname = "b"\nrun = ["spectrum-full"]\n')
        _register(self.root, "a", "completed")
        out = io.StringIO()
        with redirect_stdout(out):
            code = CLI.hedonic_main(["run", "queue", str(path), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("skip: already completed", out.getvalue())
        self.assertIn("start", out.getvalue())
        self.assertFalse(rm.entry_path("q").is_file())  # nothing was registered

    def test_snapshot_lists_every_item_with_its_status(self):
        _register(self.root, "alpha-run", "completed")
        text = runqueue.snapshot(_queue(self.root, ["alpha-run", "beta-run"]), {})
        self.assertIn("alpha-run", text)
        self.assertIn("completed", text)
        self.assertIn("beta-run", text)
        self.assertIn("pending", text)

    def test_the_running_item_shows_its_own_progress_under_its_line(self):
        _register(self.root, "done-run", "completed")
        _register_benchmark(self.root, "live-run", _plan_in_progress())
        with patch.object(rm, "worker_alive", return_value=True):
            text = runqueue.snapshot(_queue(self.root, ["done-run", "live-run"]), {"status": "running"})
        lines = text.splitlines()
        self.assertIn("live-run", text)
        self.assertIn("4/10", text)  # units done out of the total
        self.assertIn("40%", text)
        self.assertIn("ETA ~", text)
        self.assertIn("[5/10] DBLP · seed 3 · CoDeSEG", text)  # the unit in progress
        self.assertNotIn("%", next(line for line in lines if "done-run" in line))  # a finished item stays one line
        for line in lines:
            self.assertLessEqual(tui.visible_len(line), 80)

    def test_the_live_view_is_the_whole_report_with_no_separate_running_block(self):
        _register(self.root, "done-run", "completed")
        _register_benchmark(self.root, "live-run", _plan_in_progress())
        with patch.object(rm, "worker_alive", return_value=True):
            text = runqueue.snapshot(_queue(self.root, ["done-run", "live-run"]), {"status": "running"})
        self.assertTrue(text.startswith("queue q · 2 runs"))
        for part in ("overview", "all pairs", "1. done-run", "2. live-run", "total: "):
            self.assertIn(part, text)
        self.assertNotIn("running now", text)
        # the running run's progress sits inside its own section, after its heading
        self.assertLess(text.index("2. live-run"), text.index("[5/10] DBLP"))

    def test_a_running_item_without_units_yet_says_starting(self):
        _register_benchmark(self.root, "fresh-run", [])
        with patch.object(rm, "worker_alive", return_value=True):
            text = runqueue.snapshot(_queue(self.root, ["fresh-run"]), {"status": "running"})
        self.assertIn("starting", text)
        self.assertNotIn("ETA", text)

    def test_the_detail_lines_stay_within_eighty_columns(self):
        label = "CliquePerc (clique percolation)"  # the longest method label, with a 151-unit plan
        plan = [{"network": "dblp", "seed": 0, "method": "clique_percolation", "label": label, "expected": 3600.0,
                 "status": "running", "started": time.time() - 3700}]
        plan += [{"network": "dblp", "seed": s, "method": "clique_percolation", "label": label, "expected": 3600.0,
                  "status": "pending"} for s in range(1, 151)]
        _register_benchmark(self.root, "long-run", plan)
        with patch.object(rm, "worker_alive", return_value=True):
            text = runqueue.snapshot(_queue(self.root, ["long-run"]), {"status": "running"})
        self.assertIn("0/151", text)
        self.assertIn("0%", text)
        for line in text.splitlines():
            self.assertLessEqual(tui.visible_len(line), 80, line)


NOW = 1_800_000_000.0  # the clock of the progress tests; a live run in them started 300 s before it


class _FrozenClock:
    """Stands in for the time module inside runmanager: time() is fixed, every other attribute is the real module."""

    def __init__(self, now: float):
        self._now = now

    def time(self) -> float:
        return self._now

    def __getattr__(self, name: str):
        return getattr(time, name)


def _unit(method: str, label: str, seed: int, status: str, *, expected: float, seconds: float | None = None,
          detection: float | None = None, started: float | None = None) -> dict:
    """One plan unit of a DBLP run, as the worker writes it."""
    unit = {"network": "dblp", "seed": seed, "method": method, "label": label, "expected": expected,
            "status": status}
    for key, value in (("seconds", seconds), ("detection_seconds", detection), ("started", started)):
        if value is not None:
            unit[key] = value
    return unit


def _twenty_units(running_elapsed: float = 30.0, codeseg_finished: bool = False) -> list[dict]:
    """Twenty DBLP units in two methods. HOC-local (planning constant 100 s): seeds 0-2 finished (wall 40, 50 and 60 s,
    detection 30, 40 and 50 s), seed 3 running, seeds 4-9 pending. CoDeSEG (planning constant 60 s): nothing finished
    yet, unless ``codeseg_finished`` gives seed 0 a measured unit (wall 90 s, detection 80 s)."""
    hoc = [_unit("hedonic", "HOC-local", seed, "completed", expected=100.0, seconds=wall, detection=detection)
           for seed, wall, detection in ((0, 40.0, 30.0), (1, 50.0, 40.0), (2, 60.0, 50.0))]
    hoc.append(_unit("hedonic", "HOC-local", 3, "running", expected=100.0, started=NOW - running_elapsed))
    hoc += [_unit("hedonic", "HOC-local", seed, "pending", expected=100.0) for seed in range(4, 10)]
    codeseg = [_unit("codeseg", "CoDeSEG", seed, "pending", expected=60.0) for seed in range(10)]
    if codeseg_finished:
        codeseg[0] = _unit("codeseg", "CoDeSEG", 0, "completed", expected=60.0, seconds=90.0, detection=80.0)
    return hoc + codeseg


class TestProgressDisplay(unittest.TestCase):
    """What `hedonic run status` and `attach` show for a queue: the bar, the plan grid, measured times and the ETA."""

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

    def _live_run(self, name: str, plan: list[dict], log: str = "") -> None:
        """A benchmark run in progress: its registry entry, its progress with the plan, and its worker log."""
        out = self.root / "runs" / name
        out.mkdir(parents=True, exist_ok=True)
        entry = {"name": name, "config": {}, "output_dir": str(out), "created": NOW - 300, "session": "",
                 "backend": "process", "log": str(out / "worker.log")}
        rm.write_json(rm.entry_path(name), entry)
        rm.write_json(rm.progress_path(entry), {"status": "running", "started": NOW - 300, "updated": NOW,
                                                "setup": [], "plan": plan})
        (out / "worker.log").write_text(log)

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

    def test_a_queue_shows_its_bar_grid_measured_times_and_running_unit(self):
        _register(self.root, "done-run", "completed")
        self._live_run("live-run", _twenty_units(), log="setup done\n\x1b[36mseed 3 · HOC-local running\x1b[0m\n")
        entry = _queue(self.root, ["done-run", "live-run", "later-run"])
        text = self._show(entry, {"status": "running", "current": 1}, unicode_glyphs=True)

        self.assertIn("● running  · 1/3 runs completed", text)
        self.assertIn("completed", self._line(text, "1. done-run"))
        self.assertIn("running", self._line(text, "2. live-run"))
        self.assertIn("pending", self._line(text, "3. later-run"))
        # the bar: 3 of 20 units; the ETA is 6 x 50 s (measured) + 20 s left on the running unit + 10 x 60 s (planned)
        self.assertIn("█" * 4 + "░" * 26 + "  3/20 · 15%", text)
        self.assertIn("elapsed 5:00 · ETA ~15:20 (estimated)", text)
        # the grid: one row per method, one glyph per unit
        self.assertIn("HOC-local  " + "✓✓✓◉" + "·" * 6, text)
        self.assertIn("CoDeSEG    " + "·" * 10, text)
        self.assertIn("✓ done  ◉ running  · pending  ✗ failed  ■ stopped", text)
        # the measured means of the finished units; a method with none yet says that its figure is an estimate
        self.assertIn("3/10 finished · detection 40 s ± 10 s · unit 50 s", text)
        self.assertIn("0/10 finished · none measured · estimated 60 s per unit", text)
        # the running unit: place, network, seed, method, elapsed time, and its estimate from its own method
        self.assertIn("◉ [4/20] DBLP · seed 3 · HOC-local  0:30 elapsed", text)
        self.assertIn("estimate ~0:50", text)
        self.assertNotIn("(estimated)", self._line(text, "estimate ~"))
        # the last line of the child's worker log (colour codes removed) and the record directory
        self.assertIn("seed 3 · HOC-local running", self._line(text, "log"))
        self.assertEqual(self._line(text, "records").strip(), "records  ~/runs/live-run")
        for line in text.splitlines():
            self.assertLessEqual(tui.visible_len(line), 80, line)

    def test_without_a_utf8_terminal_the_glyphs_are_plain_ascii(self):
        self._live_run("live-run", _twenty_units())
        text = self._show(_queue(self.root, ["live-run"]), {"status": "running", "current": 0}, unicode_glyphs=False)
        self.assertIn("#" * 4 + "-" * 26 + "  3/20 · 15%", text)
        self.assertIn("HOC-local  +++*......", text)
        self.assertIn("CoDeSEG    ..........", text)
        self.assertIn("+ done  * running  . pending  x failed  # stopped", text)
        self.assertIn("3/10 finished · detection 40 s +/- 10 s · unit 50 s", text)
        for glyph in "✓◉■█░●✗±" + tui.SPINNER:
            self.assertNotIn(glyph, text)

    def test_once_every_method_has_a_measured_unit_the_eta_is_not_an_estimate(self):
        self._live_run("live-run", _twenty_units(codeseg_finished=True))
        text = self._show(_queue(self.root, ["live-run"]), {"status": "running", "current": 0}, unicode_glyphs=True)
        self.assertIn("4/20 · 20%", text)
        # 6 pending HOC-local units x 50 s, 20 s left on the running unit, 9 pending CoDeSEG units x 90 s
        self.assertIn("ETA ~18:50", text)
        self.assertNotIn("estimated", text)
        self.assertIn("1/10 finished · detection 80 s · unit 1:30", text)

    def test_a_running_unit_past_its_estimate_says_by_how_much(self):
        self._live_run("live-run", _twenty_units(running_elapsed=70.0))
        text = self._show(_queue(self.root, ["live-run"]), {"status": "running", "current": 0}, unicode_glyphs=True)
        self.assertIn("1:10 elapsed", text)
        self.assertIn("0:20 beyond its estimate of ~0:50", text)
        self.assertIn("ETA ~15:00 (estimated)", text)  # the running unit has no time left in its estimate

    def test_a_unit_whose_detector_has_returned_says_its_audit_is_running_not_that_detection_is_late(self):
        """BigCLAM on YouTube/top5000 detected in 16:33 and its cover's audit then ran for hours; the line used to say the
        unit was hours "beyond its estimate" of the planning constant, which prices the detection alone."""
        plan = _twenty_units()
        plan[3] = _unit("hedonic", "HOC-local", 3, "pending", expected=100.0)
        plan[10] = _unit("codeseg", "CoDeSEG", 0, "running", expected=60.0, started=NOW - 300.0, detection=20.0)
        self._live_run("live-run", plan)
        text = self._show(_queue(self.root, ["live-run"]), {"status": "running", "current": 0}, unicode_glyphs=True)
        self.assertIn("detection took 0:20 · the exact audit of its cover has run 4:40", text)
        self.assertNotIn("beyond its estimate", text)

    def test_the_status_line_gives_the_vertices_the_audit_has_done(self):
        plan = _twenty_units()
        plan[3] = _unit("hedonic", "HOC-local", 3, "pending", expected=100.0)
        plan[10] = {**_unit("codeseg", "CoDeSEG", 0, "running", expected=60.0, started=NOW - 300.0, detection=20.0),
                    "audit": {"number": 1, "policies": 2, "done": 5000, "total": 39841, "eta_seconds": 4000.0}}
        self._live_run("live-run", plan)
        text = self._show(_queue(self.root, ["live-run"]), {"status": "running", "current": 0}, unicode_glyphs=True)
        self.assertIn("exact audit: policy 1 of 2, 5,000 of 39,841 vertices · about 1:06:40 left", text)

    def test_a_measured_estimate_includes_the_audit_so_an_audit_past_it_is_late(self):
        plan = _twenty_units(running_elapsed=300.0)  # HOC-local units took 40-60 s in all, audit included
        plan[3] = {**plan[3], "detection_seconds": 30.0}
        self._live_run("live-run", plan)
        text = self._show(_queue(self.root, ["live-run"]), {"status": "running", "current": 0}, unicode_glyphs=True)
        self.assertIn("detection took 0:30 · the exact audit of its cover has run 4:30", text)
        self.assertIn("beyond its estimate of ~0:50", text)

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
