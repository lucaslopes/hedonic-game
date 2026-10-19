"""A queue's time limits (``limits``): ``timeout`` stops one run, ``max_eta`` skips a pair whose one run is estimated
longer. Both are about a single run, never the total of several, and neither changes a record's identity."""

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

from hedonic.experiments.overlapping import calibration_search as cs
from hedonic.experiments.overlapping import codeseg_reproduction as cr
from hedonic.experiments.overlapping import limits
from hedonic.experiments.overlapping import queuelimits as ql
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import runmanager as rm
from hedonic.experiments.overlapping import runqueue
from hedonic.experiments.overlapping import start_quality as sq
from hedonic.experiments.overlapping import unit_integrity as ui


class TestDurations(unittest.TestCase):
    def test_durations_read_as_seconds_and_print_back(self):
        for text, seconds in (("90s", 90), ("30m", 1800), ("2h", 7200), ("1h30m", 5400), ("1.5d", 129600),
                              (45, 45), ("600", 600)):
            self.assertEqual(limits.parse_duration(text), seconds)
        self.assertIsNone(limits.parse_duration("none"))
        self.assertEqual([limits.fmt(s) for s in (90, 5400, 7200, None)], ["1m30s", "1h30m", "2h", "none"])
        for bad in ("2x", "0", "-5", "h", True):
            with self.assertRaises(ValueError):
                limits.parse_duration(bad)

    def test_a_limit_only_lowers_a_runners_own_timeout(self):
        self.assertEqual(limits.effective(3600, 600), 600)
        self.assertEqual(limits.effective(600, 3600), 600)
        self.assertEqual(limits.effective(None, 60), 60)
        self.assertEqual(limits.effective(60, None), 60)

    def test_a_stopped_run_answers_only_under_a_limit_no_larger_than_its_own(self):
        stopped = {"status": "timeout", "limit_seconds": 600.0}
        self.assertTrue(limits.stands(stopped, 600))
        self.assertTrue(limits.stands(stopped, 300))
        self.assertFalse(limits.stands(stopped, 1200))
        self.assertFalse(limits.stands(stopped, None))
        self.assertTrue(limits.stands({"status": "completed"}, None))
        self.assertTrue(limits.stands({"status": "timeout"}, 1800, default=1800))  # written before limit_seconds


class TestEachRunOnItsOwn(unittest.TestCase):
    """The common mistake: a search of many short runs is not one long run."""

    def test_many_runs_inside_the_limit_are_never_stopped_however_long_they_take_together(self):
        started = time.monotonic()
        for _ in range(4):
            with limits.wall_clock(0.3):
                time.sleep(0.12)
        self.assertGreater(time.monotonic() - started, 0.3)  # together over the limit, each well inside it

    def test_one_run_past_the_limit_is_stopped(self):
        started = time.monotonic()
        with self.assertRaises(limits.RunLimitReached):
            with limits.wall_clock(0.1):
                time.sleep(2)
        self.assertLess(time.monotonic() - started, 1.0)

    def test_the_calibration_reruns_a_probe_its_smaller_timeout_stopped_and_reuses_it_otherwise(self):
        with tempfile.TemporaryDirectory() as tmp:
            cs.configure(tmp)
            calls = []

            def compute(spec, cfg, timeout):
                calls.append(timeout)
                return {"key": cs.cfg_key(spec, cfg), "instance": spec, "config": cfg, "status": "timeout",
                        "limit_seconds": float(timeout), "c": 1.0, "t": 0.5}

            cfg = {"phase": "local", "M": -1, "gamma": 0.01, "seed": 0}
            with patch.object(cs, "_compute", compute), patch.object(cs, "load_instance", lambda spec: None), \
                    redirect_stdout(io.StringIO()):
                cs.evaluate("dblp", cfg, timeout=60, objective="f1")
                cs.evaluate("dblp", cfg, timeout=60, objective="f1")   # same limit: the stopped probe is the answer
                cs.evaluate("dblp", cfg, timeout=30, objective="f1")   # a smaller one too
                cs.evaluate("dblp", cfg, timeout=120, objective="f1")  # a larger one runs it again
            self.assertEqual(calls, [60, 120])

    def test_the_start_quality_curve_keeps_a_stopped_condition_only_under_its_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / sq.RESULTS_NAME
            stopped = {"key": "k|0|phase=local", "phase": "local", "status": "timeout", "limit_seconds": 60.0}
            done = {"key": "j|0|phase=local", "phase": "local", "seconds": 1.0}
            sq.append_record(path, stopped)
            sq.append_record(path, done)
            self.assertEqual(set(sq.read_records(path, 60)), {"k|0|phase=local", "j|0|phase=local"})
            self.assertEqual(set(sq.read_records(path, None)), {"j|0|phase=local"})
            sq.append_record(path, {**stopped, "status": "completed", "seconds": 90.0})  # the rerun replaces it
            self.assertEqual(sq.read_records(path, None)["k|0|phase=local"]["status"], "completed")


class TestBenchmarkUnit(unittest.TestCase):
    def test_the_limit_is_not_part_of_a_units_identity(self):
        cfg = qs.config_from_args(qs.build_parser().parse_args(["--networks", "dblp", "--methods", "codeseg"]))
        method = next(m for m in qs.METHODS if m.key == "codeseg")
        before = qs.unit_digest(cfg, "dblp", 0, method, None)
        cfg.limit = 60.0
        self.assertIn("--limit-seconds", qs.unit_argv(cfg, "dblp", 0, method, None)[0])
        self.assertEqual(qs.unit_digest(cfg, "dblp", 0, method, None), before)
        self.assertIsNotNone(before)

    def test_an_in_process_detector_is_stopped_at_the_limit_and_rerun_under_a_larger_one(self):
        original, calls = cr._run_method, []

        def slow(*args, **kwargs):
            calls.append(1)
            time.sleep(0.6)
            return original(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            def run(*extra):
                with patch.object(cr, "_run_method", slow), redirect_stdout(io.StringIO()):
                    cr.main(["--smoke", "--datasets", "dblp", "--methods", "hedonic_local", "--output-dir", str(out),
                             "--seed", "0", "--resume", *extra])
                return json.loads((out / "runs" / "dblp" / "hedonic_local.json").read_text())

            record = run("--limit-seconds", "0.2")
            self.assertEqual((record["status"], record["limit_seconds"]), ("timeout", 0.2))
            self.assertTrue(ui.final(record, 0.2))
            self.assertFalse(ui.final(record, 5.0))
            self.assertEqual(run("--limit-seconds", "0.2")["status"], "timeout")
            self.assertEqual(len(calls), 1)  # the same limit reuses the stopped run
            self.assertEqual(run("--limit-seconds", "30")["status"], "completed")
            self.assertEqual(len(calls), 2)


def _queue(path: Path, text: str, root: Path, **kwargs) -> dict:
    path.write_text(text)
    return runqueue.load_queue(str(path), root=str(root), **kwargs)


class TestQueueFile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._env = patch.dict(os.environ, {"HEDONIC_RUNS_DIR": str(self.root / "registry")})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    ITEMS = ('[[items]]\nname = "bench"\nrun = ["exp", "--networks", "dblp", "--methods", "codeseg", "--name", "bench", '
             '"--output-dir", "{root}/bench"]\n'
             '[[items]]\nname = "calib"\nrun = ["calibrate", "--networks", "dblp", "--name", "calib", '
             '"--output-dir", "{root}/calib"]\nlimits = { timeout = "30m" }\n'
             '[[items]]\nname = "base"\nrun = ["calibrate-baselines", "--networks", "dblp:top5000", "--methods", "codeseg", '
             '"--name", "base", "--output-dir", "{root}/base", "--limit-seconds", "60"]\n'
             '[[items]]\nname = "curve"\nrun = ["startcurve", "--config", "start-quality-hoc-local", "--name", "curve", '
             '"--output-dir", "{root}/curve"]\nlimits = { timeout = "none" }\n'
             '[[items]]\nname = "spec"\nrun = ["spectrum", "--name", "spec", "--output-dir", "{root}/spec"]\n')

    def test_each_item_gets_the_limit_of_its_own_level_and_its_runner_gets_it_as_limit_seconds(self):
        queue = _queue(self.root / "q.toml", 'name = "q"\n[limits]\ntimeout = "2h"\nmax_eta = "6h"\n' + self.ITEMS, self.root)
        items = {item["name"]: item for item in queue["items"]}
        self.assertEqual(queue["limits"], {"timeout": 7200.0, "max_eta": 21600.0})
        flag = {name: runqueue._option(item["run"], "--limit-seconds") for name, item in items.items()}
        self.assertEqual(flag, {"bench": "7200", "calib": "1800", "base": "60", "curve": None, "spec": None})
        self.assertEqual(items["curve"]["limits"], {"max_eta": 21600.0})  # "none" removes the inherited timeout
        lines = runqueue.limit_lines(items["spec"])
        self.assertTrue(any("not applied" in line and "identity" in line for line in lines))
        self.assertTrue(any("per one detector run of a search" in line for line in runqueue.limit_lines(items["calib"])))
        self.assertTrue(any("never skipped on an estimate" in line for line in runqueue.limit_lines(items["calib"])))

    def test_the_front_doors_limits_come_under_the_files(self):
        path = self.root / "q.toml"
        path.write_text('name = "q"\n[limits]\nmax_eta = "6h"\n' + self.ITEMS)
        queue = runqueue.load_queue(str(path), root=str(self.root), inherited={"timeout": 600.0, "max_eta": 60.0})
        self.assertEqual(queue["limits"], {"timeout": 600.0, "max_eta": 21600.0})

    def test_a_queue_item_passes_its_limits_to_every_item_of_the_inner_queue(self):
        inner = self.root / "inner.toml"
        inner.write_text('name = "inner-queue"\n' + self.ITEMS)
        outer = _queue(self.root / "outer.toml", f'name = "outer-queue"\n[limits]\ntimeout = "1h"\n[[items]]\n'
                       f'run = ["queue", "{inner}", "--root", "{{root}}/in"]\nlimits = {{ max_eta = "3h" }}\n', self.root)
        item = outer["items"][0]
        self.assertEqual((runqueue._option(item["run"], "--timeout"), runqueue._option(item["run"], "--max-eta")),
                         ("3600", "10800"))
        again = runqueue.inner_queue(item["run"])  # as the started inner queue reads its own command line
        bench = next(i for i in again["items"] if i["name"] == "bench")
        self.assertEqual((runqueue._option(bench["run"], "--limit-seconds"), bench["limits"]["max_eta"]), ("3600", 10800.0))
        self.assertEqual(again["limits"], {"timeout": 3600.0, "max_eta": 10800.0})

    def test_a_wrong_limit_is_refused_before_anything_starts(self):
        for table in ('[limits]\ntimeout = "soon"\n', '[limits]\nwait = "2h"\n'):
            with self.assertRaises(SystemExit):
                _queue(self.root / "q.toml", 'name = "q"\n' + table + self.ITEMS, self.root)


class TestMaxEta(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._env = patch.dict(os.environ, {"HEDONIC_RUNS_DIR": str(self.root / "registry")})
        self._env.start()
        runqueue._DATA_CACHE.clear()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def _work(self, estimates: dict) -> tuple[list, dict]:
        queue = _queue(self.root / "q.toml", 'name = "q"\n[limits]\nmax_eta = "1h"\n[[items]]\nname = "bench"\n'
                       'run = ["exp", "--networks", "dblp,amazon", "--methods", "codeseg", "--name", "bench", '
                       f'"--output-dir", "{self.root}/bench"]\n', self.root)
        calls = []

        def launch(name, run, resuming):
            calls.append(run)
            out = self.root / "runs" / name
            out.mkdir(parents=True, exist_ok=True)
            entry = {"name": name, "kind": "benchmark", "output_dir": str(out), "created": time.time(), "session": "",
                     "backend": "process", "log": str(out / "w.log"), "argv": []}
            rm.write_json(rm.entry_path(name), entry)
            rm.write_json(rm.progress_path(entry), {"status": "completed", "plan": [], "setup": []})
            return 0

        out = self.root / "queues" / "q"
        out.mkdir(parents=True, exist_ok=True)
        entry = {"name": "q", "kind": "queue", "items": queue["items"], "file": queue["file"], "output_dir": str(out),
                 "created": time.time(), "session": "", "backend": "process", "log": str(out / "worker.log"),
                 "schedule": "file"}
        prog: dict = {}
        with patch.object(ql, "estimates", lambda run: estimates), patch.object(runqueue, "POLL_SECONDS", 0), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runqueue.work("q", entry, prog, launch_item=launch), 0)
        return calls, prog

    def test_a_pair_whose_one_run_is_estimated_over_max_eta_is_left_out(self):
        calls, prog = self._work({"dblp": (600.0, "measured on this pair"), "amazon": (7200.0, "measured on this pair")})
        self.assertEqual(runqueue._option(calls[0], "--networks"), "dblp")
        self.assertEqual([s["pair"] for s in prog["items"][0]["skipped"]], ["amazon"])

    def test_an_item_with_every_pair_over_is_skipped_and_one_without_estimates_runs_whole(self):
        calls, prog = self._work({"dblp": (4000.0, "x"), "amazon": (7200.0, "x")})
        self.assertEqual(calls, [])
        self.assertIn("skipped by max_eta 1h", prog["items"][0]["note"])
        calls, _ = self._work({})
        self.assertEqual(runqueue._option(calls[0], "--networks"), "dblp,amazon")

    def test_an_estimate_is_one_run_measured_first_and_never_longer_than_the_runs_timeout(self):
        measured = {"a": [10.0, 30.0, 20.0], "b": [100.0]}
        edges = {"a": 1000, "b": 1000, "c": 4000}
        self.assertEqual(ql._pick("a", measured, edges, None, None), (20.0, "measured on this pair"))
        seconds, basis = ql._pick("c", measured, edges, 5.0, None)
        self.assertAlmostEqual(seconds, 240.0)  # the median rate of the other pairs (0.06 s/edge) x 4000 edges
        self.assertIn("scaled by edges", basis)
        self.assertEqual(ql._pick("d", measured, edges, 5.0, None), (5.0, "the method's planning constant"))
        self.assertEqual(ql._pick("c", measured, edges, None, 60.0)[0], 60.0)
        self.assertIsNone(ql._pick("d", measured, edges, None, None))


if __name__ == "__main__":
    unittest.main()
