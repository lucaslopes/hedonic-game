"""Forked one-shot workers and the cross-process core budget, plus the equivalence of the parallel runners.

Every runner that takes ``--workers`` must give the same records with 1 worker as with several; these tests run the
real shipped study code on small graphs both ways and compare what is stored.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import igraph as ig

from hedonic.experiments.overlapping import calibrate as cal
from hedonic.experiments.overlapping import controlled_overlap_config as wrapper
from hedonic.experiments.overlapping import parallel
from hedonic.experiments.overlapping import start_quality as sq


# The start-quality fixtures (as tests/test_start_quality.py of the 1.x tree): two overlapping cliques,
# A = {0,1,2,3} and B = {3,4,5,6,7}; vertex 3 is in both.
GT = [[0, 1, 2, 3], [3, 4, 5, 6, 7]]
N = 8


def tiny_graph() -> ig.Graph:
    edges = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    edges += [(a, b) for a in range(3, 8) for b in range(a + 1, 8)]
    return ig.Graph(n=N, edges=edges)


def tiny_job():
    """A real ``gt_spectrum.Job`` on the tiny graph (only the prepared dataset is stubbed)."""
    from types import SimpleNamespace

    from hedonic.experiments.overlapping import gt_spectrum as gs

    prepared = SimpleNamespace(dataset=SimpleNamespace(graph=tiny_graph(), cover=GT))
    return gs.Job("tiny", "top5000", prepared, None)


def plan_for(folder, *, levels, seeds, cap=2, gamma=0.5, overwrite=False, phase="local") -> dict:
    return {"config": None, "phase": phase, "networks": ["tiny"], "seeds": list(seeds), "levels": list(levels),
            "output": Path(folder), "settings": {"tiny": {"cap": cap, "gamma": gamma, "source": "test"}},
            "dry_run": False, "overwrite": overwrite}




def _square(x):
    return x * x


def _nap(x):
    time.sleep(0.3)
    return x


class TestAdmission(unittest.TestCase):
    def test_admit_and_stagger_delay_starts_but_never_stall_the_queue(self):
        self.assertEqual(sorted(v for _, v in parallel.run_forked(_nap, range(4), 4, admit=lambda running: False)), [0, 1, 2, 3])
        started = time.monotonic()
        out = sorted(v for _, v in parallel.run_forked(_nap, range(3), 3, stagger=0.4,
                                                       admit=lambda running: True))
        self.assertEqual(out, [0, 1, 2])
        self.assertGreaterEqual(time.monotonic() - started, 0.8)  # three starts, 0.4 s apart


class TestWorkers(unittest.TestCase):
    def test_default_workers_prefers_the_argument_then_the_environment(self):
        self.assertEqual(parallel.default_workers(3), 3)
        self.assertEqual(parallel.default_workers(0), 1)
        with patch.dict(os.environ, {parallel.WORKERS_ENV: "5"}):
            self.assertEqual(parallel.default_workers(), 5)
            self.assertEqual(parallel.default_workers(2), 2)
        with patch.dict(os.environ, {parallel.WORKERS_ENV: "not a number"}):
            self.assertGreaterEqual(parallel.default_workers(), 1)
        self.assertGreaterEqual(parallel.performance_cores(), 1)

    def test_results_are_indexed_and_complete_for_any_worker_count(self):
        items = list(range(9))
        for workers in (1, 2, 4, 20):
            self.assertEqual(parallel.map_forked(_square, items, workers), [x * x for x in items], workers)
        self.assertEqual(list(parallel.run_forked(_square, [], 4)), [])

    def test_one_worker_runs_in_this_process_and_several_fork(self):
        here = os.getpid()
        self.assertEqual(set(parallel.map_forked(lambda _: os.getpid(), [0, 1, 2], 1)), {here})
        pids = parallel.map_forked(lambda _: os.getpid(), [0, 1, 2], 3)
        self.assertNotIn(here, pids)
        self.assertEqual(len(set(pids)), 3)

    def test_children_share_the_parents_memory_without_pickling_it(self):
        big = {"graph": list(range(1000)), "closure": lambda: 7}  # a lambda cannot be pickled
        self.assertEqual(parallel.map_forked(lambda i: big["closure"]() + big["graph"][i], [1, 2], 2), [8, 9])

    def test_a_failing_worker_reports_its_traceback_and_the_others_still_finish(self):
        def task(i):
            if i == 1:
                raise ValueError("boom in item 1")
            time.sleep(0.3)
            return i

        seen = []
        with self.assertRaises(parallel.WorkerFailed) as caught:
            for index, value in parallel.run_forked(task, [0, 1, 2], 3):
                seen.append((index, value))
        self.assertIn("boom in item 1", str(caught.exception))
        self.assertEqual(sorted(seen), [(0, 0), (2, 2)])

    def test_a_killed_worker_is_a_failure_not_a_hang(self):
        with self.assertRaises(parallel.WorkerFailed) as caught:
            parallel.map_forked(lambda i: os._exit(9) if i == 1 else i, [0, 1], 2)
        self.assertIn("without a result", str(caught.exception))

    def test_at_most_workers_run_at_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            def task(i):
                (Path(tmp) / f"{i}.start").write_text(str(time.time()))
                time.sleep(0.4)
                (Path(tmp) / f"{i}.end").write_text(str(time.time()))
                return i

            parallel.map_forked(task, list(range(4)), 2)
            spans = sorted((float((Path(tmp) / f"{i}.start").read_text()), float((Path(tmp) / f"{i}.end").read_text()))
                           for i in range(4))
            overlap = max(sum(1 for s, e in spans if s <= t < e) for t, _ in spans)
            self.assertEqual(overlap, 2)


class TestMemoryFloor(unittest.TestCase):
    """The pool stops itself before the machine runs out of memory (a runaway worker once took the machine down)."""

    def test_low_free_memory_kills_the_workers_and_says_so(self):
        marker = tempfile.mkdtemp()

        def slow(i):
            Path(marker, f"{i}.started").write_text("x")
            time.sleep(30)  # would outlive the test if the pool did not kill it
            return i

        started = time.time()
        with patch.object(parallel, "free_memory_percent", return_value=4.0):
            with self.assertRaises(parallel.MemoryLow) as caught:
                list(parallel.run_forked(slow, [0, 1, 2], 3))
        self.assertLess(time.time() - started, 10)
        self.assertIn("free memory fell to 4%", str(caught.exception))
        self.assertIsInstance(caught.exception, parallel.WorkerFailed)  # callers that handle failures handle this

    def test_a_worker_that_eats_the_memory_is_stopped_midway(self):
        readings = iter([80.0, 70.0, 6.0])  # free memory shrinks while a worker runs

        def reading():
            return next(readings, 6.0)

        started = time.time()
        with patch.object(parallel, "free_memory_percent", side_effect=reading):
            with self.assertRaises(parallel.MemoryLow):
                list(parallel.run_forked(lambda i: time.sleep(30), [0, 1], 2))
        self.assertLess(time.time() - started, 10)

    def test_finished_items_are_yielded_before_the_stop(self):
        readings = iter([80.0, 80.0, 5.0])

        def task(i):
            if i == 0:
                return "quick"
            time.sleep(30)

        seen = []
        with patch.object(parallel, "free_memory_percent", side_effect=lambda: next(readings, 5.0)):
            with self.assertRaises(parallel.MemoryLow):
                for pair in parallel.run_forked(task, [0, 1], 2):
                    seen.append(pair)
        self.assertEqual(seen, [(0, "quick")])

    def test_enough_free_memory_changes_nothing(self):
        with patch.object(parallel, "free_memory_percent", return_value=60.0):
            self.assertEqual(parallel.map_forked(_square, [1, 2, 3, 4], 3), [1, 4, 9, 16])

    def test_unknown_memory_and_a_zero_floor_never_stop_a_run(self):
        with patch.object(parallel, "free_memory_percent", return_value=None):
            self.assertEqual(parallel.map_forked(_square, [1, 2, 3], 2), [1, 4, 9])
        with patch.object(parallel, "free_memory_percent", return_value=0.0):
            self.assertEqual(list(parallel.run_forked(_square, [3], 2, floor=0)), [(0, 9)])

    def test_the_floor_comes_from_the_argument_then_the_environment_then_ten_percent(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(parallel.MEMORY_FLOOR_ENV, None)
            self.assertEqual(parallel.memory_floor(), 10.0)
        with patch.dict(os.environ, {parallel.MEMORY_FLOOR_ENV: "25"}):
            self.assertEqual(parallel.memory_floor(), 25.0)
        with patch.dict(os.environ, {parallel.MEMORY_FLOOR_ENV: "0"}):
            self.assertEqual(parallel.memory_floor(), 0.0)
        with patch.dict(os.environ, {parallel.MEMORY_FLOOR_ENV: "not a number"}):
            self.assertEqual(parallel.memory_floor(), 10.0)

    def test_this_machine_reports_a_sensible_free_memory_figure(self):
        free = parallel.free_memory_percent()
        if free is not None:
            self.assertTrue(0.0 <= free <= 100.0)


class TestSlots(unittest.TestCase):
    def test_the_budget_is_counted_and_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            slots = parallel.Slots(tmp, 2)
            with slots.hold():
                with slots.hold():
                    self.assertIsNone(slots._try())  # both taken
                extra = slots._try()  # one released
                self.assertIsNotNone(extra)
                os.close(extra)
            second = parallel.Slots(tmp, 2)  # another handle on the same folder sees them free again
            first, other = second._try(), second._try()
            self.assertTrue(first is not None and other is not None)
            self.assertIsNone(second._try())
            os.close(first)
            os.close(other)

    def test_a_forked_child_does_not_keep_its_parents_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            slots = parallel.Slots(tmp, 1)
            pid = None
            with slots.hold():
                pid = os.fork()
                if pid == 0:
                    time.sleep(2.0)
                    os._exit(0)
                time.sleep(0.3)  # the child runs its after-fork hook (closes the inherited descriptor) at once
            try:
                fd = slots._try()  # the child is still alive, but the slot is free
                self.assertIsNotNone(fd)
                os.close(fd)
            finally:
                os.kill(pid, 9)
                os.waitpid(pid, 0)


FAKE_SLOW_SCRIPT = textwrap.dedent('''
    import json, sys, time
    from pathlib import Path
    a = sys.argv
    get = lambda flag: a[a.index(flag) + 1]
    net, phase, out = get("--instance"), get("--phase"), Path(get("--out"))
    out.mkdir(parents=True, exist_ok=True)
    mark = out / f"{net}-{phase}.span"
    start = time.time()
    time.sleep(0.8)
    mark.write_text(json.dumps([start, time.time(), a[1:]]))
    gamma = 0.125 if phase == "local" else 0.0625
    record = {"kind": "tune", "instance": {"spec": net}, "phase": phase,
              "snap": {"fraction": "1/8", "kept": True},
              "final": {"gamma": gamma, "M": 6, "f1_median": 0.5, "onmi_median": 0.6, "f1_sd": 0.0,
                        "confirm_seeds": [0]},
              "detector_evaluations": 1, "wall_hours": 0.0}
    with (out / "searches.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\\n")
''')


class TestParallelCalibration(unittest.TestCase):
    def plan(self, tmp: str, *extra: str) -> dict:
        script = Path(tmp) / "slow_tune.py"
        script.write_text(FAKE_SLOW_SCRIPT)
        return cal.resolve(["--script", str(script), "--output-dir", str(Path(tmp) / "cal"), "--name", "cal-par", *extra])

    def test_workers_share_one_budget_and_are_replayed_by_a_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(tmp, "--workers", "4")
            self.assertEqual(plan["workers"], 4)
            for search in cal.searches(plan):
                argv = cal.search_argv(plan, search)
                self.assertEqual(argv[argv.index("--jobs") + 1], "4")
                self.assertEqual(argv[argv.index("--slot-count") + 1], "4")
                self.assertEqual(argv[argv.index("--slot-dir") + 1], str(plan["output"] / "ledger" / "slots"))
            self.assertEqual(cal.resolve(cal.launch_argv(plan))["workers"], 4)
            one = self.plan(tmp, "--workers", "1")
            self.assertNotIn("--jobs", cal.search_argv(one, cal.searches(one)[0]))
            with self.assertRaises(ValueError):
                self.plan(tmp, "--workers", "0")

    def test_the_four_searches_run_side_by_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(tmp, "--workers", "4", "--networks", "dblp,amazon,youtube,livejournal")
            lines: list[str] = []
            events: list[dict] = []
            started = time.time()
            code = cal.run_calibration(plan, emit=lines.append, progress=events.append)
            elapsed = time.time() - started
            self.assertEqual(code, 0)
            self.assertEqual(cal.state_of(plan), "complete")
            spans = [json.loads(path.read_text()) for path in (plan["output"] / "ledger").glob("*.span")]
            self.assertEqual(len(spans), 4)
            self.assertLess(max(s[0] for s in spans), min(s[1] for s in spans))  # all four were alive at the same moment
            self.assertLess(elapsed, 4 * 0.8)  # one after another would take at least 3.2 s
            self.assertTrue(any(line.startswith("[hedonic/dblp] ") for line in lines))  # interleaved lines say whose they are
            self.assertEqual(sorted(e["key"] for e in events if e["type"] == "unit"), sorted(s["key"] for s in cal.searches(plan)))
            self.assertEqual(max(e["done"] for e in events if e["type"] == "unit"), 4)
            data = cal.read_tuned(plan)
            self.assertTrue(data["complete"])
            self.assertEqual(set(data["searches"]), {s["key"] for s in cal.searches(plan)})

    def test_one_worker_is_the_old_one_at_a_time_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(tmp, "--workers", "1", "--phases", "local")
            lines: list[str] = []
            self.assertEqual(cal.run_calibration(plan, emit=lines.append), 0)
            spans = sorted(json.loads(path.read_text())[:2] for path in (plan["output"] / "ledger").glob("*.span"))
            self.assertLessEqual(spans[0][1], spans[1][0] + 0.05)  # the second search started after the first ended
            self.assertFalse(any(line.startswith("[") for line in lines))

    def test_a_failing_search_does_not_discard_the_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(tmp, "--workers", "4")
            Path(plan["script"]).write_text(FAKE_SLOW_SCRIPT.replace(
                'net, phase, out = get("--instance"), get("--phase"), Path(get("--out"))',
                'net, phase, out = get("--instance"), get("--phase"), Path(get("--out"))\n'
                'if (net, phase) == ("amazon", "local"): sys.exit(3)'))
            with self.assertRaises(RuntimeError):
                cal.run_calibration(plan, emit=lambda _: None)
            self.assertEqual(cal.recorded(plan), {"hedonic/dblp"})
            self.assertEqual(cal.state_of(plan), "partial")


class TestStartQualityWorkers(unittest.TestCase):
    def run_with(self, folder: str, workers: int) -> list[dict]:
        job = tiny_job()
        plan = plan_for(folder, levels=[0.0, 0.5, "random"], seeds=[0, 1, 2])
        plan["workers"] = workers
        lines: list[str] = []
        self.assertEqual(sq.run_study(plan, loader=lambda network: job, emit=lines.append), 0)
        self.assertEqual([line.split()[0] for line in lines if line.startswith("[")], [f"[{i}/9]" for i in range(1, 10)])
        return [json.loads(line) for line in (Path(folder) / sq.RESULTS_NAME).read_text().splitlines()]

    def test_records_do_not_depend_on_the_number_of_workers(self):
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as several:
            sequential = {r["key"]: r for r in self.run_with(one, 1)}
            parallel_records = {r["key"]: r for r in self.run_with(several, 3)}
        self.assertEqual(set(sequential), set(parallel_records))
        self.assertEqual(len(sequential), 9)
        for key, record in sequential.items():
            other = parallel_records[key]
            self.assertEqual(record["workers"], 1)
            self.assertEqual(other["workers"], 3)
            for field in ("seconds", "workers", "runtime"):
                record.pop(field), other.pop(field)
            self.assertEqual(record, other, key)

    def test_a_rerun_with_workers_skips_what_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_with(folder, 2)
            job = tiny_job()
            plan = plan_for(folder, levels=[0.0, 0.5, "random"], seeds=[0, 1, 2])
            plan["workers"] = 4
            lines: list[str] = []
            sq.run_study(plan, loader=lambda network: self.fail("loaded a graph with nothing to run"), emit=lines.append)
            self.assertIn("9 already", lines[0])

    def test_workers_flag_is_planned_and_replayed(self):
        plan = sq.resolve(["--networks", "dblp", "--seeds", "0", "--levels", "0", "--workers", "3"])
        self.assertEqual(plan["workers"], 3)
        self.assertEqual(sq.resolve(sq.launch_argv(plan))["workers"], 3)
        self.assertIn("workers  3", "\n".join(sq.plan_lines(plan)))
        with self.assertRaises(ValueError):
            sq.resolve(["--networks", "dblp", "--workers", "0"])


GRID = ["--n", "40", "--mus", "0.2,0.4", "--overlap-fractions", "0.1,0.2", "--overlap-memberships", "2",
        "--secondary-edge-probabilities", "0.2", "--graph-seeds", "0,1", "--phases", "local",
        "--max-memberships", "1,gt", "--starts", "singleton,gt-primary", "--resolution-multipliers", "1",
        "--timeout-per-run", "30"]


def _strip_times(value):
    if isinstance(value, dict):
        return {k: _strip_times(v) for k, v in value.items() if "elapsed_seconds" not in k}
    if isinstance(value, list):
        return [_strip_times(v) for v in value]
    return value


class TestControlledWorkers(unittest.TestCase):
    def run_with(self, folder: str, *extra: str) -> tuple[dict, list[str]]:
        output = Path(folder) / "controlled.json"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = wrapper.main([*GRID, "--output", str(output), *extra])
        self.assertEqual(code, 0)
        return json.loads(output.read_text()), buffer.getvalue().splitlines()

    def test_the_output_does_not_depend_on_the_number_of_workers(self):
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as several:
            sequential, seq_lines = self.run_with(one, "--workers", "1")
            concurrent, par_lines = self.run_with(several, "--workers", "3")
            self.assertTrue((Path(several) / "controlled.execution.json").is_file())
            self.assertFalse((Path(one) / "controlled.execution.json").exists())
            self.assertEqual(json.loads((Path(several) / "controlled.execution.json").read_text())["workers"], 3)
        self.assertEqual(len(sequential["instances"]), 8)
        self.assertEqual(_strip_times(sequential), _strip_times(concurrent))  # identity, protocol, instances, summaries
        progress = lambda lines: [line for line in lines if line.startswith("completed instance")]
        self.assertEqual(len(progress(seq_lines)), 8)
        self.assertEqual(len(progress(par_lines)), 8)  # once each: the study's own lines are dropped, the wrapper's kept
        self.assertEqual([line.split(":")[0] for line in progress(par_lines)],
                         [f"completed instance {i}/8" for i in range(1, 9)])
        self.assertTrue(any(line.startswith("wrote ") for line in par_lines))

    def test_the_workers_key_of_a_config_is_not_passed_to_the_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "grid.toml"
            output = Path(tmp) / "out.json"
            config.write_text('[overlapping_controlled]\nn = 40\nmus = "0.2,0.4"\noverlap-fractions = "0.1"\n'
                              'graph-seeds = "0"\nphases = "local"\nmax-memberships = "2"\nstarts = "singleton"\n'
                              'resolution-multipliers = "1"\ntimeout-per-run = 30.0\nworkers = 2\n'
                              f'output = "{output}"\n')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(wrapper.main(["--config", str(config)]), 0)
            self.assertEqual(json.loads((Path(tmp) / "out.execution.json").read_text())["workers"], 2)
            plan = wrapper.study_plan([*wrapper.config_arguments(str(config)), "--output", str(output)])
            self.assertEqual((plan["workers"], plan["groups"]), (2, 2))

    def test_the_study_module_itself_is_untouched(self):
        # The tracked v2 evidence is bound to this file's SHA-256; the parallel path lives in the wrapper.
        source = Path(wrapper.controlled_overlap.__file__).read_text(encoding="utf-8")
        self.assertNotIn("parallel", source)


if __name__ == "__main__":
    unittest.main()
