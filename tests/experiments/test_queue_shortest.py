"""`schedule = "shortest"`: each next step is the ready step with the smallest estimate, after the items it reads."""

from __future__ import annotations

import io
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments.overlapping import runmanager as rm
from hedonic.experiments.overlapping import runqueue


def _register(root: Path, name: str, status: str) -> None:
    out = root / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    entry = {"name": name, "kind": "spectrum", "output_dir": str(out), "created": time.time(), "session": "",
             "backend": "process", "log": str(out / "worker.log"), "argv": []}
    rm.write_json(rm.entry_path(name), entry)
    rm.write_json(rm.progress_path(entry), {"status": status, "plan": [], "setup": []})


class Launcher:
    def __init__(self, root: Path):
        self.root, self.calls = root, []

    def __call__(self, name: str, run: list[str], resuming: bool) -> int:
        self.calls.append(name)
        _register(self.root, name, "completed")
        return 0


def _step(item: str, pair: str | None = None, verb: str = "exp") -> dict:
    return {"index": 0, "item": item, "pair": pair, "run": [verb]}


class TestDependencies(unittest.TestCase):
    def test_an_item_waits_for_the_earlier_item_whose_folder_it_reads_and_for_its_after(self):
        queue = {"root": "/r", "items": [
            {"name": "calib", "run": ["calibrate", "--output-dir", "/r/calib"]},
            {"name": "other", "run": ["spectrum"]},
            {"name": "bench", "run": ["exp", "--tuned", "/r/calib/hedonic_tuned.json,/x/y.json"]},
            {"name": "late", "run": ["exp"], "after": ["other"]},
        ]}
        deps = runqueue.dependencies(queue)
        self.assertEqual(deps["bench"], {"calib"})
        self.assertEqual(deps["late"], {"other"})
        self.assertEqual(deps["calib"], set())
        self.assertEqual(deps["other"], set())

    def test_a_folder_name_that_only_shares_a_prefix_is_not_a_dependency(self):
        queue = {"root": "/r", "items": [{"name": "calib", "run": ["calibrate"]},
                                         {"name": "b", "run": ["exp", "--tuned", "/r/calibration-2/t.json"]}]}
        self.assertEqual(runqueue.dependencies(queue)["b"], set())


class TestOrder(unittest.TestCase):
    def test_the_shortest_ready_step_goes_first_and_no_estimate_goes_last(self):
        steps = [_step("a"), _step("b"), _step("c"), _step("d")]
        cost = {0: 100.0, 1: None, 2: 5.0, 3: 5.0}.get
        self.assertEqual(runqueue.shortest_order(steps, [0, 1, 2, 3], {}, cost), [2, 3, 0, 1])

    def test_a_step_waits_for_every_step_of_the_items_it_depends_on(self):
        steps = [_step("calib"), _step("bench", "p1"), _step("bench", "p2"), _step("cheap")]
        cost = {0: 50.0, 1: 1.0, 2: 2.0, 3: 10.0}.get
        order = runqueue.shortest_order(steps, [0, 1, 2, 3], {"bench": {"calib"}}, cost)
        self.assertEqual(order, [3, 0, 1, 2])

    def test_a_spectrum_takes_its_pairs_in_the_queue_order_and_the_others_in_any(self):
        steps = [_step("spec", "p1", "spectrum"), _step("spec", "p2", "spectrum"),
                 _step("bench", "p1"), _step("bench", "p2")]
        cost = {0: 90.0, 1: 1.0, 2: 80.0, 3: 2.0}.get
        self.assertEqual(runqueue.ready_steps(steps, [0, 1, 2, 3], {}), [0, 2, 3])
        self.assertEqual(runqueue.shortest_order(steps, [0, 1, 2, 3], {}, cost), [3, 2, 0, 1])


class TestQueueFile(unittest.TestCase):
    def test_the_schedule_key_and_after_are_read_and_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.toml"
            path.write_text('name = "q"\nschedule = "shortest"\n[[items]]\nname = "a"\nrun = ["spectrum-full"]\n'
                            '[[items]]\nname = "b"\nrun = ["spectrum-full"]\nafter = ["a"]\n')
            queue = runqueue.load_queue(str(path), root=tmp)
            self.assertEqual(queue["schedule"], "shortest")
            self.assertEqual(queue["items"][1]["after"], ["a"])
            path.write_text('name = "q"\n[[items]]\nname = "a"\nrun = ["spectrum-full"]\n')
            self.assertEqual(runqueue.load_queue(str(path), root=tmp)["schedule"], "file")
            path.write_text('name = "q"\nschedule = "fastest"\n[[items]]\nname = "a"\nrun = ["spectrum-full"]\n')
            with self.assertRaises(SystemExit):
                runqueue.load_queue(str(path), root=tmp)
            path.write_text('name = "q"\n[[items]]\nname = "a"\nrun = ["spectrum-full"]\nafter = ["b"]\n'
                            '[[items]]\nname = "b"\nrun = ["spectrum-full"]\n')
            with self.assertRaises(SystemExit):  # after names an item that comes later
                runqueue.load_queue(str(path), root=tmp)


class TestWorker(unittest.TestCase):
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

    def _entry(self, names: list[str], schedule: str) -> dict:
        out = self.root / "queues" / "q"
        out.mkdir(parents=True, exist_ok=True)
        return {"name": "q", "kind": "queue", "items": [{"name": n, "run": ["spectrum-full"]} for n in names],
                "file": "", "output_dir": str(out), "created": time.time(), "session": "", "backend": "process",
                "log": str(out / "worker.log"), "schedule": schedule}

    def _work(self, entry: dict, costs: dict) -> tuple[list[str], dict]:
        launcher, prog = Launcher(self.root), {}
        with patch.object(runqueue, "step_costs", lambda *_a, **_k: dict(costs)), redirect_stdout(io.StringIO()):
            self.assertEqual(runqueue.work("q", entry, prog, launch_item=launcher), 0)
        return launcher.calls, prog

    def test_the_worker_starts_the_shortest_item_first(self):
        calls, prog = self._work(self._entry(["long", "short", "unknown"], "shortest"), {0: 300.0, 1: 3.0, 2: None})
        self.assertEqual(calls, ["short", "long", "unknown"])
        self.assertEqual(prog["status"], "completed")
        self.assertEqual(prog["taken"], 3)

    def test_the_file_schedule_keeps_the_file_order(self):
        calls, _ = self._work(self._entry(["long", "short"], "file"), {0: 300.0, 1: 3.0})
        self.assertEqual(calls, ["long", "short"])


class TestBenchmarkPricedFromDisk(unittest.TestCase):
    """A pair the run has not timed is priced from the units on disk, not from the whole-graph planning constant."""

    def _family(self, history):
        from types import SimpleNamespace

        from hedonic.experiments.overlapping import queuebuild as qb

        method = SimpleNamespace(label="NEO-K-Means", key="neo_kmeans")
        units = [(net, 0, method) for net in ("dblp:top5000", "amazon:top5000", "amazon")]
        cfg = SimpleNamespace(robustness=True, order="seed,network,method")
        with patch.object(qb.qs, "expected_seconds", lambda m, c, n: 36_000.0):
            family = qb.benchmark(cfg, units, [True, False, False], {"plan": []}, now=0.0, history=history).families[0]
        return {unit.coords[0]: family.cost(unit) for unit in family.units if not unit.done}

    def test_the_same_method_on_a_pair_of_the_same_cover_prices_it_per_edge(self):
        costs = self._family({"seconds": {("dblp:top5000", "neo_kmeans"): [100.0]},
                              "edges": {"dblp:top5000": 1000, "amazon:top5000": 500}})
        self.assertEqual(costs["Amazon · top5k"], (50.0, "estimate"))
        self.assertEqual(costs["Amazon · all"], (36_000.0, "bound"))  # another cover says nothing: planning constant

    def test_without_history_the_planning_constant_stays(self):
        self.assertEqual(self._family(None)["Amazon · top5k"], (36_000.0, "bound"))


if __name__ == "__main__":
    unittest.main()
