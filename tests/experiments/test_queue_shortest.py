"""`schedule = "shortest"`: each next run is the ready run with the smallest estimate, after the runs it reads."""

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
    entry = {"name": name, "kind": "disjoint", "output_dir": str(out), "created": time.time(), "session": "",
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


def _disjoint_item(root: Path, name: str, extra: str = "") -> str:
    return (f'[[items]]\nname = "{name}"\nrun = ["disjoint", "--preset", "smoke", "--name", "{name}", '
            f'"--output-dir", "{root / "stores" / name}", "--yes"]\n{extra}')


class TestOrder(unittest.TestCase):
    def test_dependencies_come_from_folders_and_after(self):
        queue = {"root": "/r", "items": [{"name": "a", "run": ["disjoint", "--output-dir", "/r/a"]},
                                         {"name": "b", "run": ["disjoint", "--from", "/r/a/table.parquet"]},
                                         {"name": "c", "run": ["disjoint"], "after": ["a"]},
                                         {"name": "d", "run": ["disjoint", "--from", "/r/ab/x"]}]}
        deps = runqueue.dependencies(queue)
        self.assertEqual((deps["a"], deps["b"], deps["c"], deps["d"]), (set(), {"a"}, {"a"}, set()))

    def test_the_shortest_ready_run_goes_first_and_no_estimate_last(self):
        items = [{"name": n} for n in "abcd"]
        cost = {0: 100.0, 1: None, 2: 5.0, 3: 1.0}.get
        self.assertEqual(runqueue.shortest_order(items, [0, 1, 2, 3], {}, cost), [3, 2, 0, 1])
        self.assertEqual(runqueue.shortest_order(items, [0, 1, 2, 3], {"d": {"a"}}, cost), [2, 0, 3, 1])


class TestQueueFile(unittest.TestCase):
    def test_the_schedule_key_and_after_are_read_and_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, path = Path(tmp), Path(tmp) / "q.toml"
            path.write_text('name = "q"\nschedule = "shortest"\n' + _disjoint_item(root, "a")
                            + _disjoint_item(root, "b", 'after = ["a"]\n'))
            queue = runqueue.load_queue(str(path), root=tmp)
            self.assertEqual((queue["schedule"], queue["items"][1]["after"]), ("shortest", ["a"]))
            path.write_text('name = "q"\n' + _disjoint_item(root, "a"))
            self.assertEqual(runqueue.load_queue(str(path), root=tmp)["schedule"], "file")
            path.write_text('name = "q"\nschedule = "fastest"\n' + _disjoint_item(root, "a"))
            with self.assertRaises(SystemExit):
                runqueue.load_queue(str(path), root=tmp)
            path.write_text('name = "q"\n' + _disjoint_item(root, "a", 'after = ["a"]\n'))
            with self.assertRaises(SystemExit):  # an item cannot wait for itself
                runqueue.load_queue(str(path), root=tmp)

    def test_the_shipped_disjoint_queue_is_shortest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(runqueue.load_queue("disjoint", root=tmp)["schedule"], "shortest")


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

    def _work(self, names: list[str], schedule: str, costs: dict[str, float | None]) -> list[str]:
        out = self.root / "queues" / "q"
        out.mkdir(parents=True, exist_ok=True)
        entry = {"name": "q", "kind": "queue", "items": [{"name": n, "run": ["x"]} for n in names], "file": "",
                 "output_dir": str(out), "created": time.time(), "session": "", "backend": "process",
                 "log": str(out / "worker.log"), "schedule": schedule}
        launcher = Launcher(self.root)
        with patch.object(runqueue, "item_cost", lambda item: costs[item["name"]]), redirect_stdout(io.StringIO()):
            self.assertEqual(runqueue.work("q", entry, {}, launch_item=launcher), 0)
        return launcher.calls

    def test_the_worker_starts_the_shortest_run_first(self):
        self.assertEqual(self._work(["long", "short", "unknown"], "shortest",
                                    {"long": 300.0, "short": 3.0, "unknown": None}), ["short", "long", "unknown"])

    def test_the_file_schedule_keeps_the_file_order(self):
        self.assertEqual(self._work(["long", "short"], "file", {"long": 300.0, "short": 3.0}), ["long", "short"])


if __name__ == "__main__":
    unittest.main()
