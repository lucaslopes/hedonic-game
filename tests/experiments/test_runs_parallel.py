"""Forked one-shot workers, the cross-process core budget and the free-memory floor (``overlapping.parallel``).

In hedonic 1 the durable-run worker's memory guard reads the same free-memory figure; the workers and slots stay the
shared machinery they were, tested here on toy functions.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments.overlapping import parallel


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


if __name__ == "__main__":
    unittest.main()
