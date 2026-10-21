"""The per-run memory watchdog: a detector that drives the machine below its free-memory floor is stopped alone and
recorded as ``status: "memory"``, the way a time limit records a timeout; large pairs run alone; a run can be declared
out of memory without starting.

The free memory is faked from the watched process's own resident size (a fake low floor): the child really allocates,
and the watchdog really kills it, without the machine coming anywhere near its real floor.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments import timing
from hedonic.experiments.overlapping import baseline_calibration as bc
from hedonic.experiments.overlapping import codeseg_reproduction as cr
from hedonic.experiments.overlapping import limits, memwatch, parallel
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import unit_integrity

CHUNK = 16 * 1024 * 1024  # bytes a hungry child adds at each step
STEPS = 96                # at most 1.5 GB, far more than the fake floor lets it keep
FAKE_FLOOR = 90.0         # percent: with the fake measure below, about 100 MB above the start
MB_PER_POINT = 10.0       # the fake measure: one percentage point of "free memory" per 10 MB the process gained


def rss_mb(pids: list[int]) -> float:
    """Resident size of these processes in MB (``ps``; a gone process counts 0)."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", ",".join(map(str, pids))], capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split() if x.isdigit()) / 1024.0


def fake_free(pid: int):
    """``parallel.free_memory_percent`` as ``pid`` and the processes below it grow: 100% at their size now, one point
    less per 10 MB they gain (this test process, for a child it starts later)."""
    base = rss_mb([pid])

    def measure() -> float:
        tree = [pid, *memwatch.descendants(pid)]
        return 100.0 - max(0.0, rss_mb(tree) - base) / MB_PER_POINT

    return measure


def hungry(_path: str | None = None) -> None:
    """Allocate and touch memory step by step, then wait: only a kill ends it early."""
    kept = []
    for _ in range(STEPS):
        kept.append(bytearray(b"\x01") * CHUNK)
        time.sleep(0.02)
    time.sleep(60)


HUNGRY_PROGRAM = textwrap.dedent(f"""
    import time
    kept = []
    for _ in range({STEPS}):
        kept.append(bytearray(b"\\x01") * {CHUNK})
        time.sleep(0.02)
    time.sleep(60)
""")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(out) and not out.startswith("Z")


class TestStands(unittest.TestCase):
    def test_a_stop_stands_while_the_floor_is_at_least_the_one_that_stopped_it(self):
        record = memwatch.stopped(4.0, 10.0)
        self.assertEqual(record["status"], "memory")
        self.assertTrue(memwatch.stands(record, 10.0))
        self.assertTrue(memwatch.stands(record, 15.0))
        self.assertFalse(memwatch.stands(record, 5.0))   # a lower floor lets it run again
        self.assertFalse(memwatch.stands(record, 0.0))   # no floor at all: nothing would stop it, so it runs
        with patch.dict(os.environ, {memwatch.RERUN_ENV: "1"}):
            self.assertFalse(memwatch.stands(record, 10.0))

    def test_a_declared_run_stands_until_a_rerun_is_asked(self):
        record = memwatch.declared("evidence")
        self.assertTrue(memwatch.stands(record, 0.0))
        self.assertIn("evidence", record["reason"])
        with patch.dict(os.environ, {memwatch.RERUN_ENV: "1"}):
            self.assertFalse(memwatch.stands(record))

    def test_limits_and_unit_integrity_treat_a_memory_stop_like_a_timeout(self):
        record = {"status": "memory", "memory_floor_percent": 10.0, "free_memory_percent": 6.0}
        with patch.object(parallel, "memory_floor", return_value=10.0):
            self.assertTrue(limits.stands(record, None))
            self.assertTrue(unit_integrity.final(record, None))
        with patch.object(parallel, "memory_floor", return_value=5.0):
            self.assertFalse(limits.stands(record, None))
            self.assertFalse(unit_integrity.final(record, None))
        self.assertTrue(unit_integrity.final({"status": "memory", "memory_declared": True}, None))

    def test_declarations_are_read_in_the_runs_spelling_and_cite_the_evidence(self):
        got = memwatch.parse_declared("clique_percolation@livejournal:top5k, clique_percolation@friendster:top5000",
                                      qs.pair_key)
        self.assertEqual(set(got), {("clique_percolation", "livejournal:top5000"),
                                    ("clique_percolation", "friendster:top5000")})
        self.assertIn("killed by the OS for memory", got["clique_percolation", "livejournal:top5000"])
        self.assertEqual(memwatch.spell_declared(got),
                         "clique_percolation@friendster:top5000,clique_percolation@livejournal:top5000")
        with self.assertRaises(ValueError):
            memwatch.parse_declared("clique_percolation", qs.pair_key)


class TestLastResort(unittest.TestCase):
    def test_a_whole_run_guard_waits_for_the_runs_own_watchdog_unless_memory_is_critical(self):
        clock = [100.0]
        with patch.object(parallel.time, "monotonic", lambda: clock[0]):
            guard = parallel.FloorWatch(10.0)
            self.assertFalse(guard.breached(None))
            self.assertFalse(guard.breached(50.0))
            self.assertFalse(guard.breached(8.0))  # below the floor: the run's own watchdog acts first
            clock[0] += parallel.LAST_RESORT_SECONDS - 0.1
            self.assertFalse(guard.breached(8.0))
            clock[0] += 0.2
            self.assertTrue(guard.breached(8.0))  # still below the floor: the whole run is stopped
            self.assertFalse(guard.breached(30.0))  # recovered: the wait starts over
            self.assertFalse(guard.breached(8.0))
            self.assertTrue(guard.breached(4.0))  # below half the floor: at once


class TestWatchdog(unittest.TestCase):
    def test_kill_tree_reaches_a_grandchild_in_a_session_of_its_own(self):
        child = subprocess.Popen([sys.executable, "-c", textwrap.dedent("""
            import subprocess, sys, time
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            time.sleep(60)
        """)])
        try:
            deadline = time.time() + 10
            below: list[int] = []
            while time.time() < deadline and not below:
                below = memwatch.descendants(child.pid)
                time.sleep(0.05)
            self.assertEqual(len(below), 1)
            memwatch.kill_tree(child.pid)
            child.wait(timeout=10)
            deadline = time.time() + 10
            while time.time() < deadline and alive(below[0]):
                time.sleep(0.05)
            self.assertFalse(alive(below[0]))
        finally:
            memwatch.kill_tree(child.pid)
            child.wait()

    def test_a_forked_child_that_allocates_is_killed_at_the_floor_and_says_so(self):
        proc = mp.get_context("fork").Process(target=hungry)
        proc.start()
        try:
            with patch.object(parallel, "free_memory_percent", fake_free(proc.pid)):
                started = time.monotonic()
                watch = memwatch.join(proc, timeout=120, floor=FAKE_FLOOR, poll=0.05)
            self.assertIsNotNone(watch.tripped)
            self.assertLess(time.monotonic() - started, 60)  # stopped by memory, long before its own end
            self.assertEqual(watch.tripped["status"], "memory")
            self.assertEqual(watch.tripped["memory_floor_percent"], FAKE_FLOOR)
            self.assertLess(watch.tripped["free_memory_percent"], FAKE_FLOOR)
            self.assertIsNotNone(proc.exitcode)
        finally:
            if proc.is_alive():
                proc.kill()
            proc.join()

    def test_a_child_that_stays_small_is_joined_as_before(self):
        proc = mp.get_context("fork").Process(target=time.sleep, args=(0.2,))
        proc.start()
        with patch.object(parallel, "free_memory_percent", return_value=50.0):
            watch = memwatch.join(proc, timeout=30, floor=10.0, poll=0.05)
        self.assertIsNone(watch.tripped)
        self.assertEqual(proc.exitcode, 0)

    def test_the_detector_clock_kills_a_program_at_the_floor_and_raises(self):
        with patch.object(parallel, "memory_floor", return_value=FAKE_FLOOR), \
                patch.object(parallel, "free_memory_percent", fake_free(os.getpid())):
            with timing.detector_run() as clock:
                with self.assertRaises(memwatch.MemoryLimitReached):
                    clock.process([sys.executable, "-c", HUNGRY_PROGRAM], label="hungry", timeout=120)
        self.assertEqual(clock.memory_stop["status"], "memory")
        self.assertTrue(clock.regions[-1]["stopped_for_memory"])
        self.assertLess(clock.seconds, 60)

    def test_a_detector_in_this_process_is_interrupted_at_the_floor(self):
        measure = fake_free(os.getpid())
        kept = []
        with patch.object(parallel, "free_memory_percent", measure):
            with self.assertRaises(memwatch.MemoryLimitReached) as caught:
                with memwatch.in_process(floor=FAKE_FLOOR, poll=0.01) as watch:
                    self.assertIsNotNone(watch)
                    for _ in range(STEPS):
                        kept.append(bytearray(b"\x01") * CHUNK)
        del kept
        self.assertEqual(caught.exception.floor, FAKE_FLOOR)
        self.assertEqual(watch.tripped["status"], "memory")
        self.assertEqual(memwatch.signal.getitimer(memwatch.signal.ITIMER_PROF), (0.0, 0.0))  # the timer is off again

    def test_without_a_floor_nothing_is_watched(self):
        with memwatch.in_process(floor=0.0) as watch:
            self.assertIsNone(watch)


class TestBaselineCalibration(unittest.TestCase):
    def test_an_evaluation_stopped_for_memory_is_recorded_and_not_transferred(self):
        def runner(method, graph, params, seed):
            hungry()
            return []

        with patch.object(parallel, "memory_floor", return_value=FAKE_FLOOR), \
                patch.object(parallel, "free_memory_percent", fake_free(os.getpid())):
            out = bc.run_isolated(runner, "clique_percolation", None, {}, 0, timeout=120)
        self.assertEqual(out["status"], "memory")
        self.assertEqual(out["memory_floor_percent"], FAKE_FLOOR)
        self.assertLess(out["detect_seconds"], 60)
        result = bc._all_failed("clique_percolation", "rule", [{"status": "memory", "reason": out["reason"]}], "stop")
        self.assertEqual(result.status, "out_of_memory")  # --transfer-missing fills "failed" searches only


class TestAlone(unittest.TestCase):
    def test_an_item_marked_alone_never_runs_beside_another(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "spans"

            def work(item):
                start = time.monotonic()
                time.sleep(0.3)
                with open(log, "a") as handle:
                    handle.write(json.dumps([item, start, time.monotonic()]) + "\n")
                return item

            items = ["big-1", "a", "b", "big-2", "c", "d"]
            got = dict(parallel.run_forked(work, items, 3, floor=0, alone=lambda item: item.startswith("big")))
            self.assertEqual(sorted(got.values()), sorted(items))
            spans = [json.loads(line) for line in log.read_text().splitlines()]
            for name, start, end in spans:
                if not name.startswith("big"):
                    continue
                for other, s2, e2 in spans:
                    if other != name:
                        self.assertTrue(e2 <= start or s2 >= end, f"{other} ran beside {name}")
            small = [(s, e) for n, s, e in spans if not n.startswith("big")]
            self.assertTrue(any(s2 < e1 and s1 < e2 for i, (s1, e1) in enumerate(small) for s2, e2 in small[i + 1:]),
                            "the other items still run side by side")


class TestDeclared(unittest.TestCase):
    def test_a_declared_benchmark_unit_is_recorded_without_loading_and_counts_as_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = qs.build_parser().parse_args([
                "--networks", "livejournal:top5000", "--methods", "clique_percolation", "--seeds", "0",
                "--output-dir", tmp, "--network-root", str(Path(tmp) / "no-networks-here"),
                "--out-of-memory", "clique_percolation@livejournal:top5k"])
            cfg = qs.config_from_args(args)
            self.assertIn("--out-of-memory", cfg.command())
            m = qs.BY_KEY["clique_percolation"]
            argv, out = qs.unit_argv(cfg, "livejournal:top5000", 0, m, None)
            self.assertIn("--memory-declared", argv)
            plain = qs.Config(**{**cfg.__dict__, "out_of_memory": ""})
            self.assertEqual(qs.unit_digest(cfg, "livejournal:top5000", 0, m, None),
                             qs.unit_digest(plain, "livejournal:top5000", 0, m, None))  # policy, not identity
            self.assertEqual(cr.main(argv), 0)
            record = json.loads((out / "runs" / "livejournal" / f"{m.runner}.json").read_text())
            self.assertEqual(record["status"], "memory")
            self.assertTrue(record["memory_declared"])
            self.assertIn("killed by the OS for memory", record["reason"])
            self.assertTrue(unit_integrity.final(record, None))
            self.assertEqual(qs.status_text(record["status"]), "out of memory")
            self.assertIn("out of memory", qs.result_line("tag", record))

    def test_a_declaration_must_name_a_planned_unit(self):
        args = qs.build_parser().parse_args(["--networks", "dblp:top5000", "--methods", "clique_percolation",
                                             "--out-of-memory", "clique_percolation@livejournal:top5000"])
        with self.assertRaises(ValueError):
            qs.config_from_args(args)


if __name__ == "__main__":
    unittest.main()
