"""One clock for detection: detection time covers the detector and nothing else.

These tests pin the clock primitives of ``hedonic.experiments.timing`` and the rule ``detector-call-v1`` of the
disjoint store: the error fallback, reporting and validation never reach a reported time.
"""

from __future__ import annotations

import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from hedonic.experiments import timing

SLOW = 0.6  # seconds of artificial work injected outside the detector


def _slow(func, delay=SLOW):
    def wrapper(*args, **kwargs):
        time.sleep(delay)
        return func(*args, **kwargs)

    return wrapper


class TestClockPrimitives(unittest.TestCase):
    def test_a_process_is_timed_from_exec_to_exit_and_not_for_its_output(self):
        clock = timing.DetectorClock()
        code = "import sys, time; time.sleep(0.25); sys.stdout.write('x' * 5_000_000)"
        result = clock.process([sys.executable, "-c", code], label="sleeper")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(result.stdout), 5_000_000)  # a pipe that is not drained would deadlock
        self.assertGreaterEqual(clock.seconds, 0.25)
        self.assertLess(clock.seconds, 5.0)
        self.assertTrue(clock.used)

    def test_a_timeout_kills_the_whole_process_group(self):
        clock = timing.DetectorClock()
        started = time.perf_counter()
        with self.assertRaises(subprocess.TimeoutExpired):
            clock.process([sys.executable, "-c", "import time; time.sleep(60)"], label="hang", timeout=0.4)
        self.assertLess(time.perf_counter() - started, 10.0)
        self.assertGreaterEqual(clock.seconds, 0.35)  # what ran is still recorded

    def test_peak_memory_is_the_process_own_and_does_not_leak_between_runs(self):
        clock = timing.DetectorClock()
        big = clock.process([sys.executable, "-c", "b = bytearray(250 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096])"],
                            label="big")
        small = clock.process([sys.executable, "-c", "pass"], label="small")
        self.assertGreater(big.peak_rss_bytes, 200 * 1024 * 1024)
        self.assertLess(small.peak_rss_bytes, 120 * 1024 * 1024)  # RUSAGE_CHILDREN would have kept the 250 MB
        self.assertEqual(clock.report()["detector_peak_rss_scope"], "detector_process")

    def test_a_worker_is_charged_for_its_own_report_not_for_its_start_up(self):
        clock = timing.DetectorClock()
        process = clock.process([sys.executable, "-c", "import time; time.sleep(0.4)"], label="worker", counted=False)
        self.assertEqual(clock.seconds, 0.0)
        counted = clock.adopt(0.01, process, label="worker")
        self.assertAlmostEqual(clock.seconds, 0.01)
        self.assertEqual(counted, 0.01)
        self.assertGreater(clock.report()["isolation_overhead_seconds"], 0.3)

    def test_a_report_longer_than_the_process_or_missing_is_refused(self):
        clock = timing.DetectorClock()
        process = clock.process([sys.executable, "-c", "pass"], label="w", counted=False)
        for bad in (None, "abc", -1.0, process.seconds + 5.0, float("nan")):
            with self.assertRaises(ValueError, msg=repr(bad)):
                clock.adopt(bad, process, label="w")
        self.assertEqual(clock.seconds, 0.0)

    def test_an_in_process_region_reports_the_harness_peak_with_its_scope(self):
        clock = timing.DetectorClock()
        with clock.measure("block"):
            time.sleep(0.05)
        report = clock.report()
        self.assertEqual(report["detector_peak_rss_scope"], "harness_process_lifetime")
        self.assertGreaterEqual(report["detection_seconds"], 0.05)

    def test_an_unused_clock_has_no_timing(self):
        self.assertFalse(timing.DetectorClock().used)


class TestDisjointSweepClock(unittest.TestCase):
    def test_only_the_detector_call_is_inside_the_clock(self):
        import igraph as ig
        from igraph.clustering import VertexClustering

        from hedonic import Game
        from hedonic.experiments.disjoint import sbm_sweep

        g = Game(ig.Graph.Famous("Zachary"))
        gt = VertexClustering(g, [0] * g.vcount())

        class SlowMembership(VertexClustering):
            @property
            def membership(self):
                time.sleep(SLOW)
                return super().membership

        def fake_call(graph, name, params):
            return SlowMembership(graph, [i % 2 for i in range(graph.vcount())])

        method = {"method_call_name": "community_mirror", "parameters": {}}
        with patch.object(sbm_sweep, "_call_method", fake_call):
            out = sbm_sweep._execute_store_run(g, gt, method, number_of_communities=2,
                                                initial_membership=None, run_seed=1)
        self.assertEqual(out["status"], "ok", out.get("error"))
        self.assertLess(out["duration"], SLOW / 2)

    def test_the_error_fallback_and_report_are_outside_the_clock(self):
        import igraph as ig

        from hedonic import Game
        from hedonic.experiments.disjoint import sbm_sweep

        g = Game(ig.Graph.Famous("Zachary"))

        def failing(graph, name, params):
            raise RuntimeError("boom")

        slow_print = _slow(lambda *a, **k: None)
        with patch.object(sbm_sweep, "_call_method", failing), patch("builtins.print", slow_print):
            result = sbm_sweep.get_method_result(g, "community_mirror", {}, 0.1, 1.0, 10, 2, [0] * g.vcount())
        self.assertEqual(result["status"], "error")
        self.assertLess(result["duration"], SLOW / 2)

    def test_the_store_spec_names_the_clock_so_rows_timed_otherwise_never_mix(self):
        from hedonic.experiments.disjoint import store

        spec = store.build_spec(methods={}, n_runs=1, partitions="hash", generator_id="x")
        self.assertEqual(spec["timing_scope"], store.TIMING_SCOPE)


if __name__ == "__main__":
    unittest.main()
