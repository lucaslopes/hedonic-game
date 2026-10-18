"""One clock for every method: detection time covers the detector and nothing else.

These tests pin the rule ``detector-compute-v1`` of ``hedonic.experiments.timing`` (benchmark) and
``detector-call-v1`` of the disjoint store: loading, conversion, validation, scoring, audits and saving never
reach a reported time, and no adapter starts a program or a stopwatch of its own.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments import timing
from hedonic.experiments.overlapping import codeseg_reproduction as cr
from hedonic.experiments.overlapping import snap

SLOW = 0.6  # seconds of artificial work injected outside the detector
PACKAGE = Path(cr.__file__).resolve().parent


def _slow(func, delay=SLOW):
    def wrapper(*args, **kwargs):
        time.sleep(delay)
        return func(*args, **kwargs)

    return wrapper


def _run_one(method: str, *, save_covers: bool = False):
    dataset = snap.synthetic_agmfit_dataset(n_nodes=150, seed=1, n_communities=6)
    with tempfile.TemporaryDirectory() as out:
        records = cr._run_dataset(
            dataset,
            methods=[method],
            seed=0,
            method_parameters={},
            timeout_seconds=60.0,
            paper_filter=False,
            output_dir=Path(out),
            resume=False,
            config_digest="test",
            compute_omega=False,
            save_covers=save_covers,
        )
    return records[0]


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


class TestWorkerSelfTiming(unittest.TestCase):
    def test_every_shipped_worker_reports_detector_seconds_around_its_algorithm_call(self):
        workers = PACKAGE.parent / "workers"
        paths = [workers / "ncgame_worker.py", *sorted((workers / "slpa_env").glob("*_worker.py"))]
        self.assertGreaterEqual(len(paths), 5)
        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertIn("detector_seconds", source, path.name)
            self.assertIn("perf_counter", source, path.name)


class TestRunnerExcludesWorkOutsideTheDetector(unittest.TestCase):
    def test_scoring_cover_saving_and_filtering_are_not_detection(self):
        with patch.object(cr, "evaluate_cover", _slow(cr.evaluate_cover)):
            record = _run_one("louvain", save_covers=True)
        self.assertEqual(record["status"], "completed", record.get("reason"))
        self.assertLess(record["detection_seconds"], SLOW / 2)
        self.assertGreaterEqual(record["scoring_seconds"], SLOW)
        self.assertGreaterEqual(record["unit_seconds"], SLOW)
        self.assertEqual(record["runtime_seconds"], record["detection_seconds"])
        self.assertEqual(record["timing_scope"], timing.TIMING_SCOPE)

    def test_graph_conversion_is_not_detection_for_any_in_process_method(self):
        for method in ("louvain", "leiden", "flpa", "hedonic_local"):
            with self.subTest(method=method), patch.object(cr, "_undirected_graph", _slow(cr._undirected_graph)):
                record = _run_one(method)
                self.assertEqual(record["status"], "completed", record.get("reason"))
                self.assertLess(record["detection_seconds"], SLOW / 2)
                self.assertGreaterEqual(record["unit_seconds"], SLOW)

    def test_the_active_vertex_filter_is_not_detection(self):
        with patch.object(cr, "_active_graph", _slow(cr._active_graph)):
            record = _run_one("louvain")
        self.assertLess(record["detection_seconds"], SLOW / 2)

    def test_validation_and_canonicalisation_are_not_detection(self):
        from hedonic.experiments.overlapping import methods

        with patch.object(methods, "normalize_cover", _slow(methods.normalize_cover)):
            dataset = snap.synthetic_agmfit_dataset(n_nodes=150, seed=1, n_communities=6)
            _cover, info = methods.run_method_by_name(
                "hedonic_local", dataset.graph, max_memberships=3, resolution=0.05, seed=0)
        self.assertLess(info["runtime_seconds"], SLOW / 2)
        self.assertEqual(info["timing_scope"], timing.TIMING_SCOPE)

    def test_a_detector_that_never_starts_the_clock_is_a_failed_record_not_a_zero(self):
        with patch.object(cr, "_run_method", return_value=([[0, 1]], {})):
            record = _run_one("louvain")
        self.assertEqual(record["status"], "failed")
        self.assertIn("detector clock", record["reason"])

    def test_the_detector_record_carries_the_scope_and_a_peak_that_says_what_it_covers(self):
        record = _run_one("flpa")
        resources = record["resources"]
        self.assertEqual(resources["detector_peak_rss_scope"], "harness_process_lifetime")
        self.assertIn("rss_before_detector_bytes", resources)
        self.assertEqual(record["timing"]["regions"][0]["kind"], "in_process")
        self.assertLessEqual(record["detection_seconds"], record["unit_seconds"])

    def test_a_subprocess_method_is_charged_exec_to_exit_and_its_own_peak(self):
        calls = []
        real = timing.DetectorClock.process

        def fake_binary(self, command, **kwargs):
            calls.append(kwargs)
            return real(self, [sys.executable, "-c", "import time; time.sleep(0.2)"], **{
                k: v for k, v in kwargs.items() if k in {"label", "counted"}})

        clock = timing.DetectorClock()
        with patch.object(timing.DetectorClock, "process", fake_binary):
            cr._run_executable(["unused"], "fake", {}, cwd=None, env=None, input_path=None)
        # the helper must reach the clock (patched on the class) and hand it the input file for accounting
        self.assertEqual(len(calls), 1)
        self.assertIn("input_path", calls[0])
        self.assertFalse(clock.used)


class TestEveryMethodUsesTheSharedClock(unittest.TestCase):
    def test_no_adapter_spawns_a_program_or_starts_a_stopwatch_of_its_own(self):
        pattern = re.compile(r"subprocess\.(run|check_output|check_call|call)\(|time\.time\(\)")
        for name in ("codeseg_reproduction.py", "methods.py", "cdlib_env.py", "extra_methods.py"):
            text = (PACKAGE / name).read_text(encoding="utf-8")
            self.assertIsNone(pattern.search(text), f"{name} times or spawns outside timing.DetectorClock")
        # the only stopwatches of the runner are the unit and scoring clocks of _run_dataset
        runner = (PACKAGE / "codeseg_reproduction.py").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"perf_counter\(\)", runner)), 4)
        for name in ("methods.py", "cdlib_env.py", "extra_methods.py"):
            self.assertNotIn("perf_counter", (PACKAGE / name).read_text(encoding="utf-8"), name)

    def test_every_registered_runner_function_reaches_the_clock(self):
        text = (PACKAGE / "codeseg_reproduction.py").read_text(encoding="utf-8")
        functions = re.split(r"\n(?=def )", text)
        reaching = ("timing.active", "_run_executable", "_run_octave_script", "_run_template_command",
                    "run_slpa", "run_cdlib_worker", "clock.process", "clock.measure", "run_method(")
        for block in functions:
            match = re.match(r"def (_run_\w+)\(", block)
            if not match or match.group(1) in {"_run_executable", "_run_method", "_run_dataset"}:
                continue
            self.assertTrue(any(token in block for token in reaching), f"{match.group(1)} never reaches the clock")

    def test_the_protocol_version_changed_with_the_meaning_of_detection_time(self):
        self.assertTrue(cr.PROTOCOL_VERSION.endswith("-v5"))
        self.assertEqual(timing.TIMING_SCOPE, "detector-compute-v1")


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
