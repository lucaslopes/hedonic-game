"""Label-free baseline calibration: no ground truth reaches a detector, the criterion, the search, the budget and the output."""

from __future__ import annotations

import dataclasses
import io
import json
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import igraph as ig

from hedonic.experiments import CLI
from hedonic.experiments.overlapping import baseline_calibration as bc

N_BLOCKS, BLOCK = 10, 40


def planted() -> ig.Graph:
    return bc.synthetic_graph(N_BLOCKS * BLOCK)  # 10 blocks of 40, dense inside


def block_cover(n: int, k: int) -> list[list[int]]:
    """K equal contiguous blocks: with K = 10 it is the planted structure the graph was drawn from."""
    size = -(-n // k)
    return [list(range(i, min(i + size, n))) for i in range(0, n, size)]


def blocks_runner(method, graph, params, seed):
    return block_cover(graph.vcount(), int(params.get("communities", params.get("clusters", 10))))


class Fixture(unittest.TestCase):
    def ctx(self, **kw) -> bc.Context:
        kw.setdefault("budget", bc.Budget(max_evals=None, eval_timeout=60))
        kw.setdefault("runner", blocks_runner)
        return bc.Context(graph=planted(), seed=1, **kw)


class TestLabelFree(Fixture):
    def test_context_has_no_cover_and_the_default_runner_passes_an_empty_reference(self):
        self.assertFalse({f.name for f in dataclasses.fields(bc.Context)} & {"cover", "ground_truth", "reference", "truth"})
        calls = []

        def fake(name, graph, ground_truth, seed, parameters):
            calls.append(ground_truth)
            return [[0, 1, 2]], {}

        with patch("hedonic.experiments.overlapping.codeseg_reproduction._run_method", fake):
            cover = bc.default_runner("angel", planted(), {"threshold": 0.5}, 0)
        self.assertEqual(calls, [[]])
        self.assertEqual(cover, [[0, 1, 2]])

    def test_every_calibrator_is_registered_and_the_fixed_ones_run_nothing(self):
        self.assertEqual(set(bc.CALIBRATORS), {"clique_percolation", "angel", "ego_splitting", "fox", "bigclam", "ncgame",
                                               "neo_kmeans", "codeseg"})
        ctx = self.ctx(runner=lambda *a: self.fail("a fixed-protocol calibration must not run a detector"))
        for name in bc.FIXED_METHODS:
            result = bc.calibrate(name, ctx)
            self.assertEqual(result.status, "fixed")
            self.assertFalse(result.uses_ground_truth)
            self.assertEqual(result.evaluations, 0)


class TestCriterion(Fixture):
    def test_holdout_is_deterministic_disjoint_and_balanced(self):
        g = planted()
        a, b = bc.make_holdout(g, 3), bc.make_holdout(g, 3)
        self.assertEqual((a.positives, a.negatives), (b.positives, b.negatives))
        self.assertEqual(len(a.positives), len(a.negatives))
        self.assertEqual(a.train.ecount(), g.ecount() - len(a.positives))
        self.assertTrue(all(g.are_adjacent(u, v) and not a.train.are_adjacent(u, v) for u, v in a.positives))
        self.assertFalse(any(g.are_adjacent(u, v) for u, v in a.negatives))
        self.assertNotEqual(a.positives, bc.make_holdout(g, 4).positives)

    def test_planted_cover_beats_a_wrong_one_and_trivial_ones(self):
        h = bc.make_holdout(planted(), 0)
        n = h.train.vcount()
        score = lambda cover: bc.pair_loglik(cover, h.train, h.positives, h.negatives)["score"]  # noqa: E731
        planted_score = score(block_cover(n, N_BLOCKS))
        self.assertGreater(planted_score, score(block_cover(n, 2)))
        self.assertGreater(planted_score, score([list(range(n))]))
        self.assertGreater(planted_score, score([]))
        shuffled = [[(v * 7 + 3) % n for v in c] for c in block_cover(n, N_BLOCKS)]
        self.assertGreater(planted_score, score(shuffled))

    def test_cover_summary(self):
        s = bc.cover_summary([[0, 1, 2], [2, 3], [9]], 10)
        self.assertEqual(s["communities"], 2)  # the singleton is not a community
        self.assertAlmostEqual(s["coverage"], 0.4)
        self.assertAlmostEqual(s["mean_memberships"], 1.25)


class TestSearchAndBudget(Fixture):
    def test_k_search_finds_the_planted_scale_and_brackets_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self.ctx(ledger_dir=Path(tmp))
            result = bc.calibrate_bigclam(ctx)
        self.assertEqual(result.status, "calibrated")
        self.assertIn(result.params["communities"], (8, 16))  # grid points either side of the true 10
        self.assertTrue(result.bracketed)
        self.assertEqual(result.criterion, bc.CRITERION)
        self.assertTrue(all(c["status"] == "completed" for c in result.candidates))

    def test_budget_stops_before_a_new_evaluation_and_the_ledger_makes_a_rerun_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self.ctx(ledger_dir=Path(tmp), budget=bc.Budget(max_evals=2, eval_timeout=60))
            first = bc.calibrate_neo_kmeans(ctx)
            self.assertEqual(first.evaluations, 2)
            self.assertTrue(first.stop_reason.startswith("budget"))
            ctx2 = self.ctx(ledger_dir=Path(tmp), budget=bc.Budget(max_evals=2, eval_timeout=60))
            second = bc.calibrate_neo_kmeans(ctx2)
            self.assertEqual([c["params"] for c in second.candidates[:2]], [c["params"] for c in first.candidates])
            self.assertEqual(second.candidates[0]["score"], first.candidates[0]["score"])  # reused, not rerun
            lines = (Path(tmp) / "evals-neo_kmeans.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 4)  # two from the first run, two new in the second (budget again 2)

    def test_failures_are_recorded_and_unavailable_is_reported(self):
        def broken(method, graph, params, seed):
            raise RuntimeError("boom")

        result = bc.calibrate_angel(self.ctx(runner=broken))
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.params)
        self.assertIn("boom", " ".join(result.notes))

    def test_a_missing_runtime_stops_the_search_after_one_evaluation(self):
        class MethodUnavailable(Exception):
            pass

        def missing(method, graph, params, seed):
            raise MethodUnavailable("no binary")

        result = bc.calibrate_bigclam(self.ctx(runner=missing, budget=bc.Budget(max_evals=8, eval_timeout=60)))
        self.assertEqual((result.status, result.evaluations, result.stop_reason), ("unavailable", 1, "runtime unavailable"))

    def test_a_timeout_is_a_failed_probe_not_a_crash(self):
        import time

        def slow(method, graph, params, seed):
            time.sleep(30)

        ctx = self.ctx(runner=slow, budget=bc.Budget(max_evals=1, eval_timeout=0.5))
        rec = bc.evaluate(ctx, "angel", {"threshold": 0.5})
        self.assertEqual(rec["status"], "timeout")


class TestTimeoutKillsGrandchildren(unittest.TestCase):
    def test_a_timed_out_probe_leaves_no_orphan_behind(self):
        import os
        import subprocess
        import tempfile
        import time

        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "pid"

            def runner(method, graph, params, seed):  # like the CDlib worker: a subprocess of the probe
                child = subprocess.Popen(["sleep", "300"])
                pidfile.write_text(str(child.pid))
                time.sleep(300)

            out = bc.run_isolated(runner, "angel", planted(), {}, 0, 1.5)
            self.assertEqual(out["status"], "timeout")
            pid = int(pidfile.read_text())
            time.sleep(0.3)
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)  # killed with the probe, not left as an orphan of init


class TestParallelReplicas(Fixture):
    def test_replicas_run_side_by_side_and_score_like_one_after_another(self):
        import time

        def slow(method, graph, params, seed):
            time.sleep(1.0)
            return block_cover(graph.vcount(), 10)

        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            serial = self.ctx(ledger_dir=Path(a), runner=slow, replicates={"bigclam": 3})
            t0 = time.perf_counter()
            one = bc.evaluate(serial, "bigclam", {"communities": 10})
            t_serial = time.perf_counter() - t0
            par = self.ctx(ledger_dir=Path(b), runner=slow, replicates={"bigclam": 3}, replica_workers=3)
            t0 = time.perf_counter()
            two = bc.evaluate(par, "bigclam", {"communities": 10})
            t_par = time.perf_counter() - t0
            self.assertEqual((one["score"], one["replicas"]), (two["score"], two["replicas"]))
            self.assertLess(t_par, t_serial - 1.0)
            self.assertEqual(par.evaluations, 1)  # a setting counts once, however many replicas
            self.assertGreater(par.seconds, 2.5)  # the children's seconds are added in the parent
            self.assertEqual(len((Path(b) / "evals-bigclam.jsonl").read_text().splitlines()), 3)


class TestFoxThresholds(Fixture):
    def test_both_published_thresholds_are_tried_and_the_better_one_kept(self):
        def runner(method, graph, params, seed):  # the planted blocks at 0.01, a coarser cover at 0.1
            return block_cover(graph.vcount(), 10 if params["wcc_threshold"] == 0.01 else 3)
        ctx = self.ctx(runner=runner)
        result = bc.calibrate("fox", ctx)
        self.assertEqual(result.status, "calibrated")
        self.assertEqual([c["params"]["wcc_threshold"] for c in result.candidates], [0.1, 0.01])
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.params["wcc_threshold"], 0.01)  # the one that explains held-out pairs better
        self.assertEqual(result.params["queue_size"], 1)
        self.assertNotIn("fox", bc.FIXED_METHODS)


class TestFixedAndStructural(Fixture):
    def test_published_settings(self):
        self.assertEqual(bc.calibrate_codeseg().params, {"tau": 0.3, "alpha": 1.0, "iterations": 10})
        fox = bc.calibrate_fox().params  # no graph: nothing to compare, the CoDeSEG paper's setting
        self.assertEqual((fox["wcc_threshold"], fox["threads"]), (0.1, 1))
        self.assertEqual(bc.calibrate_ncgame().params, {"similarity_type": "HP", "stop_criterion": 0.01})

    def test_ego_splitting_is_flagged_unfaithful(self):
        r = bc.calibrate_ego_splitting()
        self.assertIs(r.faithful_to_paper, False)
        self.assertIn("Absolute Potts", " ".join(r.notes))

    def test_clique_percolation_picks_the_first_k_without_a_giant_community(self):
        n = 100

        def runner(method, graph, params, seed):
            return [list(range(80)), [80, 81, 82]] if params["clique_size"] == 3 else [list(range(10, 20)), list(range(30, 40)), list(range(50, 60))]

        ctx = self.ctx(runner=runner)
        r = bc.calibrate_clique_percolation(ctx)
        self.assertEqual(r.params, {"clique_size": 4})
        self.assertEqual(r.evaluations, 2)  # k = 5, 6 are not tried once the transition is passed
        self.assertTrue(r.bracketed)
        self.assertEqual(ctx.graph.vcount(), n * 4)

    def test_clique_percolation_without_a_transition_says_so(self):
        r = bc.calibrate_clique_percolation(self.ctx(runner=lambda *a: [list(range(300))]))
        self.assertEqual(r.params, {"clique_size": 6})
        self.assertFalse(r.bracketed)

    def test_angel_grid_is_the_registered_one_walked_from_the_largest_phi(self):
        result = bc.calibrate_angel(self.ctx(runner=lambda m, g, p, s: block_cover(g.vcount(), 10)))
        self.assertEqual([c["params"]["threshold"] for c in result.candidates], sorted(bc.ANGEL_THRESHOLDS, reverse=True))
        self.assertTrue(all(c["params"]["min_community_size"] == 3 for c in result.candidates))
        self.assertEqual(result.params["threshold"], min(bc.ANGEL_THRESHOLDS))  # equal scores: the smaller phi

    def test_an_angel_setting_stopped_at_the_timeout_ends_the_walk(self):
        import time

        def runner(method, graph, params, seed):  # a smaller phi merges more and runs longer
            if params["threshold"] < 0.5:
                time.sleep(30)
            return block_cover(graph.vcount(), 10 if params["threshold"] == 0.5 else 5)

        result = bc.calibrate_angel(self.ctx(runner=runner, budget=bc.Budget(max_evals=None, eval_timeout=1.0)))
        self.assertEqual([(c["params"]["threshold"], c["status"]) for c in result.candidates],
                         [(0.8, "completed"), (0.66, "completed"), (0.5, "completed"), (0.33, "timeout")])  # 0.2 not tried
        self.assertTrue(result.stop_reason.startswith("timeout at phi 0.33"))
        self.assertEqual((result.status, result.params["threshold"]), ("calibrated", 0.5))
        self.assertFalse(result.bracketed)
        self.assertIn("did not finish within the evaluation timeout", " ".join(result.notes))


class TestOutput(Fixture):
    def test_ncgame_flags_reach_the_command(self):
        got = {}

        def inner(method, graph, params, seed):
            got.update(params)
            return []

        bc.ncgame_runner(inner)("ncgame", planted(), {"command": "python w.py --input {input}", "stop_criterion": 0.005,
                                                      "similarity_type": "HP"}, 0)
        self.assertTrue(got["command"].endswith("--stop-criterion 0.005 --similarity-type HP"))

    def test_toml_snippet_parses_into_method_params_tables(self):
        results = [bc.calibrate_codeseg(), bc.calibrate_ego_splitting()]
        text = bc.toml_snippet("dblp:top5000", results)
        data = tomllib.loads(text)
        self.assertEqual(data["hedonic_run"]["method_params"]["codeseg"]["dblp:top5000"]["tau"], 0.3)
        self.assertIn("NOT the authors' protocol", text)

    def test_plan_lists_the_k_grid_without_loading_anything(self):
        rows = bc.plan(300_000, 0, ["bigclam", "fox", "angel"], bc.Budget(max_evals=8))
        self.assertEqual({r["method"]: r["kind"] for r in rows}, {"bigclam": "search", "fox": "search", "angel": "search"})
        self.assertIn(6000, [t["communities"] for t in rows[0]["tries"]])


class TestCLI(unittest.TestCase):
    def test_registered_with_working_help(self):
        self.assertIn("baseline-calibrate", CLI.COMMANDS)
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stop:
            bc.main(["--help"])
        self.assertEqual(stop.exception.code, 0)

    def test_dry_run_and_smoke_write_the_outputs(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(bc.main(["--dry-run", "--methods", "bigclam,fox"]), 0)
        self.assertIn('"method": "bigclam"', buffer.getvalue())
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            self.assertEqual(bc.main(["--smoke", "--methods", "codeseg,fox,ncgame,ego_splitting", "--output-dir", tmp]), 0)
            record = json.loads((Path(tmp) / "smoke" / "baseline_tuned.json").read_text())
            self.assertFalse(record["results"]["fox"]["uses_ground_truth"])
            tomllib.loads((Path(tmp) / "smoke" / "baseline_tuned.toml").read_text())

    def test_unknown_method_is_an_argument_error(self):
        with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            bc.main(["--methods", "nope", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
