"""The packaged Hedonic calibration search (``calibration_search``), its label-free audit and the lazy runner on it.

Data-free: every search runs on a small planted overlapping instance registered in the process. The tests assert the
search's properties (where its steps go, what it keeps, what it reuses), never numbers of a particular build.
"""

from __future__ import annotations

import io
import json
import math
import random
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import igraph as ig
import numpy as np

from hedonic import Game
from hedonic.experiments.overlapping import calibration_search as cs
from hedonic.experiments.overlapping.calibration_audit import (
    CoverAudit,
    active_density,
    golden_min,
    internal_densities,
)

PLANTED = "planted"


def planted(n_comm: int = 6, size: int = 12, overlap: int = 2, p_in: float = 0.7, p_out: float = 0.02, seed: int = 0):
    """A ring of ``n_comm`` dense communities of ``size`` vertices, consecutive ones sharing ``overlap`` vertices."""
    rng = random.Random(seed)
    step = size - overlap
    n = n_comm * step
    comms = [[(i * step + j) % n for j in range(size)] for i in range(n_comm)]
    member = [set() for _ in range(n)]
    for c, vs in enumerate(comms):
        for v in vs:
            member[v].add(c)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < (p_in if member[u] & member[v] else p_out)]
    return ig.Graph(n=n, edges=edges), comms


def register(spec: str = PLANTED) -> dict:
    graph, gt = planted()
    return cs.register_instance(spec, graph, gt)


def evals_lines(out: Path) -> list[str]:
    return [line for path in sorted(out.glob("evals*.jsonl")) for line in path.read_text().splitlines() if line.strip()]


def last_search(out: Path) -> dict:
    return json.loads((out / "searches.jsonl").read_text().splitlines()[-1])


def tune_argv(out: Path, *extra: str) -> list[str]:
    return ["tune", "--instance", PLANTED, "--out", str(out), "--confirm-seeds", "0,1,2", "--M-max", "16",
            "--timeout", "60", *extra]


class TestSimplestRational(unittest.TestCase):
    def brute(self, lo: float, hi: float) -> Fraction:
        for q in range(1, 10_000):
            p = math.ceil(Fraction(lo) * q)
            if Fraction(p, q) <= Fraction(hi):
                return Fraction(p, q)
        raise AssertionError("no fraction found")

    def test_it_is_the_smallest_denominator_fraction_inside_the_interval(self):
        rng = random.Random(1)
        self.assertEqual(cs.simplest_rational(0.3, 0.3), Fraction(0.3))  # a point: the float itself
        self.assertEqual(cs.simplest_rational(0.25, 0.25), Fraction(1, 4))
        cases = [(0.14, 0.145), (1 / 723 / 1.1, 1 / 723 * 1.1), (0.999, 1.0), (2.5, 3.5), (0.01, 0.0101)]
        cases += [(x, x * (1 + rng.uniform(0.001, 0.2))) for x in (rng.uniform(1e-4, 1.0) for _ in range(40))]
        for lo, hi in cases:
            frac = cs.simplest_rational(lo, hi)
            self.assertTrue(Fraction(lo) <= frac <= Fraction(hi), (lo, hi, frac))
            self.assertEqual(frac.denominator, self.brute(lo, hi).denominator, (lo, hi))

    def test_an_empty_or_non_positive_interval_is_refused(self):
        for lo, hi in ((0.0, 0.1), (-0.1, 0.1), (0.2, 0.1)):
            with self.assertRaises(ValueError):
                cs.simplest_rational(lo, hi)


class TestGoldenMaxLog(unittest.TestCase):
    @staticmethod
    def peak_at(centre: float, width: float = 1.0):
        calls: list[float] = []

        def f(gamma: float) -> float:
            calls.append(gamma)
            return -((math.log(gamma) - math.log(centre)) / width) ** 2

        return f, calls

    def test_the_optimum_inside_the_bracket_is_located_to_the_tolerance(self):
        f, _ = self.peak_at(0.05)
        out = cs.golden_max_log(f, 0.01, 0.3, tol_u=0.05, max_evals=40, floor=1e-4, ceil=0.9)
        lo, hi = out["bracket"]
        self.assertLessEqual(math.log(hi / lo), 0.05 + 1e-12)
        self.assertLess(abs(math.log(out["gamma"] / 0.05)), 0.05)
        self.assertEqual(out["evaluations"], len(out["points"]))

    def test_the_gallop_leaves_the_bracket_but_never_the_floor_and_ceiling(self):
        f, calls = self.peak_at(0.004)
        out = cs.golden_max_log(f, 0.05, 0.4, tol_u=0.05, max_evals=40, floor=1e-3, ceil=0.9)
        self.assertLess(out["gamma"], 0.05)  # found below the first bracket
        self.assertLess(abs(math.log(out["gamma"] / 0.004)), 0.1)
        self.assertGreaterEqual(min(calls), 1e-3 * (1 - 1e-9))
        f, calls = self.peak_at(0.8)
        out = cs.golden_max_log(f, 0.01, 0.1, tol_u=0.05, max_evals=40, floor=1e-3, ceil=0.5)
        self.assertLessEqual(max(calls), 0.5 * (1 + 1e-9))
        self.assertAlmostEqual(out["gamma"], 0.5, places=6)  # the best it may look at

    def test_a_plateau_stops_the_gallop(self):
        calls: list[float] = []

        def flat_below(gamma: float) -> float:  # rises toward small gamma, but by less than the plateau
            calls.append(gamma)
            return -1e-4 * math.log(gamma)

        cs.golden_max_log(flat_below, 0.05, 0.4, tol_u=0.05, max_evals=40, floor=1e-6, ceil=0.9, plateau=0.01)
        self.assertGreater(min(calls), 0.05 / 4 * (1 - 1e-9))  # one step out, then stopped

    def test_the_evaluation_budget_is_respected(self):
        f, calls = self.peak_at(0.01, width=5.0)
        out = cs.golden_max_log(f, 0.02, 0.5, tol_u=1e-6, max_evals=9, floor=1e-5, ceil=0.9)
        self.assertLessEqual(out["evaluations"], 9)
        self.assertEqual(len(set(round(math.log(g), 9) for g in calls)), out["evaluations"])  # memoized: no repeats

    def test_prefetch_is_given_the_scan_points_before_the_first_one_is_asked(self):
        f, calls = self.peak_at(0.05)
        fetched: list[list[float]] = []

        def prefetch(gammas):
            self.assertEqual(calls, [])
            fetched.append(list(gammas))

        cs.golden_max_log(f, 0.01, 0.16, tol_u=0.1, max_evals=20, floor=1e-4, ceil=0.9, prefetch=prefetch)
        self.assertEqual(len(fetched), 1)
        scan = fetched[0]
        self.assertAlmostEqual(scan[0], 0.01)
        self.assertAlmostEqual(scan[-1], 0.16)
        self.assertEqual(calls[:len(scan)], scan)  # the loop asks for exactly those, in that order


class FakeProbe:
    """The part of ``Probe`` that ``cap_search`` uses, with a known score profile in the cap."""

    def __init__(self, profile):
        self.spec, self.base, self.seeds = "fake", {"phase": "local"}, [0]
        self.profile = profile
        self.asked: list[int] = []
        self.prefetched: list[dict] = []

    def __call__(self, *, gamma, M):
        self.asked.append(M)
        return self.profile(M)

    def prefetch(self, overs):
        self.prefetched.extend(overs)


class TestCapSearch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cs.configure(self.tmp.name)

    def tearDown(self):
        cs.CACHE.clear()
        self.tmp.cleanup()

    def test_the_smallest_cap_within_delta_of_the_best_is_chosen(self):
        probe = FakeProbe(lambda M: min(M, 11) / 11)  # rises to cap 11, flat after
        out = cs.cap_search(probe, 0.1, 8, 128, delta=1e-3)
        self.assertEqual(out["M"], 11)
        self.assertEqual(out["J"], 1.0)
        self.assertIn("up:", out["gallop_stop"])
        self.assertEqual([o["M"] for o in probe.prefetched], [8, 16, 4])

    def test_a_cap_that_regularizes_is_found_below_the_start(self):
        probe = FakeProbe(lambda M: 1.0 - abs(math.log2(M) - 2) / 10)  # best at 4, worse on both sides
        self.assertEqual(cs.cap_search(probe, 0.1, 16, 128, delta=1e-3)["M"], 4)
        self.assertIn(3, probe.asked)  # the binary search between the last bad cap and the first good one

    def test_the_uncapped_certificate_stops_the_upward_gallop(self):
        probe = FakeProbe(lambda M: math.log2(M))  # doubling always pays
        key = cs.cfg_key("fake", {"phase": "local", "gamma": 0.1, "M": 8, "seed": 0})
        cs.CACHE[key] = {"label_free": {"saturated_fraction": 0.0, "uncapped_stable_fraction": 1.0}}
        out = cs.cap_search(probe, 0.1, 8, 128, delta=1e-3)
        self.assertIn("cap certificate at M=8", out["gallop_stop"])
        self.assertNotIn(32, probe.asked)
        uncertified = FakeProbe(lambda M: math.log2(M))
        self.assertIn("reached M_max", cs.cap_search(uncertified, 0.2, 8, 64, delta=1e-3)["gallop_stop"])


class TestAudit(unittest.TestCase):
    def test_internal_densities_equal_the_induced_subgraph_densities(self):
        graph, gt = planted()
        self.assertTrue(np.allclose(internal_densities(graph, gt), [graph.induced_subgraph(c).density() for c in gt]))
        self.assertEqual(list(internal_densities(graph, [[0]])), [1.0])
        rho, n_act = active_density(graph)
        self.assertEqual(n_act, graph.vcount())
        self.assertAlmostEqual(rho, graph.density())
        with self.assertRaises(ValueError):
            active_density(ig.Graph(n=3))

    def test_the_detectors_returns_have_no_regret_and_a_moved_vertex_has_some(self):
        graph, _ = planted()
        game = Game(graph)
        for gamma, cap in ((0.6, 1), (0.3, 2), (0.15, 16)):
            game.community_hedonic(None, max_memberships=cap, resolution=gamma, allow_isolation=True, seed=0)
            rows = [list(r) for r in game.memberships]
            audit = CoverAudit(graph, rows)
            self.assertLess(audit.evaluate(gamma, cap)["regret"].max(), 1e-9, (gamma, cap))
            self.assertTrue(audit.stable(gamma, cap).all())
            self.assertEqual(audit.window_robust_fraction(gamma, 1.0, cap), 1.0)
        fresh = max(c for r in rows for c in r) + 1
        rows[0] = [fresh]  # vertex 0 alone, away from its dense community
        self.assertGreater(CoverAudit(graph, rows).evaluate(0.15, 16)["regret"][0], 0.0)

    def test_golden_min_finds_the_minimum_of_a_convex_function(self):
        x, fx, trace = golden_min(lambda x: (x - 0.3) ** 2 + 1.0, 0.0, 1.0, tol=1e-6)
        self.assertAlmostEqual(x, 0.3, places=5)
        self.assertAlmostEqual(fx, 1.0, places=9)
        self.assertGreater(len(trace), 2)


class TestInstances(unittest.TestCase):
    def test_a_network_is_the_whole_graph_and_a_pair_the_covered_induced_one(self):
        self.assertEqual(cs.parse_instance("dblp"), ("dblp", None))
        self.assertEqual(cs.parse_instance("DBLP:top5k"), ("dblp", "top5000"))
        self.assertEqual(cs.parse_instance("amazon:all"), ("amazon", "all"))

    def test_derived_instances_unknown_networks_and_missing_covers_are_refused(self):
        for bad in ("dblp@5000s0", "dblp:top5000@100u3", "nosuchnet", "dblp:nosuchcover", "wikipedia:top5000",
                    "dblp:top5000:x", ""):
            with self.assertRaises(ValueError, msg=bad):
                cs.parse_instance(bad)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cs.main(["tune", "--instance", "dblp@5000s0", "--out", "/nonexistent-never-written"])
        self.assertEqual(caught.exception.code, 2)

    def test_a_snap_instance_is_provisioned_and_loaded_like_a_covered_benchmark_unit(self):
        from hedonic.experiments.overlapping import codeseg_reproduction as cr
        from hedonic.experiments.overlapping import snap

        graph, gt = planted()
        ensured: list[tuple] = []
        loaded: list[dict] = []

        def fake_ensure(network, cover=None, root=None, *, download=True, say=None, **_):
            ensured.append((network, cover, Path(root), download))

        def fake_load(name, **kwargs):
            loaded.append({"name": name, **kwargs})
            return SimpleNamespace(graph=graph.copy(), cover=gt, report={"n": graph.vcount(), "m": graph.ecount()})

        with tempfile.TemporaryDirectory() as tmp, patch.object(snap, "ensure_pair", fake_ensure), \
                patch.object(cr, "_load_dataset", fake_load), \
                patch.object(cr, "_cover_for_paper_metrics", lambda d: (d.cover, {v for c in d.cover for v in c}, {})), \
                redirect_stdout(io.StringIO()):
            cs.configure(tmp, network_root=Path(tmp) / "nets", snap_cache_dir=Path(tmp) / "cache", download=False)
            for spec in ("dblp:top5000", "amazon"):
                cs.INST.pop(spec, None)
                inst = cs.load_instance(spec)
                self.assertEqual(inst["m"], graph.ecount())
            cs.INST.pop("dblp:top5000", None)
            cs.INST.pop("amazon", None)
        self.assertEqual(ensured, [("dblp", "top5000", Path(tmp) / "nets", False), ("amazon", "all", Path(tmp) / "nets", False)])
        self.assertEqual([(d["name"], d["cover_variant"], d["induce"]) for d in loaded],
                         [("dblp", "top5000", "covered"), ("amazon", "all", "full")])
        self.assertTrue(all(d["max_nodes"] is None and d["allow_catalog"] is False for d in loaded))
        self.assertEqual(loaded[0]["snap_cache_dir"], Path(tmp) / "cache")

    def test_a_registered_graph_is_simplified_and_must_be_undirected(self):
        graph, gt = planted()
        doubled = graph.copy()
        doubled.add_edges([graph.es[0].tuple, (2, 2)])  # a parallel edge and a loop
        inst = cs.register_instance("planted-doubled", doubled, gt)
        self.assertEqual(inst["m"], graph.ecount())
        self.assertIsInstance(inst["graph"], Game)
        self.assertEqual(cs.instance_summary(inst)["induce"], "given")
        with self.assertRaises(ValueError):
            cs.register_instance("directed", ig.Graph(n=2, edges=[(0, 1)], directed=True), [[0, 1]])
        with self.assertRaises(ValueError):
            cs.register_instance("no-cover", graph, [])


class TestLedger(unittest.TestCase):
    """Evaluations are memoized by configuration and build, survive a torn write and are reused by a resume."""

    def setUp(self):
        register()
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "ledger"
        cs.configure(self.out, tag="t")
        self.cfg = {"phase": "local", "gamma": 0.15, "M": 4, "seed": 0}

    def tearDown(self):
        cs.CACHE.clear()
        self.tmp.cleanup()

    def evaluate(self, cfg=None):
        with redirect_stdout(io.StringIO()) as buffer:
            rec = cs.evaluate(PLANTED, cfg or self.cfg, timeout=60, objective="f1")
        return rec, buffer.getvalue()

    def test_a_record_is_written_once_with_its_build_and_reused_after_a_restart(self):
        rec, printed = self.evaluate()
        self.assertEqual(rec["status"], "completed")
        self.assertEqual(rec["build"], cs.build())
        self.assertTrue(0.0 <= rec["metrics"]["f1"] <= 1.0)
        self.assertLess(rec["label_free"]["max_regret"], 1e-9)  # the detector returns an equilibrium
        self.assertRegex(printed, r"\[planted local M=4 gamma=0\.15 .*seed=0\] completed")
        cs.configure(self.out, tag="t")  # a new process: empty cache, the same folder
        again, printed = self.evaluate({**self.cfg, "gamma": 0.1500000001})  # equal at 6 significant digits
        self.assertEqual(again["key"], rec["key"])
        self.assertEqual(printed, "")
        self.assertEqual(len(evals_lines(self.out)), 1)

    def test_a_torn_last_line_is_cut_before_the_next_record(self):
        self.evaluate()
        path = self.out / "evals-t.jsonl"
        with path.open("a") as fh:
            fh.write('{"key": "torn by a ki')
        cs.configure(self.out, tag="t")
        self.evaluate({**self.cfg, "M": 2})
        lines = path.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(json.loads(line)["status"] == "completed" for line in lines))

    def test_records_of_another_build_are_never_reused(self):
        rec, _ = self.evaluate()
        path = self.out / "evals-t.jsonl"
        stale = {**rec, "build": {"hedonic": "0.0.0", "hedonic_core_sha256": "0" * 64}, "metrics": {"f1": 2.0}}
        path.write_text(json.dumps(stale) + "\n")
        cs.configure(self.out, tag="t")
        again, printed = self.evaluate()
        self.assertIn("ignored 1 evaluations made by another build", printed)
        self.assertLessEqual(again["metrics"]["f1"], 1.0)
        self.assertEqual(len(evals_lines(self.out)), 2)

    def test_a_failure_in_the_detector_is_recorded_and_never_scored(self):
        rec, _ = self.evaluate({**self.cfg, "M": 0})
        self.assertEqual(rec["status"], "failed")
        self.assertNotIn("metrics", rec)
        self.assertEqual(cs.objective_value(rec, "f1"), -math.inf)

    def test_nothing_is_evaluated_without_a_ledger_folder(self):
        with patch.object(cs, "OUT", None), self.assertRaises(RuntimeError):
            cs.evaluate_many(PLANTED, [self.cfg], timeout=60, objective="f1")


class TestTune(unittest.TestCase):
    """The whole search on the planted instance: a sensible setting, a resume that evaluates nothing, and the same
    answer whatever the number of side-by-side evaluations."""

    @classmethod
    def setUpClass(cls):
        register()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "one"
        with redirect_stdout(io.StringIO()) as buffer:
            cls.code = cs.main(tune_argv(cls.out, "--tag", "planted-local"))
        cls.printed = buffer.getvalue()
        cls.summary = last_search(cls.out)

    @classmethod
    def tearDownClass(cls):
        cs.CACHE.clear()
        cls.tmp.cleanup()

    def test_the_search_finds_a_good_setting_inside_its_bounds(self):
        self.assertEqual(self.code, 0)
        final = self.summary["final"]
        inst = cs.INST[PLANTED]
        self.assertTrue(inst["rho"] * (1 - 1e-9) <= final["gamma"] < 1.0)
        self.assertTrue(1 <= final["M"] <= 16)
        self.assertGreater(final["f1_median"], 0.8)  # six dense planted communities
        self.assertEqual(final["completed"], 3)
        self.assertEqual(final["confirm_seeds"], [0, 1, 2])
        self.assertEqual(self.summary["build"], cs.build())
        self.assertEqual(self.summary["phase"], "local")
        self.assertEqual(self.summary["instance"]["spec"], PLANTED)
        self.assertIn(f"== {PLANTED} local: gamma*=", self.printed)

    def test_the_snap_is_the_simplest_fraction_of_its_bracket_and_kept_only_within_delta(self):
        snap = self.summary["snap"]
        lo, hi = snap["bracket"]
        frac = Fraction(snap["fraction"])
        self.assertTrue(Fraction(lo) <= frac <= Fraction(hi))
        self.assertEqual(frac, cs.simplest_rational(lo, hi))
        best = max(v for _, v in self.summary["stages"].get("gamma_at_chosen_M", self.summary["stages"]["gamma_at_M"])["points"])
        if not self.summary["branch_check"]:  # a branch check may move the setting after the snap
            self.assertEqual(snap["kept"], snap["J"] >= best - self.summary["delta"])
            self.assertEqual(self.summary["final"]["gamma"] == float(frac), snap["kept"])

    def test_every_evaluation_is_in_the_ledger_and_a_rerun_evaluates_nothing(self):
        before = evals_lines(self.out)
        self.assertEqual(len(before), len({json.loads(line)["key"] for line in before}))  # no configuration twice
        self.assertEqual(self.summary["detector_evaluations"], len(before))
        with redirect_stdout(io.StringIO()):
            cs.main(tune_argv(self.out, "--tag", "planted-local"))
        self.assertEqual(evals_lines(self.out), before)
        again = last_search(self.out)
        self.assertEqual((again["final"]["gamma"], again["final"]["M"]), (self.summary["final"]["gamma"], self.summary["final"]["M"]))

    def test_side_by_side_evaluations_and_a_slot_budget_change_nothing_but_the_clock(self):
        out = Path(self.tmp.name) / "jobs"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cs.main(tune_argv(out, "--jobs", "3", "--slot-dir", str(out / "slots"), "--slot-count", "2")), 0)
        summary = last_search(out)
        self.assertEqual((summary["final"]["gamma"], summary["final"]["M"]),
                         (self.summary["final"]["gamma"], self.summary["final"]["M"]))
        self.assertEqual(summary["final"]["f1_median"], self.summary["final"]["f1_median"])
        keys = lambda folder: sorted(json.loads(line)["key"] for line in evals_lines(folder))  # noqa: E731
        self.assertEqual(keys(out), keys(self.out))

    def test_a_warm_start_scans_its_bracket_at_its_cap_and_skips_the_extremal_probe(self):
        out = Path(self.tmp.name) / "warm"
        with redirect_stdout(io.StringIO()):
            cs.main(tune_argv(out, "--bracket", "0.1,0.2", "--M-start", "4", "--no-extremal"))
        summary = last_search(out)
        stage = summary["stages"]["gamma_at_M"]
        self.assertEqual(stage["M"], 4)
        self.assertTrue(any(abs(g - 0.1) < 1e-6 for g, _ in stage["points"]))
        self.assertTrue(any(abs(g - 0.2) < 1e-6 for g, _ in stage["points"]))
        self.assertEqual((summary["probe_trace"][0]["M"], round(summary["probe_trace"][0]["gamma"], 6)), (4, 0.1))
        self.assertTrue(math.isnan(summary["stages"]["extremal"]["J"]))
        configs = [json.loads(line)["config"] for line in evals_lines(out)]
        self.assertNotIn(cs.EXTREMAL, {c["gamma"] for c in configs})


class TestUnlimitedCap(unittest.TestCase):
    """``--fixed-M -1``: the resolution only, no cap, on the wide log scan."""

    def test_the_resolution_only_search_scans_the_wide_grid_and_keeps_no_cap(self):
        register()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "unlimited"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cs.main(tune_argv(out, "--fixed-M", "-1", "--no-extremal")), 0)
            summary = last_search(out)
            final = summary["final"]
            self.assertEqual(final["M"], -1)
            self.assertNotIn("rounds", summary["stages"])  # no cap search
            gammas = [g for g, _ in summary["stages"]["gamma_at_M"]["points"]]
            lo, hi = cs.WIDE_SCAN
            self.assertTrue(any(abs(g / lo - 1) < 1e-6 for g in gammas))
            self.assertTrue(any(abs(g / hi - 1) < 1e-6 for g in gammas))
            self.assertGreater(final["f1_median"], 0.8)
            configs = [json.loads(line) for line in evals_lines(out)]
            self.assertEqual({c["config"]["M"] for c in configs}, {-1})
            done = [c for c in configs if c["status"] == "completed"]
            self.assertTrue(done)
            # the label-free audit plays the uncapped game: no vertex is "at the cap"
            self.assertEqual({c["label_free"]["saturated_fraction"] for c in done}, {0.0})


class TestCommandLine(unittest.TestCase):
    def test_the_multilevel_phase_is_refused_early(self):
        register()
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as caught:
                cs.main(["tune", "--instance", PLANTED, "--phase", "multilevel", "--out", tmp])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("not available in hedonic 2.0", err.getvalue())
            self.assertEqual(list(Path(tmp).iterdir()), [])
            cs.configure(tmp)
            with self.assertRaises(ValueError):
                cs.tune(PLANTED, "multilevel", M0=8, M_max=16, seeds=[0], objective="f1", timeout=60, delta=0.002,
                        tol_u=0.1, max_evals=14, confirm_seeds=[0])

    def test_out_is_required_and_probe_and_grid_need_their_values(self):
        register()
        with tempfile.TemporaryDirectory() as tmp:
            for argv in (["tune", "--instance", PLANTED], ["probe", "--instance", PLANTED, "--out", tmp, "--gamma", "0.1"],
                         ["grid", "--instance", PLANTED, "--out", tmp, "--M", "4"]):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cs.main(argv)

    def test_probe_and_grid_modes_evaluate_through_the_same_ledger(self):
        register()
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(io.StringIO()) as buffer:
                self.assertEqual(cs.main(["probe", "--instance", PLANTED, "--out", tmp, "--gamma", "0.15", "--M", "4"]), 0)
                self.assertEqual(cs.main(["grid", "--instance", PLANTED, "--out", tmp, "--gammas", "3/20,0.3", "--M", "4"]), 0)
            self.assertEqual(len(evals_lines(Path(tmp))), 2)  # 3/20 is the probe's configuration
            self.assertEqual(buffer.getvalue().count("GRID planted local M=4"), 2)
        cs.CACHE.clear()


class TestLazyOnThePackagedSearch(unittest.TestCase):
    def run_lazy(self, out: Path, *flags: str) -> dict:
        from hedonic.experiments.overlapping import lazy_search

        register()
        original = cs.Probe.__call__
        with redirect_stdout(io.StringIO()):
            self.assertEqual(lazy_search.main([*flags, *tune_argv(out)]), 0)
        self.assertIs(cs.Probe.__call__, original)  # the shared module is unbudgeted again
        self.assertNotIn("_unbudgeted", cs.Probe.__dict__)
        return last_search(out)

    def test_a_budget_stops_the_search_and_keeps_the_best_observed_selection_setting(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self.run_lazy(Path(tmp), "--budget", "4")
            lazy = record["lazy"]
            self.assertEqual((lazy["stopped_by"], lazy["evaluations"]), ("budget", 4))
            self.assertEqual(record["final"]["f1_median"], max(a["f1"] for a in lazy["asked"]))
            self.assertEqual(len(evals_lines(Path(tmp))), 4)
            self.assertEqual(record["instance"]["spec"], PLANTED)
        cs.CACHE.clear()

    def test_a_search_that_ends_inside_the_budget_is_the_full_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self.run_lazy(Path(tmp), "--budget", "10000")
            self.assertNotIn("lazy", record)
            self.assertIn("snap", record)
        cs.CACHE.clear()


if __name__ == "__main__":
    unittest.main()
