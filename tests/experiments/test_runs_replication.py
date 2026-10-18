"""Replication: a deterministic method runs once per network (the lowest seed), the others at every seed.

One table (``overlapping.replication``) serves the calibration and the benchmark. The plan, the queue's judgement of the
files, the robustness stage, the result tables, the morning check and the joint plots all follow it, and the setting
never changes a unit's identity, so records are shared by ``--replication auto`` and ``all``. Data-free.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments.overlapping import baseline_calibration as bc
from hedonic.experiments.overlapping import cover_audit as ca
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import replication
from hedonic.experiments.overlapping import runmanager as rm
from hedonic.experiments.overlapping import runqueue
from hedonic.experiments.plots import benchmark_tradeoffs as bt
from hedonic.experiments.resources import CONFIG_DIR

DETERMINISTIC = ["codeseg", "clique_percolation", "fox", "bigclam", "ncgame", "neo_kmeans"]
EVERY_SEED = ["hedonic", "angel", "ego_splitting"]


def config(*argv: str) -> qs.Config:
    return qs.config_from_args(qs.build_parser().parse_args(list(argv)))


class TestTable(unittest.TestCase):
    def test_every_method_and_control_of_the_catalogue_is_classified(self):
        for method in qs.METHODS + qs.CONTROLS:
            self.assertIn(method.key, replication.REPLICATION, method.key)
            info = replication.REPLICATION[method.key]
            self.assertIn(info["kind"], replication.KINDS)
            self.assertTrue(info["note"])
        self.assertEqual(sorted(m for m in replication.REPLICATION if replication.kind(m) == "deterministic"),
                         sorted([*DETERMINISTIC, "disjoint_oracle"]))
        self.assertEqual(replication.kind("hedonic_local"), "seeded")  # the runner's name of Hedonic
        self.assertEqual(replication.kind("not-a-method"), "seeded")  # unclassified: every seed, never fewer runs

    def test_the_calibration_reads_the_same_table(self):
        self.assertIs(bc.REPLICATION, replication.REPLICATION)
        # CoDeSEG runs at one thread in both, where it repeats: neither replicates it
        self.assertEqual(replication.kind("codeseg"), "deterministic")
        self.assertEqual(replication.calibration_kind("codeseg"), "deterministic")
        self.assertNotIn("threads", qs.method_params(qs.BY_KEY["codeseg"], qs.Config(threads=8), "dblp"))
        ctx = bc.with_replicates(bc.Context(graph=bc.synthetic_graph(50)), 3)
        self.assertEqual(ctx.replicates, {})

    def test_planned_seeds(self):
        self.assertEqual(replication.planned_seeds("fox", [5, 3, 9]), [3])
        self.assertEqual(replication.planned_seeds("fox", [5, 3, 9], "all"), [3, 5, 9])
        self.assertEqual(replication.planned_seeds("angel", [5, 3, 9]), [3, 5, 9])
        self.assertEqual(replication.planned_seeds("fox", []), [])
        with self.assertRaises(ValueError):
            replication.check_mode("some")


class TestPlan(unittest.TestCase):
    def test_a_deterministic_method_runs_once_per_network_at_the_lowest_seed(self):
        cfg = config("--networks", "dblp,amazon", "--methods", "all", "--seeds", "0-9")
        plan = qs.build_plan(cfg, {})
        runs = Counter(m.key for _net, _seed, m in plan)
        self.assertEqual({m: runs[m] for m in DETERMINISTIC}, {m: 2 for m in DETERMINISTIC})
        self.assertEqual({m: runs[m] for m in EVERY_SEED}, {m: 20 for m in EVERY_SEED})
        self.assertEqual(len(plan), 2 * (3 * 10 + 6))
        self.assertEqual({seed for _net, seed, m in plan if m.key in DETERMINISTIC}, {0})
        # the lowest seed of the plan, whatever it is; the nesting is kept
        cfg = config("--networks", "dblp", "--methods", "hedonic,fox", "--seeds", "3,5,7", "--order", "seed,method,network")
        self.assertEqual([(s, m.key) for _n, s, m in qs.build_plan(cfg, {})],
                         [(3, "hedonic"), (3, "fox"), (5, "hedonic"), (7, "hedonic")])

    def test_replication_all_restores_every_seed(self):
        cfg = config("--networks", "dblp,amazon", "--methods", "all", "--seeds", "0-9", "--replication", "all")
        self.assertEqual(cfg.replication, "all")
        self.assertEqual(len(qs.build_plan(cfg, {})), 2 * 9 * 10)
        self.assertIn("--replication all", cfg.command())
        self.assertNotIn("--replication", config("--seeds", "0-9").command())
        self.assertIn("replication all", qs.describe(cfg))
        again = config(*cfg.command().split()[3:])  # the command a resume would run plans the same units
        self.assertEqual(len(qs.build_plan(again, {})), 180)

    def test_a_config_file_sets_it_and_a_bad_value_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "c.toml"
            good.write_text('[hedonic_run]\nmethods = ["hedonic", "fox"]\nseeds = "0-2"\nreplication = "all"\n')
            self.assertEqual(len(qs.build_plan(qs.load_config(str(good)), {})), 6)
            bad = Path(tmp) / "b.toml"
            bad.write_text('[hedonic_run]\nreplication = "some"\n')
            with self.assertRaises(ValueError):
                qs.load_config(str(bad))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            qs.build_parser().parse_args(["--replication", "some"])

    def test_the_paper_benchmark_plan_and_its_estimate(self):
        cfg = config("--config", str(CONFIG_DIR / "paper-benchmark.toml"))
        self.assertEqual(cfg.seeds, list(range(10)))  # the config keeps seeds 0-9
        plan = qs.build_plan(cfg, {})
        self.assertEqual(len(plan), 8 * (3 * 10 + 6))  # 8 pairs; 288 units instead of 720
        every = config("--config", str(CONFIG_DIR / "paper-benchmark.toml"), "--replication", "all")
        seconds = lambda c: sum(qs.expected_seconds(m, c, net) for net, _s, m in qs.build_plan(c, {}))  # noqa: E731
        self.assertLess(seconds(cfg), seconds(every))
        text = qs.render_plan(cfg)
        self.assertIn("288 runs", text)
        self.assertIn("deterministic, one run at seed 0: CoDeSEG, FOX, CliquePerc, BigCLAM, NcGame, NEO-K-Means", text)

    def test_the_setting_never_changes_a_units_arguments_identity_or_folder(self):
        auto = config("--networks", "amazon:top5000", "--induce", "covered", "--full", "--methods", "all",
                      "--seeds", "0-9", "--save-covers")
        every = config("--networks", "amazon:top5000", "--induce", "covered", "--full", "--methods", "all",
                       "--seeds", "0-9", "--save-covers", "--replication", "all")
        for key, seed in (("clique_percolation", 0), ("hedonic", 0), ("hedonic", 7), ("angel", 3)):
            m = qs.BY_KEY[key]
            self.assertEqual(qs.unit_argv(auto, "amazon:top5000", seed, m, None),
                             qs.unit_argv(every, "amazon:top5000", seed, m, None))
            digest = qs.unit_digest(auto, "amazon:top5000", seed, m, None)
            self.assertIsNotNone(digest)
            self.assertEqual(digest, qs.unit_digest(every, "amazon:top5000", seed, m, None))
        # every unit of the auto plan is a unit of the all plan, with the same coordinates
        units = lambda c: {(net, seed, m.key) for net, seed, m in qs.build_plan(c, {})}  # noqa: E731
        self.assertLess(units(auto), units(every))


class _Folders(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.overlap = self.root / "overlap"
        self._patches = [
            patch.dict(os.environ, {"HEDONIC_RUNS_DIR": str(self.root / "registry")}),
            patch("hedonic.experiments.overlapping.userconfig.defaults",
                  return_value={"output_dir": str(self.overlap), "cache_dir": str(self.root / "cache")}),
        ]
        for patcher in self._patches:
            patcher.start()
        runqueue._DATA_CACHE.clear()

    def tearDown(self):
        for patcher in reversed(self._patches):
            patcher.stop()
        runqueue._DATA_CACHE.clear()
        self._tmp.cleanup()

    def record(self, cfg: qs.Config, seed: int, key: str) -> None:
        """A certified record of one unit, as the runner writes it (its digest is the one the runner would stamp)."""
        m = qs.BY_KEY[key]
        _argv, out = qs.unit_argv(cfg, "dblp", seed, m, None)
        path = out / "runs" / "dblp" / f"{m.runner}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "completed", "config_digest": qs.unit_digest(cfg, "dblp", seed, m, None),
                                    "dataset": "dblp", "method": m.runner}))


class TestQueue(_Folders):
    RUN = ["exp", "--networks", "dblp", "--methods", "hedonic,clique_percolation", "--seeds", "0-2", "--name", "bench"]

    def state(self, run: list[str]) -> str:
        runqueue._DATA_CACHE.clear()
        return runqueue.data_state(run)[0]

    def test_a_benchmark_is_complete_with_only_the_lowest_seed_of_a_deterministic_method(self):
        cfg = config(*self.RUN[1:])
        cfg.output_dir = str(self.overlap / "bench")
        for seed in (0, 1, 2):
            self.record(cfg, seed, "hedonic")
        self.assertEqual(self.state(self.RUN), "partial")
        self.record(cfg, 0, "clique_percolation")
        self.assertEqual(self.state(self.RUN), "complete")
        self.assertEqual(runqueue.unit_counts(self.RUN), (4, 4, "units"))
        folder = runqueue.data_state(self.RUN)[1]
        parts = {(group, label): flags for group, label, flags in runqueue._PARTS[folder]}
        self.assertEqual(parts[("DBLP · all", qs.BY_KEY["clique_percolation"].label)], [True])  # one mark: one planned run
        model = runqueue._MODEL[folder]
        units = [u.coords for family in model.families for u in family.units]
        self.assertEqual(len(units), 4)  # the tree prices no seed the run will not execute
        self.assertNotIn(("DBLP · all", qs.BY_KEY["clique_percolation"].label, "1"), units)
        # every seed asked for: the deterministic method's other seeds are work again, and the records are shared
        every = [*self.RUN, "--replication", "all"]
        self.assertEqual(self.state(every), "partial")
        self.assertEqual(runqueue.unit_counts(every), (4, 6, "units"))
        self.record(cfg, 1, "clique_percolation")
        self.record(cfg, 2, "clique_percolation")
        self.assertEqual(self.state(every), "complete")
        self.assertEqual(self.state(self.RUN), "complete")  # extra records are harmless: not counted, not missing
        self.assertEqual(runqueue.unit_counts(self.RUN), (4, 4, "units"))


class TestRobustnessStage(unittest.TestCase):
    def test_the_stage_and_the_standalone_audit_expect_a_deterministic_cover_at_the_lowest_seed_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            stage = ca.Stage(run, ["dblp"], ["hedonic", "clique_percolation"], [0, 1, 2])
            keys = [c["key"] for c in ca.conditions(stage.plan)]
            self.assertEqual(keys, [f"dblp/seed{s}/hedonic/{p}" for s in range(3) for p in ("cover", "labelled")]
                             + ["dblp/seed0/clique_percolation/cover"])
            self.assertEqual(stage.unit_conditions("dblp", 1, "clique_percolation"), [])
            every = ca.Stage(run, ["dblp"], ["hedonic", "clique_percolation"], [0, 1, 2], replication_mode="all")
            self.assertEqual(len(ca.conditions(every.plan)), 3 * 2 + 3)
            plan = ca.resolve(["--source", str(run), "--networks", "dblp", "--methods", "hedonic,clique_percolation",
                               "--seeds", "0-2", "--output-dir", str(Path(tmp) / "audit")])
            self.assertEqual(len(ca.conditions(plan)), 7)
            self.assertIn("deterministic methods: seed 0 only", ca.describe(plan))
            again = ca.resolve(ca.launch_argv({**plan, "replication": "all"}))  # a resume rebuilds the same plan
            self.assertEqual((again["replication"], len(ca.conditions(again))), ("all", 9))
            # the benchmark opens its stage with the run's own setting
            cfg = config("--networks", "dblp", "--methods", "hedonic,clique_percolation", "--seeds", "0-2", "--full",
                         "--robustness", "--output-dir", str(run))
            self.assertEqual(len(ca.conditions(qs.open_stage(cfg).plan)), 7)


class TestResults(unittest.TestCase):
    def records(self) -> list[dict]:
        metrics = lambda f1: {key: f1 for key, _ in qs.COLUMNS}  # noqa: E731
        recs = [{"network": "dblp", "method_key": "hedonic", "seed": s, "status": "completed", "metrics": metrics(0.4 + s / 100),
                 "detection_seconds": 2.0 + s} for s in (0, 1, 2)]
        recs.append({"network": "dblp", "method_key": "clique_percolation", "seed": 0, "status": "completed",
                     "metrics": metrics(0.3), "detection_seconds": 1.0})
        return recs

    def test_a_single_run_method_shows_its_value_without_sd_and_runs_1_of_1(self):
        cfg = qs.Config(networks=["dblp"], methods=["hedonic", "clique_percolation"], seeds=[0, 1, 2])
        rows = {r["method"]: r for r in qs.summarise(self.records(), cfg)}
        once, many = rows["clique_percolation"], rows["hedonic"]
        self.assertEqual((once["completed"], once["runs"], once["planned_runs"], once["replication"]),
                         (1, 1, 1, "deterministic"))
        self.assertIsNone(once["f1_sd"])
        self.assertEqual(once["sd_note"], "n/a: one planned run")
        self.assertEqual((many["runs"], many["planned_runs"], many["sd_note"]), (3, 3, None))
        self.assertIsNotNone(many["f1_sd"])
        table = qs.render(list(rows.values()), cfg, columns=400)
        line = next(row for row in table.splitlines() if "CliquePerc" in row)
        self.assertIn("1/1", line)
        self.assertNotIn("±", line)
        self.assertIn("±", next(row for row in table.splitlines() if "Hedonic" in row))
        self.assertIn("a deterministic method: its one run (seed 0)", table)


class TestCheck(unittest.TestCase):
    def test_the_morning_check_leaves_out_methods_whose_runs_are_replicas(self):
        cfg = qs.Config(networks=["dblp"], methods=["codeseg"], seeds=[0, 1])
        entry = {"name": "bench", "config": cfg.__dict__}
        prog = {"plan": [{"network": "dblp", "seed": s, "method": "codeseg", "label": "CoDeSEG", "status": "completed",
                          "detection_seconds": 1.0} for s in (0, 1)]}
        out = io.StringIO()
        replicas = {"kind": "replicas", "runs": 10, "note": "a method whose runs differ (CoDeSEG on several threads)"}
        with patch.object(rm, "read_progress", return_value=prog), contextlib.redirect_stdout(out), \
                patch.dict(replication.REPLICATION, {"codeseg": replicas}), \
                patch.object(qs, "run_one", side_effect=AssertionError("no rerun of a replica")):
            self.assertEqual(rm.check_benchmark(entry, 2), 1)
        self.assertIn("independent replicas", out.getvalue())
        self.assertIn("nothing to check yet", out.getvalue())


class TestPlots(unittest.TestCase):
    def test_a_deterministic_method_is_one_point_unless_every_seed_is_asked_for(self):
        from tests.experiments.test_studies_benchmark_tradeoffs import write_record

        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp) / "bench"
            for seed in (0, 1, 2):  # an older run that ran the deterministic method at every seed
                write_record(bench, "dblp", seed, "hedonic", f1=0.4 + seed / 100, seconds=10.0)
                write_record(bench, "dblp", seed, "clique_percolation", f1=0.3, seconds=5.0)
            rows, _ = bt.load_benchmark([bench], "f1")
            self.assertEqual(sorted((r["method"], r["seed"]) for r in rows),
                             [("clique_percolation", 0), ("hedonic", 0), ("hedonic", 1), ("hedonic", 2)])
            agg = {a["method"]: a for a in bt.aggregate(rows)}
            self.assertEqual((agg["clique_percolation"]["n"], agg["clique_percolation"]["accuracy_ci95"]), (1, None))
            self.assertEqual(agg["clique_percolation"]["replication"], "deterministic")
            rows, _ = bt.load_benchmark([bench], "f1", replication_mode="all")
            self.assertEqual(len(rows), 6)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(bt.main(["--benchmark", str(bench), "--output-dir", str(Path(tmp) / "fig"),
                                          "--formats", "png"]), 0)


if __name__ == "__main__":
    unittest.main()
