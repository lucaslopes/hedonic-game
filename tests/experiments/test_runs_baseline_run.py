"""The baselines' calibration as a run: plan, resume, the settings file, replicas, and the benchmark's ``--tuned``."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments import CLI
from hedonic.experiments.overlapping import baseline_calibration as bc
from hedonic.experiments.overlapping import replication
from hedonic.experiments.overlapping import baseline_run as br
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import runqueue

BLOCKS = 10


def blocks_runner(method, graph, params, seed):
    if method == "bigclam" and params.get("communities") == 7:
        raise RuntimeError("boom")
    k = int(params.get("communities", params.get("clusters", BLOCKS)))
    size = -(-graph.vcount() // k)
    return [list(range(i, min(i + size, graph.vcount()))) for i in range(0, graph.vcount(), size)]


class Loader:
    """Stands in for ``baseline_calibration.load_context``: the planted graph, a fake detector, the run's own ledger."""

    def __init__(self):
        import os, tempfile as _t
        handle, self.path = _t.mkstemp(prefix="loader-")  # a pair's graph is loaded in a forked child: count through a file
        os.close(handle)

    @property
    def targets(self) -> list[str]:
        return sorted(Path(self.path).read_text().split())

    def __call__(self, target, *, network_root, induce, seed, budget, ledger_dir, runtime):
        with open(self.path, "a") as handle:
            handle.write(target + "\n")
        return bc.Context(graph=bc.synthetic_graph(400), seed=seed, budget=budget, ledger_dir=ledger_dir,
                          runner=blocks_runner, runtime=runtime, label=target)


def plan_for(tmp: str, *extra: str) -> dict:
    return br.resolve(["--networks", "dblp:top5000,amazon", "--methods", "clique_percolation,angel,codeseg,ncgame,bigclam",
                       "--output-dir", tmp, "--max-evals", "8", *extra])


class TestPlan(unittest.TestCase):
    def test_searches_go_method_by_method_in_the_methods_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_for(tmp)
        keys = [s["key"] for s in br.searches(plan)]
        self.assertEqual(keys[:4], ["clique_percolation/dblp:top5000", "clique_percolation/amazon",
                                    "angel/dblp:top5000", "angel/amazon"])  # a plain name stays the network (its `all` cover)
        self.assertEqual([k.split("/")[0] for k in keys[::2]], ["clique_percolation", "angel", "codeseg", "ncgame", "bigclam"])
        self.assertEqual(len(keys), 10)

    def test_covers_cross_the_networks_and_bad_arguments_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = br.resolve(["--networks", "dblp,amazon", "--covers", "top5000,all", "--output-dir", tmp])
            self.assertEqual(len(plan["networks"]), 4)
            self.assertEqual(plan["methods"], list(bc.CALIBRATORS))
            for bad in (["--methods", "nope"], ["--methods", "angel,angel"], ["--replicates", "0"], ["--max-evals", "-1"]):
                with self.assertRaises(ValueError, msg=bad):
                    br.resolve(["--output-dir", tmp, *bad])

    def test_dry_run_prints_the_plan_and_the_replication_policy_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(br.front_main(["--dry-run", "--output-dir", tmp, "--networks", "dblp:top5000"]), 0)
            text = out.getvalue()
            self.assertIn("G[S] only", text)
            self.assertIn("bigclam", text)
            self.assertIn("10 runs per pair (seeded)", text)
            self.assertIn("1 run per pair (deterministic)", text)
            self.assertNotIn("replicas 3", text)  # no method of the registry has replica runs now
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_launch_argv_rebuilds_the_same_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_for(tmp, "--sensitivity", "--replicates", "2", "--seed", "3")
            again = br.resolve(br.launch_argv(plan))
        for key in ("networks", "methods", "seed", "max_evals", "replicates", "sensitivity", "eval_timeout_minutes"):
            self.assertEqual(plan[key], again[key], key)


class TestConfigFile(unittest.TestCase):
    def write(self, text: str) -> str:
        d = tempfile.mkdtemp()
        path = Path(d) / "c.toml"
        path.write_text(text)
        return str(path)

    def test_a_toml_table_gives_the_options_and_flags_override_it(self):
        path = self.write('[calibrate_baselines]\nname = "x"\nnetworks = ["dblp:top5000", "amazon:all"]\n'
                          'methods = ["fox", "angel"]\nmax_evals = 5\nworkers = 3\noutput_dir = "' + tempfile.mkdtemp() + '"\n')
        plan = br.resolve(["--config", path])
        self.assertEqual((plan["networks"], plan["methods"], plan["max_evals"], plan["workers"], plan["name"]),
                         (["dblp:top5000", "amazon:all"], ["fox", "angel"], 5, 3, "x"))
        plan = br.resolve(["--config", path, "--max-evals", "2", "--methods", "fox"])
        self.assertEqual((plan["max_evals"], plan["methods"]), (2, ["fox"]))

    def test_unknown_key_is_refused_and_the_paper_config_parses(self):
        with self.assertRaisesRegex(ValueError, "unknown key"):
            br.resolve(["--config", self.write("[calibrate_baselines]\nnetwrks = 1\n")])
        from hedonic.experiments.resources import CONFIG_DIR

        plan = br.resolve(["--config", str(CONFIG_DIR / "baseline-calibration-paper.toml")])
        self.assertEqual(len(plan["networks"]), 8)
        self.assertNotIn("friendster:all", plan["networks"])
        self.assertIn("friendster:top5000", plan["networks"])
        self.assertEqual(len(plan["methods"]), 8)


class TestRun(unittest.TestCase):
    def run_plan(self, plan, loader):
        with patch.object(bc, "load_context", loader), redirect_stdout(io.StringIO()):
            return br.run_calibration(plan)

    def test_a_run_writes_the_settings_file_and_a_resume_loads_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_for(tmp)
            loader = Loader()
            self.assertEqual(self.run_plan(plan, loader), 0)
            # each searched method loads each pair, in a child; CoDeSEG and NcGame take the authors' setting and load nothing
            self.assertEqual(loader.targets, ["amazon"] * 3 + ["dblp:top5000"] * 3)
            data = br.read_tuned(plan)
            self.assertTrue(data["complete"])
            self.assertFalse(data["uses_ground_truth"])
            self.assertEqual(br.state_of(plan), "complete")
            self.assertEqual(set(data["searches"]), {s["key"] for s in br.searches(plan)})
            settings = data["settings"]
            self.assertEqual(settings["codeseg"]["dblp:top5000"], {"tau": 0.3, "alpha": 1.0, "iterations": 10})
            self.assertEqual(settings["ncgame"]["amazon"], {})   # the runner's own defaults are the setting
            self.assertEqual(set(settings["bigclam"]["dblp:top5000"]), {"communities"})  # no threads: the benchmark's CPU policy
            self.assertIn("clique_size", settings["clique_percolation"]["amazon"])
            self.assertEqual(data["searches"]["bigclam/amazon"]["replication"]["kind"], "deterministic")
            self.assertEqual(data["searches"]["angel/amazon"]["replication"]["kind"], "seeded")
            again = Loader()
            self.assertEqual(self.run_plan(plan, again), 0)
            self.assertEqual(again.targets, [])  # everything is recorded: no graph is loaded

    def test_a_fixed_setting_loads_a_graph_only_for_a_sensitivity_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader = Loader()
            self.assertEqual(self.run_plan(plan_for(tmp, "--methods", "codeseg,ncgame,ego_splitting"), loader), 0)
            self.assertEqual(loader.targets, [])
            data = br.read_tuned(plan_for(tmp, "--methods", "codeseg,ncgame,ego_splitting"))
            self.assertEqual(data["searches"]["codeseg/amazon"]["status"], "fixed")
            self.assertEqual(data["settings"]["codeseg"]["amazon"], {"tau": 0.3, "alpha": 1.0, "iterations": 10})
            loader = Loader()
            self.run_plan(plan_for(tmp + "/s", "--methods", "codeseg", "--sensitivity"), loader)
            self.assertEqual(loader.targets, ["amazon", "dblp:top5000"])

    def test_a_run_goes_method_by_method_and_the_pairs_of_a_method_may_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            for workers in ("1", "2"):
                plan = plan_for(tmp + "/w" + workers, "--workers", workers, "--methods", "codeseg,ncgame,clique_percolation")
                events: list[str] = []
                with patch.object(bc, "load_context", Loader()), redirect_stdout(io.StringIO()):
                    br.run_calibration(plan, progress=lambda e: events.append(e["key"]) if e["type"] == "unit" else None)
                self.assertEqual([k.split("/")[0] for k in events],
                                 ["codeseg"] * 2 + ["ncgame"] * 2 + ["clique_percolation"] * 2)  # a method finishes before the next starts

    def test_a_partial_run_resumes_only_what_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_for(tmp)
            self.run_plan(plan, Loader())
            path = Path(tmp) / br.TUNED_NAME
            data = json.loads(path.read_text())
            del data["searches"]["angel/amazon"]
            data["complete"] = False
            path.write_text(json.dumps(data))
            self.assertEqual(br.state_of(plan), "partial")
            loader = Loader()
            self.assertEqual(self.run_plan(plan, loader), 0)
            self.assertEqual(loader.targets, ["amazon"])
            self.assertEqual(br.state_of(plan), "complete")

    def test_a_calibration_copied_from_a_larger_plan_is_reused_whole(self):
        """The 1.x ``baseline-paper`` run had a ninth pair: its file says ``complete: false`` for that plan, but every
        search of the 2.0 queue's plan is recorded, so nothing is loaded or run and the file is marked complete."""
        with tempfile.TemporaryDirectory() as earlier, tempfile.TemporaryDirectory() as tmp:
            larger = br.resolve(["--networks", "dblp:top5000,amazon,youtube:top5000", "--methods", "codeseg,ncgame",
                                 "--output-dir", earlier])
            self.run_plan(larger, Loader())
            data = json.loads((Path(earlier) / br.TUNED_NAME).read_text())
            del data["searches"]["ncgame/youtube:top5000"]  # the earlier run stopped before its last search
            data["complete"] = False
            data["environment"] = {"hedonic": "1.4.0", "igraph_c": "1.0.0.6"}  # another build: never consulted
            (Path(tmp) / br.TUNED_NAME).write_text(json.dumps(data))
            plan = br.resolve(["--networks", "dblp:top5000,amazon", "--methods", "codeseg,ncgame", "--output-dir", tmp])
            self.assertEqual(br.state_of(plan), "complete")
            loader = Loader()
            self.assertEqual(self.run_plan(plan, loader), 0)
            self.assertEqual(loader.targets, [])
            self.assertTrue(br.read_tuned(plan)["complete"])

    def test_a_method_that_cannot_run_is_recorded_without_a_setting_and_does_not_stop_the_others(self):
        class Missing(Exception):
            pass

        Missing.__name__ = "MethodUnavailable"

        def runner(method, graph, params, seed):
            if method == "angel":
                raise Missing("no cdlib")
            return blocks_runner(method, graph, params, seed)

        def loader(target, **kw):
            ctx = Loader()(target, **kw)
            ctx.runner = runner
            return ctx

        with tempfile.TemporaryDirectory() as tmp:
            plan = br.resolve(["--networks", "dblp:top5000", "--methods", "angel,codeseg", "--output-dir", tmp])
            self.run_plan(plan, loader)
            data = br.read_tuned(plan)
            self.assertTrue(data["complete"])
            self.assertEqual(data["searches"]["angel/dblp:top5000"]["status"], "unavailable")
            self.assertNotIn("dblp:top5000", data["settings"].get("angel", {}))
            self.assertIn("dblp:top5000", data["settings"]["codeseg"])

    def test_the_overwrite_flag_deletes_the_results_and_the_ledger_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_for(tmp)
            self.run_plan(plan, Loader())
            br.reset(plan)
            self.assertFalse((Path(tmp) / br.TUNED_NAME).exists())
            self.assertEqual([p for p in (Path(tmp) / "ledger").rglob("*") if p.is_file()], [])


class SizedLoader(Loader):
    """A loader whose pairs differ in size (``:all`` twice the vertices), so a transferred K must be scaled."""

    def __call__(self, target, **kw):
        ctx = super().__call__(target, **kw)
        ctx.graph = bc.synthetic_graph(800 if target.endswith(":all") else 400)
        return ctx


class TestAdoptAndTransfer(unittest.TestCase):
    run_plan = TestRun.run_plan

    def test_method_budget_is_per_method_and_survives_a_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_for(tmp, "--method-budget", "angel=3:60")
            self.assertEqual(plan["method_budgets"], {"angel": (3, 60.0)})
            self.assertEqual(br.budget_of(plan, "angel").max_evals, 3)
            self.assertEqual(br.budget_of(plan, "angel").max_seconds, 3600)
            self.assertEqual(br.budget_of(plan, "bigclam").max_evals, 8)
            again = br.resolve(br.launch_argv(plan))
            self.assertEqual(again["method_budgets"], plan["method_budgets"])
            for bad in ("nope=3", "angel", "angel=x", "fox=3"):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    plan_for(tmp, "--method-budget", bad)  # fox is not in plan_for's methods
            plan = plan_for(tmp, "--method-budget", "angel=3")
            with patch.object(bc, "load_context", Loader()), redirect_stdout(io.StringIO()):
                br.run_calibration(plan)
            self.assertLessEqual(len(br.read_tuned(plan)["searches"]["angel/amazon"]["candidates"]), 3)

    def test_a_transferred_pair_is_not_searched_and_its_k_is_scaled(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = br.resolve(["--networks", "dblp:top5000,amazon:top5000,dblp:all", "--methods", "codeseg,bigclam,angel",
                               "--transfer", "dblp:all", "--output-dir", tmp])
            self.assertIn("transfer", "\n".join(br.plan_lines(plan)))
            loader = SizedLoader()
            with patch.object(bc, "load_context", loader), redirect_stdout(io.StringIO()):
                self.assertEqual(br.run_calibration(plan), 0)
            self.assertEqual(loader.targets.count("dblp:all"), 1)  # codeseg's published setting only: no search there
            data = br.read_tuned(plan)
            moved = data["searches"]["bigclam/dblp:all"]
            self.assertEqual(moved["status"], "transferred")
            self.assertEqual(moved["transferred_from"], "dblp:top5000")  # the same network's other cover first
            source = data["settings"]["bigclam"]["dblp:top5000"]["communities"]
            self.assertEqual(data["settings"]["bigclam"]["dblp:all"]["communities"], round(source * 800 / 400))
            self.assertEqual(data["settings"]["angel"]["dblp:all"], data["settings"]["angel"]["dblp:top5000"])
            self.assertEqual(data["searches"]["codeseg/dblp:all"]["status"], "fixed")
            self.assertTrue(data["complete"])
            self.assertEqual(br.resolve(br.launch_argv(plan))["transfer"], ["dblp:all"])
        with self.assertRaises(ValueError):
            br.resolve(["--networks", "dblp:top5000", "--transfer", "amazon:all"])
        with self.assertRaises(ValueError):
            br.resolve(["--networks", "dblp:top5000", "--transfer", "dblp:top5000"])

    def test_a_search_without_a_setting_is_transferred_when_asked(self):
        def runner(method, graph, params, seed):
            if method == "bigclam" and graph.vcount() == 800:
                raise RuntimeError("every probe fails here")
            return blocks_runner(method, graph, params, seed)

        def loader(target, **kw):
            ctx = SizedLoader()(target, **kw)
            ctx.runner = runner
            return ctx

        with tempfile.TemporaryDirectory() as tmp:
            plan = br.resolve(["--networks", "dblp:top5000,amazon:all", "--methods", "bigclam", "--transfer-missing",
                               "--output-dir", tmp])
            with patch.object(bc, "load_context", loader), redirect_stdout(io.StringIO()):
                self.assertEqual(br.run_calibration(plan), 0)
            entry = br.read_tuned(plan)["searches"]["bigclam/amazon:all"]
            self.assertEqual(entry["status"], "transferred")
            self.assertEqual(entry["transferred_from"], "dblp:top5000")
            self.assertEqual(entry["failed_search"]["status"], "failed")  # the search that failed is kept beside it

    def test_finished_searches_are_adopted_with_their_ledgers_and_the_source_is_untouched(self):
        with tempfile.TemporaryDirectory() as earlier, tempfile.TemporaryDirectory() as tmp:
            first = br.resolve(["--networks", "dblp:top5000,amazon", "--methods", "codeseg,bigclam", "--output-dir", earlier])
            self.run_plan(first, Loader())
            data = json.loads((Path(earlier) / br.TUNED_NAME).read_text())
            del data["searches"]["bigclam/amazon"]  # stopped before its last search: its ledger stays
            data["environment"] = {"hedonic": "1.0.5"}
            for entry in data["searches"].values():
                entry.pop("environment", None)
            (Path(earlier) / br.TUNED_NAME).write_text(json.dumps(data))
            before = sorted((p.name, p.stat().st_size) for p in Path(earlier).rglob("*") if p.is_file())
            plan = br.resolve(["--networks", "dblp:top5000,amazon", "--methods", "codeseg,bigclam,angel",
                               "--adopt-from", earlier, "--output-dir", tmp])
            self.assertEqual(br.adoptable(plan), {str(Path(earlier).resolve()): ["codeseg/dblp:top5000", "codeseg/amazon",
                                                                                 "bigclam/dblp:top5000"]})
            self.assertIn("adopt    3 finished searches", "\n".join(br.plan_lines(plan)))
            loader = Loader()
            calls = []

            def counting(method, graph, params, seed):
                calls.append(method)
                return blocks_runner(method, graph, params, seed)

            def load(target, **kw):
                ctx = loader(target, **kw)
                ctx.runner = counting
                return ctx

            with patch.object(bc, "load_context", load), redirect_stdout(io.StringIO()):
                self.assertEqual(br.run_calibration(plan), 0)
            self.assertEqual(sorted(loader.targets), ["amazon", "amazon", "dblp:top5000"])  # bigclam/amazon, angel x 2
            self.assertNotIn("bigclam", calls)  # bigclam/amazon resumed entirely from the copied ledger
            got = br.read_tuned(plan)
            self.assertEqual(got["searches"]["codeseg/amazon"]["adopted_from"]["environment"], {"hedonic": "1.0.5"})
            self.assertNotIn("adopted_from", got["searches"]["angel/amazon"])
            self.assertTrue(got["complete"])
            after = sorted((p.name, p.stat().st_size) for p in Path(earlier).rglob("*") if p.is_file())
            self.assertEqual(before, after)
            bad = dict(data, protocol="other")
            (Path(earlier) / br.TUNED_NAME).write_text(json.dumps(bad))
            other = br.resolve(["--networks", "dblp:top5000", "--methods", "codeseg", "--adopt-from", earlier,
                                "--output-dir", tmp + "/x"])
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                br.adopt(other)


REGISTERED = {m: dict(i) for m, i in bc.REPLICATION.items()}  # the policy as shipped
REPLICAS_POLICY = {m: {"kind": "replicas", "runs": 10, "note": "test"} for m in ("bigclam", "fox", "neo_kmeans")}


class TestReplicas(unittest.TestCase):
    """The replica mechanism, which no method of the registry uses now (BigCLAM, NEO-K-Means and FOX repeat identically)."""

    def setUp(self):
        patcher = patch.dict(bc.REPLICATION, REPLICAS_POLICY)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_replica_method_is_scored_by_the_mean_and_costs_one_setting_of_budget(self):
        ctx = bc.Context(graph=bc.synthetic_graph(400), seed=1, runner=blocks_runner,
                         budget=bc.Budget(max_evals=None, eval_timeout=60))
        with tempfile.TemporaryDirectory() as tmp:
            ctx.ledger_dir = Path(tmp)
            bc.with_replicates(ctx, 3)
            self.assertEqual(ctx.replicates, {"bigclam": 3, "fox": 3, "neo_kmeans": 3})
            rec = bc.evaluate(ctx, "bigclam", {"communities": 10})
            self.assertEqual((rec["replicas"], rec["replicas_completed"]), (3, 3))
            self.assertEqual(ctx.evaluations, 1)
            lines = (Path(tmp) / "evals-bigclam.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(bc.evaluate(ctx, "bigclam", {"communities": 10})["score"], rec["score"])  # memoised
            self.assertEqual(len((Path(tmp) / "evals-bigclam.jsonl").read_text().splitlines()), 3)

    def test_deterministic_methods_are_never_replicated(self):
        ctx = bc.with_replicates(bc.Context(graph=bc.synthetic_graph(400)), 5)
        for method in ("angel", "clique_percolation", "ncgame", "codeseg", "ego_splitting"):
            self.assertNotIn(method, ctx.replicates)

    def test_the_registered_policy(self):
        # one table for the calibration and the benchmark (``replication``); CoDeSEG runs at one thread in both, where
        # its runs repeat
        self.assertIs(bc.REPLICATION, replication.REPLICATION)
        self.assertEqual({m: (i["kind"], i["runs"]) for m, i in REGISTERED.items()},
                         {"hedonic": ("seeded", 10), "clique_percolation": ("deterministic", 1), "angel": ("seeded", 10),
                          "ncgame": ("deterministic", 1), "codeseg": ("deterministic", 1), "ego_splitting": ("seeded", 10),
                          "bigclam": ("deterministic", 1), "fox": ("deterministic", 1), "neo_kmeans": ("deterministic", 1),
                          "disjoint_oracle": ("deterministic", 1)})
        self.assertEqual([m for m, i in REGISTERED.items() if i.get("calibration_kind", i["kind"]) == "replicas"], [])

    def test_runner_params_keep_only_what_the_runner_reads(self):
        self.assertEqual(bc.runner_params("fox", bc.FOX_FIXED), {"wcc_threshold": 0.1, "queue_size": 1})
        self.assertEqual(bc.runner_params("ncgame", bc.NCGAME_FIXED), {})
        self.assertEqual(bc.runner_params("bigclam", {"communities": 50, "threads": 1}), {"communities": 50})


class TestTuned(unittest.TestCase):
    def files(self, tmp: str) -> tuple[Path, Path]:
        hoc = Path(tmp) / "hoc_tuned.json"
        hoc.write_text(json.dumps({"complete": True, "settings": {"hedonic": {"dblp:top5000": {
            "max_memberships": 9, "absolute_resolution": 0.5}}}}))
        base = Path(tmp) / "baseline_tuned.json"
        base.write_text(json.dumps({"complete": True, "settings": {
            "codeseg": {"dblp:top5000": {"tau": 0.3, "alpha": 1.0, "iterations": 10}},
            "bigclam": {"dblp:top5000": {"communities": 120}}, "ncgame": {"dblp:top5000": {}}}}))
        return hoc, base

    def test_a_baseline_file_is_read_as_the_runner_parameters_and_unfinished_is_awaited(self):
        with tempfile.TemporaryDirectory() as tmp:
            hoc, base = self.files(tmp)
            tuned = qs.load_tuned_file(base)
            self.assertEqual(tuned["bigclam"]["dblp:top5000"], {"communities": 120})
            self.assertNotIn("ncgame", tuned)  # nothing to lay over the runner's defaults
            half = Path(tmp) / "half.json"
            half.write_text(json.dumps({"complete": False, "settings": {}}))
            with self.assertRaises(qs.TunedMissing):
                qs.load_tuned_file(half)
            with self.assertRaises(qs.TunedMissing):
                qs.load_tuned_file(Path(tmp) / "absent.json")

    def test_two_files_are_laid_over_the_methods_of_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            hoc, base = self.files(tmp)
            cfg = qs.Config(networks=["dblp:top5000"], methods=["hedonic", "codeseg", "bigclam"])
            qs.apply_tuned(cfg, f"{hoc},{base}")
            self.assertEqual(cfg.method_params["hedonic"]["dblp:top5000"]["max_memberships"], 9)
            self.assertEqual(cfg.method_params["codeseg"]["dblp:top5000"]["tau"], 0.3)
            self.assertEqual(cfg.tuned, f"{hoc.resolve()},{base.resolve()}")
            only = qs.Config(networks=["dblp:top5000"], methods=["codeseg"])
            qs.apply_tuned(only, [str(hoc), str(base)])
            self.assertEqual(set(only.method_params), {"codeseg"})  # a method outside the run is left alone
            params = qs.method_params(qs.BY_KEY["codeseg"], only, "dblp:top5000")
            self.assertEqual((params["tau"], params["iterations"]), (0.3, 10))   # over the screened 0.5 / 20

    def test_the_tuned_flag_repeats_and_takes_a_comma_list(self):
        args = qs.build_parser().parse_args(["--tuned", "a.json,b.json", "--tuned", "c.json"])
        self.assertEqual(qs.tuned_files(args.tuned), ["a.json", "b.json", "c.json"])

    def test_the_toml_snippet_has_only_runner_parameters(self):
        result = bc.calibrate_ncgame()
        text = bc.toml_snippet("dblp:top5000", [bc.calibrate_codeseg(), result])
        self.assertIn("tau = 0.3", text)
        self.assertNotIn("similarity_type", text)


class TestQueueAndCLI(unittest.TestCase):
    ARGS = ["calibrate-baselines", "--networks", "dblp:top5000", "--methods", "angel,codeseg"]

    def test_the_verb_is_registered_and_its_arguments_are_checked_by_its_own_parser(self):
        self.assertEqual(CLI.RUN_ALIASES["calibrate-baselines"], "baseline-calibrate")
        runqueue._check_arguments(self.ARGS)
        with self.assertRaises(ValueError):
            runqueue._check_arguments(["calibrate-baselines", "--methods", "nope"])

    def test_the_queue_judges_the_item_by_its_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = [*self.ARGS, "--output-dir", tmp]
            runqueue._DATA_CACHE.clear()
            self.assertEqual(runqueue.data_state(run, fresh=True)[0], "missing")
            plan = br.front_study(run[1:])
            with patch.object(bc, "load_context", Loader()), redirect_stdout(io.StringIO()):
                br.run_calibration(plan)
            self.assertEqual(runqueue.data_state(run, fresh=True)[0], "complete")
            self.assertEqual(runqueue.unit_counts(run), (2, 2, "calibrations"))

    def test_front_main_refuses_a_bad_plan_with_exit_2(self):
        with redirect_stderr(io.StringIO()):
            self.assertEqual(br.front_main(["--methods", "nope"]), 2)


if __name__ == "__main__":
    unittest.main()


class TestTree(unittest.TestCase):
    """The interactive tree of a calibration: pair > method > every setting tried, read from the files."""

    @patch.dict(bc.REPLICATION, REPLICAS_POLICY)
    def test_the_tree_counts_calibrations_and_lists_the_settings_tried(self):
        from hedonic.experiments.overlapping import queuebuild as qb
        from hedonic.experiments.overlapping import queuemodel as qm

        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_for(tmp)
            with patch.object(bc, "load_context", Loader()), redirect_stdout(io.StringIO()):
                br.run_calibration(plan)
            todo = br.searches(plan)
            results = br.read_tuned(plan)["searches"]
            node = qm.run_node(1, "t", "", "done", qb.baselines(plan, todo, results), (len(todo), len(todo), "calibrations"))
            qb.decorate_baselines(node, plan, results)
            methods = [n for pair in node.children for n in pair.children]
            self.assertEqual(len(methods), len(todo))
            self.assertEqual(node.done, node.total)
            self.assertEqual(node.total, sum(m.total for m in methods))  # pair and run count what is under them
            big = next(n for n in methods if n.key[-1] == "BigCLAM")
            # each setting is a leaf with its own bar: done when its replicas are in, and the method counts them
            self.assertTrue(big.children and all(not c.log and (c.done, c.total) == (1, 1) for c in big.children))
            self.assertEqual((big.noun, big.done, big.total), ("settings", len(big.children), len(big.children)))
            self.assertTrue(any("chosen" in c.label for c in big.children))
            self.assertTrue(any("replica 2" in line for c in big.children for line in c.detail))

    def test_a_pending_search_is_priced_from_its_own_ledger(self):
        from hedonic.experiments.overlapping import queuebuild as qb

        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_for(tmp)
            todo = br.searches(plan)
            model = qb.baselines(plan, todo, {})
            costs = {u.coords: model.families[0].cost(u) for u in model.families[0].units}
            self.assertEqual(costs[("DBLP · top5k", "CoDeSEG")], (0.0, "estimate"))  # published: nothing runs
            self.assertIsNone(costs[("DBLP · top5k", "ANGEL")][0])  # nothing measured yet: no invented figure


class TestSingleRunTree(unittest.TestCase):
    def test_a_run_without_a_planner_keeps_its_panel_and_one_with_a_planner_gets_the_tree(self):
        from hedonic.experiments.overlapping import runqueue

        with self.assertRaises(ValueError):
            runqueue.single_tree({"name": "x", "command": "hedonic-exp something"}, {})
        with tempfile.TemporaryDirectory() as tmp:
            command = f"hedonic run calibrate-baselines --networks dblp:top5000 --methods angel,codeseg --output-dir {tmp}"
            runqueue._DATA_CACHE.clear()
            root = runqueue.single_tree({"name": "x", "command": command}, {})
            self.assertEqual(root.label, "run x")
            self.assertGreaterEqual(root.children[0].total, 2)  # the two methods, and the settings a search may try


class TestUnitProgress(unittest.TestCase):
    @patch.dict(bc.REPLICATION, REPLICAS_POLICY)
    def test_a_search_under_way_fills_its_bar_by_settings_tried(self):
        from hedonic.experiments.overlapping import queuebuild as qb
        from hedonic.experiments.overlapping import queuemodel as qm
        from hedonic.experiments.overlapping import queuetui as qt
        from hedonic.experiments.overlapping import tui

        with tempfile.TemporaryDirectory() as tmp:
            plan = br.resolve(["--networks", "dblp:top5000", "--methods", "bigclam", "--output-dir", tmp, "--replicates", "3"])
            ledger = Path(tmp) / "ledger" / "dblp_top5000"
            ledger.mkdir(parents=True)
            rows = [{"method": "bigclam", "params": {"communities": k}, "replica": r, "status": "completed", "score": -2.0,
                     "detect_seconds": 1.0, "score_seconds": 0.1} for k in (10, 20) for r in range(3)]
            rows.append({**rows[0], "params": {"communities": 40}, "replica": 0})  # a third setting, one replica in
            (ledger / "evals-bigclam.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            todo = br.searches(plan)
            node = qm.run_node(1, "t", "", "running", qb.baselines(plan, todo, {}), (0, 1, "calibrations"))
            qb.decorate_baselines(node, plan, {})
            method = node.children[0].children[0]
            # two settings finished, a third with one replica of three in, and what the search may still try (budget 8)
            self.assertEqual((method.done, method.total, method.noun), (2, 8, "settings"))
            self.assertEqual((node.done, node.total), (2, 8))  # the run counts the same settings
            tried = [c for c in method.children if c.key[-2] == "setting"]
            self.assertEqual([(c.done, c.total) for c in tried], [(1, 1), (1, 1), (0, 1)])
            leaf = tried[-1]
            self.assertEqual(leaf.sub[:2], (1, 3))
            G = {**tui.glyphs(False), "open": "v", "closed": ">"}
            self.assertIn("1/3 replicas", qt.right_columns(leaf, 120, G))
