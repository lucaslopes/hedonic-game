"""SNAP ground-truth robustness spectrum: discovery, audit, runner, resume, front door."""

from __future__ import annotations

import gzip
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import igraph as ig

from hedonic.experiments import CLI
from hedonic.experiments.overlapping import gt_spectrum as gs
from hedonic.experiments.overlapping import robustness as rb
from hedonic.experiments.overlapping import runmanager


def _options(out: str, *extra: str) -> dict:
    args = gs.build_parser().parse_args(["--profile", "smoke", "--output-dir", out, *extra])
    return gs.resolve_options(args)


class _FixedBudget(unittest.TestCase):
    """A fixed memory budget, so a tiny study is never refused because other work fills the machine meanwhile."""

    def setUp(self):
        budget = patch.dict(os.environ, {"HEDONIC_MEMORY_BUDGET_GB": "16"})
        budget.start()
        self.addCleanup(budget.stop)


def _touch(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class TestSpectrumAudit(unittest.TestCase):
    def test_sweep_matches_the_v3_endpoint_audit_on_random_tiny_graphs(self):
        rng = random.Random(3)
        for _ in range(12):
            n = rng.randint(6, 9)
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5]
            graph = ig.Graph(n=n, edges=edges)
            labels = [sorted(rng.sample(range(3), rng.randint(1, 2))) for _ in range(n)]
            used = sorted({label for row in labels for label in row})
            remap = {label: i for i, label in enumerate(used)}
            memberships = [[remap[label] for label in row] for row in labels]
            for isolation in (False, True):
                for gamma in (0.0, 0.05, 0.3, 1.0):
                    swept = gs.spectrum_audit(graph, memberships, cap=2, allow_isolation=isolation,
                                              gammas=[gamma], atol=1e-10, rtol=1e-9)[0]
                    reference = rb.audit_cover(graph, memberships, max_memberships=2, allow_isolation=isolation,
                                               gamma=gamma, atol=1e-10, rtol=1e-9)
                    self.assertAlmostEqual(swept["stable_fraction"], reference["stable_fraction_at_resolution"])
                    self.assertAlmostEqual(swept["max_positive_regret"], reference["max_positive_regret_at_resolution"])
                    self.assertEqual(swept["is_nash_equilibrium"], reference["is_local_equilibrium_at_resolution"])

    def test_nash_means_every_vertex_is_stable(self):
        graph = ig.Graph.Full(4)
        row = gs.spectrum_audit(graph, [[0]] * 4, cap=2, allow_isolation=False, gammas=[0.0], atol=1e-10,
                                rtol=1e-9)[0]
        self.assertEqual(row["stable_fraction"], 1.0)
        self.assertTrue(row["is_nash_equilibrium"])
        self.assertEqual(row["profitable_vertex_count"], 0)

    def test_grid_syntax(self):
        grid = gs._parse_grid("0+geom:1e-4:1:9")
        self.assertEqual(len(grid), 10)
        self.assertEqual((grid[0], grid[-1]), (0.0, 1.0))
        self.assertAlmostEqual(grid[3], 1e-3, places=9)
        self.assertEqual(gs._parse_grid("0:1:3+0.5"), [0.0, 0.5, 1.0])
        for bad in ("2", "geom:0:1:4", "-0.1"):
            with self.assertRaises(ValueError):
                gs._parse_grid(bad)


class TestDiscovery(unittest.TestCase):
    def test_present_missing_and_unsupported_pairs_without_touching_the_network(self):
        with tempfile.TemporaryDirectory() as d, patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            root = Path(d)
            _touch(root / "Amazon" / "com-amazon.ungraph.txt.gz")
            _touch(root / "Amazon" / "com-amazon.top5000.cmty.txt.gz")      # only the top5000 cover exists
            _touch(root / "Wikipedia" / "wiki-topcats.txt.gz")               # graph without its cover
            rows = {(r["dataset"], r["cover"]): r for r in gs.discover(root)}
            self.assertEqual(rows["amazon", "top5000"]["status"], "present")
            self.assertEqual(rows["amazon", "top5000"]["source_kind"], "streamed_raw_gzip")
            self.assertEqual(rows["amazon", "all"]["status"], "missing")
            self.assertIn("cover", rows["amazon", "all"]["reason"])
            self.assertEqual(rows["wikipedia", "top5000"]["status"], "unsupported")   # a variant that cannot exist
            self.assertEqual(rows["wikipedia", "all"]["status"], "missing")
            self.assertEqual(rows["orkut", "all"]["status"], "missing")
            self.assertIn("dataset directory", rows["orkut", "all"]["reason"])
            self.assertIn("present", gs.render_cohort(gs.select_cohort(list(rows.values()), "auto", "auto")))

    def test_absence_is_a_gap_only_when_the_user_named_it(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(Path(d) / "Amazon" / "com-amazon.ungraph.txt.gz")
            _touch(Path(d) / "Amazon" / "com-amazon.top5000.cmty.txt.gz")
            auto = gs.select_cohort(gs.discover(d), "auto", "auto")
            self.assertEqual([r["dataset"] + "/" + r["cover"] for r in auto if r["eligible"]], ["amazon/top5000"])
            self.assertFalse(any(r["explicitly_requested"] for r in auto))
            named = gs.select_cohort(gs.discover(d), ["amazon", "dblp"], "auto")
            self.assertTrue(any(r["explicitly_requested"] and r["dataset"] == "dblp" for r in named))
            pairs = gs.select_cohort(gs.discover(d), "auto", "auto", pairs=[("amazon", "top5000"), ("orkut", "all")])
            self.assertEqual([r["dataset"] for r in pairs if r["explicitly_requested"]], ["orkut"])

    def test_loading_never_enables_the_catalogue_without_provisioning(self):
        captured = {}

        def fake_loader(name, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop")

        row = {"dataset": "amazon", "cover": "top5000", "status": "present", "root_kind": "networks_dir"}
        with tempfile.TemporaryDirectory() as d:
            for provision in (False, True):
                options = {**_options(d), "smoke": False, "provision": provision}
                with patch.object(gs.snap, "load_snap_dataset", fake_loader), self.assertRaises(RuntimeError):
                    gs.load_job(options, row)
                self.assertEqual(captured["allow_catalog"], provision)


class TestScoring(_FixedBudget):
    def test_cached_base_plus_omega_equals_evaluate_cover(self):
        from hedonic.experiments.overlapping.metrics import evaluate_cover

        with tempfile.TemporaryDirectory() as d:
            options = _options(os.path.join(d, "o"))
            identity = gs.implementation_identity(options)
            job = gs.load_job(options, gs._smoke_rows(options)[0])
            rng = random.Random(1)
            n = job.graph.vcount()
            cover = rb.canonicalize_cover([rng.sample(range(n), rng.randint(2, 5)) for _ in range(6)], n)
            digest = rb.cover_hash(cover, n)
            for seed in (0, 3):
                fast = gs._with_omega(gs._base_metrics(job, options, identity, cover, digest), job, cover, options, seed)
                slow = evaluate_cover(cover, job.gt_cover, n, compute_omega=True, omega_sample_size=options["omega_sample_size"],
                                      omega_seed=seed)
                self.assertEqual(json.dumps(fast, sort_keys=True, default=str), json.dumps(slow, sort_keys=True, default=str))
            # the start-vs-final transition equals final-vs-GT: the start *is* the ground truth
            transition = evaluate_cover(cover, job.gt_cover, n, compute_omega=False)
            self.assertEqual(json.dumps(gs._with_omega(gs._base_metrics(job, {**options, "omega": False}, identity, cover, digest),
                                                       job, cover, {**options, "omega": False}, 0), sort_keys=True, default=str),
                             json.dumps(transition, sort_keys=True, default=str))
            self.assertTrue(list((Path(options["output_dir"]) / "metric_cache").glob("*.json")))

    def test_concurrent_writers_of_one_content_addressed_artifact_do_not_collide(self):
        import concurrent.futures

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            memberships = [[0], [0, 1], [1], [2]]
            with concurrent.futures.ThreadPoolExecutor(8) as pool:
                digests = list(pool.map(lambda _: gs._persist_memberships(root, memberships, 4), range(64)))
                covers = list(pool.map(lambda _: gs._persist_cover(root, [[0, 1], [1, 2]]), range(64)))
            self.assertEqual(len(set(digests)), 1)
            self.assertEqual(len(set(covers)), 1)
            self.assertFalse(list(root.rglob("*.tmp")))

    def test_worker_pool_gives_the_same_records_as_in_process(self):
        with tempfile.TemporaryDirectory() as d:
            serial, pooled = os.path.join(d, "a"), os.path.join(d, "b")
            extra = ("--datasets", "amazon", "--seeds", "0-1", "--resolutions", "0,0.5")
            gs.run_study(_options(serial, *extra, "--workers", "1"))
            gs.run_study(_options(pooled, *extra, "--workers", "2"))

            def table(root):
                out = {}
                for line in (Path(root) / "results.jsonl").read_text().splitlines():
                    row = json.loads(line)
                    for volatile in ("detector_runtime_seconds", "detector_peak_rss_bytes"):
                        row.pop(volatile, None)
                    out[(row["row_type"], row.get("action_policy"), row.get("gamma"), row.get("seed"))] = row
                return out

            self.assertEqual(table(serial), table(pooled))
            a = json.loads((Path(serial) / "coverage_report.json").read_text())
            b = json.loads((Path(pooled) / "coverage_report.json").read_text())
            self.assertEqual(a["status_counts"], b["status_counts"])
            self.assertEqual(sorted(csv_rows(serial)), sorted(csv_rows(pooled)))


def csv_rows(root: str) -> list[tuple]:
    import csv

    with (Path(root) / "spectrum_audit.csv").open() as stream:
        return [tuple(sorted(row.items())) for row in csv.DictReader(stream)]


class TestSpectrumStudy(_FixedBudget):
    def test_pairs_run_in_the_order_they_are_listed_not_in_the_catalogues(self):
        # the smoke catalogue lists amazon before dblp; the pairs name dblp first
        with tempfile.TemporaryDirectory() as d:
            events = []
            options = _options(os.path.join(d, "out"), "--pairs", "dblp/top5000,amazon/top5000")
            self.assertEqual(gs.run_study(options, events.append), 0)
            plan = next(event for event in events if event.get("type") == "plan")
            self.assertEqual(plan["jobs"], ["dblp/top5000", "amazon/top5000"])

    def test_smoke_study_writes_a_complete_verified_ledger_and_resumes_without_detection(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out")
            self.assertEqual(gs.run_study(_options(out)), 0)
            root = Path(out)
            for name in ("manifest.json", "plan.json", "cohort.json", "coverage_report.json", "results.jsonl",
                         "results.csv.gz", "spectrum_audit.csv", "condition_summary.csv", "coverage_report.csv"):
                self.assertTrue((root / name).is_file(), name)
            coverage = json.loads((root / "coverage_report.json").read_text())
            self.assertTrue(coverage["complete"])
            self.assertEqual(coverage["expected_detector_conditions"], 2 * 2 * 3 * 2)
            self.assertEqual(coverage["recorded_detector_conditions"], 24)
            rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
            reference = [r for r in rows if r["row_type"] == "ground_truth_reference"]
            runs = [r for r in rows if r["row_type"] == "detector_run"]
            self.assertEqual(len(reference), 2)
            self.assertTrue(all(r["detector_runtime_seconds"] is None and not r["runtime_applicable"] for r in reference))
            for run in runs:
                self.assertIn(run["status"], ("completed", "completed_non_equilibrium"))
                self.assertEqual(run["audit_final_is_nash_equilibrium"], run["status"] == "completed")
                self.assertIn("delta_f1", run)
                self.assertTrue((root / "covers" / f"{run['initial_cover_hash']}.json.gz").is_file())
                self.assertTrue((root / "raw_memberships" / f"{run['final_membership_hash']}.json.gz").is_file())
            # every detector call is exact-GT and local moving (to an equilibrium: no iteration budget)
            record = next((root / "runs").glob("*/*.json"))
            condition = json.loads(record.read_text())["condition"]
            self.assertEqual((condition["phase"], condition["start"]), ("local", "exact_canonical_ground_truth"))
            self.assertNotIn("n_iterations", condition)
            manifest = json.loads((root / "plan.json").read_text())
            self.assertIs(manifest["locked"], False)
            self.assertEqual(manifest["lock"], gs.LOCK_NOTE)
            self.assertEqual(manifest["identity"]["tracked_files_sha256"], gs.tracked_digest())
            provenance = json.loads((root / "jobs" / "amazon-top5000.json").read_text())
            self.assertEqual(provenance["synthetic_memberships_added"], 0)
            self.assertEqual(provenance["completion_policy"], "covered-induced")
            # resume: compatible stored records are reused; detection must not run
            with patch.object(gs, "run_in_subprocess", side_effect=AssertionError("detector invoked")):
                self.assertEqual(gs.run_study(_options(out)), 0)
                self.assertEqual(gs.run_study(_options(out, "--rescore-only")), 0)
            # a changed tolerance changes the condition identity: the cache is not reused
            with patch.object(gs, "run_in_subprocess", side_effect=AssertionError("detector invoked")), \
                    self.assertRaises(AssertionError):
                gs.run_study(_options(out, "--robustness-atol", "1e-8"))

    def test_resource_outcomes_are_preserved_and_excluded_from_metric_means(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out")
            timeout = gs.ProcessOutcome("timeout", None, 1.0, "worker exceeded timeout") \
                if hasattr(gs, "ProcessOutcome") else None
            from hedonic.experiments.overlapping.execution import ProcessOutcome

            timeout = ProcessOutcome("timeout", None, 1.0, "worker exceeded timeout")
            with patch.object(gs, "run_in_subprocess", return_value=timeout):
                self.assertEqual(gs.run_study(_options(out, "--datasets", "amazon", "--seeds", "0")), 0)
            coverage = json.loads((Path(out) / "coverage_report.json").read_text())
            self.assertEqual(coverage["timeouts"], 6)          # 2 policies x 3 resolutions, none imputed
            self.assertFalse(coverage["verified_equilibria"])
            summary = (Path(out) / "condition_summary.csv").read_text().splitlines()
            self.assertIn("timeouts", summary[0])
            self.assertNotIn("delta_f1_mean", summary[0])      # no returned cover, no metric mean
            # preserved unless the user asks to retry
            with patch.object(gs, "run_in_subprocess", side_effect=AssertionError("retried")):
                self.assertEqual(gs.run_study(_options(out, "--datasets", "amazon", "--seeds", "0")), 0)

    def test_requested_but_absent_pair_makes_the_run_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out")
            options = _options(out, "--pairs", "amazon/top5000,dblp/top5000")
            options["pairs"] = [("amazon", "top5000"), ("dblp", "top5000")]
            rows = gs._smoke_rows(options)
            rows[1]["status"] = "missing"
            rows[1]["reason"] = "gone"
            with patch.object(gs, "_smoke_rows", return_value=rows):
                self.assertEqual(gs.run_study(options), 0)
            coverage = json.loads((Path(out) / "coverage_report.json").read_text())
            self.assertFalse(coverage["complete"])
            self.assertEqual(coverage["requested_but_unavailable_pairs"], ["dblp/top5000 (missing)"])

    def test_a_pair_that_does_not_fit_is_refused_before_anything_is_built(self):
        from hedonic.experiments.overlapping import capacity

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out")
            with patch.object(capacity, "budget_bytes", return_value=1), \
                    patch.object(rb, "build_fractional_state", side_effect=AssertionError("built despite no capacity")):
                self.assertEqual(gs.run_study(_options(out, "--datasets", "amazon", "--seeds", "0")), 0)
            coverage = json.loads((Path(out) / "coverage_report.json").read_text())
            self.assertFalse(coverage["complete"])
            failure = coverage["load_failures"][0]
            self.assertEqual(failure["status"], "capacity_refused")
            self.assertIn("can be used now", failure["error"])

    def test_a_tight_budget_shrinks_the_pool_and_the_records_do_not_change(self):
        from hedonic.experiments.overlapping import capacity

        plans = []
        real = capacity.plan_workers

        def spy(need, requested, budget=None):
            plan = real(need, requested, budget)
            plans.append(plan)
            return plan

        with tempfile.TemporaryDirectory() as d:
            wide, tight = os.path.join(d, "wide"), os.path.join(d, "tight")
            extra = ("--datasets", "amazon", "--seeds", "0-1")
            with patch.object(capacity, "plan_workers", side_effect=spy):
                gs.run_study(_options(wide, *extra, "--workers", "2"))
            self.assertFalse(any(plan.reduced for plan in plans))
            floor = max(plan.need.total(1) for plan in plans)
            plans.clear()
            with patch.object(capacity, "plan_workers", side_effect=spy), \
                    patch.object(capacity, "budget_bytes", return_value=int(floor * 1.01)):
                gs.run_study(_options(tight, *extra, "--workers", "2"))
            self.assertTrue(any(plan.reduced for plan in plans), [plan.line() for plan in plans])
            rows = lambda root: [(r["action_policy"], r["gamma"], r.get("seed"), r.get("final_f1")) for r in
                                 map(json.loads, (Path(root) / "results.jsonl").read_text().splitlines())
                                 if r.get("row_type") == "detector_run"]
            self.assertEqual(sorted(map(str, rows(wide))), sorted(map(str, rows(tight))))

    def test_output_safety_and_dry_run_write_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            nets = Path(d) / "nets"
            nets.mkdir()
            with self.assertRaisesRegex(ValueError, "raw network data"):  # never write inside the SNAP data
                gs.run_study(_options(str(nets / "out"), "--data-root", str(nets)))
            self.assertFalse((nets / "out").exists())
            out = Path(d) / "dry"
            self.assertEqual(gs.run_study(_options(str(out), "--dry-run")), 0)
            self.assertEqual(gs.run_study(_options(str(out), "--discover")), 0)
            self.assertFalse(out.exists())

    def test_runs_are_unlocked_and_the_identity_is_package_relative_and_path_free(self):
        identity = gs.implementation_identity({"config_sha256": None})
        self.assertIs(identity["locked"], False)
        self.assertTrue(identity["tracked_files"])
        for key in identity["tracked_files"]:
            self.assertFalse(Path(key).is_absolute(), key)
            self.assertTrue(key.startswith("hedonic/"), key)
        for module in ("gt_spectrum", "ground_truth_robustness", "robustness", "snap"):
            self.assertIn(f"hedonic/experiments/overlapping/{module}.py", identity["tracked_files"])
        self.assertIn("hedonic/Game.py", identity["tracked_files"])
        runtime = identity["runtime"]
        self.assertEqual(set(runtime), {"hedonic", "hedonic_core_source_sha256", "algorithm_identity"})
        self.assertNotIn("/", json.dumps(runtime).replace("community_hedonic/", ""))
        # the digest is of the package source plus the version and the core build, nothing else
        self.assertEqual(identity["tracked_files_sha256"], gs.tracked_digest())
        self.assertEqual(gs.canonical_digest(identity["tracked_files_sha256"]), identity["tracked_files_sha256"])
        with patch.object(gs, "_runtime_identity", return_value={**runtime, "hedonic_core_sha256": "0" * 64}):
            self.assertNotEqual(gs.tracked_digest(), identity["tracked_files_sha256"])
        compact = gs._compact_identity(identity)
        self.assertNotIn("runtime_versions", compact)

    def test_lock_flags_are_gone_or_harmless(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            gs.build_parser().parse_args(["--write-lock"])
        args = gs.build_parser().parse_args(["--profile", "smoke", "--allow-unlocked"])  # accepted, no effect
        self.assertNotIn("allow_unlocked", gs.resolve_options(args))

    def test_provisioning_defaults_on_for_named_pairs_and_off_for_discovery_runs(self):
        named = gs.resolve_options(gs.build_parser().parse_args(["--pairs", "dblp/top5000"]))
        self.assertTrue(named["provision"])
        self.assertFalse(gs.resolve_options(gs.build_parser().parse_args(["--pairs", "dblp/top5000",
                                                                           "--no-provision"]))["provision"])
        self.assertFalse(gs.resolve_options(gs.build_parser().parse_args([]))["provision"])
        self.assertFalse(_options("/tmp/unused-smoke")["provision"])


class TestSpectrumCli(_FixedBudget):
    def test_registered_and_help(self):
        self.assertIn("overlapping-gt-spectrum", CLI.COMMANDS)
        self.assertEqual(CLI.main(["overlapping-gt-spectrum", "--help"]), 0)
        self.assertEqual(CLI.hedonic_main(["run", "spectrum", "--help"]), 0)

    def test_front_door_foreground_and_strip(self):
        self.assertEqual(gs._strip_front(["--name", "x", "--detach", "--seeds", "0", "-y"]), ["--seeds", "0"])
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_CONFIG": os.path.join(d, "c.toml")}):
            out = os.path.join(d, "o")
            self.assertEqual(CLI.hedonic_main(["run", "spectrum", "--profile", "smoke", "--foreground", "--output-dir", out,
                                               "--seeds", "0", "--resolutions", "0,1"]), 0)
            self.assertTrue((Path(out) / "coverage_report.json").is_file())

    def test_show_covers(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(CLI.hedonic_main(["show", "covers", "--network-root", d, "--cache-dir", d, "--json"]), 0)

    def test_durable_spectrum_run_snapshot_and_resume_entry(self):
        class Proc:
            pid = 999_999_999

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_RUNS_DIR": os.path.join(d, "runs")}):
            with patch.object(runmanager.shutil, "which", return_value=None), \
                    patch.object(runmanager.subprocess, "Popen", return_value=Proc()):
                entry = runmanager.launch_spectrum(["--profile", "smoke"], "sp", os.path.join(d, "out"))
            self.assertEqual(entry["kind"], "spectrum")
            self.assertTrue(entry["argv"][-2:] == ["--output-dir", entry["output_dir"]] or "--resume" in entry["argv"])
            self.assertIn("--resume", entry["argv"])
            # the worker publishes its plan and events; the snapshot shows per-pair denominators
            prog = {"kind": "spectrum", "status": "running", "pid": os.getpid(), "started": 0, "jobs": ["amazon/top5000"],
                    "describe": "x", "plan": [
                        {"job": "amazon-top5000", "dataset": "amazon", "cover": "top5000", "policy": "fixed_labels",
                         "gamma": 0.0, "seed": 0, "status": "completed", "seconds": 1.0},
                        {"job": "amazon-top5000", "dataset": "amazon", "cover": "top5000", "policy": "fixed_labels",
                         "gamma": 0.1, "seed": 0, "status": "pending"}]}
            text = runmanager.snapshot_spectrum(entry, prog, "●", True)
            self.assertIn("amazon/top5000", text)
            self.assertIn("1/2", text)



class TestNamedConfig(unittest.TestCase):
    def test_full_config_resolves_by_name_to_a_named_complete_graph_run(self):
        from hedonic.experiments import CLI
        from hedonic.experiments.overlapping import gt_spectrum

        self.assertEqual(gt_spectrum.named_config("full").name, "overlapping-gt-spectrum-full.toml")
        self.assertTrue(gt_spectrum.named_config("full").is_file())
        self.assertEqual(gt_spectrum.named_config("x/y.toml"), Path("x/y.toml"))
        with self.assertRaisesRegex(ValueError, "no spectrum config"):
            gt_spectrum.resolve_options(gt_spectrum.build_parser().parse_args(["--config", "no-such-config"]))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"HEDONIC_CONFIG": str(Path(tmp) / "c.toml")}):
            (Path(tmp) / "c.toml").write_text(f'[defaults]\noutput_dir = "{tmp}"\n')
            options = gt_spectrum.resolve_options(gt_spectrum.build_parser().parse_args(["--config", "full"]))
            self.assertEqual(options["output_dir"], Path(tmp) / "gt-robustness")
        self.assertEqual(options["run_name"], "gt-robustness")
        self.assertIs(options["locked"], False)
        self.assertTrue(options["provision"])  # the queue's step names its pairs; a missing one is downloaded
        self.assertIsNone(options["max_nodes"])
        self.assertEqual(options["pairs"], [(net, "top5000") for net in ("dblp", "amazon", "youtube", "friendster", "livejournal")]
                         + [(net, "all") for net in ("dblp", "amazon", "youtube")])
        self.assertIsNone(options["tuned_resolutions"])  # the queue passes --tuned; the config alone reads none
        self.assertEqual(options["workers"], 8)
        self.assertIn("spectrum-full", CLI.RUN_ALIASES)
        default = gt_spectrum.resolve_options(gt_spectrum.build_parser().parse_args([]))
        self.assertTrue(str(default["output_dir"]).endswith("gt_spectrum_v1"))
        self.assertIs(default["locked"], False)


class TestTunedSettings(_FixedBudget):
    """Explicit detector settings: GT start at the tuned benchmark (gamma, cap), cap-projected when needed."""

    def test_cap_projection_keeps_the_strongest_attachments_deterministically(self):
        # vertex 0 is in communities a={0,1,2}, b={0,3}, c={0,4,5}; it has 2 neighbours in a, 1 in b, 2 in c
        graph = ig.Graph(6, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (4, 5)])
        cover = rb.canonicalize_cover([[0, 1, 2], [0, 3], [0, 4, 5]], 6)
        memberships = rb.cover_to_vertex_memberships(cover, 6, require_covered=True)
        projected, info = gs.project_memberships(graph, memberships, 2)
        again, _ = gs.project_memberships(graph, memberships, 2)
        self.assertEqual(projected, again)
        self.assertTrue(all(1 <= len(labels) <= 2 for labels in projected))
        kept = sorted(tuple(c) for c in rb.vertex_memberships_to_cover(projected) if 0 in c)
        self.assertEqual(kept, [(0, 1, 2), (0, 4, 5)])          # {0,3} dropped: fewest neighbours
        self.assertEqual((info["truncated_vertices"], info["memberships_removed"]), (1, 1))
        self.assertAlmostEqual(info["truncated_vertex_fraction"], 1 / 6)
        self.assertEqual(info["reference_max_memberships"], 3)
        unchanged, info = gs.project_memberships(graph, memberships, 3)
        self.assertEqual((unchanged, info["truncated_vertices"]), (memberships, 0))

    def test_tuned_config_mirrors_the_benchmark_settings(self):
        from hedonic.experiments.overlapping import quickstart

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"HEDONIC_CONFIG": str(Path(tmp) / "c.toml")}):
            (Path(tmp) / "c.toml").write_text(f'[defaults]\noutput_dir = "{tmp}"\n')
            settings = ",\n".join(
                f'  {{ pair = "{net}/all", label = "{method}", phase = "local", gamma = {float(p["absolute_resolution"])!r}, '
                f'cap = {int(p["max_memberships"])} }}'
                for method, nets in quickstart.HEDONIC_TUNED.items() for net, p in nets.items())
            config = Path(tmp) / "tuned.toml"
            config.write_text('[overlapping_gt_spectrum]\nname = "gt-tuned"\nworkers = 8\n'
                              'pairs = ["dblp/all", "amazon/all"]\nmax_nodes = 0\n'
                              'isolation_policies = ["open_labels"]\nseeds = "0-9"\n'
                              f'detector_settings = [\n{settings},\n]\n')
            options = gs.resolve_options(gs.build_parser().parse_args(["--config", str(config)]))
            self.assertEqual(options["output_dir"], Path(tmp) / "gt-tuned")
        self.assertEqual((options["run_name"], options["workers"], options["max_nodes"]), ("gt-tuned", 8, None))
        self.assertIs(options["locked"], False)
        self.assertEqual(options["isolation_policies"], ["open_labels"])
        self.assertEqual(options["detector_seeds"], list(range(10)))
        phases = {"hedonic": "local"}
        expected = {(f"{net}/all", method, phases[method], float(p["absolute_resolution"]), int(p["max_memberships"]))
                    for method, nets in quickstart.HEDONIC_TUNED.items() for net, p in nets.items()}
        got = {(s["pair"], s["label"], s["phase"], s["gamma"], s["cap"]) for s in options["detector_settings"]}
        self.assertEqual(got, expected)
        grid = gs.scientific_grid(options, [("dblp", "all"), ("amazon", "all")])
        self.assertEqual(len(grid["detector_settings"]), 2)
        self.assertFalse(hasattr(gs, "write_lock"))              # hedonic 2.0 has no protocol lock to rewrite
        with self.assertRaises(ValueError):
            gs.parse_detector_settings([{"pair": "dblp/all", "phase": "global", "gamma": 0.1, "cap": 2}])
        with self.assertRaisesRegex(ValueError, "not available in hedonic 2.0"):  # the 1.x multilevel phase
            gs.parse_detector_settings([{"pair": "dblp/all", "phase": "multilevel", "gamma": 0.1, "cap": 2}])

    def test_smoke_study_with_explicit_settings_projects_scores_and_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            config = Path(d) / "tuned.toml"
            config.write_text(
                '[overlapping_gt_spectrum]\npairs = ["amazon/top5000"]\nisolation_policies = ["open_labels"]\n'
                'detector_settings = [\n'
                '  { pair = "amazon/top5000", label = "loc", phase = "local", gamma = 0.25, cap = 1 },\n'
                '  { pair = "amazon/top5000", label = "wide", phase = "local", gamma = 0.25, cap = 2 },\n]\n')
            out = os.path.join(d, "out")
            self.assertEqual(gs.run_study(_options(out, "--config", str(config), "--seeds", "0-1")), 0)
            root = Path(out)
            coverage = json.loads((root / "coverage_report.json").read_text())
            self.assertTrue(coverage["complete"])
            self.assertEqual(coverage["expected_detector_conditions"], 2 * 2)
            runs = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
            runs = [r for r in runs if r["row_type"] == "detector_run"]
            by_setting = {r["setting"]: r for r in runs}
            self.assertEqual(set(by_setting), {"loc", "wide"})
            self.assertEqual((by_setting["loc"]["phase"], by_setting["loc"]["max_memberships"]), ("local", 1))
            self.assertEqual(by_setting["wide"]["start"], "exact_canonical_ground_truth")    # cap 2 = multiplicity
            loc = by_setting["loc"]
            self.assertEqual(loc["start"], "cap_projected_canonical_ground_truth")
            self.assertGreater(loc["start_truncated_vertex_fraction"], 0)
            for key in ("vs_start_f1", "vs_start_onmi", "final_f1", "final_onmi", "audit_final_max_positive_regret",
                        "cover_distance_initial_to_final", "changed_vertex_fraction", "detector_peak_rss_bytes"):
                self.assertIn(key, loc)
            conditions = [json.loads(p.read_text())["condition"] for p in (root / "runs").glob("*/*.json")]
            self.assertEqual({(c["setting"], c["phase"], c["max_memberships"]) for c in conditions},
                             {("loc", "local", 1), ("wide", "local", 2)})
            audit = (root / "start_setting_audit.csv").read_text().splitlines()
            self.assertEqual(len(audit), 1 + 2)
            summary = (root / "condition_summary.csv").read_text().splitlines()
            self.assertEqual(len(summary), 1 + 2)
            with patch.object(gs, "run_in_subprocess", side_effect=AssertionError("detector invoked")):
                self.assertEqual(gs.run_study(_options(out, "--config", str(config), "--seeds", "0-1")), 0)



def _calibration(path: Path, settings: dict, complete: bool = True) -> Path:
    path.write_text(json.dumps({"complete": complete, "settings": {"hedonic": settings}}))
    return path


class TestCalibratedResolutions(_FixedBudget):
    """``--tuned hedonic_tuned.json``: each pair's calibrated resolution joins that pair's detector and audit grids."""

    def test_a_pair_gets_its_own_calibrated_resolution_and_nothing_else_changes(self):
        with tempfile.TemporaryDirectory() as d:
            tuned = _calibration(Path(d) / "hedonic_tuned.json", {
                "amazon:top5000": {"max_memberships": 3, "absolute_resolution": 0.35},
                "dblp": {"max_memberships": 9, "absolute_resolution": 0.2}})  # a plain name is the `all` cover
            options = _options(os.path.join(d, "out"), "--tuned", str(tuned))
            self.assertEqual(options["tuned_resolutions"], {"amazon/top5000": 0.35, "dblp/all": 0.2})
            self.assertEqual(gs.detector_resolutions(options, "amazon", "top5000"), [0.0, 0.35, 0.5, 1.0])
            self.assertEqual(gs.detector_resolutions(options, "dblp", "all"), [0.0, 0.2, 0.5, 1.0])
            self.assertIn(0.35, gs.audit_resolutions(options, "amazon", "top5000"))
            self.assertNotIn(0.35, gs.audit_resolutions(options, "dblp", "all"))
            self.assertEqual([s["gamma"] for s in gs.job_settings(options, "amazon", "top5000")], [0.0, 0.35, 0.5, 1.0])
            with self.assertRaisesRegex(ValueError, "no Hedonic setting for dblp:top5k"):
                gs.detector_resolutions(options, "dblp", "top5000")
            grid = gs.scientific_grid(options, [("amazon", "top5000")])
            self.assertEqual(grid["tuned_resolutions"], {"amazon/top5000": 0.35})
            # without a file the grid, and so the hash of every earlier study, is unchanged
            plain = _options(os.path.join(d, "out"))
            self.assertNotIn("tuned_resolutions", gs.scientific_grid(plain, [("amazon", "top5000")]))
            self.assertEqual(gs.detector_resolutions(plain, "amazon", "top5000"), [0.0, 0.5, 1.0])

    def test_a_missing_or_unfinished_calibration_is_tuned_missing(self):
        from hedonic.experiments.overlapping import quickstart as qs

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(qs.TunedMissing):
                _options(os.path.join(d, "out"), "--tuned", str(Path(d) / "absent.json"))
            half = _calibration(Path(d) / "half.json", {}, complete=False)
            with self.assertRaises(qs.TunedMissing):
                _options(os.path.join(d, "out"), "--tuned", str(half))

    def test_smoke_study_runs_the_calibrated_resolution_audits_it_and_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            tuned = _calibration(Path(d) / "hedonic_tuned.json",
                                 {"amazon:top5000": {"max_memberships": 3, "absolute_resolution": 0.35}})
            out = os.path.join(d, "out")
            args = ("--pairs", "amazon/top5000", "--tuned", str(tuned))
            self.assertEqual(gs.run_study(_options(out, *args)), 0)
            root = Path(out)
            coverage = json.loads((root / "coverage_report.json").read_text())
            self.assertTrue(coverage["complete"])
            self.assertEqual(coverage["expected_detector_conditions"], 2 * 4 * 2)  # policies x (3 + 1) resolutions x seeds
            rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
            self.assertEqual(sum(r["row_type"] == "detector_run" and r["gamma"] == 0.35 for r in rows), 2 * 2)
            audit = json.loads((root / "audit" / "amazon-top5000.json").read_text())
            self.assertIn(0.35, audit["identity"]["grid"])
            self.assertEqual(sum(r["gamma"] == 0.35 for r in audit["rows"]), 2)  # one row per action policy
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["effective_grid"]["tuned_resolutions"], {"amazon/top5000": 0.35})
            with patch.object(gs, "run_in_subprocess", side_effect=AssertionError("detector invoked")):
                self.assertEqual(gs.run_study(_options(out, *args)), 0)
            self.assertEqual(gs.main(["--replot", "--output-dir", out, "--profile", "smoke", *args]), 0)
            summary = (root / "condition_summary.csv").read_text().splitlines()
            self.assertEqual(len(summary), 1 + 2 * 4)

    def test_the_queue_waits_for_the_calibration_but_still_knows_the_pairs(self):
        from hedonic.experiments.overlapping import runqueue

        with tempfile.TemporaryDirectory() as d:
            run = ["spectrum", "--config", "full", "--tuned", str(Path(d) / "absent" / "hedonic_tuned.json"),
                   "--name", "gt-robustness", "--output-dir", str(Path(d) / "gt")]
            self.assertEqual(runqueue.data_state(run, fresh=True)[0], "awaiting")
            options = runqueue._spectrum_options("spectrum", run[1:])
            self.assertIn(("friendster", "top5000"), options["pairs"])
            self.assertNotIn(("livejournal", "all"), options["pairs"])


if __name__ == "__main__":
    unittest.main()
