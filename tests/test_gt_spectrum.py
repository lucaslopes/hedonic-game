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


class TestScoring(unittest.TestCase):
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


class TestSpectrumStudy(unittest.TestCase):
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
            # every detector call is exact-GT, local-only and n_iterations=-1
            record = next((root / "runs").glob("*/*.json"))
            condition = json.loads(record.read_text())["condition"]
            self.assertEqual((condition["phase"], condition["n_iterations"], condition["start"]),
                             ("local", -1, "exact_canonical_ground_truth"))
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

    def test_output_safety_and_dry_run_write_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in (Path(d) / "ground_truth_robustness_v9", Path("artifacts/overlapping/ground_truth_robustness_v3/x")):
                with self.assertRaises(ValueError):
                    gs.run_study(_options(str(bad)))
            out = Path(d) / "dry"
            self.assertEqual(gs.run_study(_options(str(out), "--dry-run")), 0)
            self.assertEqual(gs.run_study(_options(str(out), "--discover")), 0)
            self.assertFalse(out.exists())

    def test_unlocked_real_run_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            options = {**_options(os.path.join(d, "o")), "smoke": False, "allow_unlocked": False}
            row = [{"dataset": "amazon", "cover": "top5000", "status": "present", "root_kind": "networks_dir",
                    "bytes": 1, "graph_file": "g", "cover_file": "c", "source_kind": "x"}]
            fake = {"tracked_files_match_lock": False, "runtime_matches_lock": True, "config_matches_lock": True,
                    "tracked_files": {}, "tracked_files_sha256": "x", "runtime": {}, "schema_version": 1,
                    "protocol_name": gs.PROTOCOL_NAME, "protocol_lock_sha256": None}
            with patch.object(gs, "discover", return_value=row), \
                    patch.object(gs, "implementation_identity", return_value=fake), \
                    self.assertRaisesRegex(ValueError, "lock"):
                gs.run_study(options)


class TestSpectrumCli(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
