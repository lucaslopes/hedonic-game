"""Archive-free coverage for the TOML/tmux overlapping-paper orchestrator."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments import CLI
from hedonic.experiments.overlapping import benchmark, reproduce_paper


def _smoke_config(root: Path) -> Path:
    paper = root / "paper"
    paper.mkdir()
    (paper / "main.tex").write_text("\\documentclass{article}\\begin{document}x\\end{document}\n")
    config = root / "paper.toml"
    config.write_text(
        "\n".join(
            [
                "[paths]",
                f'networks_dir = "{root / "networks"}"',
                "",
                "[overlapping_paper]",
                'profile = "smoke"',
                f'data_root = "{root / "networks"}"',
                f'output_dir = "{paper / "artifacts" / "full"}"',
                f'paper_dir = "{paper}"',
                'tmux_session = "hedonic-paper-test"',
                'methods = ["hedonic_multiphase", "cpm"]',
                'seeds = "0-1"',
                'resolutions = "auto"',
                "max_nodes = 0",
                "timeout_per_run = 30",
                "omega = true",
                "omega_sample_size = 1000",
                "resume = true",
                "retry_attempts = 0",
                "max_parallel_workers = 1",
                "max_parallel_cap = 1",
                "memory_budget_gb = 8",
                "memory_reserve_gb = 0",
                "memory_safety_factor = 1.5",
                "poll_seconds = 0.01",
                "compile_paper = false",
                "",
                "[[overlapping_paper.jobs]]",
                'name = "amazon-all"',
                'dataset = "amazon"',
                'cover = "all"',
                "",
                "[[overlapping_paper.jobs]]",
                'name = "wikipedia-all"',
                'dataset = "wikipedia"',
                'cover = "all"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


class TestOverlappingPaperReproduction(unittest.TestCase):
    def test_external_baseline_policy_is_recorded_in_plan_and_worker_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _smoke_config(root)
            config.write_text(
                config.read_text(encoding="utf-8").replace(
                    'methods = ["hedonic_multiphase", "cpm"]',
                    'methods = ["hedonic_multiphase", "cpm"]\n'
                    'not_rerun_external_methods = ["cpm"]',
                ),
                encoding="utf-8",
            )
            args = reproduce_paper.build_parser().parse_args(["--config", str(config)])
            options, _ = reproduce_paper._load_options(args)
            self.assertEqual(options["not_rerun_external_methods"], ["cpm"])
            plan = reproduce_paper._make_plan(options)
            argv = reproduce_paper._benchmark_argv(
                plan, plan["jobs"][0], root / "shard", execution="fresh"
            )
            self.assertEqual(argv[argv.index("--skip-methods") + 1], "cpm")
            self.assertEqual(
                argv[argv.index("--expected_dataset_metadata_sha256") + 1],
                plan["jobs"][0]["dataset_metadata_identity"]["sha256"],
            )

    def test_memory_plan_uses_per_node_membership_capacity_and_budgeted_waves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _smoke_config(root)
            args = reproduce_paper.build_parser().parse_args(["--config", str(config)])
            options, _ = reproduce_paper._load_options(args)
            self.assertTrue(
                all(job["max_memberships"] == 2 for job in options["jobs"])
            )
            plan = reproduce_paper._make_plan(options)
            for wave in plan["assignments"]:
                used = sum(
                    int(job["memory"]["estimated_peak_bytes"])
                    for job in wave
                    if isinstance(job, dict)
                )
                self.assertLessEqual(used, int(plan["memory_budget_bytes"]))

    def test_memory_scheduler_packs_only_jobs_that_fit_together(self):
        jobs = [
            {"name": "large", "index": 0, "memory": {"estimated_peak_bytes": 6}},
            {"name": "medium", "index": 1, "memory": {"estimated_peak_bytes": 4}},
            {"name": "small", "index": 2, "memory": {"estimated_peak_bytes": 3}},
        ]
        waves = reproduce_paper._schedule_memory_waves(
            jobs, memory_budget_bytes=7, worker_limit=2
        )
        self.assertEqual([[job["name"] for job in wave] for wave in waves], [["small", "medium"], ["large"]])

    def test_memory_scheduler_never_co_schedules_livejournal(self):
        jobs = [
            {"name": "livejournal", "dataset": "livejournal", "index": 0, "memory": {"estimated_peak_bytes": 2}},
            {"name": "youtube", "dataset": "youtube", "index": 1, "memory": {"estimated_peak_bytes": 2}},
            {"name": "amazon", "dataset": "amazon", "index": 2, "memory": {"estimated_peak_bytes": 1}},
        ]
        waves = reproduce_paper._schedule_memory_waves(jobs, memory_budget_bytes=10, worker_limit=3)
        self.assertTrue(all(len(wave) == 1 for wave in waves if any(j["dataset"] == "livejournal" for j in wave)))
        self.assertTrue(all(not (any(j["dataset"] == "livejournal" for j in wave) and len(wave) > 1) for wave in waves))

    def test_single_worker_scheduler_keeps_every_dataset_in_its_own_wave(self):
        jobs = [
            {"name": name, "dataset": name, "index": index,
             "memory": {"estimated_peak_bytes": 1}}
            for index, name in enumerate(("amazon", "dblp", "livejournal", "youtube", "wikipedia"))
        ]
        waves = reproduce_paper._schedule_memory_waves(
            jobs, memory_budget_bytes=64, worker_limit=1
        )
        self.assertEqual(
            [[job["dataset"] for job in wave] for wave in waves],
            [["amazon"], ["dblp"], ["livejournal"], ["youtube"], ["wikipedia"]],
        )

    def test_scheduler_budget_and_process_tree_rss(self):
        jobs = [
            {"name": "root", "index": 0, "memory": {"estimated_peak_bytes": 6}},
            {"name": "child", "index": 1, "memory": {"estimated_peak_bytes": 1}},
        ]
        waves = reproduce_paper._schedule_memory_waves(jobs, memory_budget_bytes=7, worker_limit=2)
        self.assertTrue(all(sum(j["memory"]["estimated_peak_bytes"] for j in wave) <= 7 for wave in waves))
        with patch.object(
            benchmark,
            "_process_tree_snapshot",
            return_value={10: {"ppid": 1, "rss_bytes": 100}, 11: {"ppid": 10, "rss_bytes": 250}},
        ):
            rss, pids = benchmark._process_tree_rss_bytes(10)
        self.assertEqual(rss, 350)
        self.assertEqual(pids, {10, 11})

    def test_exit_minus_nine_is_oom_and_cache_parameters_are_strict(self):
        self.assertEqual(benchmark._classify_exit(-9)[0], "oom")
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory)
            artifact = benchmark._persist_final_cover(artifact_root, [[0, 1]])
            expected = {
                "experiment_identity": benchmark.current_experiment_identity(),
                "dataset": "amazon", "cover": "all", "method": "cpm", "seed": 0,
                "resolution": 0.1, "max_memberships": 2, "timeout_seconds": 10.0,
                "memory_limit_bytes": 100,
                "method_parameters": benchmark.METHODS["cpm"].parameters,
                "method_dependency": benchmark.method_dependency_identity("cpm"),
                "run_options": {"omega": True, "omega_sample_size": 10},
                "_artifact_root": artifact_root,
                "_method_available": False,
            }
            existing = {
                **expected,
                "protocol_version": benchmark.RUN_PROTOCOL_VERSION,
                "status": "timeout",
                "method_metadata": {
                    "parameters": benchmark.METHODS["cpm"].parameters,
                    "dependency": benchmark.method_dependency_identity("cpm"),
                },
            }
            existing.pop("_artifact_root")
            self.assertFalse(benchmark._cache_compatible(existing, expected))
            completed = {
                **existing,
                "status": "completed",
                "final_cover_artifact": artifact["artifact"],
                "final_cover_sha256": artifact["content_sha256"],
                "final_cover_artifact_sha256": artifact["artifact_sha256"],
            }
            # A cover artifact without bound/recomputed metrics is not a
            # resumable completed result.
            self.assertFalse(benchmark._cache_compatible(completed, expected))
            skipped = {**existing, "status": "skipped_unsupported", "failure_kind": "unsupported"}
            self.assertTrue(benchmark._cache_compatible(skipped, expected))
            self.assertFalse(
                benchmark._cache_compatible(
                    {
                        **skipped,
                        "method_metadata": {
                            "parameters": benchmark.METHODS["cpm"].parameters,
                            "dependency": {
                                "distribution": "networkx",
                                "version": "0.0.invalid",
                            },
                        },
                    },
                    expected,
                )
            )
            self.assertFalse(
                benchmark._cache_compatible(
                    skipped, {**expected, "_method_available": True}
                )
            )
            self.assertFalse(benchmark._cache_compatible(existing, {**expected, "timeout_seconds": 11.0}))
            self.assertFalse(benchmark._cache_compatible({**existing, "protocol_version": 2}, expected))
            self.assertFalse(
                benchmark._cache_compatible(
                    {
                        **existing,
                        "status": "failed",
                        "error": "TypeError: _record() got multiple values for keyword argument 'timeout_seconds'",
                    },
                    expected,
                )
            )

    def test_paper_summary_excludes_failed_numeric_values_and_reports_coverage(self):
        records = [
            {"dataset": "amazon", "cover": "all", "method": "hedonic_multiphase", "seed": 0,
             "resolution": 0.1, "status": "completed", "metrics": {"symmetric_best_match_f1": 0.8}},
            {"dataset": "amazon", "cover": "all", "method": "hedonic_multiphase", "seed": 1,
             "resolution": 0.1, "status": "timeout", "runtime_seconds": 12.0},
            {"dataset": "amazon", "cover": "all", "method": "hedonic_multiphase", "seed": 2,
             "resolution": 0.1, "status": "skipped_not_scalable",
             "resource_status": "timeout", "runtime_seconds": 12.0},
        ]
        summary = reproduce_paper._paper_summary(records)
        self.assertEqual(summary[0]["n_completed"], 1)
        self.assertEqual(summary[0]["mean_symmetric_best_match_f1"], 0.8)
        self.assertNotIn("mean_runtime_seconds", summary[0])
        plan = {
            "profile": "full", "jobs": [{"dataset": "amazon", "cover": "all"}],
            "methods": ["hedonic_multiphase"], "seeds": "0-2", "resolutions": "auto",
        }
        coverage = reproduce_paper._coverage_report(
            plan, records, reproduce_paper._audit_records(plan, records)
        )
        self.assertEqual(coverage["completed"], 1)
        self.assertEqual(coverage["timeout"], 2)

    def test_partial_compile_status_defines_a_visible_tex_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            reproduce_paper._write_paper_status(
                output, {"ready_for_paper": False}, partial=True
            )
            status = (output / "paper_status.tex").read_text()
            self.assertIn("EXPLICIT PARTIAL COMPILE", status)
            self.assertIn(r"\partialcompilewarning", status)
    def test_toml_driven_smoke_foreground_merges_and_preserves_smoke_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _smoke_config(root)
            code = CLI.main(
                [
                    "reproduce-overlapping-paper",
                    "--config",
                    str(config),
                    "--no-tmux",
                ]
            )
            self.assertEqual(code, 0)
            output = root / "paper" / "artifacts" / "full"
            manifest = json.loads((output / "paper_manifest.json").read_text())
            self.assertEqual(manifest["methods"], ["hedonic_multiphase", "cpm"])
            self.assertTrue(manifest["experiment_identity"]["tracked_files_match_lock"])
            self.assertEqual(manifest["audit"]["expected_records"], 8)
            self.assertEqual(manifest["audit"]["observed_records"], 8)
            self.assertFalse(manifest["audit"]["ready_for_paper"])
            self.assertTrue((output / "paper_summary.csv").is_file())
            self.assertTrue((output / "condition_summary.csv").is_file())
            self.assertEqual(len(json.loads((output / "condition_summary.json").read_text())["rows"]), 8)
            self.assertTrue((output / "failure_report.json").is_file())
            self.assertTrue((output / "paper_results.tex").is_file())
            self.assertIn("\\smokeresultstrue", (output / "paper_status.tex").read_text())
            self.assertTrue((output / "plots" / "accuracy_by_dataset.pdf").is_file())
            self.assertFalse(any("hedonic_local" in path.parts for path in (output / "shards").rglob("*.json")))
            records = [
                json.loads(path.read_text())
                for path in (output / "shards").rglob("runs/**/*.json")
            ]
            self.assertTrue(records)
            self.assertTrue(all(record["memory_limit_bytes"] > 0 for record in records))
            self.assertTrue(
                all(record["detector_memory"]["limit_bytes"] == record["memory_limit_bytes"] for record in records)
            )

    def test_dry_run_never_launches_workers_or_detectors(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _smoke_config(Path(directory))
            with patch.object(reproduce_paper, "launch_tmux", side_effect=AssertionError("tmux launched")), patch.object(
                reproduce_paper, "run_foreground", side_effect=AssertionError("foreground detector launched")
            ):
                self.assertEqual(
                    CLI.main(["reproduce-overlapping-paper", "--config", str(config), "--dry-run"]),
                    0,
                )

    def test_local_only_variant_is_rejected_by_the_paper_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _smoke_config(root)
            text = config.read_text(encoding="utf-8").replace(
                'methods = ["hedonic_multiphase", "cpm"]',
                'methods = ["hedonic_local", "hedonic_multiphase", "cpm"]',
            )
            config.write_text(text, encoding="utf-8")
            self.assertEqual(
                CLI.main(["reproduce-overlapping-paper", "--config", str(config), "--dry-run"]),
                1,
            )

    def test_tmux_launcher_creates_worker_and_coordinator_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _smoke_config(root)
            args = reproduce_paper.build_parser().parse_args(["--config", str(config)])
            options, _ = reproduce_paper._load_options(args)
            plan = reproduce_paper._write_new_plan(options)

            calls: list[list[str]] = []

            def run(command, **_kwargs):
                calls.append(list(command))
                if command[1] == "has-session":
                    return subprocess.CompletedProcess(command, 1)
                return subprocess.CompletedProcess(command, 0)

            with patch.object(reproduce_paper.shutil, "which", return_value="/usr/bin/tmux"), patch.object(
                reproduce_paper.subprocess, "run", side_effect=run
            ):
                self.assertEqual(reproduce_paper.launch_tmux(plan), 0)

            self.assertTrue(any(call[1] == "new-session" for call in calls))
            self.assertTrue(any(call[1] == "new-window" and "coordinator" in call for call in calls))
            command_text = "\n".join(" ".join(call) for call in calls)
            self.assertIn("--worker-index", command_text)
            self.assertIn("--coordinator", command_text)

    def test_worker_liveness_repair_terminalizes_unstarted_waves(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            plan = {
                "plan_id": "liveness-test",
                "output_dir": str(output),
                "worker_startup_timeout_seconds": 0,
            }
            state = {
                "plan_id": "liveness-test",
                "status": "pending",
                "worker_pid": None,
                "waves": [
                    {"index": 0, "status": "pending"},
                    {"index": 1, "status": "pending"},
                ],
            }
            repaired = reproduce_paper._repair_worker_liveness(
                plan, [state], started=0.0
            )[0]
            self.assertEqual(repaired["status"], "error")
            self.assertEqual(
                [wave["status"] for wave in repaired["waves"]],
                ["error", "error"],
            )
            persisted = json.loads(
                reproduce_paper._state_path(output, 0).read_text()
            )
            self.assertEqual(persisted["status"], "error")

    def test_full_paper_switch_requires_every_condition_to_complete(self):
        plan = {
            "profile": "full",
            "jobs": [{"dataset": "amazon", "cover": "top5000"}],
            "methods": ["hedonic_multiphase", "cpm"],
            "seeds": "0",
            "resolutions": "auto",
        }
        records = [
            {
                "dataset": "amazon",
                "cover": "top5000",
                "method": method,
                "seed": 0,
                "resolution": 0.1,
                "status": "completed",
            }
            for method in plan["methods"]
        ]
        self.assertTrue(reproduce_paper._audit_records(plan, records)["ready_for_paper"])
        records[1]["status"] = "unavailable"
        self.assertFalse(reproduce_paper._audit_records(plan, records)["ready_for_paper"])

    def test_failure_summary_keeps_non_scalable_baseline_explicit(self):
        plan = {
            "profile": "full",
            "jobs": [{"dataset": "livejournal", "cover": "top5000"}],
            "methods": ["cpm", "demon"],
            "seeds": "0",
            "resolutions": "auto",
        }
        records = [
            {"dataset": "livejournal", "cover": "top5000", "method": "cpm", "seed": 0,
             "resolution": 0.1, "status": "timeout", "runtime_seconds": 4.0},
            {"dataset": "livejournal", "cover": "top5000", "method": "demon", "seed": 0,
             "resolution": 0.1, "status": "skipped_unsupported", "reason": "known non-scalable"},
        ]
        audit = reproduce_paper._audit_records(plan, records)
        self.assertEqual(audit["status_counts"]["timeout"], 1)
        self.assertEqual(audit["status_counts"]["skipped_unsupported"], 1)
        self.assertEqual(len(audit["failure_records"]), 2)

    def test_paper_summary_uses_only_digest_bound_nested_metrics(self):
        records = [{
            "dataset": "amazon",
            "cover": "all",
            "method": "cpm",
            "seed": 0,
            "resolution": 0.1,
            "status": "completed",
            "symmetric_best_match_f1": 0.999,
            "metrics": {"symmetric_best_match_f1": 0.25},
        }]
        summary = reproduce_paper._paper_summary(records)
        self.assertEqual(summary[0]["mean_symmetric_best_match_f1"], 0.25)

    def test_finalize_fails_closed_when_supplied_plan_differs_from_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifacts" / "full"
            paper = root / "paper"
            paper.mkdir()
            (paper / "main.tex").write_text(
                "\\documentclass{article}\\begin{document}x\\end{document}\n"
            )
            disk_plan = {
                "plan_id": "bound-plan",
                "experiment_identity": {},
                "config_path": str(root / "missing.toml"),
                "profile": "full",
                "output_dir": str(output),
                "paper_dir": str(paper),
                "methods": ["cpm"],
                "jobs": [{
                    "name": "amazon-all", "dataset": "amazon", "cover": "all"
                }],
                "seeds": "0",
                "resolutions": "auto",
                "compile_paper": False,
                "assignments": [],
            }
            plan_path = reproduce_paper._plan_path(output)
            reproduce_paper._write_json(plan_path, disk_plan)
            supplied = reproduce_paper._load_plan(plan_path)
            supplied["paper_dir"] = str(root / "unbound-paper")
            supplied["compile_paper"] = True
            with patch.object(
                reproduce_paper.shutil,
                "which",
                side_effect=AssertionError("an unbound plan must never compile"),
            ):
                self.assertEqual(reproduce_paper.finalize(supplied), 0)
            manifest = json.loads((output / "paper_manifest.json").read_text())
            reconciliation = manifest["audit"]["protocol_reconciliation"]
            self.assertFalse(manifest["audit"]["ready_for_paper"])
            self.assertIn(
                "finalize_plan_binding:supplied_plan_content_mismatch",
                reconciliation["global_rejection_reasons"],
            )
            self.assertFalse((root / "unbound-paper").exists())

    def test_skeletal_full_records_cannot_enable_or_compile_the_paper_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paper = root / "paper"
            paper.mkdir()
            (paper / "main.tex").write_text(
                "\\documentclass{article}\\begin{document}x\\end{document}\n"
            )
            output = paper / "artifacts" / "full"
            plan = {
                "plan_id": "render-test",
                "experiment_identity": {},
                "config_path": "test.toml",
                "profile": "full",
                "output_dir": str(output),
                "paper_dir": str(paper),
                "methods": ["hedonic_multiphase"],
                "jobs": [{"name": "amazon-top5000", "dataset": "amazon", "cover": "top5000"}],
                "seeds": "0",
                "resolutions": "auto",
                "compile_paper": True,
            }
            run = benchmark._run_path(
                output / "shards" / "amazon-top5000",
                "amazon",
                "top5000",
                "hedonic_multiphase",
                0,
                0.1,
            )
            reproduce_paper._write_json(
                run,
                {
                    "dataset": "amazon",
                    "cover": "top5000",
                    "method": "hedonic_multiphase",
                    "seed": 0,
                    "resolution": 0.1,
                    "status": "completed",
                    "metrics": {
                        "symmetric_best_match_f1": 0.5,
                        "matching_f1": 0.4,
                        "node_membership_micro_f1": 0.3,
                        "size_weighted_f1": 0.2,
                        "omega": 0.1,
                        "runtime_seconds": 1.0,
                        "predicted_overlapping_node_fraction": 0.25,
                        "coverage_rate": 0.5,
                        "inclusion_rate": 0.4,
                    },
                },
            )
            with patch.object(
                reproduce_paper.shutil,
                "which",
                side_effect=AssertionError("latexmk must not be queried"),
            ):
                self.assertEqual(reproduce_paper.finalize(plan), 0)
            self.assertIn("\\smokeresultstrue", (output / "paper_status.tex").read_text())
            manifest = json.loads((output / "paper_manifest.json").read_text())
            self.assertFalse(manifest["audit"]["ready_for_paper"])
            self.assertEqual(
                manifest["audit"]["protocol_reconciliation"]["admissible_records"],
                0,
            )
            self.assertFalse((output.parent / "build" / "main.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
