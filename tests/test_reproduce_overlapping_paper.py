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
                "memory_gb_per_worker = 1",
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
            self.assertEqual(manifest["audit"]["expected_records"], 8)
            self.assertEqual(manifest["audit"]["observed_records"], 8)
            self.assertFalse(manifest["audit"]["ready_for_paper"])
            self.assertTrue((output / "paper_summary.csv").is_file())
            self.assertTrue((output / "paper_results.tex").is_file())
            self.assertIn("\\smokeresultstrue", (output / "paper_status.tex").read_text())
            self.assertTrue((output / "plots" / "accuracy_by_dataset.pdf").is_file())
            self.assertFalse(any("hedonic_local" in path.parts for path in (output / "shards").rglob("*.json")))

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

    @unittest.skipUnless(shutil.which("latexmk"), "latexmk is required for manuscript render test")
    def test_complete_full_records_generate_and_compile_the_paper_branch(self):
        source_paper = Path(__file__).resolve().parents[1] / "docs" / "papers" / "overlapping_communities"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paper = root / "paper"
            paper.mkdir()
            shutil.copy2(source_paper / "main.tex", paper / "main.tex")
            shutil.copy2(source_paper / "reference.bib", paper / "reference.bib")
            output = paper / "artifacts" / "full"
            plan = {
                "plan_id": "render-test",
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
            self.assertEqual(reproduce_paper.finalize(plan), 0)
            self.assertIn("\\smokeresultsfalse", (output / "paper_status.tex").read_text())
            self.assertTrue((paper / "main.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
