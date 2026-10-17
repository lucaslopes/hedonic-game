"""Focused tests for the LFR-derived controlled-overlap experiment."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import hedonic
from hedonic.experiments import CLI
from hedonic.experiments.overlapping import controlled_overlap
from hedonic.experiments.resources import config_path


class TestControlledOverlap(unittest.TestCase):
    def assert_implementation_identity_current(self, identity):
        self.assertEqual(identity["algorithm"], "sha256")
        self.assertEqual(identity["path_scope"], "package-relative")
        package_parent = Path(hedonic.__file__).resolve().parent.parent
        self.assertEqual(
            set(identity["source_files"]),
            {controlled_overlap._module_source(module)[0]
             for module in controlled_overlap.IMPLEMENTATION_MODULES},
        )
        for relative, recorded_digest in identity["source_files"].items():
            # keys are relative to the installed package, never an absolute path
            self.assertFalse(Path(relative).is_absolute(), relative)
            self.assertTrue(relative.startswith("hedonic/"), relative)
            self.assertEqual(
                recorded_digest,
                hashlib.sha256((package_parent / relative).read_bytes()).hexdigest(),
                relative,
            )
        descriptor = {
            key: value
            for key, value in identity.items()
            if key != "identity_sha256"
        }
        self.assertEqual(
            identity["identity_sha256"],
            controlled_overlap._sha256_json(descriptor),
        )
        self.assertEqual(
            identity["environment"], controlled_overlap._environment_identity()
        )

    def _instance(self):
        return controlled_overlap.generate_controlled_cover(
            n=80,
            mu=0.2,
            overlap_fraction=0.2,
            memberships_per_overlapping_vertex=2,
            secondary_edge_probability=0.2,
            seed=0,
            average_degree=5,
            min_community=15,
            max_community=30,
        )

    def test_construction_is_deterministic_and_realizes_overlap(self):
        first = self._instance()
        second = self._instance()
        self.assertEqual(first.graph.get_edgelist(), second.graph.get_edgelist())
        self.assertEqual(first.cover, second.cover)
        self.assertEqual(len(first.overlapping_vertices), 16)
        memberships = [
            sum(vertex in community for community in first.cover)
            for vertex in range(first.graph.vcount())
        ]
        self.assertTrue(all(memberships[v] == 2 for v in first.overlapping_vertices))
        self.assertTrue(
            all(
                memberships[v] == 1
                for v in range(first.graph.vcount())
                if v not in first.overlapping_vertices
            )
        )
        self.assertEqual(
            first.metadata["construction_label"],
            "LFR-derived controlled overlap",
        )
        self.assertIn("not canonical overlapping LFR", first.metadata["construction_caveat"])

    def test_one_real_detector_run_records_ablation_and_metrics(self):
        instance = self._instance()
        rows = controlled_overlap.run_ablations(
            instance,
            phases=["local"],
            cap_specs=["gt"],
            starts=["gt-primary"],
            resolution_multipliers=[1.0],
            seed=0,
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["max_memberships"], 2)
        self.assertEqual(row["initialization_supervision"], "ground-truth primary labels")
        self.assertNotIn("local_move_only", row)  # hedonic 2.0 runs local moving only
        self.assertNotIn("ensure_equilibrium", row)
        self.assertEqual(row["allow_isolation"], True)
        self.assertIn("matching_f1", row)
        self.assertEqual(row["omega"], None)
        summary = controlled_overlap._summary(rows)
        self.assertEqual(summary[0]["sd_matching_f1"], 0.0)
        self.assertEqual(
            summary[0]["ci95_low_matching_f1"],
            summary[0]["mean_matching_f1"],
        )

    def test_known_native_offender_reaches_equilibrium_after_native_fix(self):
        instance = controlled_overlap.generate_controlled_cover(
            n=80,
            mu=0.2,
            overlap_fraction=0.1,
            memberships_per_overlapping_vertex=2,
            secondary_edge_probability=0.1,
            seed=0,
            average_degree=5,
            min_community=15,
            max_community=30,
            lfr_max_iters=500,
        )
        rows = controlled_overlap.run_ablations(
            instance,
            phases=["local"],
            cap_specs=["2"],
            starts=["singleton"],
            resolution_multipliers=[1.0],
            seed=0,
            timeout_seconds=0.1,
        )
        # The historical non-terminating cell completes within the watchdog.
        self.assertEqual(rows[0]["status"], "completed")
        summary = controlled_overlap._summary(rows)
        self.assertEqual(summary[0]["n_expected"], 1)
        self.assertEqual(summary[0]["n_completed"], 1)
        self.assertEqual(summary[0]["n_timeout"], 0)
        self.assertIsNotNone(summary[0]["mean_matching_f1"])

    def test_detector_seed_is_paired_across_ablation_cells(self):
        # Overlapping multilevel ("multiphase") is not part of hedonic 2.0: a clear error, never a fallback.
        with self.assertRaisesRegex(ValueError, "not available in hedonic 2.0"):
            controlled_overlap.run_ablations(
                self._instance(), phases=["local", "multiphase"], cap_specs=["1"],
                starts=["singleton"], resolution_multipliers=[1.0], seed=7,
            )
        rows = controlled_overlap.run_ablations(
            self._instance(),
            phases=["local"],
            cap_specs=["1", "gt"],
            starts=["singleton", "gt-primary"],
            resolution_multipliers=[1.0, 10.0],
            seed=7,
        )
        self.assertEqual({row["seed"] for row in rows}, {7})
        self.assertEqual({row["allow_isolation"] for row in rows}, {True})

    def test_cli_registration_help_and_csv_output(self):
        command = CLI.COMMANDS["overlapping-controlled"]
        self.assertEqual(
            command.module,
            "hedonic.experiments.overlapping.controlled_overlap_config",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            code = CLI.main(["overlapping-controlled", "--help"])
        self.assertEqual(code, 0)

        # Restrict the grid to one run; this tests the shipped JSON/CSV path
        # without turning a unit test into the 16-run smoke experiment.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            status = controlled_overlap.main(
                [
                    "--n",
                    "80",
                    "--mus",
                    "0.2",
                    "--overlap-fractions",
                    "0.2",
                    "--overlap-memberships",
                    "2",
                    "--secondary-edge-probabilities",
                    "0.2",
                    "--graph-seeds",
                    "0",
                    "--average-degree",
                    "5",
                    "--min-community",
                    "15",
                    "--max-community",
                    "30",
                    "--phases",
                    "local",
                    "--max-memberships",
                    "gt",
                    "--starts",
                    "gt-primary",
                    "--resolution-multipliers",
                    "1",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 0)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".csv").is_file())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assert_implementation_identity_current(
                payload["implementation_identity"]
            )
            self.assertEqual(
                payload["environment"],
                payload["implementation_identity"]["environment"],
            )
            self.assertEqual(
                payload["implementation_identity"],
                payload["protocol"]["implementation_identity"],
            )


class TestControlledOverlapConfig(unittest.TestCase):
    def test_config_file_supplies_options_and_flags_override_it(self):
        from hedonic.experiments.overlapping import controlled_overlap_config as wrapper

        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "grid.toml"
            output = Path(tmp) / "out" / "controlled.json"
            config.write_text(
                "[overlapping_controlled]\n"
                "n = 40\nmus = \"0.2,0.4\"\noverlap-fractions = \"0.2\"\ngraph-seeds = \"0\"\n"
                "phases = \"local\"\nmax-memberships = \"2\"\nstarts = \"neutral-disjoint\"\n"
                "resolution-multipliers = \"1\"\ntimeout-per-run = 60.0\n"
                f"output = \"{output}\"\n"
            )
            arguments = wrapper.config_arguments(str(config))
            self.assertEqual(arguments[arguments.index("--mus") + 1], "0.2,0.4")
            self.assertEqual(arguments[arguments.index("--timeout-per-run") + 1], "60.0")
            with contextlib.redirect_stdout(io.StringIO()):
                code = wrapper.main(["--config", str(config), "--mus", "0.2", "--timeout-per-run", "30"])
            self.assertEqual(code, 0)
            construction = json.loads(output.read_text())["protocol"]["construction"]
            self.assertEqual(construction["n"], 40)
            self.assertEqual(construction["mus"], [0.2])  # the flag won over the file


GRID_TABLE = (
    "[overlapping_controlled]\n"
    "n = 40\n"
    'mus = "0.2,0.4"\n'
    'overlap-fractions = "0.1,0.2"\n'
    'overlap-memberships = "2"\n'
    'secondary-edge-probabilities = "0.2"\n'
    'graph-seeds = "0,1"\n'
    'phases = "local"\n'
    'max-memberships = "1,gt"\n'
    'starts = "singleton,gt-primary"\n'
    'resolution-multipliers = "1"\n'
    "timeout-per-run = 30.0\n"
    'output = "{output}"\n'
)


class TestControlledRunFrontDoor(unittest.TestCase):
    """``hedonic run controlled``: plan, run folder, progress, registry and resume, with no detector or worker started."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        env = patch.dict(os.environ, {"HEDONIC_RUNS_DIR": str(self.root / "runs"),
                                      "HEDONIC_CONFIG": str(self.root / "no-config.toml")})
        env.start()
        self.addCleanup(env.stop)

    def write_config(self, name="grid.toml"):
        path = self.root / name
        path.write_text(GRID_TABLE.format(output=self.root / "out" / "controlled.json"))
        return path

    def worker_entry(self, config, folder):
        return {"name": "grid", "kind": "controlled", "command": "hedonic run controlled",
                "argv": ["--config", str(config), "--name", "grid", "--output-dir", str(folder)],
                "output_dir": str(folder), "created": time.time(), "session": "", "backend": "process",
                "log": str(folder / "worker.log")}

    def test_help_alias_and_run_listing(self):
        self.assertEqual(CLI.RUN_ALIASES["controlled"], "overlapping-controlled")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(CLI.hedonic_main(["run", "controlled", "--help"]), 0)
            self.assertEqual(CLI.hedonic_main(["run", "--help"]), 0)
        self.assertIn("--dry-run", out.getvalue())
        self.assertIn("hedonic run controlled --config", out.getvalue())

    def test_missing_config_and_bad_name_are_usage_errors(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(CLI.hedonic_main(["run", "controlled"]), 2)
            missing = ["run", "controlled", "--config", str(self.root / "missing.toml"), "--dry-run"]
            self.assertEqual(CLI.hedonic_main(missing), 2)
            bad_name = ["run", "controlled", "--config", str(self.write_config()), "--name", "bad name", "--dry-run"]
            self.assertEqual(CLI.hedonic_main(bad_name), 2)

    def test_dry_run_prints_the_plan_and_writes_nothing(self):
        folder = self.root / "run"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = CLI.hedonic_main(["run", "controlled", "--config", str(self.write_config()),
                                     "--output-dir", str(folder), "--dry-run"])
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("8 instances × 4 detector run(s) = 32 runs", text)
        self.assertIn("timeout  30 s per run", text)
        self.assertIn("dry run: nothing started, nothing written", text)
        self.assertFalse(folder.exists())
        self.assertFalse((self.root / "runs").exists())

    def test_plan_places_the_json_where_the_config_says(self):
        from hedonic.experiments.overlapping import controlled_overlap_config as wrapper

        config = self.write_config()
        run = wrapper.resolve(["--config", str(config)])
        self.assertEqual(run["name"], "grid")
        self.assertEqual(run["folder"], str(self.root / "out"))
        self.assertEqual(run["json"], str(self.root / "out" / "controlled.json"))
        run = wrapper.resolve(["--config", str(config), "--name", "alt",
                               "--output-dir", str(self.root / "elsewhere")])
        self.assertEqual(run["name"], "alt")
        self.assertEqual(run["json"], str(self.root / "elsewhere" / "controlled.json"))

    def test_tracked_heatmap_config_is_planned_as_640_runs(self):
        from hedonic.experiments.overlapping import controlled_overlap_config as wrapper

        heatmap = config_path("overlap-heatmap")
        self.assertTrue(heatmap.is_file())
        plan = wrapper.resolve(["--config", "overlap-heatmap"])["plan"]
        self.assertEqual((plan["instances"], plan["runs_per_instance"], plan["runs"]), (640, 1, 640))

    def test_study_plan_counts_the_study_loops(self):
        from hedonic.experiments.overlapping import controlled_overlap_config as wrapper

        plan = wrapper.study_plan([
            "--graph-seeds", "0,1", "--mus", "0.2,0.4", "--overlap-fractions", "0.1",
            "--overlap-memberships", "2", "--secondary-edge-probabilities", "0.1,0.3",
            "--phases", "local", "--max-memberships", "1,1,gt",
            "--starts", "singleton", "--resolution-multipliers", "1,10",
        ])
        # 2 seeds x 2 mus x 1 fraction x 1 membership count x 2 probabilities = 8 instances; per instance
        # 1 phase x 2 distinct cap labels (the repeated 1 counts once) x 1 start x 2 multipliers = 4 runs
        self.assertEqual((plan["instances"], plan["runs_per_instance"], plan["runs"]), (8, 4, 32))
        with self.assertRaisesRegex(ValueError, "not available in hedonic 2.0"):
            wrapper.study_plan(["--phases", "local,multiphase"])

    def test_progress_line_is_the_line_the_study_prints(self):
        from hedonic.experiments.overlapping import controlled_overlap_config as wrapper

        source = Path(controlled_overlap.__file__).read_text(encoding="utf-8")
        self.assertIn('f"completed instance {instance_index}/{total_instances}: "', source)
        found = wrapper.PROGRESS_LINE.match(
            "completed instance 3/8: seed=0 mu=0.2 overlap=0.1 memberships=2 secondary_p=0.3")
        self.assertEqual((found[1], found[2]), ("3", "8"))

    def test_progress_tap_forwards_everything_and_reports_whole_lines(self):
        from hedonic.experiments.overlapping.controlled_overlap_config import ProgressTap

        real, seen = io.StringIO(), []
        tap = ProgressTap(real, seen.append)
        tap.write("completed inst")
        tap.write("ance 1/2: seed=0\nrun phase=local")
        tap.write("\n")
        print("completed instance 2/2: seed=1", file=tap, flush=True)
        self.assertEqual(seen, ["completed instance 1/2: seed=0", "run phase=local",
                                "completed instance 2/2: seed=1"])
        self.assertEqual(real.getvalue(),
                         "completed instance 1/2: seed=0\nrun phase=local\ncompleted instance 2/2: seed=1\n")

    def test_worker_mirrors_the_study_progress_and_records_completion(self):
        from hedonic.experiments.overlapping import controlled_overlap_config as wrapper
        from hedonic.experiments.overlapping import runmanager as rm

        config = self.write_config()
        folder = self.root / "run"
        folder.mkdir()
        entry = self.worker_entry(config, folder)
        seen_arguments = []

        def fake_study(arguments):  # stands in for the study: prints its progress lines and writes its JSON
            seen_arguments.append(arguments)
            for index in range(1, 9):
                print(f"completed instance {index}/8: seed=0 mu=0.2 overlap=0.1 memberships=2 secondary_p=0.3",
                      flush=True)
            Path(arguments[arguments.index("--output") + 1]).write_text("{}")
            return 0

        with patch.object(wrapper, "main", fake_study), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(wrapper.work("grid", entry, {}), 0)
        saved = json.loads(rm.progress_path(entry).read_text())
        self.assertEqual(saved["status"], "completed")
        self.assertEqual((saved["done"], saved["total"]), (8, 8))
        self.assertEqual(saved["output"], str(folder / "controlled.json"))
        self.assertTrue((folder / "controlled.json").is_file())
        self.assertEqual(seen_arguments[0][:2], ["--config", str(config)])

    def test_worker_fails_when_the_study_returns_an_error_code(self):
        from hedonic.experiments.overlapping import controlled_overlap_config as wrapper

        folder = self.root / "run"
        folder.mkdir()
        entry = self.worker_entry(self.write_config(), folder)
        with patch.object(wrapper, "main", lambda arguments: 3), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "returned 3"):
                wrapper.work("grid", entry, {})

    def test_launch_registers_the_run_kind_without_starting_a_worker(self):
        from hedonic.experiments.overlapping import runmanager as rm

        folder = self.root / "run"
        argv = ["--config", str(self.write_config()), "--name", "grid", "--output-dir", str(folder)]
        with patch.object(rm, "_spawn", side_effect=lambda entry: entry) as spawn:
            entry = rm.launch_controlled(argv, "grid", str(folder), backend="process")
        spawn.assert_called_once()
        self.assertEqual((entry["kind"], entry["argv"], entry["output_dir"], entry["backend"]),
                         ("controlled", argv, str(folder), "process"))
        self.assertTrue(folder.is_dir())

    def test_resume_relaunches_a_stopped_controlled_run_from_its_argv(self):
        from hedonic.experiments.overlapping import runmanager as rm

        folder = self.root / "run"
        folder.mkdir()
        argv = ["--config", str(self.write_config()), "--name", "grid", "--output-dir", str(folder)]
        entry = self.worker_entry(self.write_config(), folder)
        entry["argv"] = argv
        rm.write_json(rm.entry_path("grid"), entry)
        rm.write_json(folder / "progress.json", {"status": "stopped"})
        calls = []
        with patch.object(rm, "launch_controlled",
                          side_effect=lambda *args, **kwargs: calls.append((args, kwargs)) or entry):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(rm.cmd_resume("grid", attach=False), 0)
        self.assertEqual(calls, [((argv, "grid", str(folder)), {"backend": "process"})])

    def test_resume_leaves_a_completed_controlled_run_alone(self):
        from hedonic.experiments.overlapping import runmanager as rm

        folder = self.root / "run"
        folder.mkdir()
        entry = self.worker_entry(self.write_config(), folder)
        rm.write_json(rm.entry_path("grid"), entry)
        rm.write_json(folder / "progress.json", {"status": "completed"})
        with patch.object(rm, "launch_controlled") as launch, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(rm.cmd_resume("grid", attach=False), 0)
        launch.assert_not_called()
        self.assertIn("is completed", out.getvalue())

    def test_snapshot_shows_instances_and_the_resume_hint(self):
        from hedonic.experiments.overlapping import controlled_overlap_config as wrapper

        entry = {"name": "grid", "kind": "controlled", "output_dir": str(self.root), "created": time.time(),
                 "session": "", "backend": "process"}
        now = time.time()
        finished = wrapper.snapshot(entry, {"status": "completed", "done": 8, "total": 8, "runs_per_instance": 4,
                                            "started": now - 60, "finished": now, "describe": "grid",
                                            "output": "/x/controlled.json"}, "●")
        self.assertIn("8/8 instances (32/32 detector runs)", finished)
        self.assertIn("/x/controlled.json", finished)
        stopped = wrapper.snapshot(entry, {"status": "stopped", "done": 3, "total": 8, "runs_per_instance": 4,
                                           "started": now - 60, "finished": now, "describe": "grid"}, "●")
        self.assertIn("3/8 instances", stopped)
        self.assertIn("hedonic run resume grid", stopped)

    def test_existing_output_refuses_a_start_unless_overwrite_is_given(self):
        from hedonic.experiments.overlapping import runmanager as rm

        output = self.root / "out" / "controlled.json"
        output.parent.mkdir(parents=True)
        output.write_text("{}")
        config = self.write_config()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(CLI.hedonic_main(["run", "controlled", "--config", str(config), "--dry-run"]), 0)
        self.assertIn("a start needs --overwrite", out.getvalue())
        with patch.object(rm, "_spawn", side_effect=lambda entry: entry) as spawn, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(CLI.hedonic_main(["run", "controlled", "--config", str(config), "--detach"]), 2)
            spawn.assert_not_called()
            self.assertEqual(CLI.hedonic_main(["run", "controlled", "--config", str(config), "--detach",
                                               "--overwrite"]), 0)
        spawn.assert_called_once()
        self.assertEqual(spawn.call_args[0][0]["kind"], "controlled")

    def test_runmanager_routes_controlled_runs_to_their_worker_and_snapshot(self):
        from hedonic.experiments.overlapping import controlled_overlap_config as wrapper
        from hedonic.experiments.overlapping import runmanager as rm

        folder = self.root / "run"
        folder.mkdir()
        entry = self.worker_entry(self.write_config(), folder)
        now = time.time()
        rm.write_json(rm.progress_path(entry), {"status": "completed", "done": 8, "total": 8, "runs_per_instance": 4,
                                                "started": now - 5, "finished": now, "describe": "grid",
                                                "output": str(folder / "controlled.json")})
        self.assertIn("8/8 instances", rm.snapshot(entry))
        with patch.object(wrapper, "main", lambda arguments: 0), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rm._work("grid", entry, {}), 0)
        self.assertEqual(json.loads(rm.progress_path(entry).read_text())["kind"], "controlled")

    def test_progress_counts_are_instances_for_controlled_runs(self):
        from hedonic.experiments.overlapping import runmanager as rm

        self.assertEqual(rm.progress_counts({"kind": "controlled"}, {"done": 3, "total": 8}), (3, 8))


if __name__ == "__main__":
    unittest.main()
