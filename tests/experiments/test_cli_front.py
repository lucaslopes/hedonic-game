"""The `hedonic-exp` registry and the `hedonic` front door."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


from hedonic.experiments import CLI


class TestCLI(unittest.TestCase):
    def test_help_lists_subcommands(self):
        code = CLI.main(["--help"])
        self.assertEqual(code, 0)

    def test_list_commands(self):
        code = CLI.main(["list"])
        self.assertEqual(code, 0)

    def test_version(self):
        code = CLI.main(["--version"])
        self.assertEqual(code, 0)

    def test_unknown_command(self):
        code = CLI.main(["not-a-command"])
        self.assertEqual(code, 2)

    def test_registry_covers_expected_commands(self):
        expected = {"smoke", "disjoint", "disjoint-load", "disjoint-store", "disjoint-stats", "plots", "reproduce-disjoint"}
        self.assertEqual(set(CLI.COMMANDS), expected)
        removed = {"overlapping-small", "overlapping-controlled", "start-quality", "overlapping-cover-audit",
                   "overlapping-calibrate", "baseline-calibrate", "overlapping-tradeoffs", "participation-budget",
                   "overlapping-codeseg", "codeseg-setup", "codeseg-doctor", "snap-pickle", "overlapping-gt-spectrum",
                   "overlapping-gt-posthoc", "info", "quickstart"}
        self.assertFalse(removed & set(CLI.COMMANDS))
        self.assertEqual(CLI.RUN_EQUIVALENTS, {})
        for name in removed:
            with patch("sys.stderr"):
                self.assertEqual(CLI.main([name, "--help"]), 2)

    def test_every_registered_command_has_working_help(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout

        for name, command in CLI.COMMANDS.items():
            if not command.has_argparse_help:
                continue
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(CLI.main([name, "--help"]), 0, name)

    def test_the_run_aliases_and_top_level_commands(self):
        self.assertEqual(set(CLI.RUN_VERBS), {"list", "status", "attach", "logs", "stop", "resume", "check", "queue"})
        self.assertEqual(CLI.RUN_ALIASES, {"disjoint": "disjoint"})
        self.assertEqual(set(CLI.TOP_LEVEL), {"run", "detect", "show", "guide", "config", "update"})

    def test_the_overlap_verbs_fail_and_name_hedonic_2(self):
        import io
        from contextlib import redirect_stderr

        for argv in (["setup", "--dry-run"], ["pickle", "--all"], *(["run", verb] for verb in CLI.HEDONIC_2_RUN_VERBS)):
            err = io.StringIO()
            with self.subTest(argv=argv), redirect_stderr(err):
                self.assertEqual(CLI.hedonic_main(list(argv)), 2)
            self.assertIn("hedonic 2", err.getvalue())
        for verb in ("exp", "paper", "spectrum", "calibrate", "calibrate-baselines", "startcurve", "controlled",
                     "audit", "posthoc"):
            self.assertIn(verb, CLI.HEDONIC_2_RUN_VERBS)

    def test_hedonic_front_door(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout

        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(CLI.hedonic_main(["--help"]), 0)
            self.assertEqual(CLI.hedonic_main(["run", "--help"]), 0)
        for line in ("hedonic run disjoint", "hedonic run queue", "hedonic show disjoint", "hedonic config"):
            self.assertIn(line, out.getvalue())
        self.assertNotIn("hedonic run exp", out.getvalue())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(CLI.hedonic_main(["run", "disjoint", "--help"]), 0)
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(CLI.hedonic_main(["run", "nope"]), 2)
            self.assertEqual(CLI.hedonic_main(["run"]), 2)  # no favourite: say what can run
            self.assertEqual(CLI.hedonic_main(["nope"]), 2)
        self.assertIn("hedonic run disjoint", err.getvalue())

    def test_hedonic_show(self):
        import io
        from contextlib import redirect_stdout

        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(CLI.hedonic_main(["show", "disjoint", "--json"]), 0)
        self.assertIn("paper", json.loads(out.getvalue()))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(CLI.hedonic_main(["show"]), 0)  # the one topic is the default
        for what in ("methods", "networks", "metrics", "covers", "table", "figure"):
            with self.subTest(what=what), self.assertRaises(SystemExit) as caught:
                CLI.hedonic_main(["show", what])
            self.assertIn("hedonic 2", str(caught.exception))

    def test_hedonic_run_verbs_without_runs(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_RUNS_DIR": d}):
            self.assertEqual(CLI.hedonic_main(["run", "list"]), 0)
            with self.assertRaises(SystemExit):
                CLI.hedonic_main(["run", "status"])

    def test_run_manager_never_trusts_a_recycled_pid_or_idle_tmux_pane(self):
        from hedonic.experiments.overlapping import runmanager as rm

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_RUNS_DIR": d}):
            out = os.path.join(d, "out")
            os.makedirs(out)
            entry = {"name": "r1", "config": {}, "output_dir": out, "created": 0, "session": "",
                     "backend": "process", "log": os.path.join(out, "worker.log"), "pid": os.getpid()}
            # this test process is alive but is not a hedonic worker: the run is interrupted, not "running"
            rm.write_json(rm.progress_path(entry), {"status": "running", "pid": os.getpid(), "plan": []})
            self.assertFalse(rm.is_worker(os.getpid(), "r1"))
            self.assertEqual(rm.effective_status(entry, rm.read_progress(entry)), "interrupted")
            self.assertFalse(rm.tmux_session_exists(""))  # an empty target must never match some other session
            rm.write_json(rm.entry_path("r1"), entry)
            self.assertEqual(rm.cmd_stop("r1"), 0)  # must not signal this process
            self.assertEqual(rm.effective_status(entry, rm.read_progress(entry)), "interrupted")

    def test_run_lookup_is_exact_then_unique_fragment(self):
        from hedonic.experiments.overlapping import runmanager as rm

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_RUNS_DIR": d}):
            for name in ("paper-20260101-000000", "paper-20260202-000000", "exp-1"):
                rm.write_json(rm.entry_path(name), {"name": name, "output_dir": d, "created": 0})
            self.assertEqual(rm.load_entry("exp-1")["name"], "exp-1")
            self.assertEqual(rm.load_entry("20260202")["name"], "paper-20260202-000000")
            with self.assertRaises(SystemExit):
                rm.load_entry("paper")  # ambiguous: refuse rather than stop the wrong run
            with self.assertRaises(SystemExit):
                rm.load_entry("missing")

    def test_worker_error_is_recorded_as_failed_and_resumable(self):
        from hedonic.experiments.disjoint import front
        from hedonic.experiments.overlapping import runmanager as rm

        settings = front.resolve(front.build_parser().parse_args(["--preset", "smoke"]))
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_RUNS_DIR": d}):
            entry = {"name": "bad", "kind": "disjoint", "settings": settings, "output_dir": d, "created": 0, "session": "",
                     "backend": "process", "log": os.path.join(d, "worker.log")}
            rm.write_json(rm.entry_path("bad"), entry)
            with patch.object(front, "work", side_effect=RuntimeError("not-a-method")), \
                    patch("sys.stderr"), patch("sys.stdout"):
                self.assertEqual(rm.worker("bad"), 1)  # fails inside the worker, after it has started
            progress = rm.read_progress(entry)
            self.assertEqual(progress["status"], "failed")
            self.assertIn("not-a-method", progress["reason"])
            self.assertEqual(rm.effective_status(entry, progress), "failed")
            self.assertIn("hedonic run resume bad", rm.snapshot(entry))

    def test_a_registered_run_of_a_removed_kind_is_refused_with_hedonic_2(self):
        from hedonic.experiments.overlapping import runmanager as rm

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_RUNS_DIR": d}):
            entry = {"name": "old", "kind": "spectrum", "output_dir": d, "created": 0, "session": "",
                     "backend": "process", "log": os.path.join(d, "worker.log")}
            rm.write_json(rm.entry_path("old"), entry)
            with patch("sys.stderr"), patch("sys.stdout"):
                self.assertEqual(rm.worker("old"), 1)
            self.assertIn("hedonic 2", rm.read_progress(entry)["reason"])
            self.assertIn("hedonic 2", rm.snapshot(entry))
            with self.assertRaises(SystemExit) as caught:
                rm.cmd_resume("old")
            self.assertIn("hedonic 2", str(caught.exception))

    def test_run_launch_records_absolute_paths_and_resets_progress(self):
        from hedonic.experiments.disjoint import front
        from hedonic.experiments.overlapping import runmanager as rm

        class Proc:
            pid = 999_999_999

        settings = front.resolve(front.build_parser().parse_args(["--preset", "smoke"]))
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_RUNS_DIR": d}):
            cwd = os.getcwd()
            os.chdir(d)
            try:
                stale = Path(d) / "rel" / "progress.json"
                stale.parent.mkdir(parents=True)
                stale.write_text('{"status": "completed"}')
                with patch.object(rm.shutil, "which", return_value=None), \
                        patch.object(rm.subprocess, "Popen", return_value=Proc()):
                    entry = rm.launch_disjoint(settings, "job", "rel")
                    self.assertTrue(os.path.isabs(entry["output_dir"]))  # `hedonic run list` works from any directory
                    self.assertEqual((entry["kind"], entry["backend"]), ("disjoint", "process"))
                    self.assertFalse(stale.exists())  # a resumed run must not look finished
                    with self.assertRaises(ValueError):
                        rm.launch_disjoint(settings, "bad name", "rel2")
            finally:
                os.chdir(cwd)

    def test_run_option_validation_and_friendly_errors(self):
        from hedonic.experiments.overlapping import common, userconfig

        self.assertEqual(common.parse_seeds("0-2,7"), [0, 1, 2, 7])
        for bad in ("3-1", "x", "-2", ""):
            with self.assertRaises(ValueError):
                common.parse_seeds(bad)
        self.assertEqual(common.parse_name("paper-1_a"), "paper-1_a")
        for bad in ("a b", "a.b", "../x", ""):
            with self.assertRaises(ValueError):
                common.parse_name(bad)
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_CONFIG": os.path.join(d, "c.toml")}):
            for argv in (["set", "threads", "abc"], ["set", "threads", "0"], ["set", "disjoint_output_root", "true"],
                         ["set", "profiles.q.seeds", "1"], ["set", "run", '"paper"'], ["set", "output_dir", d]):
                with self.assertRaises(SystemExit):
                    userconfig.main(argv)
            self.assertFalse(os.path.exists(userconfig.path()))  # nothing invalid was written
            self.assertEqual(userconfig.main(["set", "threads", "4"]), 0)
            self.assertEqual(userconfig.main(["set", "defaults.disjoint_output_root", d]), 0)
            self.assertEqual(userconfig.defaults(), {"threads": 4, "disjoint_output_root": d})
            self.assertEqual(userconfig.main(["unset", "threads"]), 0)
            self.assertEqual(userconfig.defaults(), {"disjoint_output_root": d})

    def test_the_environment_a_run_records_names_its_builds(self):
        from hedonic.experiments.overlapping import common

        env = common.environment()
        self.assertEqual(list(env), ["hedonic", "hedonic_core_sha256", "python_igraph", "igraph_c",
                                     "igraph_extension_sha256", "igraph_extension_path", "algorithm_identity",
                                     "python", "platform", "machine", "executable"])
        self.assertIn(env["python_igraph"], common.environment_line())

    def test_tui_fallback_reasks_on_garbage_and_fit_truncates_by_visible_width(self):
        from hedonic.experiments.overlapping import tui

        with patch.object(tui, "interactive", return_value=False), patch("sys.stdout"):
            with patch("builtins.input", side_effect=["x", "9", "2"]):
                self.assertEqual(tui.select("t", ["a", "b"]), 1)
            with patch("builtins.input", side_effect=["0", "1,3"]):
                self.assertEqual(tui.multiselect("t", ["a", "b", "c"]), [0, 2])
            with patch("builtins.input", side_effect=EOFError), self.assertRaises(tui.Cancelled):
                tui.select("t", ["a", "b"])
        colored = "\033[1m" + "x" * 50 + "\033[0m"
        fitted = tui.fit(colored, 20)
        self.assertEqual(tui.visible_len(fitted), 20)
        self.assertTrue(fitted.endswith("\033[0m"))
        self.assertEqual(tui.fit("short", 20), "short")

    def test_disjoint_help(self):
        code = CLI.main(["disjoint", "--help"])
        self.assertEqual(code, 0)

    def test_disjoint_load_help(self):
        code = CLI.main(["disjoint-load", "--help"])
        self.assertEqual(code, 0)

    def test_smoke_help(self):
        code = CLI.main(["smoke", "--help"])
        self.assertEqual(code, 0)

    def test_smoke_isolated_run(self):
        code = CLI.main(["smoke"])
        self.assertEqual(code, 0)

    def test_disjoint_smoke_to_tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            code = CLI.main(
                ["disjoint", "--smoke", "--output_root", d, "--folder_name", "cli"]
            )
            self.assertEqual(code, 0)
            self.assertGreaterEqual(len(list(Path(d).rglob("*.json"))), 1)

    def test_disjoint_v1020_smoke_via_cli(self):
        with tempfile.TemporaryDirectory() as d:
            code = CLI.main(
                ["disjoint", "--preset", "v1020-smoke", "--output_root", d]
            )
            self.assertEqual(code, 0)
            jsons = list(Path(d).rglob("partition_*.json"))
            self.assertGreaterEqual(len(jsons), 1)
            code = CLI.main(
                [
                    "disjoint-load",
                    "--results_folder",
                    str(Path(d) / "resultados"),
                    "--output",
                    str(Path(d) / "out.csv.gzip"),
                    "--simple",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((Path(d) / "out.csv.gzip").is_file())

    def test_plots_help(self):
        code = CLI.main(["plots", "--help"])
        self.assertEqual(code, 0)

    def test_plots_smoke_via_cli(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "figures"
            code = CLI.main(
                [
                    "plots",
                    "--smoke",
                    "--output_dir",
                    str(out),
                    "--format",
                    "png",
                    "--no-persist",
                ]
            )
            self.assertEqual(code, 0)
            from hedonic.experiments.plots.paper_figures import FIGURE_NAMES

            for stem in FIGURE_NAMES:
                matches = list(out.glob(f"{stem}.*"))
                self.assertTrue(matches, f"missing figure stem {stem}")

    def test_reproduce_disjoint_smoke_via_cli(self):
        with tempfile.TemporaryDirectory() as d:
            code = CLI.main(
                [
                    "reproduce-disjoint",
                    "--preset",
                    "v1020-smoke",
                    "--output_root",
                    d,
                    "--format",
                    "png",
                    "--no-persist",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((Path(d) / "resultados.csv.gzip").is_file())
            figs = list((Path(d) / "figures").glob("*.png"))
            self.assertGreaterEqual(len(figs), 4)




class TestForeground(unittest.TestCase):
    """``hedonic run disjoint --foreground`` runs the worker in this terminal, under its --name and --output-dir."""

    def test_a_foreground_run_works_in_this_process(self):
        import io
        from contextlib import redirect_stdout

        from hedonic.experiments.disjoint import front
        from hedonic.experiments.overlapping import runmanager as rm

        seen = {}

        def fake_work(name, entry, prog, progress_file=None):
            seen.update(name=name, kind=entry["kind"], output=entry["output_dir"], progress=progress_file)
            return 0

        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"HEDONIC_RUNS_DIR": str(Path(tmp) / "registry")}), \
                patch.object(front, "work", fake_work), redirect_stdout(io.StringIO()):
            code = CLI.hedonic_main(["run", "disjoint", "--preset", "smoke", "--foreground", "--yes",
                                     "--name", "fg-test", "--output-dir", tmp + "/out"])
            self.assertEqual(code, 0)
            self.assertEqual((seen["name"], seen["kind"]), ("fg-test", "disjoint"))
            self.assertEqual(Path(seen["output"]), (Path(tmp) / "out").resolve())
            self.assertEqual(Path(seen["progress"]), (Path(tmp) / "out").resolve() / "progress.json")
            self.assertEqual(rm.all_entries(), [])  # a foreground run is not recoverable, so it is not registered


class TestTmuxEnvironment(unittest.TestCase):
    """A tmux session starts from the server's environment: the worker must not inherit an earlier shell's HEDONIC_*."""

    def test_stale_server_settings_are_unset_and_the_registry_is_pinned(self):
        import os
        import subprocess
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from hedonic.experiments.overlapping import runmanager as rm

        calls = []

        def fake_run(argv, **kw):
            calls.append(argv)
            if argv[:2] == ["tmux", "show-environment"]:
                return subprocess.CompletedProcess(argv, 0, "HEDONIC_OUTPUT_DIR=/elsewhere\nHEDONIC_RUNS_DIR=/old\nPATH=/x\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"HEDONIC_RUNS_DIR": str(Path(tmp) / "reg")}, clear=False), \
                patch.object(rm.subprocess, "run", fake_run), patch.object(rm, "tmux_session_exists", return_value=False):
            os.environ.pop("HEDONIC_OUTPUT_DIR", None)
            rm._spawn({"name": "t", "backend": "tmux", "session": "hedonic-t", "log": str(Path(tmp) / "w.log")})
            inner = [c for c in calls if c[:2] == ["tmux", "new-session"]][0][-1]
            self.assertIn("-u HEDONIC_OUTPUT_DIR", inner)
            self.assertIn(f"HEDONIC_RUNS_DIR={Path(tmp) / 'reg'}", inner)
            self.assertNotIn("-u HEDONIC_RUNS_DIR", inner)
