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
        expected = {
            "smoke", "disjoint", "disjoint-load", "disjoint-store", "disjoint-stats", "plots", "reproduce-disjoint",
            "overlapping-small", "overlapping-controlled", "start-quality", "overlapping-cover-audit",
            "overlapping-calibrate", "baseline-calibrate", "overlapping-tradeoffs", "participation-budget",
            "overlapping-codeseg", "codeseg-setup", "codeseg-doctor", "snap-pickle", "overlapping-gt-spectrum",
            "overlapping-gt-posthoc", "info", "quickstart",
        }
        self.assertEqual(set(CLI.COMMANDS), expected)
        removed = {"overlapping-full", "overlapping-subgraph", "overlapping-scale", "overlapping-resolution",
                   "reproduce-overlapping-paper", "overlapping-gt-robustness", "overlapping-dnn", "overlapping-audit"}
        self.assertFalse(removed & set(CLI.COMMANDS))
        for name in removed:
            with patch("sys.stderr"):
                self.assertNotEqual(CLI.main([name, "--help"]), 0)

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
        self.assertNotIn("youtube-hoc", CLI.RUN_ALIASES)
        self.assertNotIn("spectrum-tuned", CLI.RUN_ALIASES)
        for verb in ("exp", "paper", "spectrum", "spectrum-full", "calibrate", "calibrate-baselines", "startcurve",
                     "controlled", "audit", "posthoc", "disjoint"):
            self.assertIn(verb, CLI.RUN_ALIASES)
        self.assertEqual(set(CLI.TOP_LEVEL), {"run", "setup", "show", "guide", "config", "pickle", "update"})

    def test_hedonic_setup_dry_run_is_wired(self):
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_CODESEG_CACHE": d + "/cache"}):
            out = io.StringIO()
            with redirect_stdout(out):
                code = CLI.hedonic_main(["setup", "--dry-run", "--json", "--network-root", d + "/nets"])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(out.getvalue().strip().splitlines()[-1])["dry_run"])
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_hedonic_front_door(self):
        self.assertEqual(CLI.hedonic_main(["--help"]), 0)
        self.assertEqual(CLI.hedonic_main(["run", "exp", "--help"]), 0)
        self.assertEqual(CLI.hedonic_main(["run", "nope"]), 2)
        self.assertEqual(CLI.main(["quickstart", "--help"]), 0)

    def test_quickstart_parsing_and_catalogue(self):
        from hedonic.experiments.overlapping import quickstart as qs

        self.assertEqual(qs.parse_seeds("0-2,7"), [0, 1, 2, 7])
        self.assertEqual(len(qs.NETWORKS), 7)
        self.assertEqual(len(qs.METHODS), 9)
        self.assertEqual(qs.parse_choice("all", list(qs.BY_KEY), qs.ALIASES, "method"), list(qs.BY_KEY))
        self.assertEqual(qs.parse_choice("hoc,lazyfox", list(qs.BY_KEY), qs.ALIASES, "method"), ["hedonic", "fox"])
        with self.assertRaises(ValueError):
            qs.parse_choice("nope", list(qs.NETWORKS), {}, "network")
        cfg = qs.config_from_args(qs.build_parser().parse_args(["--networks", "dblp", "--nodes", "1000", "--threads", "4"]))
        # community counts tuned on full DBLP are scaled to the subgraph; threads reach threaded methods
        self.assertLess(qs.method_params(qs.BY_KEY["bigclam"], cfg, "dblp")["communities"], 25000)
        self.assertEqual(qs.method_params(qs.BY_KEY["codeseg"], cfg, "dblp")["threads"], 4)
        self.assertEqual(qs.method_params(qs.BY_KEY["hedonic"], cfg, "dblp")["max_memberships"], 64)
        self.assertIn("--nodes 1000", cfg.command())

    def test_quickstart_table_lists_every_accuracy_metric(self):
        from hedonic.experiments.overlapping import quickstart as qs

        cfg = qs.Config(methods=["hedonic", "codeseg"], seeds=[0, 1])
        metrics = {key: 0.5 for key, _ in qs.COLUMNS}
        records = [{"network": "dblp", "method_key": "hedonic", "seed": s, "status": "completed",
                    "metrics": metrics, "detection_seconds": 1.0} for s in (0, 1)]
        records.append({"network": "dblp", "method_key": "codeseg", "status": "unavailable",
                        "reason": "no C++ compiler found"})
        table = qs.render(qs.summarise(records, cfg), cfg)
        for _, label in qs.COLUMNS:
            self.assertIn(label, table)
        self.assertIn("no C++ compiler found", table)
        self.assertIn("mean ± sd over 2 seeds", table)

    def test_paper_config_plan_order(self):
        from hedonic.experiments.overlapping import quickstart as qs

        cfg = qs.load_config("paper")
        plan = qs.build_plan(cfg, {})
        self.assertEqual(len(plan), 2 * 9 * 10)
        # seed -> method -> network: seed 0, method 1 on DBLP then Amazon, then method 2 ...
        self.assertEqual([(n, s, m.key) for n, s, m in plan[:3]],
                         [("dblp", 0, "hedonic"), ("amazon", 0, "hedonic"), ("dblp", 0, "codeseg")])
        self.assertEqual(plan[20][1], 1)
        self.assertEqual(cfg.nodes, 0)
        self.assertEqual(cfg.command(), "hedonic run exp --config paper")
        # flags override a named config; a saved file round-trips
        args = qs.build_parser().parse_args(["--config", "paper", "--seeds", "0-1"])
        self.assertEqual(qs.config_from_args(args).seeds, [0, 1])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            self.assertEqual(qs.main(["--config", "paper", "--save-config", path]), 0)
            self.assertEqual(len(qs.build_plan(qs.load_config(path), {})), 180)
            toml_path = os.path.join(d, "c.toml")  # the extension decides the format, so it must load back
            self.assertEqual(qs.main(["--config", "paper", "--seeds", "0-2", "--save-config", toml_path]), 0)
            self.assertEqual(qs.load_config(toml_path).seeds, [0, 1, 2])
            self.assertEqual(qs.load_config(toml_path).networks, ["dblp", "amazon"])
        self.assertEqual(CLI.hedonic_main(["run", "paper", "--dry-run"]), 0)

    def test_user_config_defaults_and_profiles(self):
        from hedonic.experiments.overlapping import quickstart as qs

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.toml")
            with patch.dict(os.environ, {"HEDONIC_CONFIG": path}):
                self.assertEqual(CLI.hedonic_main(["config", "set", "output_dir", d]), 0)
                self.assertEqual(CLI.hedonic_main(["config", "set", "run", '"paper"']), 0)
                self.assertEqual(CLI.hedonic_main(["config", "set", "profiles.mine.seeds", '"0-2"']), 0)
                self.assertEqual(CLI.hedonic_main(["config", "set", "profiles.mine.config", '"paper"']), 0)
                cfg = qs.config_from_args(qs.build_parser().parse_args(["--config", "paper"]))
                self.assertEqual(cfg.output_dir, d)  # machine default applies to named configs
                cfg = qs.config_from_args(qs.build_parser().parse_args(["--config", "mine"]))
                self.assertEqual((cfg.seeds, cfg.networks), ([0, 1, 2], ["dblp", "amazon"]))
                cfg = qs.config_from_args(qs.build_parser().parse_args(["--config", "paper", "--output-dir", "x"]))
                self.assertEqual(cfg.output_dir, "x")  # flags win
                self.assertEqual(CLI.hedonic_main(["run"]), 0)  # no TTY: shows the plan, does not start

    def test_hedonic_show(self):
        from hedonic.experiments.overlapping import info

        with tempfile.TemporaryDirectory() as d:
            rows = info.method_status(Path(d))
            self.assertEqual(len(rows), 9)
            self.assertEqual({r["key"]: r["status"] for r in rows}["hedonic"], "ready")
            nets = info.network_status(Path(d), d, remote=False)
            self.assertEqual(len(nets), 7)
            self.assertTrue(all(n["status"] == "not downloaded" for n in nets))
            for what in ("methods", "networks", "metrics"):
                self.assertEqual(CLI.hedonic_main(["show", what, "--cache-dir", d, "--json"] +
                                                  (["--network-root", d] if what == "networks" else [])), 0)

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
        from hedonic.experiments.overlapping import quickstart as qs
        from hedonic.experiments.overlapping import runmanager as rm

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_RUNS_DIR": d}):
            cfg = qs.Config(methods=["hedonic"], output_dir=d, cache_dir=d)
            cfg.methods = ["not-a-method"]  # fails inside the worker, after it has started
            entry = {"name": "bad", "config": cfg.__dict__, "output_dir": d, "created": 0, "session": "",
                     "backend": "process", "log": os.path.join(d, "worker.log")}
            rm.write_json(rm.entry_path("bad"), entry)
            with patch("sys.stderr"), patch("sys.stdout"):
                self.assertEqual(rm.worker("bad"), 1)
            progress = rm.read_progress(entry)
            self.assertEqual(progress["status"], "failed")
            self.assertIn("not-a-method", progress["reason"])
            self.assertEqual(rm.effective_status(entry, progress), "failed")
            self.assertIn("hedonic run resume bad", rm.snapshot(entry))

    def test_run_launch_records_absolute_paths_and_resets_progress(self):
        from hedonic.experiments.overlapping import quickstart as qs
        from hedonic.experiments.overlapping import runmanager as rm

        class Proc:
            pid = 999_999_999

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_RUNS_DIR": d}):
            cwd = os.getcwd()
            os.chdir(d)
            try:
                stale = Path(d) / "rel" / "job" / "progress.json"
                stale.parent.mkdir(parents=True)
                stale.write_text('{"status": "completed"}')
                with patch.object(rm.shutil, "which", return_value=None), \
                        patch.object(rm.subprocess, "Popen", return_value=Proc()):
                    entry = rm.launch(qs.Config(output_dir="rel", cache_dir=os.path.join(d, "c")), "job")
                    self.assertTrue(os.path.isabs(entry["output_dir"]))  # `hedonic run list` works from any directory
                    self.assertFalse(stale.exists())  # a resumed run must not look finished
                    with self.assertRaises(ValueError):
                        rm.launch(qs.Config(output_dir="rel"), "bad name")
            finally:
                os.chdir(cwd)

    def test_run_option_validation_and_friendly_errors(self):
        from hedonic.experiments.overlapping import quickstart as qs
        from hedonic.experiments.overlapping import userconfig

        for bad in ("3-1", "x", "-2", ""):
            with self.assertRaises(ValueError):
                qs.parse_seeds(bad)
        self.assertEqual(qs.parse_name("paper-1_a"), "paper-1_a")
        for bad in ("a b", "a.b", "../x", ""):
            with self.assertRaises(ValueError):
                qs.parse_name(bad)
        for flags in (["--threads", "0"], ["--timeout", "0"], ["--nodes", "-1"]):
            with self.assertRaises(ValueError):
                qs.config_from_args(qs.build_parser().parse_args(flags))
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_CONFIG": os.path.join(d, "c.toml")}):
            for argv in (["set", "threads", "abc"], ["set", "threads", "0"], ["set", "output_dir", "true"],
                         ["set", "profiles.q.bogus", "1"]):
                with self.assertRaises(SystemExit):
                    userconfig.main(argv)
            self.assertFalse(os.path.exists(userconfig.path()))  # nothing invalid was written
            self.assertEqual(userconfig.main(["set", "threads", "4"]), 0)
            self.assertEqual(userconfig.defaults()["threads"], 4)

    def test_run_exp_list_uses_the_configured_cache_dir(self):
        from hedonic.experiments.overlapping import quickstart as qs

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_CONFIG": os.path.join(d, "c.toml")}):
            self.assertEqual(CLI.hedonic_main(["config", "set", "cache_dir", d]), 0)
            with patch("hedonic.experiments.overlapping.info.method_status") as method_status, \
                    patch("hedonic.experiments.overlapping.info.network_status", return_value=[]), \
                    patch("hedonic.experiments.overlapping.info.render_networks", return_value=""), \
                    patch("hedonic.experiments.overlapping.info.render_methods", return_value=""):
                self.assertEqual(qs.main(["--list"]), 0)
                self.assertEqual(method_status.call_args.args[0], Path(d))  # not the literal path "None"

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

    def test_overlapping_small_via_cli(self):
        code = CLI.main(["overlapping-small"])
        self.assertEqual(code, 0)

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


