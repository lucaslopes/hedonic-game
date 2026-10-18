"""Experiment paths: data and run roots, environment overrides and the TOML config file."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


from hedonic.experiments.config import (
    load_config_file,
    read_toml,
    reload_paths,
    resolve_experiment_paths,
)


class TestExperimentsConfig(unittest.TestCase):
    def test_defaults(self):
        """With no environment and no user config, everything lives under one folder, ~/Databases/Hedonic."""
        import subprocess
        import sys

        code = ("import json; from hedonic.experiments import config as c; "
                "print(json.dumps({k: str(getattr(c, k)) for k in ('DATA_ROOT', 'RUNS_ROOT', 'DISJOINT_RUNS_ROOT', "
                "'DEFAULT_NETWORKS_DIR', 'DEFAULT_DBLP_DIR', 'DEFAULT_SYNTHETIC_DIR', 'DEFAULT_OUTPUT_DIR', "
                "'SNAP_CACHE_DIR', 'NETWORKS_DIR')}))")
        with tempfile.TemporaryDirectory() as d:
            env = {k: v for k, v in os.environ.items() if not k.startswith("HEDONIC_") or k == "HIG_PINNED"}
            env["HEDONIC_CONFIG"] = str(Path(d) / "absent.toml")
            out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        paths = {k: Path(v) for k, v in json.loads(out.stdout).items()}
        root = Path.home() / "Databases" / "Hedonic"
        self.assertEqual(paths["DATA_ROOT"], root)
        self.assertEqual(paths["RUNS_ROOT"], root / "Overlap")
        self.assertEqual(paths["DISJOINT_RUNS_ROOT"], root / "Disjoint")
        self.assertEqual(paths["DEFAULT_NETWORKS_DIR"], root / "Networks")
        self.assertEqual(paths["NETWORKS_DIR"], paths["DEFAULT_NETWORKS_DIR"])
        self.assertEqual(paths["DEFAULT_DBLP_DIR"], paths["DEFAULT_NETWORKS_DIR"] / "DBLP")
        self.assertEqual(paths["DEFAULT_SYNTHETIC_DIR"], root / "PHYSA" / "Synthetic_Networks" / "V1020")
        self.assertEqual(paths["DEFAULT_OUTPUT_DIR"], paths["RUNS_ROOT"])
        self.assertEqual(paths["SNAP_CACHE_DIR"], root / "cache" / "snap")
        for value in paths.values():
            self.assertTrue(value.is_relative_to(root), value)

    def test_environment_and_user_config_move_the_roots(self):
        import subprocess
        import sys

        code = ("import json; from hedonic.experiments import config as c; "
                "print(json.dumps([str(c.DATA_ROOT), str(c.RUNS_ROOT), str(c.NETWORKS_DIR), str(c.expand_path('artifacts/x'))]))")
        with tempfile.TemporaryDirectory() as d:
            user = Path(d) / "config.toml"
            user.write_text(f'[defaults]\nnetwork_root = "{d}/snap"\n')
            env = {k: v for k, v in os.environ.items() if not k.startswith("HEDONIC_") or k == "HIG_PINNED"}
            env.update(HEDONIC_CONFIG=str(user), HEDONIC_DATA_DIR=f"{d}/data", HEDONIC_OUTPUT_DIR=f"{d}/runs")
            out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
            data, runs, networks, artifact = json.loads(out.stdout)
            self.assertEqual((data, runs, networks), (f"{d}/data", f"{d}/runs", f"{d}/snap"))
            self.assertEqual(artifact, f"{d}/runs/artifacts/x")

    def test_env_override(self):
        with patch.dict(
            os.environ,
            {
                "HEDONIC_DBLP_DIR": "/tmp/custom_dblp",
                "HEDONIC_NETWORKS_DIR": "/tmp/custom_networks",
                "HEDONIC_SYNTHETIC_DIR": "/tmp/custom_synth",
                "HEDONIC_OUTPUT_DIR": "/tmp/custom_out",
            },
            clear=False,
        ):
            dblp, synth = reload_paths()
            self.assertEqual(str(dblp), "/tmp/custom_dblp")
            self.assertEqual(str(synth), "/tmp/custom_synth")
            from hedonic.experiments import config as cfg

            self.assertEqual(str(cfg.OUTPUT_DIR), "/tmp/custom_out")
            self.assertEqual(str(cfg.NETWORKS_DIR), "/tmp/custom_networks")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HEDONIC_DBLP_DIR", None)
            os.environ.pop("HEDONIC_NETWORKS_DIR", None)
            os.environ.pop("HEDONIC_SYNTHETIC_DIR", None)
            os.environ.pop("HEDONIC_OUTPUT_DIR", None)
            reload_paths()

    def test_toml_paths_and_section(self):
        with tempfile.TemporaryDirectory() as d:
            toml_path = Path(d) / "hedonic.toml"
            toml_path.write_text(
                "\n".join(
                    [
                        "[paths]",
                        'dblp_dir = "/toml/dblp"',
                        'networks_dir = "/toml/networks"',
                        'synthetic_dir = "/toml/synth"',
                        'output_dir = "/toml/out"',
                        "",
                        "[overlapping_resolution]",
                        'output_dir = "/toml/res-f1"',
                        'resolutions = "0,1"',
                        'seeds = "0,1"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            data = read_toml(toml_path)
            self.assertIn("paths", data)
            with patch.dict(os.environ, {}, clear=False):
                for k in (
                    "HEDONIC_DBLP_DIR",
                    "HEDONIC_SYNTHETIC_DIR",
                    "HEDONIC_OUTPUT_DIR",
                    "HEDONIC_CONFIG",
                ):
                    os.environ.pop(k, None)
                loaded = load_config_file(toml_path, search_cwd=False, apply=True)
                self.assertEqual(loaded["paths"]["dblp_dir"], "/toml/dblp")
                from hedonic.experiments import config as cfg

                self.assertEqual(str(cfg.DBLP_DIR), "/toml/dblp")
                self.assertEqual(str(cfg.NETWORKS_DIR), "/toml/networks")
                self.assertEqual(str(cfg.OUTPUT_DIR), "/toml/out")
                resolved = resolve_experiment_paths(
                    config_path=toml_path,
                    data_dir=None,
                    output_dir=None,
                    experiment_section="overlapping_resolution",
                    search_cwd=False,
                )
                self.assertEqual(str(resolved["output_dir"]), "/toml/res-f1")
                self.assertEqual(
                    resolved["section"]["resolutions"], "0,1"
                )
                # CLI explicit output wins over TOML section
                resolved_cli = resolve_experiment_paths(
                    config_path=toml_path,
                    data_dir=None,
                    output_dir="/cli/out",
                    experiment_section="overlapping_resolution",
                    search_cwd=False,
                )
                self.assertEqual(str(resolved_cli["output_dir"]), "/cli/out")
            reload_paths()

    def test_relative_artifact_paths_are_anchored_at_repo_root(self):
        from hedonic.experiments.config import ARTIFACTS_DIR

        with tempfile.TemporaryDirectory() as d:
            toml_path = Path(d) / "hedonic.toml"
            toml_path.write_text(
                "\n".join(
                    [
                        "[paths]",
                        'output_dir = "artifacts"',
                        "",
                        "[overlapping_resolution]",
                        'output_dir = "artifacts/overlapping/resolution_f1"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HEDONIC_OUTPUT_DIR", None)
                resolved = resolve_experiment_paths(
                    config_path=toml_path,
                    data_dir=None,
                    output_dir=None,
                    experiment_section="overlapping_resolution",
                    search_cwd=False,
                )
            self.assertEqual(
                resolved["output_dir"],
                ARTIFACTS_DIR / "overlapping" / "resolution_f1",
            )
        reload_paths()

    def test_every_shipped_config_parses_and_names_no_machine_path(self):
        import tomllib

        from hedonic.experiments.resources import shipped_configs

        configs = shipped_configs()
        self.assertTrue({"queue-paper.toml", "paper-benchmark.toml", "overlap-heatmap.toml",
                         "start-quality-hoc-local.toml", "baseline-calibration-paper.toml",
                         "overlapping-gt-spectrum-full.toml", "hedonic-run.example.toml"} <= {p.name for p in configs})
        for path in configs:
            text = path.read_text()
            tomllib.loads(text)
            for word in ("/Users/", "-1006", "docs/papers"):
                self.assertNotIn(word, text, f"{path.name}: {word}")
            # The one data folder may be named, as the documented default (~/Databases/Hedonic), never elsewhere.
            self.assertEqual(text.count("Databases"), text.count("~/Databases/Hedonic"), path.name)

    def test_find_config_prefers_configs_dir(self):
        from hedonic.experiments.config import DEFAULT_TOML_NAMES, find_config_file

        self.assertEqual(DEFAULT_TOML_NAMES[0], "configs/hedonic.toml")
        old = Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "configs").mkdir()
            (Path(d) / "configs" / "hedonic.toml").write_text('[paths]\noutput_dir = "~/x"\n')
            try:
                os.chdir(d)
                found = find_config_file(search_cwd=True)
                self.assertIsNotNone(found)
                self.assertTrue(str(found).endswith("configs/hedonic.toml"))
            finally:
                os.chdir(old)


class TestCodesegCache(unittest.TestCase):
    """``hedonic setup``, ``hedonic run exp`` and ``hedonic show methods`` find the external methods in one folder."""

    def resolve(self, env, cache_dir=None):
        """(config's resolver, hedonic setup's default, hedonic run exp's folder) under ``env``."""
        from hedonic.experiments.config import codeseg_cache_dir
        from hedonic.experiments.overlapping import codeseg_setup, userconfig
        from hedonic.experiments.overlapping import quickstart as qs

        with patch.dict(os.environ, env, clear=False):
            if "HEDONIC_CODESEG_CACHE" not in env:
                os.environ.pop("HEDONIC_CODESEG_CACHE", None)
            cfg = qs.Config()
            cfg.cache_dir = cache_dir or userconfig.defaults().get("cache_dir") or cfg.cache_dir  # as the front door sets it
            return codeseg_cache_dir(cache_dir), codeseg_setup.default_cache_dir(), qs.codeseg_tools(cfg)

    def test_the_order_is_environment_then_user_config_then_the_default(self):
        with tempfile.TemporaryDirectory() as d:
            user = Path(d) / "config.toml"
            user.write_text(f'[defaults]\ncache_dir = "{d}/user-cache"\n', encoding="utf-8")
            absent = str(Path(d) / "absent.toml")
            default = Path("~/.cache/hedonic/codeseg").expanduser()
            self.assertEqual(set(self.resolve({"HEDONIC_CONFIG": absent})), {default})
            self.assertEqual(set(self.resolve({"HEDONIC_CONFIG": str(user)})), {Path(d) / "user-cache" / "codeseg"})
            self.assertEqual(set(self.resolve({"HEDONIC_CONFIG": str(user), "HEDONIC_CODESEG_CACHE": f"{d}/env"})),
                             {Path(d) / "env"})

    def test_a_cache_dir_flag_wins(self):
        with tempfile.TemporaryDirectory() as d:
            explicit, _, run = self.resolve({"HEDONIC_CONFIG": str(Path(d) / "absent.toml"),
                                             "HEDONIC_CODESEG_CACHE": f"{d}/env"}, cache_dir=f"{d}/flag")
            self.assertEqual(explicit, Path(d) / "flag" / "codeseg")
            self.assertEqual(run, Path(d) / "flag" / "codeseg")


if __name__ == "__main__":
    unittest.main()
