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
                "'DEFAULT_SYNTHETIC_DIR', 'DEFAULT_OUTPUT_DIR', 'DISJOINT_ARTIFACTS_DIR')}))")
        with tempfile.TemporaryDirectory() as d:
            env = {k: v for k, v in os.environ.items() if not k.startswith("HEDONIC_") or k == "HIG_PINNED"}
            env["HEDONIC_CONFIG"] = str(Path(d) / "absent.toml")
            out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        paths = {k: Path(v) for k, v in json.loads(out.stdout).items()}
        root = Path.home() / "Databases" / "Hedonic"
        self.assertEqual(paths["DATA_ROOT"], root)
        self.assertEqual(paths["RUNS_ROOT"], root / "Overlap")
        self.assertEqual(paths["DISJOINT_RUNS_ROOT"], root / "Disjoint")
        self.assertEqual(paths["DEFAULT_SYNTHETIC_DIR"], root / "PHYSA" / "Synthetic_Networks" / "V1020")
        self.assertEqual(paths["DEFAULT_OUTPUT_DIR"], paths["RUNS_ROOT"])
        self.assertEqual(paths["DISJOINT_ARTIFACTS_DIR"], root / "Overlap" / "artifacts" / "disjoint")
        for value in paths.values():
            self.assertTrue(value.is_relative_to(root), value)

    def test_environment_moves_the_roots(self):
        import subprocess
        import sys

        code = ("import json; from hedonic.experiments import config as c; "
                "print(json.dumps([str(c.DATA_ROOT), str(c.RUNS_ROOT), str(c.DISJOINT_RUNS_ROOT), "
                "str(c.expand_path('artifacts/x'))]))")
        with tempfile.TemporaryDirectory() as d:
            env = {k: v for k, v in os.environ.items() if not k.startswith("HEDONIC_") or k == "HIG_PINNED"}
            env.update(HEDONIC_CONFIG=str(Path(d) / "absent.toml"), HEDONIC_DATA_DIR=f"{d}/data",
                       HEDONIC_OUTPUT_DIR=f"{d}/runs")
            out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
            data, runs, disjoint, artifact = json.loads(out.stdout)
            self.assertEqual((data, runs, disjoint), (f"{d}/data", f"{d}/runs", f"{d}/data/Disjoint"))
            self.assertEqual(artifact, f"{d}/runs/artifacts/x")

    def test_env_override(self):
        with patch.dict(
            os.environ,
            {"HEDONIC_SYNTHETIC_DIR": "/tmp/custom_synth", "HEDONIC_OUTPUT_DIR": "/tmp/custom_out"},
            clear=False,
        ):
            synth, out = reload_paths()
            self.assertEqual(str(synth), "/tmp/custom_synth")
            self.assertEqual(str(out), "/tmp/custom_out")
            from hedonic.experiments import config as cfg

            self.assertEqual(str(cfg.OUTPUT_DIR), "/tmp/custom_out")
        with patch.dict(os.environ, {}, clear=False):
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
                        'synthetic_dir = "/toml/synth"',
                        'output_dir = "/toml/out"',
                        "",
                        "[disjoint_sweep]",
                        'output_dir = "/toml/sweep"',
                        'seeds = "0,1"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            data = read_toml(toml_path)
            self.assertIn("paths", data)
            with patch.dict(os.environ, {}, clear=False):
                for k in ("HEDONIC_SYNTHETIC_DIR", "HEDONIC_OUTPUT_DIR", "HEDONIC_CONFIG"):
                    os.environ.pop(k, None)
                loaded = load_config_file(toml_path, search_cwd=False, apply=True)
                self.assertEqual(loaded["paths"]["synthetic_dir"], "/toml/synth")
                from hedonic.experiments import config as cfg

                self.assertEqual(str(cfg.SYNTHETIC_DIR), "/toml/synth")
                self.assertEqual(str(cfg.OUTPUT_DIR), "/toml/out")
                resolved = resolve_experiment_paths(
                    config_path=toml_path,
                    data_dir=None,
                    output_dir=None,
                    experiment_section="disjoint_sweep",
                    search_cwd=False,
                )
                self.assertEqual(str(resolved["output_dir"]), "/toml/sweep")
                self.assertEqual(str(resolved["data_dir"]), "/toml/synth")
                self.assertEqual(resolved["section"]["seeds"], "0,1")
                # CLI explicit output wins over TOML section
                resolved_cli = resolve_experiment_paths(
                    config_path=toml_path,
                    data_dir=None,
                    output_dir="/cli/out",
                    experiment_section="disjoint_sweep",
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
                        "[disjoint_sweep]",
                        'output_dir = "artifacts/disjoint/sweep"',
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
                    experiment_section="disjoint_sweep",
                    search_cwd=False,
                )
            self.assertEqual(
                resolved["output_dir"],
                ARTIFACTS_DIR / "disjoint" / "sweep",
            )
        reload_paths()

    def test_every_shipped_config_parses_and_names_no_machine_path(self):
        import tomllib

        from hedonic.experiments.resources import shipped_configs

        configs = shipped_configs()
        self.assertEqual({p.name for p in configs}, {"queue-disjoint.toml", "hedonic-run.example.toml"})
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


class TestUserConfigTemplate(unittest.TestCase):
    def test_the_example_holds_machine_settings_only(self):
        import tomllib

        from hedonic.experiments.overlapping import userconfig
        from hedonic.experiments.resources import CONFIG_DIR

        text = (CONFIG_DIR / "hedonic-run.example.toml").read_text()
        uncommented = "\n".join(line.lstrip("# ") for line in text.splitlines() if "=" in line)
        keys = set(tomllib.loads(uncommented))
        self.assertEqual(keys, set(userconfig.DEFAULT_KEYS))
        self.assertEqual(set(tomllib.loads(text)), {"defaults"})


if __name__ == "__main__":
    unittest.main()
