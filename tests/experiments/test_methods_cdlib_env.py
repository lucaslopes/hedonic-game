"""The isolated CDlib environment (``overlapping.cdlib_env``): shipped workers, receipts, fail-closed runs.

Data-free: no CDlib is imported and no worker runs unless a test builds a fake environment for it.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import igraph as ig

from hedonic.experiments import resources
from hedonic.experiments.overlapping import cdlib_env as ce
from hedonic.experiments.overlapping import methods as om


class TestShippedWorkers(unittest.TestCase):
    def test_every_worker_ships_inside_the_package(self):
        for name in ce.WORKERS:
            path = ce.worker_script(name)
            self.assertTrue(path.is_file(), name)
            self.assertTrue(path.is_relative_to(resources.WORKER_DIR), path)
        project = ce.shipped_project_dir()
        self.assertTrue((project / "pyproject.toml").is_file())
        self.assertTrue((project / "uv.lock").is_file())
        self.assertTrue(resources.worker_path("ncgame_worker.py").is_file())
        with self.assertRaises(FileNotFoundError):
            ce.worker_script("no_such_worker.py")

    def test_the_old_worker_modules_are_gone_from_the_package(self):
        package = Path(om.__file__).parent
        for name in ("baselines", "overlap_lfr", "protocol", "paper_figures", *[w[:-3] for w in ce.WORKERS],
                     "ncgame_worker"):
            self.assertFalse((package / f"{name}.py").exists(), name)

    def test_the_environment_root_follows_its_variables(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HEDONIC_SLPA_ENV": str(Path(tmp) / "env")}):
                self.assertEqual(ce.environment_root(), (Path(tmp) / "env").resolve())
            env = {k: v for k, v in os.environ.items() if k != "HEDONIC_SLPA_ENV"}
            env["HEDONIC_CODESEG_CACHE"] = tmp
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(ce.environment_root(), Path(tmp).resolve() / "cdlib_env")
            self.assertIs(ce._slpa_environment_root, ce.environment_root)


class TestReceiptAndFailClosed(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve() / "env"
        patcher = patch.dict(os.environ, {"HEDONIC_SLPA_ENV": str(self.root)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_an_empty_environment_is_not_ready_and_a_run_says_why(self):
        with patch.object(ce.shutil, "which", return_value=None):
            receipt = ce.slpa_environment_receipt()
        self.assertFalse(receipt["ready"])
        self.assertEqual(receipt["protocol_version"], ce.SLPA_PROTOCOL_VERSION)
        self.assertEqual(receipt["direct_pins"], dict(ce.SLPA_DIRECT_PINS))
        self.assertIsNone(receipt["files"]["project"]["sha256"])
        self.assertTrue(receipt["files"]["worker"]["sha256"])  # the worker always ships
        with self.assertRaisesRegex(ce.MethodUnavailable, "not ready.*hedonic setup"):
            ce.run_worker("cdlib_worker.py", {"method": "angel"}, receipt=receipt)
        link = ce.link_clustering_environment_receipt()
        self.assertFalse(link["ready"])
        self.assertEqual(link["api"], "cdlib.algorithms.hierarchical_link_community")

    def test_a_worker_that_is_not_shipped_is_unavailable(self):
        with self.assertRaisesRegex(ce.MethodUnavailable, "not shipped"):
            ce.run_worker("missing_worker.py", {}, receipt={"ready": True})

    def test_a_ready_environment_runs_the_worker_with_its_python_and_reads_the_last_json_line(self):
        self.root.mkdir(parents=True)
        (self.root / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        (self.root / "uv.lock").write_text("version = 1\n")
        python = self.root / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        receipt = ce.slpa_environment_receipt()
        self.assertTrue(receipt["ready"])
        self.assertEqual(receipt["execution_preference"], "project_venv")
        command, mode = ce.worker_command(receipt, ce.worker_script("cdlib_worker.py"))
        self.assertEqual((command[0], mode), (str(python), "project_venv"))
        fake = self.root.parent / "echo_worker.py"
        fake.write_text("import json, sys\nrequest = json.load(sys.stdin)\nprint('chatter')\n"
                        "print(json.dumps({'echo': request}))\n")
        with patch.object(ce, "_worker_or_none", return_value=fake):
            payload, run = ce.run_worker("cdlib_worker.py", {"a": 1}, receipt=receipt)
        self.assertEqual(payload, {"echo": {"a": 1}})
        self.assertEqual(run["execution_mode"], "project_venv")
        self.assertEqual(run["worker_source_identity"]["path"], fake.name)  # a name, never an absolute path


class TestHelpers(unittest.TestCase):
    def test_last_json_line_skips_chatter_and_fails_closed(self):
        self.assertEqual(ce.last_json_line("warning\n{\"a\": 1}\nnot json\n"), {"a": 1})
        self.assertIs(ce._last_json_line, ce.last_json_line)
        with self.assertRaises(ce.MethodUnavailable):
            ce.last_json_line("[1, 2]\nnothing\n")

    def test_pins_are_checked_against_what_the_worker_reports(self):
        pins = {"cdlib": "0.4.0", "networkx": "3.4.2"}
        self.assertEqual(ce.check_pins({"cdlib_version": "0.4.0", "networkx_version": "3.4.2"}, pins, "w"), pins)
        with self.assertRaisesRegex(ce.MethodUnavailable, "drift"):
            ce.check_pins({"cdlib_version": "0.3.0", "networkx_version": "3.4.2"}, pins, "w")

    def test_the_method_registry_routes_cdlib_methods_through_the_environment(self):
        self.assertIs(om.MethodUnavailable, ce.MethodUnavailable)
        availability = om.method_availability()
        for name in ("angel", "link_clustering", "slpa"):
            self.assertIn(name, availability)
            if not availability[name]["available"]:
                self.assertTrue(availability[name].get("reason"))

    def test_an_unready_environment_never_falls_back_to_an_ambient_cdlib(self):
        graph = ig.Graph.Famous("Zachary")
        with patch.dict(os.environ, {"HEDONIC_SLPA_ENV": tempfile.mkdtemp()}), \
                patch.object(ce.shutil, "which", return_value=None):
            with self.assertRaises(ce.MethodUnavailable):
                ce.run_cdlib_method(graph, method="angel", parameters={"threshold": 0.5}, seed=0)


if __name__ == "__main__":
    unittest.main()
