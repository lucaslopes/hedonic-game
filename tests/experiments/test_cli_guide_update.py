"""`hedonic guide` and `hedonic update`."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from hedonic.experiments import CLI
from hedonic.experiments.overlapping import guide, selfupdate


def run(argv):
    out = io.StringIO()
    with redirect_stdout(out):
        code = CLI.hedonic_main(argv)
    return code, out.getvalue()


class TestGuide(unittest.TestCase):
    def test_topics_and_index(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_RUNS_DIR": d, "HEDONIC_CONFIG": d + "/c.toml"}):
            code, text = run(["guide"])
            self.assertEqual(code, 0)
            for topic in guide.TOPICS:
                self.assertIn(f"hedonic guide {topic}", text)
            self.assertIn("no runs yet", text)  # state-aware suggestion
            for topic in guide.TOPICS:
                self.assertEqual(run(["guide", topic])[0], 0)
            self.assertEqual(run(["guide", "--list"])[1].split(), list(guide.TOPICS))
            self.assertEqual(set(json.loads(run(["guide", "--json"])[1])), set(guide.TOPICS))
            with patch("sys.stderr"):
                self.assertEqual(run(["guide", "nope"])[0], 2)

    def test_every_command_a_topic_mentions_exists(self):
        import re

        from hedonic.experiments.CLI import RUN_ALIASES, RUN_VERBS

        verbs = set(RUN_ALIASES) | set(RUN_VERBS)
        for _, text in guide.TOPICS.values():
            for verb in re.findall(r"hedonic run ([\w-]+)", text):
                self.assertIn(verb, verbs)
            for command in re.findall(r"hedonic (?!run\b|guide\b)([\w-]+)", text):
                self.assertIn(command, CLI.TOP_LEVEL, command)
            for command in re.findall(r"hedonic-exp ([\w-]+)", text):
                self.assertIn(command, CLI.COMMANDS, command)

    def test_named_queues_and_configs_are_shipped(self):
        import re

        from hedonic.experiments.overlapping import quickstart as qs
        from hedonic.experiments.overlapping import runqueue
        from hedonic.experiments.resources import config_path

        for _, text in guide.TOPICS.values():
            for name in re.findall(r"hedonic run queue ([a-z][\w-]*)", text):  # FILE is a placeholder
                self.assertIn(name, runqueue.shipped_queues())
            for name in re.findall(r"--config ([\w.-]+)", text):
                if name in ("NAME", "FILE") or name.endswith(".toml"):  # placeholders and the user's own files
                    continue
                spectrum = config_path(f"overlapping-gt-spectrum-{name}")  # `hedonic run spectrum --config NAME`
                self.assertTrue(name in qs.PROFILES or config_path(name).is_file() or spectrum.is_file(), name)

    def test_no_topic_or_help_names_what_2_0_removed(self):
        removed = ("1006", "docs/", "--script", "1.0.0.", "youtube-hoc", "spectrum-tuned", "lucas-igraph",
                   "hoc_multilevel", "multi-phase")
        for _, text in guide.TOPICS.values():
            self.assertNotIn("codeseg-setup", text)  # the guide names `hedonic setup`
        texts = [text for _, text in guide.TOPICS.values()]
        for argv in (["--help"], ["run", "--help"]):
            out = io.StringIO()
            with redirect_stdout(out), patch("sys.stderr"):
                try:
                    CLI.hedonic_main(argv)
                except SystemExit:
                    pass
            texts.append(out.getvalue())
        out = io.StringIO()
        with redirect_stdout(out):
            try:
                CLI.main(["--help"])
            except SystemExit:
                pass
        texts.append(out.getvalue())
        for text in texts:
            for word in removed:
                self.assertNotIn(word, text)


class TestUpdate(unittest.TestCase):
    def test_version_ordering(self):
        self.assertGreater(selfupdate._key("1.0.5"), selfupdate._key("0.1.1"))
        self.assertGreater(selfupdate._key("1.0.5"), selfupdate._key("1.0.5rc1"))
        self.assertEqual(selfupdate._key("1.0.5"), selfupdate._key("1.0.5"))

    def _report(self, installed, latest, editable=False):
        with patch.object(selfupdate, "installed_version", return_value=installed), \
                patch.object(selfupdate, "is_editable", return_value=editable):
            return selfupdate.report(latest)

    def test_states(self):
        self.assertEqual(self._report("1.0.5", "1.0.5")["state"], "up_to_date")
        self.assertEqual(self._report("1.0.5", "0.1.1")["state"], "ahead_of_pypi")
        self.assertEqual(self._report("1.0.5", "1.1.0")["state"], "update_available")
        self.assertEqual(self._report(None, "1.1.0")["state"], "unknown")

    def test_check_never_installs_and_yes_upgrades_only_non_editable(self):
        for editable, yes, expect_install in ((False, False, False), (False, True, True), (True, True, False)):
            with patch.object(selfupdate, "latest_version", return_value="9.9.9"), \
                    patch.object(selfupdate, "installed_version", return_value="1.0.5"), \
                    patch.object(selfupdate, "is_editable", return_value=editable), \
                    patch.object(selfupdate, "upgrade", return_value=0) as upgrade:
                code, text = run(["update", *(["--yes"] if yes else [])])
            self.assertEqual(code, 0)
            self.assertEqual(upgrade.called, expect_install, (editable, yes))
            self.assertIn("9.9.9", text)

    def test_offline_is_a_clean_failure_and_json(self):
        with patch.object(selfupdate, "latest_version", side_effect=OSError("offline")):
            code, text = run(["update"])
            self.assertEqual(code, 1)
            self.assertIn("could not reach PyPI", text)
            code, text = run(["update", "--json"])
            self.assertEqual(code, 1)
            self.assertIn("offline", json.loads(text)["error"])


class TestCliExtra(unittest.TestCase):
    """`pip install hedonic` is the library alone; the command line names the extra it needs."""

    def test_the_command_line_refuses_without_the_cli_extra(self):
        import importlib.util

        real = importlib.util.find_spec
        with patch("importlib.util.find_spec", lambda name, *a: None if name in ("pandas", "duckdb") else real(name, *a)):
            for entry in (CLI.hedonic_main, CLI.main):
                err = io.StringIO()
                with redirect_stderr(err):
                    self.assertEqual(entry(["run", "list"]), 2)
                self.assertIn('pip install "hedonic[cli]"', err.getvalue())
                self.assertIn("pandas, duckdb", err.getvalue())

    def test_the_checked_modules_are_the_cli_extra(self):
        import re
        import tomllib
        from pathlib import Path

        extra = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
        names = {re.split(r"[<>=!~;\[ ]", req, maxsplit=1)[0]
                 for req in extra["project"]["optional-dependencies"]["cli"]}
        self.assertEqual(names, set(CLI.CLI_MODULES))

    def test_update_keeps_the_extra(self):
        from types import SimpleNamespace

        with patch.object(selfupdate.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(selfupdate.upgrade(), 0)
        self.assertEqual(run.call_args.args[0][-2:], ["--upgrade", "hedonic[cli]"])


if __name__ == "__main__":
    unittest.main()
