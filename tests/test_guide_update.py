"""`hedonic guide` and `hedonic update`."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
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

        verbs = {"exp", "paper", "spectrum", "list", "status", "attach", "logs", "stop", "resume"}
        for _, text in guide.TOPICS.values():
            for verb in re.findall(r"hedonic run (\w+)", text):
                self.assertIn(verb, verbs)


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


if __name__ == "__main__":
    unittest.main()
