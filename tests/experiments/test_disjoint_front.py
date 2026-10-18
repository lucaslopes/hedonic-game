"""`hedonic run disjoint`: where a run goes when no folder is named."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments import config
from hedonic.experiments.disjoint import front


class TestDefaultOutput(unittest.TestCase):
    def check(self, defaults: dict, expected: Path):
        with patch("hedonic.experiments.overlapping.userconfig.defaults", return_value=defaults):
            self.assertEqual(Path(front.default_output("sbm100")), expected)

    def test_the_runs_root_never_the_working_directory(self):
        self.check({}, Path(config.DISJOINT_RUNS_ROOT) / "sbm100")

    def test_the_user_config_moves_it(self):
        # A stray output_dir (a [defaults] key of earlier releases) never moves the disjoint runs' folder.
        self.check({"output_dir": "/data/runs"}, Path(config.DISJOINT_RUNS_ROOT) / "sbm100")
        self.check({"output_dir": "/data/runs", "disjoint_output_root": "/data/disjoint"}, Path("/data/disjoint/sbm100"))


if __name__ == "__main__":
    unittest.main()
