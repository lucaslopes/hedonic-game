"""The shipped paper queue on 2.0: found by name, written under ``{root}``, checked against what `hedonic setup` prepared."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments import CLI
from hedonic.experiments.overlapping import codeseg_setup as cs
from hedonic.experiments.overlapping import runqueue
from hedonic.experiments.resources import CONFIG_DIR

SMALL = ["dblp:top5000", "dblp:all", "amazon:top5000", "amazon:all", "youtube:top5000", "youtube:all"]
LARGE = ["livejournal:top5000", "friendster:top5000"]
PAIRS = SMALL + LARGE
BASELINES = ["codeseg", "ncgame", "ego_splitting", "fox", "clique_percolation", "angel", "bigclam", "neo_kmeans"]


class _Isolated(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._patches = [
            patch.dict(os.environ, {"HEDONIC_RUNS_DIR": str(self.root / "registry"),
                                    "HEDONIC_CONFIG": str(self.root / "config.toml"),
                                    "HEDONIC_CODESEG_CACHE": str(self.root / "cache"),
                                    "HEDONIC_NETWORKS_DIR": str(self.root / "nets")}),
        ]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self._patches):
            patcher.stop()
        self._tmp.cleanup()


class TestShippedQueue(_Isolated):
    def test_the_queue_is_found_by_name_and_by_file_name(self):
        self.assertIn("paper", runqueue.shipped_queues())
        self.assertEqual(runqueue.queue_file("paper"), CONFIG_DIR / "queue-paper.toml")
        self.assertEqual(runqueue.queue_file("queue-paper"), CONFIG_DIR / "queue-paper.toml")
        with self.assertRaisesRegex(SystemExit, "shipped queues: disjoint, paper"):
            runqueue.queue_file("nope")

    def test_root_is_substituted_everywhere_and_every_item_writes_under_it(self):
        root = self.root / "q"
        queue = runqueue.load_queue("paper", root=str(root))
        self.assertEqual(queue["pairs"], SMALL)
        # the benchmark items share one folder and the large spectrum the first one's, so their records are reused
        shared = {"gt-robustness-large": "gt-robustness"}
        for item in queue["items"]:
            text = " ".join(item["run"])
            self.assertNotIn("{root}", text)
            self.assertNotIn("Databases", text)
            self.assertNotIn("1006", text)
            folder = ("benchmark" if item["name"].startswith("benchmark-") else shared.get(item["name"], item["name"]))
            self.assertEqual(runqueue._option(item["run"], "--output-dir"), str(root.resolve() / folder), item["name"])
        tuned = runqueue._option(next(i for i in queue["items"] if i["name"] == "benchmark-fox")["run"], "--tuned")
        self.assertEqual(tuned.split(","), [str(root.resolve() / "calibration-fox" / "baseline_tuned.json")])

    def test_the_default_root_is_the_output_dir_then_the_runs_root(self):
        with patch("hedonic.experiments.overlapping.userconfig.defaults", return_value={"output_dir": str(self.root / "o")}):
            self.assertEqual(runqueue.default_root("paper-queue"), self.root / "o" / "paper-queue")
        with patch("hedonic.experiments.overlapping.userconfig.defaults", return_value={}):
            from hedonic.experiments import config

            self.assertEqual(runqueue.default_root("paper-queue"), Path(config.RUNS_ROOT) / "paper-queue")

    def test_the_baseline_items_have_the_settings_of_the_earlier_calibration(self):
        """The same budget as the earlier calibration, whose finished searches each item adopts (see baseline_run)."""
        root = self.root / "q"
        queue = runqueue.load_queue("paper", root=str(root))
        items = [i for i in queue["items"] if i["name"].startswith("calibration-") and "hedonic" not in i["name"]]
        self.assertEqual(sorted(runqueue._option(i["run"], "--methods") for i in items), sorted(2 * BASELINES))
        for item in items:
            run = item["run"]
            self.assertEqual(run[0], "calibrate-baselines")
            self.assertEqual(runqueue._option(run, "--networks").split(","),
                             LARGE if item["name"].endswith("-large") else SMALL)
            self.assertEqual([runqueue._option(run, flag) for flag in
                              ("--max-evals", "--max-minutes", "--eval-timeout-minutes", "--replicates")],
                             ["8", "240", "60", "1"])
            self.assertEqual(runqueue._option(run, "--adopt-from"), str(root.resolve() / "baseline-calibration-rc1"))

    def test_no_item_asks_for_the_multilevel_phase(self):
        queue = runqueue.load_queue("paper", root=str(self.root / "q"))
        for item in queue["items"]:
            self.assertNotIn("multilevel", " ".join(item["run"]))
        self.assertNotIn("multilevel", (CONFIG_DIR / "start-quality-hoc-local.toml").read_text().split("#")[0])


class TestSetupCheck(_Isolated):
    def test_requirements_are_the_pairs_and_the_methods_of_the_items(self):
        queue = runqueue.load_queue("paper", root=str(self.root / "q"))
        pairs, methods = runqueue.requirements(queue)
        self.assertEqual(list(dict.fromkeys(pairs)), PAIRS)
        self.assertEqual(set(methods), {"hedonic", *BASELINES})

    def test_nothing_prepared_names_the_setup_command(self):
        queue = runqueue.load_queue("paper", root=str(self.root / "q"))
        check = runqueue.setup_check(queue)
        self.assertTrue(check["command"].startswith("hedonic setup --pairs " + ",".join(PAIRS)))
        self.assertNotIn("hedonic", check["command"].split("--methods")[1].split(","))  # built in, never set up
        self.assertEqual(set(check["command"].split("--methods ")[1].split(",")), {m for m in BASELINES if cs.needs_setup(m)})

    def test_a_ready_manifest_and_present_files_need_nothing(self):
        queue = runqueue.load_queue("paper", root=str(self.root / "q"))
        manifest = {"methods": {m: {"status": "ready"} for m in BASELINES}}
        with patch.object(cs, "load_setup_manifest", return_value=manifest), \
                patch.object(cs, "slpa_environment_receipt", return_value={"ready": True}), \
                patch("hedonic.experiments.overlapping.snap.pair_files", return_value=object()):
            check = runqueue.setup_check(queue)
        self.assertIsNone(check["command"])

    def test_an_items_network_root_is_where_its_pairs_are_looked_for(self):
        queue = runqueue.load_queue("paper", root=str(self.root / "q"))
        elsewhere = self.root / "elsewhere"
        for item in queue["items"]:
            if item["run"][0] == "calibrate-baselines":
                item["run"] = [*item["run"], "--network-root", str(elsewhere)]
        roots = runqueue.item_network_roots(queue)
        self.assertEqual(set(roots), set(PAIRS))
        self.assertEqual(set(roots.values()), {elsewhere})
        seen = []

        def pair_files(network, cover, root=None):
            seen.append(Path(root))
            return None
        with patch("hedonic.experiments.overlapping.snap.pair_files", pair_files):
            check = runqueue.setup_check(queue)
        self.assertEqual(set(seen), {elsewhere})
        self.assertIn("--network-root " + str(elsewhere), check["command"])

    def test_a_queue_that_needs_nothing_prints_no_setup_line(self):
        out = io.StringIO()
        with redirect_stdout(out):
            runqueue._print_setup_check({"pairs": [], "methods": [], "command": None, "network_root": "x"})
        self.assertEqual(out.getvalue(), "")

    def test_the_setup_cache_follows_the_environment_then_the_user_config(self):
        self.assertEqual(cs.default_cache_dir(), self.root / "cache")
        with patch.dict(os.environ, {"HEDONIC_CODESEG_CACHE": ""}), \
                patch("hedonic.experiments.overlapping.userconfig.defaults", return_value={"cache_dir": str(self.root / "c")}):
            self.assertEqual(cs.default_cache_dir(), self.root / "c" / "codeseg")
        with patch.dict(os.environ, {"HEDONIC_CODESEG_CACHE": ""}), \
                patch("hedonic.experiments.overlapping.userconfig.defaults", return_value={}):
            self.assertEqual(cs.default_cache_dir(), Path("~/.cache/hedonic/codeseg").expanduser())

    def test_the_dry_run_prints_the_plan_and_the_setup_command_and_starts_nothing(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = CLI.hedonic_main(["run", "queue", "paper", "--dry-run", "--root", str(self.root / "q")])
        self.assertEqual(code, 0, err.getvalue())
        text = out.getvalue()
        self.assertIn("hedonic setup --pairs", text)
        self.assertIn("benchmark", text)
        self.assertFalse((self.root / "q").exists())
        self.assertFalse((self.root / "registry" / "paper-queue.json").exists())

    def test_a_start_with_nothing_prepared_is_refused_with_the_setup_command(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                self.assertRaisesRegex(SystemExit, "not started: run `hedonic setup --pairs"):
            runqueue.front_main(["paper", "--root", str(self.root / "q")])
        self.assertFalse((self.root / "q").exists())
        self.assertFalse(any((self.root / "registry").glob("**/paper-queue*")) if (self.root / "registry").exists() else False)


class TestRoot(_Isolated):
    """A queue never writes into a folder of runs it did not start (the 1.x paper queue's folder, for instance)."""

    def ready(self):
        return patch.object(runqueue, "setup_check", return_value={"command": None, "pairs": [], "methods": [],
                                                                    "network_root": "x", "missing_pairs": [],
                                                                    "missing_methods": []})

    def start(self, *extra):
        launched = []
        with self.ready(), patch.object(runqueue, "launch", side_effect=lambda q: launched.append(q) or
                                        {"name": q["name"], "backend": "process", "log": "x", "session": ""}), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            runqueue.front_main(["paper", "--root", str(self.root / "q"), "--detach", *extra])
        return launched

    def test_a_new_or_empty_root_is_taken_and_stamped(self):
        self.assertEqual(len(self.start()), 1)
        stamp = json.loads((self.root / "q" / runqueue.ROOT_STAMP).read_text())
        self.assertEqual(stamp["queue"], "paper-queue")
        from hedonic.experiments import runtime

        self.assertEqual(stamp["hedonic"], runtime.hedonic_version())
        self.assertEqual(len(self.start()), 1)  # its own stamped root: started again (continued) without a question

    def test_a_root_with_unstamped_runs_is_refused_unless_adopted(self):
        (self.root / "q" / "benchmark").mkdir(parents=True)
        with self.assertRaisesRegex(SystemExit, "not started: .*benchmark.*--adopt-root"):
            self.start()
        self.assertFalse((self.root / "q" / runqueue.ROOT_STAMP).exists())
        self.assertEqual(len(self.start("--adopt-root")), 1)
        self.assertTrue((self.root / "q" / runqueue.ROOT_STAMP).exists())

    def test_a_root_stamped_by_another_major_version_is_refused(self):
        (self.root / "q").mkdir()
        (self.root / "q" / runqueue.ROOT_STAMP).write_text(json.dumps({"queue": "paper-queue", "hedonic": "1.6.0"}))
        with self.assertRaisesRegex(SystemExit, "hedonic 1.6.0 queue"):
            self.start()

    def test_the_dry_run_says_so_and_writes_nothing(self):
        (self.root / "q" / "benchmark").mkdir(parents=True)
        out = io.StringIO()
        with self.ready(), redirect_stdout(out), redirect_stderr(io.StringIO()):
            self.assertEqual(runqueue.front_main(["paper", "--root", str(self.root / "q"), "--dry-run"]), 0)
        self.assertIn("--adopt-root", out.getvalue())
        self.assertFalse((self.root / "q" / runqueue.ROOT_STAMP).exists())


if __name__ == "__main__":
    unittest.main()
