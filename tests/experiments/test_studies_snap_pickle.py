"""``hedonic pickle``: the Game of a network:cover pair round-trips through a pickle, in a worker process of its own."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hedonic import Game
from hedonic.experiments import CLI
from hedonic.experiments.overlapping import snap_pickle as sp
from tests.experiments.test_provision_covered_stream import edge_multiset, load, write_root

# The fixture graphs are tiny: a fixed budget and no free-memory floor, so a busy machine (a CI runner below the 10%
# floor) neither skips a pair nor stops its worker.
_MEMORY = patch.dict(os.environ, {"HEDONIC_MEMORY_BUDGET_GB": "16", "HEDONIC_MEMORY_FLOOR": "0"})


def setUpModule():
    _MEMORY.start()


def tearDownModule():
    _MEMORY.stop()


class TestSnapPickle(unittest.TestCase):
    def test_registered_in_the_cli_and_help_works(self):
        self.assertIn("snap-pickle", CLI.COMMANDS)
        with self.assertRaises(SystemExit) as stop, redirect_stdout(io.StringIO()):
            sp.main(["--help"])
        self.assertEqual(stop.exception.code, 0)

    def test_table_pairs_are_the_eleven_of_table_1(self):
        self.assertEqual(len(sp.TABLE_PAIRS), 11)
        self.assertNotIn("wikipedia", {network for network, _ in sp.TABLE_PAIRS})
        self.assertNotIn(("orkut", "top5000"), sp.TABLE_PAIRS)
        self.assertEqual(set(sp.TABLE_PAIRS) - set(sp.SIZES), set())

    def test_a_pair_is_written_reopened_and_equal_to_the_loaded_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cache, out = Path(tmp) / "nets", Path(tmp) / "cache", Path(tmp) / "out"
            write_root(root)
            with patch.dict(os.environ, {"HEDONIC_SNAP_STREAM_COVERED": "0"}), redirect_stdout(io.StringIO()):
                code = sp.main(["orkut:top5000", "--network-root", str(root), "--cache-dir", str(cache),
                                "--out-dir", str(out), "--json"])
            self.assertEqual(code, 0)
            game = sp.load_game(out / "orkut-top5000.game.pkl")
            self.assertIsInstance(game, Game)
            expected = load(root, cache / "ref", "0")
            self.assertEqual(edge_multiset(game), edge_multiset(expected.graph))
            self.assertEqual(game["network"], "orkut")
            self.assertEqual(game["cover_variant"], "top5000")
            self.assertTrue(game["cover"])
            self.assertEqual(game["pickle_schema"], sp.SCHEMA)

    def test_rows_report_size_and_reopen_time_and_an_existing_pickle_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cache, out = Path(tmp) / "nets", Path(tmp) / "cache", Path(tmp) / "out"
            write_root(root)
            argv = ["orkut:top5000", "--network-root", str(root), "--cache-dir", str(cache), "--out-dir", str(out), "--json"]
            buffer = io.StringIO()
            with redirect_stdout(buffer), patch.dict(os.environ, {"HEDONIC_SNAP_STREAM_COVERED": "0"}):
                sp.main(argv)
            row = json.loads(buffer.getvalue())[0]
            self.assertEqual(row["result"], "written")
            self.assertGreater(row["bytes"], 0)
            self.assertTrue(row["is_game"])
            self.assertGreater(row["m"], 0)
            self.assertIn("load_seconds", row)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                sp.main(argv)
            self.assertEqual(json.loads(buffer.getvalue())[0]["action"], "exists")

    def test_unknown_pair_and_missing_cover_are_skipped_with_a_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nets"
            write_root(root)
            rows = [sp.plan_pair("nosuch", "all", root, Path(tmp)), sp.plan_pair("orkut", "all", root, Path(tmp))]
            self.assertEqual([r["action"] for r in rows], ["skipped", "skipped"])
            self.assertTrue(all(r["reason"] for r in rows))

    def test_a_pair_that_does_not_fit_is_skipped_unless_forced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nets"
            write_root(root)
            with patch.dict(os.environ, {"HEDONIC_MEMORY_BUDGET_GB": "0.0001", "HEDONIC_SNAP_STREAM_COVERED": "0"}):
                skipped = sp.plan_pair("orkut", "top5000", root, Path(tmp))
                forced = sp.plan_pair("orkut", "top5000", root, Path(tmp), force=True)
            self.assertEqual(skipped["action"], "skipped")
            self.assertIn("free above the floor", skipped["reason"])
            self.assertEqual(forced["action"], "write")


class TestPickleRun(unittest.TestCase):
    def _written(self, tmp):
        root, cache, out = Path(tmp) / "nets", Path(tmp) / "cache", Path(tmp) / "out"
        write_root(root)
        with patch.dict(os.environ, {"HEDONIC_SNAP_STREAM_COVERED": "0"}), redirect_stdout(io.StringIO()):
            sp.main(["orkut:top5000", "--network-root", str(root), "--cache-dir", str(cache), "--out-dir", str(out)])
        return out

    @staticmethod
    def _tuned(path, **settings):
        path.write_text(json.dumps({"complete": True, "settings": {"hedonic": settings}}))
        return path

    def test_nearest_density_is_chosen_on_a_log_scale(self):
        from hedonic.experiments.overlapping import pickle_run as pr

        known = {("a", "x"): {"density": 1e-5, "setting": {"cap": 1}}, ("b", "x"): {"density": 1e-3, "setting": {"cap": 2}}}
        self.assertEqual(pr.borrow(("c", "x"), 2e-4, known)[0], ("b", "x"))
        self.assertEqual(pr.borrow(("c", "x"), 5e-5, known)[0], ("a", "x"))
        with self.assertRaises(ValueError):
            pr.borrow(("c", "x"), 1e-4, {})

    @staticmethod
    def _loopless(out):
        """The fixture's pickle without its self-loop (hedonic 2.0's detector refuses self-loops at every cap)."""
        import pickle

        path = out / "orkut-top5000.game.pkl"
        game = sp.load_game(path)
        game.simplify(multiple=False, loops=True)
        path.write_bytes(pickle.dumps(game, protocol=pickle.HIGHEST_PROTOCOL))
        return out

    def test_run_scores_a_pickle_and_records_time_and_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._loopless(self._written(tmp))
            tuned = self._tuned(Path(tmp) / "t.json", **{"orkut:top5000": {"max_memberships": 1, "absolute_resolution": 0.5}})
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = sp.main(["run", "orkut:top5000", "--tuned", str(tuned), "--out-dir", str(out), "--json"])
            self.assertEqual(code, 0, buffer.getvalue())
            row = json.loads(buffer.getvalue())[0]
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["setting_source"], "calibrated")
            self.assertGreater(row["rss_peak_bytes"], 0)
            self.assertGreaterEqual(row["detection_seconds"], 0)
            self.assertTrue(0 <= row["f1"] <= 1 and 0 <= row["onmi"] <= 1)
            self.assertTrue((out / "hoc-runs" / "results.json").is_file())
            self.assertIn("hedonic_core_sha256", row["runtime"])
            self.assertNotIn("lucas_igraph", json.dumps(row))

    def test_a_pair_without_calibration_borrows_and_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._written(tmp)
            (out / "dblp-top5000.game.pkl").write_bytes((out / "orkut-top5000.game.pkl").read_bytes())
            tuned = self._tuned(Path(tmp) / "t.json", **{"orkut:top5000": {"max_memberships": 2, "absolute_resolution": 0.5}})
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                sp.main(["run", "orkut:top5000", "dblp:top5000", "--tuned", str(tuned), "--out-dir", str(out), "--dry-run"])
            self.assertIn("borrowed from orkut:top5000", buffer.getvalue())

    def test_a_failing_pair_is_a_row_with_its_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._written(tmp)  # the fixture has a self-loop, which the detector refuses
            tuned = self._tuned(Path(tmp) / "t.json", **{"orkut:top5000": {"max_memberships": 2, "absolute_resolution": 0.5}})
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = sp.main(["run", "orkut:top5000", "--tuned", str(tuned), "--out-dir", str(out), "--json"])
            row = json.loads(buffer.getvalue())[0]
            self.assertEqual((code, row["status"]), (1, "failed"))
            self.assertIn("self-loops", row["reason"])

    def test_a_tuned_file_is_required(self):
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
            sp.main(["run", "dblp:top5000"])


if __name__ == "__main__":
    unittest.main()
