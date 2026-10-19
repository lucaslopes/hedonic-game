"""What the studies became on hedonic 2.0: local moving only, seeded calls, package-relative identities, run folders.

Data-free and fast: tiny graphs only, and no study reads ``~/Databases`` or writes outside a temporary folder.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import igraph as ig

import hedonic.metrics
from hedonic.experiments.overlapping import controlled_overlap as co
from hedonic.experiments.overlapping import cover_audit
from hedonic.experiments.overlapping import ground_truth_robustness as v3
from hedonic.experiments.overlapping import metrics as om
from hedonic.experiments.overlapping import small_graphs

PACKAGE = Path(om.__file__).resolve().parents[2]  # .../hedonic (the installed package, not the repository)
OWNED = [
    "experiments/overlapping/" + name for name in (
        "methods.py", "codeseg_reproduction.py", "ground_truth_robustness.py", "gt_spectrum.py",
        "gt_spectrum_report.py", "start_quality.py", "controlled_overlap.py", "controlled_overlap_config.py",
        "cover_audit.py", "gt_posthoc.py", "pickle_run.py", "snap_pickle.py", "small_graphs.py", "robustness.py",
        "stability.py", "disjoint_oracle.py", "extra_methods.py", "metrics.py", "cdlib_env.py")
] + ["experiments/plots/benchmark_tradeoffs.py", "experiments/plots/participation_budget.py"]


def two_cliques() -> ig.Graph:
    edges = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    edges += [(a, b) for a in range(3, 8) for b in range(a + 1, 8)]
    return ig.Graph(n=8, edges=edges)


class TestSources(unittest.TestCase):
    def test_no_study_names_the_1x_runtime_or_reaches_the_repository(self):
        removed = re.compile(r"lucas[-_]igraph|HEDONIC_REQUIRE_IGRAPH_C|igraph-runtime\.toml|_hedonic_raw_memberships|"
                             r"local_move_only\s*=|ensure_equilibrium\s*=|phase_policy\s*=|debug_trace\s*=")
        for relative in OWNED:
            source = (PACKAGE / relative).read_text(encoding="utf-8")
            self.assertIsNone(removed.search(source), relative)
            self.assertIsNone(re.search(r"__file__\)[^\n]*\.parents\[", source), relative)
            self.assertNotIn("set_random_number_generator", source, relative)
            self.assertNotIn("~/Databases", source, relative)

    def test_the_v3_helpers_import_neither_the_protocol_nor_the_paper_figures(self):
        source = Path(v3.__file__).read_text(encoding="utf-8")
        self.assertNotRegex(source, r"import (protocol|paper_figures)|overlapping\.(protocol|paper_figures)")
        self.assertFalse(hasattr(v3, "main"))  # the v3 study itself is not part of 2.0; its helpers are


class TestLocalPhase(unittest.TestCase):
    def test_the_detector_worker_is_seeded_and_returns_labelled_rows(self):
        graph = two_cliques()
        start = [[0], [0], [0], [0, 1], [1], [1], [1], [1]]
        runs = [v3._detector_worker(graph, start, 2, 0.3, "local", True, 5) for _ in range(2)]
        self.assertEqual(runs[0], runs[1])
        rows = runs[0]["raw_memberships"]
        self.assertEqual(len(rows), graph.vcount())
        self.assertTrue(all(1 <= len(row) <= 2 for row in rows))
        self.assertEqual(runs[0]["final_memberships"], rows)

    def test_a_multilevel_phase_is_refused_before_the_detector_runs(self):
        self.assertEqual(v3.PHASES, ("local",))
        self.assertEqual(v3.require_local_phase("local"), "local")
        for phase in ("multilevel", "multi-phase", "global"):
            with self.assertRaisesRegex(ValueError, "not available in hedonic 2.0"):
                v3.require_local_phase(phase)
        with patch.object(v3, "Game", side_effect=AssertionError("detector built")):
            with self.assertRaises(ValueError):
                v3._detector_worker(two_cliques(), None, 2, 0.3, "multilevel", True, 0)

    def test_the_controlled_grid_is_local_only(self):
        self.assertEqual(co.check_phases(["local"]), ["local"])
        for phase in ("multiphase", "multi-phase", "multilevel"):
            with self.assertRaisesRegex(ValueError, "not available in hedonic 2.0"):
                co.check_phases(["local", phase])


class TestMetricsShim(unittest.TestCase):
    def test_every_score_is_the_library_one(self):
        for name in ("evaluate_cover", "partition_to_cover_lists", "cover_quality",
                     "overlapping_normalized_mutual_information_lfk", "symmetric_best_match_f1"):
            self.assertIs(getattr(om, name), getattr(hedonic.metrics, name), name)

    def test_the_two_experiment_helpers(self):
        graph = two_cliques()
        cover = [[0, 1, 2, 3], [3, 4, 5, 6, 7]]
        quality = om.quality_overlapping_cpm(graph, cover, resolution=0.5)
        # e_c - gamma * C(n_c, 2): (6 - 3) + (10 - 5) = 8, over 2m = 32
        self.assertAlmostEqual(quality, 8 / 32)
        self.assertTrue(om.in_equilibrium_overlapping(graph, cover, resolution=0.5))
        self.assertFalse(om.in_equilibrium_overlapping(graph, [[0, 1, 2, 3, 4, 5, 6, 7]], resolution=0.9))


class TestSmallGraphs(unittest.TestCase):
    def test_the_smoke_is_seeded_and_prints_the_same_numbers_every_time(self):
        outputs = []
        for _ in range(2):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.assertEqual(small_graphs.main([]), 0)
            outputs.append(buffer.getvalue())
        self.assertEqual(outputs[0], outputs[1])
        self.assertIn("All small-graph tests completed.", outputs[0])


class TestRunFolders(unittest.TestCase):
    def test_the_cover_audit_writes_under_the_flag_else_the_user_output_dir_else_the_runs_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bench"
            source.mkdir()
            base = ["--source", str(source), "--networks", "dblp:top5000", "--methods", "hedonic", "--seeds", "0"]
            user = Path(tmp) / "config.toml"
            user.write_text(f'[defaults]\noutput_dir = "{tmp}/user"\n')
            with patch.dict(os.environ, {"HEDONIC_CONFIG": str(user)}):
                self.assertEqual(cover_audit.resolve(base)["output"], (Path(tmp) / "user" / "cover-audit").resolve())
            with patch.dict(os.environ, {"HEDONIC_CONFIG": str(Path(tmp) / "none.toml")}), \
                    patch.object(cover_audit, "RUNS_ROOT", Path(tmp) / "runs"):
                self.assertEqual(cover_audit.resolve(base)["output"], (Path(tmp) / "runs" / "cover-audit").resolve())
            here = cover_audit.resolve([*base, "--output-dir", str(Path(tmp) / "here")])
            self.assertEqual(here["output"], (Path(tmp) / "here").resolve())


if __name__ == "__main__":
    unittest.main()
