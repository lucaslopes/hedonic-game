"""The durable post-hoc analysis: exact counts agree with the grid audit, progress is reported, configs load."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments import CLI, resources
from hedonic.experiments.overlapping import gt_posthoc as gp
from hedonic.experiments.overlapping import robustness as rb
from hedonic.experiments.overlapping import stability as st
from tests.experiments.test_studies_stability_exact import random_instance


class TestAnalyseState(unittest.TestCase):
    def test_exact_counts_match_grid_audit(self):
        grid = [0.0, 0.1, 0.35, 0.6, 1.0]
        for seed in range(6):
            graph, memberships = random_instance(seed)
            state = rb.build_fractional_state(graph, memberships)
            cap = max(2, max(len(row) for row in memberships))
            out = gp.analyse_state(state, n_vertices=graph.vcount(), cap=cap, grid=grid, workers=1)
            for policy, block in out.items():
                self.assertEqual(len(block["per_gamma"]), len(grid))
                for row in block["per_gamma"]:
                    self.assertEqual(row["exact_stable_vertex_count"], row["grid_stable_vertex_count"],
                                     (seed, policy, row["gamma"]))

    def test_progress_reaches_total(self):
        graph, memberships = random_instance(3)
        state = rb.build_fractional_state(graph, memberships)
        cap = max(2, max(len(row) for row in memberships))
        events = []
        gp.analyse_state(state, n_vertices=graph.vcount(), cap=cap, grid=[0.5], progress=events.append)
        vertex_events = [e for e in events if e["type"] == "vertices"]
        self.assertTrue(vertex_events)
        last = vertex_events[-1]
        self.assertEqual(last["done"], last["total"])
        self.assertEqual(last["total"], 2 * graph.vcount())
        self.assertTrue(all(0 < e["done"] <= e["total"] for e in vertex_events))

    def test_intervals_are_stored_by_vertex_not_by_completion_order(self):
        graph, memberships = random_instance(4)
        state = rb.build_fractional_state(graph, memberships)
        cap = max(2, max(len(row) for row in memberships))
        out = gp.analyse_state(state, n_vertices=graph.vcount(), cap=cap, grid=[0.5], workers=1)
        stored = out["fixed_labels"]["intervals"]
        for v in range(graph.vcount()):
            expected = [[float(lo), float(hi)]
                        for lo, hi in st.exact_stable_intervals(state, v, cap, False, atol=gp.ATOL, rtol=gp.RTOL)]
            self.assertEqual(stored[v], expected, v)


class TestEngines(unittest.TestCase):
    def test_python_and_compiled_engines_store_identical_intervals(self):
        if not st.HAVE_NUMBA:
            self.skipTest("numba is not installed")
        for seed in range(5):
            graph, memberships = random_instance(seed)
            state = rb.build_fractional_state(graph, memberships)
            cap = max(2, max(len(row) for row in memberships))
            kwargs = {"n_vertices": graph.vcount(), "cap": cap, "grid": [0.0, 0.3, 1.0], "workers": 1}
            reference = gp.analyse_state(state, engine="python", **kwargs)
            compiled = gp.analyse_state(state, engine="numba", **kwargs)
            for policy in reference:
                self.assertEqual(compiled[policy]["intervals"], reference[policy]["intervals"], (seed, policy))
                self.assertEqual(compiled[policy]["per_gamma"], reference[policy]["per_gamma"], (seed, policy))


AMAZON_FULL = ('[overlapping_gt_posthoc]\nname = "gt-posthoc-amazon-all-full"\ndataset = "amazon"\ncover = "all"\n'
               'max_nodes = 0\nworkers = 8\n')


class TestConfiguration(unittest.TestCase):
    def test_a_name_resolves_through_the_shipped_configs_and_uses_the_registered_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            shipped = Path(tmp) / "configs"
            shipped.mkdir()
            (shipped / "overlapping-gt-posthoc-amazon-full.toml").write_text(AMAZON_FULL, encoding="utf-8")
            with patch.object(resources, "CONFIG_DIR", shipped):
                options = gp.load_options("amazon-full", default_output_root=tmp)
            self.assertEqual(options["config"], str((shipped / "overlapping-gt-posthoc-amazon-full.toml").resolve()))
            self.assertEqual(options["dataset"], "amazon")
            self.assertIsNone(options["max_nodes"])
            self.assertEqual(len(options["grid"]), 131)
            self.assertEqual(options["grid"], gp.locked_grid())
            self.assertEqual((options["grid"][0], options["grid"][-1]), (0.0, 1.0))
            self.assertIn(1e-4, options["grid"])
            self.assertEqual(Path(options["output_dir"]), (Path(tmp) / "gt-posthoc-amazon-all-full").resolve())

    def test_a_missing_config_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(resources, "CONFIG_DIR", Path(tmp)):
            with self.assertRaisesRegex(ValueError, "no such config: amazon-full"):
                gp.load_options("amazon-full")

    def test_the_default_run_folder_is_under_the_runs_root_not_the_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "posthoc.toml"
            path.write_text(AMAZON_FULL, encoding="utf-8")
            with patch.object(gp, "RUNS_ROOT", Path(tmp) / "runs"):
                options = gp.load_options(str(path))
            self.assertEqual(Path(options["output_dir"]), (Path(tmp) / "runs" / "gt-posthoc-amazon-all-full").resolve())
            explicit = gp.load_options(str(path), output_dir=str(Path(tmp) / "here"))
            self.assertEqual(Path(explicit["output_dir"]), (Path(tmp) / "here").resolve())

    def test_flags_override_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "posthoc.toml"
            path.write_text('[overlapping_gt_posthoc]\ndataset = "dblp"\nworkers = 2\n'
                            'audit_resolutions = "0:1:3"\n', encoding="utf-8")
            options = gp.load_options(str(path), name="t-run", workers=4, default_output_root=tmp)
            self.assertEqual(options["name"], "t-run")
            self.assertEqual(options["workers"], 4)
            self.assertEqual(options["grid"], [0.0, 0.5, 1.0])
            self.assertEqual(options["cover"], "all")

    def test_bad_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "posthoc.toml"
            path.write_text(AMAZON_FULL, encoding="utf-8")
            with self.assertRaises(ValueError):
                gp.load_options(str(path), name="bad name", default_output_root=tmp)


class TestRegistration(unittest.TestCase):
    def test_registered_in_cli(self):
        self.assertIn("overlapping-gt-posthoc", CLI.COMMANDS)
        self.assertEqual(CLI.COMMANDS["overlapping-gt-posthoc"].module, "hedonic.experiments.overlapping.gt_posthoc")
        self.assertEqual(CLI.RUN_ALIASES["posthoc"], "overlapping-gt-posthoc")


if __name__ == "__main__":
    unittest.main()
