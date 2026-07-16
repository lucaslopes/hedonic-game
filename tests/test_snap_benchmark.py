"""Small, archive-free tests for the shared SNAP benchmark surface."""

from __future__ import annotations

import gzip
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import igraph as ig

from hedonic.experiments import CLI
from hedonic.experiments.overlapping.methods import (
    METHODS,
    effective_resolution,
    method_availability,
    normalize_cover,
    run_method,
)
from hedonic.experiments.overlapping.metrics import structural_overlap_metrics
from hedonic.experiments.overlapping.snap import (
    bounded_induced_dataset,
    load_snap_dataset,
    smoke_dataset,
)


def _write_gzip(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(text)


class TestSnapLoader(unittest.TestCase):
    def test_streamed_raw_remaps_original_ids_and_reports_drops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "networks"
            amazon = root / "Amazon"
            _write_gzip(
                amazon / "com-amazon.ungraph.txt.gz",
                "# source target\n10 20\n20 30\n",
            )
            _write_gzip(
                amazon / "top5000.cmty.txt.gz",
                "10 20 999\n20 30\n999\n",
            )
            cache = Path(directory) / "cache"
            dataset = load_snap_dataset(
                "amazon", cover_variant="top5000", data_root=root, cache_dir=cache
            )
            self.assertEqual(dataset.cover, [[0, 1], [1, 2]])
            self.assertEqual(dataset.report["id_mapping_strategy"], "raw_sorted_original_id_to_contiguous_index")
            self.assertEqual(dataset.report["validation"]["missing_member_count"], 2)
            self.assertEqual(dataset.report["validation"]["dropped_communities_lt_2_members"], 1)
            self.assertTrue(Path(dataset.report["normalized_cache_path"]).is_file())

            cached = load_snap_dataset(
                "amazon", cover_variant="top5000", data_root=root, cache_dir=cache
            )
            self.assertEqual(cached.cover, dataset.cover)
            self.assertEqual(cached.report["source_kind"], "validated_normalized_cache")

    def test_dblp_label_attribute_remaps_cached_cover_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "networks"
            pkl = root / "DBLP" / "pkl"
            pkl.mkdir(parents=True)
            graph = ig.Graph(n=3, edges=[(0, 1), (1, 2)], directed=False)
            graph.vs["label"] = [100, 300, 700]
            with (pkl / "com-dblp.ungraph.pkl").open("wb") as f:
                pickle.dump(graph, f)
            with (pkl / "top5000.cmty.pkl").open("wb") as f:
                pickle.dump([[100, 300], [700, 999]], f)

            dataset = load_snap_dataset(
                "dblp",
                cover_variant="top5000",
                data_root=root,
                cache_dir=Path(directory) / "cache",
            )
            self.assertEqual(dataset.cover, [[0, 1]])
            self.assertEqual(dataset.report["id_mapping_strategy"], "cached_vertex_label_attribute")
            self.assertEqual(dataset.report["validation"]["missing_member_count"], 1)

    def test_wikipedia_category_cover_preserves_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "networks"
            wikipedia = root / "Wikipedia"
            _write_gzip(wikipedia / "wiki-topcats.txt.gz", "1 2\n2 3\n")
            _write_gzip(
                wikipedia / "wiki-topcats-categories.txt.gz",
                "Category:Graphs; 1 2 99\nCategory:Other; 2 3\n",
            )
            dataset = load_snap_dataset(
                "wikipedia", cover_variant="all", data_root=root, cache_dir=Path(directory) / "cache"
            )
            self.assertTrue(dataset.graph.is_directed())
            self.assertEqual(dataset.cover, [[0, 1], [1, 2]])

    def test_bounded_dataset_reindexes_cover(self):
        dataset = smoke_dataset("amazon", cover_variant="all")
        bounded = bounded_induced_dataset(dataset, 4)
        self.assertEqual(bounded.graph.vcount(), 4)
        self.assertTrue(all(0 <= v < 4 for community in bounded.cover for v in community))
        self.assertIn("bounded_subgraph", bounded.report)


class TestBenchmarkHelpers(unittest.TestCase):
    def test_cover_normalization_and_method_registry(self):
        cover, validation = normalize_cover([[0, 0, 1, 9, "bad"], []], 3)
        self.assertEqual(cover, [[0, 1]])
        self.assertEqual(validation["invalid_members_dropped"], 2)
        self.assertEqual(validation["duplicate_members_removed"], 1)
        availability = method_availability()
        self.assertTrue(
            {
                "hedonic_local",
                "hedonic_multiphase",
                "hedonic_multiphase_x10",
                "hedonic_multiphase_x100",
                "cpm",
                "demon",
            }.issubset(METHODS)
        )
        self.assertIn("available", availability["demon"])
        self.assertIn("install_requirement", availability["cpm"])

    def test_multiphase_density_variants_fix_resolution_and_enable_isolation(self):
        dataset = smoke_dataset("amazon", cover_variant="all")
        density = dataset.graph.density()
        expected = {
            "hedonic_multiphase": min(density, 1.0),
            "hedonic_multiphase_x10": min(density * 10, 1.0),
            "hedonic_multiphase_x100": min(density * 100, 1.0),
        }
        for name, resolution in expected.items():
            adapter = METHODS[name]
            self.assertEqual(
                effective_resolution(adapter, dataset.graph, requested_resolution=0.001),
                resolution,
            )
            self.assertTrue(adapter.parameters["allow_isolation"])

    def test_structural_metrics_are_overlap_aware(self):
        metrics = structural_overlap_metrics(
            [[0, 1, 2], [2, 3]], [[0, 1, 2], [2, 3]]
        )
        self.assertAlmostEqual(metrics["inclusion_rate"], 1.0)
        self.assertAlmostEqual(metrics["coverage_rate"], 1.0)
        self.assertGreater(metrics["overlapping_rate"], 0.0)
        self.assertGreater(metrics["distribution_rate"], 0.0)

    def test_hedonic_multiphase_applies_the_recorded_igraph_seed(self):
        dataset = smoke_dataset("amazon", cover_variant="all")
        first, first_meta = run_method(
            METHODS["hedonic_multiphase"],
            dataset.graph,
            max_memberships=3,
            resolution=dataset.graph.density(),
            seed=17,
        )
        second, second_meta = run_method(
            METHODS["hedonic_multiphase"],
            dataset.graph,
            max_memberships=3,
            resolution=dataset.graph.density(),
            seed=17,
        )
        self.assertEqual(first, second)
        self.assertEqual(first_meta["parameters"]["igraph_rng"], "random.Random(seed)")
        self.assertEqual(second_meta["seed"], 17)

    def test_cli_list_help_smoke_resume_and_manifest(self):
        self.assertIn("overlapping-benchmark", CLI.COMMANDS)
        self.assertEqual(CLI.main(["overlapping-benchmark", "--help"]), 0)
        self.assertEqual(CLI.main(["overlapping-benchmark", "--list-networks"]), 0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"
            argv = [
                "overlapping-benchmark",
                "--profile",
                "smoke",
                "--datasets",
                "amazon,dblp",
                "--methods",
                "hedonic_multiphase,hedonic_multiphase_x10,hedonic_multiphase_x100",
                "--output_dir",
                str(output),
                "--no-plots",
            ]
            self.assertEqual(CLI.main(argv), 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["run_status_counts"].get("completed"), 6)
            self.assertTrue((output / "results.jsonl").is_file())
            self.assertTrue((output / "results.csv.gz").is_file())
            self.assertTrue((output / "summary.csv").is_file())
            hedonic_records = [
                json.loads(path.read_text())
                for path in (output / "runs").rglob("*.json")
            ]
            self.assertTrue(all(record["allow_isolation"] for record in hedonic_records))
            self.assertTrue(all(record["max_memberships"] == 2 for record in hedonic_records))
            self.assertTrue(
                all(record["ground_truth_max_memberships"] == 2 for record in hedonic_records)
            )
            self.assertTrue(
                all(
                    record["resolution"] == min(
                        smoke_dataset(record["dataset"], cover_variant="all").graph.density()
                        * {"hedonic_multiphase": 1, "hedonic_multiphase_x10": 10,
                           "hedonic_multiphase_x100": 100}[record["method"]],
                        1.0,
                    )
                    for record in hedonic_records
                )
            )
            run = next((output / "runs").rglob("*.json"))
            before = run.read_text()
            self.assertEqual(CLI.main(argv + ["--resume"]), 0)
            self.assertEqual(run.read_text(), before)


if __name__ == "__main__":
    unittest.main()
