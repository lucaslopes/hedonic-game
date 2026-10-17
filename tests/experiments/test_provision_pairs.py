"""`snap.ensure_pair`: a network:cover pair on disk, found where every loader looks, downloaded only when asked."""

from __future__ import annotations

import gzip
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments.overlapping import snap


def _write_amazon(root: Path, cover: str = "top5000") -> None:
    folder = root / snap.SPECS["amazon"].directory
    folder.mkdir(parents=True, exist_ok=True)
    with gzip.open(folder / "com-amazon.ungraph.txt.gz", "wt") as handle:
        handle.write("# a comment\n1\t2\n2\t3\n3\t1\n")
    with gzip.open(folder / f"com-amazon.{cover}.cmty.txt.gz", "wt") as handle:
        handle.write("1\t2\t3\n")


class TestPairs(unittest.TestCase):
    def test_a_pair_spelled_either_way_is_one_target(self):
        self.assertEqual(snap.split_pair("dblp:top5k"), ("dblp", "top5000"))
        self.assertEqual(snap.split_pair("dblp", "top5000"), ("dblp", "top5000"))
        self.assertEqual(snap.split_pair("dblp")[0], "dblp")

    def test_files_on_disk_are_used_and_nothing_is_downloaded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_amazon(root)
            with patch.object(snap, "_download_catalog_file", side_effect=AssertionError("downloaded")):
                found = snap.ensure_pair("amazon:top5000", root=root)
            self.assertEqual(found.downloaded, ())
            self.assertEqual(found.root, root)
            self.assertTrue(found.graph.is_file() and found.cover_file.is_file())
            self.assertEqual(snap.pair_downloads("amazon:top5000", root=root), [])
            dataset = snap.load_snap_dataset("amazon", cover_variant="top5000", data_root=root)
            self.assertEqual(dataset.graph.vcount(), 3)

    def test_a_missing_pair_without_download_names_the_setup_command(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(snap.SnapLoadError, "hedonic setup --pairs dblp:top5"):
                snap.ensure_pair("dblp:top5000", root=Path(d), download=False)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_a_missing_pair_is_downloaded_under_its_official_names(self):
        def fake(url, destination, *, max_download_bytes=None):
            destination.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(destination, "wt") as handle:
                handle.write("1\t2\n")

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            plan = snap.pair_downloads("amazon:top5000", root=root)
            self.assertEqual(len(plan), 2)
            self.assertTrue(all(url.startswith("https://snap.stanford.edu/") for url, _ in plan))
            lines = []
            with patch.object(snap, "_download_catalog_file", side_effect=fake):
                found = snap.ensure_pair("amazon:top5000", root=root, say=lines.append)
            self.assertEqual(len(found.downloaded), 2)
            self.assertEqual(len(lines), 2)
            on_disk = snap.pair_files("amazon", "top5000", root)
            self.assertEqual((found.graph, found.cover_file), (on_disk.graph, on_disk.cover_file))
            self.assertEqual(sorted(found.downloaded), sorted(str(dest) for _url, dest in plan))
            # a second call finds both files and fetches nothing
            with patch.object(snap, "_download_catalog_file", side_effect=AssertionError("downloaded")):
                self.assertEqual(snap.ensure_pair("amazon:top5000", root=root).downloaded, ())

    def test_only_the_missing_cover_is_fetched_when_the_graph_is_there(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_amazon(root, cover="top5000")
            plan = snap.pair_downloads("amazon:all", root=root)
            self.assertEqual([dest.name for _url, dest in plan], ["com-amazon.all.dedup.cmty.txt.gz"])

    def test_the_default_root_is_the_networks_dir(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_NETWORKS_DIR": d}):
            self.assertEqual(snap.networks_root(), Path(d))


if __name__ == "__main__":
    unittest.main()
