"""The covered-induced graph read as a stream equals the full load followed by ``covered_induced_dataset``."""

from __future__ import annotations

import gzip
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments.overlapping import codeseg_reproduction as cr
from hedonic.experiments.overlapping import ground_truth_data as gtd

# ids are not contiguous; 40 and 50 are covered but have no covered neighbour (isolated in the induced graph); 7 is covered
# but absent from the graph; 99 is a cover member of a one-member community (dropped); a duplicate edge and a self loop.
EDGES = [(10, 20), (20, 30), (10, 30), (30, 40), (40, 60), (50, 70), (10, 20), (20, 20), (60, 70), (80, 10), (30, 80)]
COVER = [[10, 20, 30], [30, 20, 10, 7], [40], [50, 40, 50], [99], [20, 30, 50]]


def write_root(root: Path) -> None:
    folder = root / "Orkut"
    folder.mkdir(parents=True)
    with gzip.open(folder / "com-orkut.ungraph.txt.gz", "wt") as handle:
        handle.write("# a comment\n")
        for a, b in EDGES:
            handle.write(f"{a}\t{b}\n")
    with gzip.open(folder / "com-orkut.top5000.cmty.txt.gz", "wt") as handle:
        for community in COVER:
            handle.write("\t".join(map(str, community)) + "\n")


def load(root: Path, cache: Path, stream: str):
    with patch.dict(os.environ, {"HEDONIC_SNAP_STREAM_COVERED": stream}):
        return cr._load_dataset("orkut", cover_variant="top5000", network_root=root, max_nodes=None, smoke=False,
                                snap_cache_dir=cache, allow_catalog=False, induce="covered")


def edge_multiset(graph):
    return sorted(tuple(sorted(edge)) for edge in graph.get_edgelist())


class TestCoveredStream(unittest.TestCase):
    def test_the_stream_gives_the_same_labelled_graph_and_cover(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cache = Path(tmp) / "nets", Path(tmp) / "cache"
            write_root(root)
            full, streamed = load(root, cache / "a", "0"), load(root, cache / "b", "1")
            self.assertEqual(full.graph.vcount(), streamed.graph.vcount())
            self.assertEqual(edge_multiset(full.graph), edge_multiset(streamed.graph))
            self.assertEqual(full.cover, streamed.cover)
            self.assertEqual(full.report["ground_truth_completion"]["original_vertex_ids"],
                             streamed.report["ground_truth_completion"]["original_vertex_ids"])
            self.assertEqual(full.report["n"], streamed.report["n"])
            self.assertEqual(full.report["m"], streamed.report["m"])
            self.assertGreater(streamed.graph.ecount(), 0)

    def test_the_result_is_cached_and_reloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cache = Path(tmp) / "nets", Path(tmp) / "cache"
            write_root(root)
            first = load(root, cache, "1")
            self.assertEqual(len(list(cache.glob("*covered-stream*.pkl"))), 1)
            with patch("pandas.read_csv", side_effect=AssertionError("must not read the edges again")):
                again = load(root, cache, "1")
            self.assertEqual(edge_multiset(first.graph), edge_multiset(again.graph))

    def test_only_a_graph_too_large_to_build_is_streamed_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nets"
            write_root(root)
            with patch.dict(os.environ, {"HEDONIC_SNAP_STREAM_COVERED": ""}):
                self.assertFalse(gtd.stream_covered_wanted("orkut", root))
                with patch.object(gtd, "STREAM_COVERED_BYTES", 10):
                    self.assertTrue(gtd.stream_covered_wanted("orkut", root))
                self.assertFalse(gtd.stream_covered_wanted("dblp", root))



class TestEveryCoveredLoaderStreams(unittest.TestCase):
    """The spectrum (and the start curve, through it), the cover audit and the baselines' calibration (through the
    audit's loader) read a streamed pair as the same graph and cover the full load gives them."""

    def test_the_spectrum_job_is_the_same_streamed_or_not(self):
        from hedonic.experiments.overlapping import gt_spectrum as gs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nets"
            write_root(root)
            jobs = {}
            for stream in ("0", "1"):
                options = {"smoke": False, "policy": "covered-induced", "max_nodes": None, "data_root": str(root),
                           "provision": False, "max_download_bytes": None, "snap_cache_dir": str(Path(tmp) / f"c{stream}")}
                with patch.dict(os.environ, {"HEDONIC_SNAP_STREAM_COVERED": stream}):
                    jobs[stream] = gs.load_job(options, {"dataset": "orkut", "cover": "top5000"})
            self.assertTrue(list((Path(tmp) / "c1").glob("*covered-stream*.pkl")))  # the stream ran, into the study's cache
            full, streamed = jobs["0"], jobs["1"]
            self.assertEqual(edge_multiset(full.graph), edge_multiset(streamed.graph))
            self.assertEqual(full.memberships, streamed.memberships)
            self.assertEqual((full.cap, full.prepared.ground_truth_identity),
                             (streamed.cap, streamed.prepared.ground_truth_identity))

    def test_the_start_curve_job_is_the_same_streamed_or_not(self):
        """The start curve builds its own options for the spectrum's loader; a streamed pair needs its cache folder."""
        from hedonic.experiments.overlapping import start_quality as sq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nets"
            write_root(root)
            config = Path(tmp) / "config.toml"
            config.write_text(f'[defaults]\ncache_dir = "{tmp}/cache"\n')
            jobs = {}
            for stream in ("0", "1"):
                env = {"HEDONIC_SNAP_STREAM_COVERED": stream, "HEDONIC_CONFIG": str(config),
                       "HEDONIC_NETWORKS_DIR": str(root)}
                with patch.dict(os.environ, env):
                    jobs[stream] = sq.load_network_job("orkut:top5000")
            self.assertTrue(list((Path(tmp) / "cache" / "snap").glob("*covered-stream*.pkl")))  # the runner's cache
            full, streamed = jobs["0"], jobs["1"]
            self.assertEqual(edge_multiset(full.graph), edge_multiset(streamed.graph))
            self.assertEqual(full.memberships, streamed.memberships)

    def test_the_cover_audit_graph_is_the_same_streamed_or_not(self):
        from hedonic.experiments.overlapping import cover_audit as ca

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nets"
            write_root(root)
            config = Path(tmp) / "config.toml"
            config.write_text(f'[defaults]\ncache_dir = "{tmp}/cache"\n')
            graphs = {}
            for stream in ("0", "1"):
                with patch.dict(os.environ, {"HEDONIC_SNAP_STREAM_COVERED": stream, "HEDONIC_CONFIG": str(config)}):
                    graphs[stream] = ca.load_graph("orkut", "top5000", root, induce="covered")
            self.assertTrue(list((Path(tmp) / "cache" / "snap").glob("*covered-stream*.pkl")))  # the runner's cache
            self.assertEqual(graphs["0"].vcount(), graphs["1"].vcount())
            self.assertEqual(edge_multiset(graphs["0"]), edge_multiset(graphs["1"]))

if __name__ == "__main__":
    unittest.main()
