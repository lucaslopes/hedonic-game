"""The SNAP network table covers the six com-* networks, including Orkut's top5000 cover (curated counts)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hedonic.experiments.overlapping import snap_table as st
from tests.experiments.test_provision_covered_stream import write_root


class TestSnapTable(unittest.TestCase):
    def test_the_networks_are_the_six_com_datasets(self):
        self.assertEqual(st.NETWORK_ORDER, ("dblp", "amazon", "youtube", "friendster", "livejournal", "orkut"))
        self.assertNotIn("wikipedia", st.DATASET_SEMANTICS)

    def test_orkut_top5000_is_reported_without_rescanning_the_large_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nets"
            write_root(root)  # an Orkut graph with only a top5000 cover
            rows, _signature, missing = st.analyze(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dataset"], "orkut")
            self.assertEqual(rows[0]["cover"], "top5000")
            self.assertEqual(rows[0]["vertices"], 731_514)
            self.assertEqual(rows[0]["edges"], 21_992_510)
            self.assertNotIn("orkut:top5000", missing)
            self.assertIn("orkut:all", missing)
            self.assertIn("dblp:top5000", missing)


if __name__ == "__main__":
    unittest.main()
