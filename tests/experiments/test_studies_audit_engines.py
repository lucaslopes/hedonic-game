"""Every study that audits a cover writes the same output on the C engine as on the Python reference.

The experiments run the C audit (``stability.prepare``'s default); the Python engine is the reference implementation.
These tests run each consumer on both, on small instances, and compare everything except wall-clock fields.
"""

from __future__ import annotations

import unittest

import igraph as ig

from hedonic.experiments.overlapping import cover_audit as ca
from hedonic.experiments.overlapping import gt_posthoc as gp
from hedonic.experiments.overlapping import gt_spectrum as gs
from hedonic.experiments.overlapping import stability as st
from tests.experiments.test_studies_stability_exact import many_label_instance, random_instance

TIMINGS = {"exact_seconds", "regret_seconds"}


def without_timings(blocks: dict) -> dict:
    return {policy: {k: v for k, v in block.items() if k not in TIMINGS} for policy, block in blocks.items()}


def instances():
    for seed in range(8):
        yield f"random {seed}", *random_instance(seed)
    yield "many labels", *many_label_instance(1, communities=60)
    yield "tied", ig.Graph.Full(6), [[0, 1], [0, 1], [0, 1], [1, 2], [1, 2], [2]]
    weighted = ig.Graph.Ring(12)
    weighted.es["weight"] = [0.5 + (i % 3) for i in range(weighted.ecount())]
    yield "weighted", weighted, [[i // 4] for i in range(12)]


def engines(graph, memberships):
    return st.prepare(graph, memberships), st.prepare(graph, memberships, engine="python")


class TestConsumersOnBothEngines(unittest.TestCase):
    grid = [0.0, 0.02, 0.1, 0.3, 0.55, 1.0]

    def test_cover_audit_records(self):
        for label, graph, memberships in instances():
            cap = max(2, max(len(row) for row in memberships))
            c, py = (ca.audit_state(cover, cap=cap, grid=self.grid, regret_gammas=self.grid[::2], workers=1,
                                    atol=gp.ATOL, rtol=gp.RTOL) for cover in engines(graph, memberships))
            self.assertEqual(without_timings(c), without_timings(py), label)

    def test_cover_audit_records_with_forked_workers(self):
        graph, memberships = many_label_instance(0, communities=80)
        cap = max(len(row) for row in memberships) + 1
        c = ca.audit_state(st.prepare(graph, memberships), cap=cap, grid=self.grid, regret_gammas=self.grid,
                           workers=3, atol=gp.ATOL, rtol=gp.RTOL)
        py = ca.audit_state(st.prepare(graph, memberships, engine="python"), cap=cap, grid=self.grid,
                            regret_gammas=self.grid, workers=1, atol=gp.ATOL, rtol=gp.RTOL)
        self.assertEqual(without_timings(c), without_timings(py))

    def test_posthoc_analysis(self):
        for label, graph, memberships in instances():
            cap = max(2, max(len(row) for row in memberships))
            c, py = (gp.analyse_state(cover, cap=cap, grid=self.grid) for cover in engines(graph, memberships))
            self.assertEqual(without_timings(c), without_timings(py), label)

    def test_spectrum_rows(self):
        for label, graph, memberships in instances():
            top = max(len(row) for row in memberships)
            for cap in (top, top + 1):
                for allow in (False, True):
                    c, py = (gs.spectrum_audit(graph, memberships, cap=cap, allow_isolation=allow, gammas=self.grid,
                                               atol=gp.ATOL, rtol=gp.RTOL, cover=cover)
                             for cover in engines(graph, memberships))
                    self.assertEqual(c, py, (label, cap, allow))

    def test_endpoint_audit_of_a_returned_cover(self):
        for label, graph, memberships in instances():
            cap = max(len(row) for row in memberships) + 1
            for allow in (False, True):
                c, py = (st.audit_cover(graph, memberships, max_memberships=cap, allow_isolation=allow, gamma=0.2,
                                        engine=engine) for engine in st.ENGINES)
                self.assertEqual(c, py, (label, allow))


if __name__ == "__main__":
    unittest.main()
