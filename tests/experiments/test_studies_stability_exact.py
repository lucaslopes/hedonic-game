"""Parity of the stability engines (the C audit and the Python reference) with the best-response audit."""

from __future__ import annotations

import random
import unittest

import igraph as ig
import numpy as np

from hedonic.experiments.overlapping import robustness as rb
from hedonic.experiments.overlapping import stability as st

ATOL, RTOL = 1e-10, 1e-9


def random_cover(rng: random.Random, n: int, k: int, max_labels: int) -> list[list[int]]:
    """A valid cover: contiguous labels 0..k-1, each vertex 1..max_labels labels."""
    memberships = [sorted(rng.sample(range(k), rng.randint(1, max_labels))) for _ in range(n)]
    for label in range(k):
        if not any(label in row for row in memberships):
            memberships[rng.randrange(n)] = sorted(set(memberships[rng.randrange(n)]) | {label})
    return memberships


def random_instance(seed: int) -> tuple[ig.Graph, list[list[int]]]:
    rng = random.Random(seed)
    n = rng.randint(6, 14)
    p = rng.choice([0.2, 0.4, 0.7])
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    graph = ig.Graph(n=n, edges=edges)
    k = rng.randint(2, 4)
    return graph, random_cover(rng, n, k, min(k, rng.choice([1, 2, 3])))


def reference_stable(state: rb.FractionalState, v: int, gamma: float, cap: int, allow_isolation: bool) -> bool:
    diagnostic = rb.best_response(state, v, gamma, cap, allow_isolation)
    tolerance = rb._tolerance(diagnostic["current_utility"], ATOL, RTOL)
    return max(0.0, float(diagnostic["regret"])) <= tolerance


COUNT_KEYS = ("stable_vertex_count", "profitable_vertex_count", "is_nash_equilibrium")
REGRET_KEYS = ("mean_positive_regret", "max_positive_regret", "p95_positive_regret", "max_tolerance")


def reference_rows(graph, memberships, cap, gammas, allow):
    """Spectrum rows computed vertex by vertex from ``robustness.best_response`` (the original audit)."""
    state = rb.build_fractional_state(graph, memberships)
    rows = []
    for gamma in gammas:
        regret, tolerance = [], []
        for v in range(graph.vcount()):
            diagnostic = rb.best_response(state, v, gamma, cap, allow)
            regret.append(max(0.0, float(diagnostic["regret"])))
            tolerance.append(rb._tolerance(diagnostic["current_utility"], ATOL, RTOL))
        rows.append(st._spectrum_row(gamma, np.array(regret), np.array(tolerance), cap, ATOL, RTOL, allow))
    return rows


def engine_rows(graph, memberships, cap, gammas, allow, engine, workers=1):
    cover = st.prepare(graph, memberships, engine=engine)
    return st.spectrum_rows(cover, cap, gammas, atol=ATOL, rtol=RTOL, allow_isolation=allow, workers=workers)


class TestSpectrumRows(unittest.TestCase):
    def assert_close(self, fast, ref, label):
        for f, r in zip(fast, ref, strict=True):
            for key in COUNT_KEYS:
                self.assertEqual(f[key], r[key], (label, f["gamma"], key))
            for key in REGRET_KEYS:
                self.assertAlmostEqual(f[key], r[key], delta=1e-12, msg=(label, key))

    def test_rows_match_the_best_response_audit(self):
        gammas = [0.0, 0.05, 0.2, 0.5, 0.8, 1.0]
        for seed in range(25):
            graph, memberships = random_instance(seed)
            top = max(len(row) for row in memberships)
            for cap in (top, top + 1, top + 2):
                for allow in (False, True):
                    ref = reference_rows(graph, memberships, cap, gammas, allow)
                    for engine in st.ENGINES:
                        self.assert_close(engine_rows(graph, memberships, cap, gammas, allow, engine), ref,
                                          (seed, cap, allow, engine))

    def test_the_c_engine_returns_the_reference_floats(self):
        gammas = [0.0, 0.1, 0.4, 0.75, 1.0]
        for seed in range(15):
            graph, memberships = random_instance(seed)
            top = max(len(row) for row in memberships)
            for allow in (False, True):
                self.assertEqual(engine_rows(graph, memberships, top, gammas, allow, "c"),
                                 engine_rows(graph, memberships, top, gammas, allow, "python"), (seed, allow))

    def test_rows_do_not_depend_on_the_worker_count(self):
        graph, memberships = many_label_instance(2)
        top = max(len(row) for row in memberships)
        gammas = [i / 20 for i in range(21)]
        one = engine_rows(graph, memberships, top, gammas, True, "c")
        self.assertEqual(engine_rows(graph, memberships, top, gammas, True, "c", workers=3), one)


class TestExactIntervals(unittest.TestCase):
    def test_intervals_agree_with_reference_away_from_boundaries(self):
        grid = [i / 400 for i in range(401)]
        for seed in range(30):
            graph, memberships = random_instance(seed)
            state = rb.build_fractional_state(graph, memberships)
            top = max(len(row) for row in memberships)
            for cap in (top, top + 1):
                for allow in (False, True):
                    for v in range(graph.vcount()):
                        intervals = st.exact_stable_intervals(state, v, cap, allow, atol=ATOL, rtol=RTOL)
                        for gamma in grid:
                            near = min([abs(gamma - e) for lo, hi in intervals for e in (lo, hi)] + [1.0])
                            if near < 1e-6:
                                continue
                            expected = reference_stable(state, v, gamma, cap, allow)
                            self.assertEqual(st.in_intervals(intervals, gamma), expected,
                                             (seed, cap, allow, v, gamma, intervals))

    def test_certificate_is_sound(self):
        rng = random.Random(7)
        checked = 0
        for seed in range(40):
            graph, memberships = random_instance(seed)
            state = rb.build_fractional_state(graph, memberships)
            for _ in range(4):
                g0 = rng.random()
                g1 = min(1.0, g0 + rng.random() * 0.3)
                for v in range(graph.vcount()):
                    top = max(len(row) for row in memberships)
                    cap, allow = rng.choice([top, top + 1]), rng.choice([False, True])
                    if not st.certify_interval(state, v, cap, allow, g0=g0, g1=g1, atol=ATOL, rtol=RTOL):
                        continue
                    checked += 1
                    intervals = st.exact_stable_intervals(state, v, cap, allow, atol=ATOL, rtol=RTOL)
                    for step in range(21):
                        gamma = g0 + (g1 - g0) * step / 20
                        self.assertTrue(st.in_intervals(intervals, gamma, slack=1e-9),
                                        (seed, v, cap, allow, g0, g1, gamma, intervals))
        self.assertGreater(checked, 50)


class TestEngineIntervals(unittest.TestCase):
    """The C audit returns the reference's interval endpoints, not merely close ones."""

    def check_instance(self, graph, memberships, label):
        state = rb.build_fractional_state(graph, memberships)
        cover = st.prepare(graph, memberships)
        top = max(len(row) for row in memberships)
        for cap in (top, top + 1, top + 2):
            for allow in (False, True):
                fast = st.stable_intervals(cover, cap, allow, atol=ATOL, rtol=RTOL)
                for v in range(graph.vcount()):
                    ref = st.exact_stable_intervals(state, v, cap, allow, atol=ATOL, rtol=RTOL)
                    self.assertEqual([tuple(i) for i in fast[v]], ref, (label, cap, allow, v))

    def test_c_intervals_equal_reference_on_random_instances(self):
        for seed in range(40):
            graph, memberships = random_instance(seed)
            self.check_instance(graph, memberships, seed)

    def test_c_intervals_equal_reference_on_tied_scores(self):
        # identical rows make many candidate scores equal at every gamma: the index tie-break decides
        graph = ig.Graph.Full(6)
        memberships = [[0, 1], [0, 1], [0, 1], [1, 2], [1, 2], [2]]
        self.check_instance(graph, memberships, "tied")

    def test_intervals_do_not_depend_on_workers_or_chunks(self):
        graph, memberships = many_label_instance(0)
        cover = st.prepare(graph, memberships)
        one = st.stable_intervals(cover, 4, True, atol=ATOL, rtol=RTOL)
        self.assertEqual(st.stable_intervals(cover, 4, True, atol=ATOL, rtol=RTOL, workers=3), one)
        streamed = {}
        for vertices, found in st.stream_intervals(cover, 4, True, atol=ATOL, rtol=RTOL, chunk=7,
                                                   order=list(reversed(range(cover.n_vertices)))):
            streamed.update(zip(vertices.tolist(), found))
        self.assertEqual([streamed[v] for v in range(cover.n_vertices)], one)

    def test_unknown_engine_is_rejected(self):
        graph, memberships = random_instance(1)
        with self.assertRaises(ValueError):
            st.prepare(graph, memberships, engine="fortran")

    def test_a_vertex_over_the_cap_is_rejected_by_both_engines(self):
        graph, memberships = ig.Graph.Full(3), [[0, 1], [0], [1]]
        for engine in st.ENGINES:
            cover = st.prepare(graph, memberships, engine=engine)
            with self.assertRaises(ValueError):
                cover.intervals(np.arange(3), 1, False, atol=ATOL, rtol=RTOL)


def many_label_instance(seed: int, communities: int = 300) -> tuple[ig.Graph, list[list[int]]]:
    """More communities than vertices can see: most candidate labels have zero support at every vertex.

    300 communities is above the 256 below which ``rb._candidate_labels`` lists every label, so this exercises the
    sparse branch (own, neighbour and ``cap + |labels| + 1`` lightest labels) as well.
    """
    rng = random.Random(seed)
    n = communities + 10
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < 0.03]
    memberships = []
    for v in range(n):
        labels = {v % communities} | {rng.randrange(communities) for _ in range(rng.choice([0, 0, 1, 2]))}
        memberships.append(sorted(labels))
    return ig.Graph(n=n, edges=edges), memberships


class TestPrunedCandidates(unittest.TestCase):
    """Dropping the zero-support labels a vertex does not hold (all but the lightest) changes nothing it computes."""

    @staticmethod
    def expected_labels(state, v, full):
        """The pruning rule written out from the full label list, independently of ``stability._relevant_labels``."""
        current = set(state.memberships[v])
        keep = [label for label in full.labels
                if label == rb.FRESH_LABEL or label in current or state.support[v].get(label, 0.0) > 0.0]
        filler = [label for label in full.labels if label not in keep]
        if filler:
            keep.append(min(filler, key=lambda label: (state.masses[label], label)))
        return sorted(keep)

    def test_the_label_set_is_the_full_one_without_its_filler(self):
        for seed in range(4):
            graph, memberships = many_label_instance(seed)
            state = rb.build_fractional_state(graph, memberships)
            for cap in (3, 12):
                for allow in (False, True):
                    for v in range(0, graph.vcount(), 11):
                        full = st.vertex_affine(state, v, cap, allow, prune=False)
                        pruned = st.vertex_affine(state, v, cap, allow)
                        self.assertEqual(list(pruned.labels), self.expected_labels(state, v, full), (seed, cap, allow, v))
                        self.assertLess(len(pruned.labels), len(full.labels))
                        self.assertEqual((pruned.cur_a, pruned.cur_b), (full.cur_a, full.cur_b))

    def test_the_label_set_does_not_grow_with_the_cap(self):
        graph, memberships = many_label_instance(1)
        state = rb.build_fractional_state(graph, memberships)
        for v in range(0, graph.vcount(), 17):
            small = st.vertex_affine(state, v, 3, True)
            large = st.vertex_affine(state, v, 200, True)
            self.assertEqual(small.labels, large.labels)
            self.assertGreater(len(st.vertex_affine(state, v, 200, True, prune=False).labels), 150)

    def test_the_lightest_filler_is_the_smaller_label_on_equal_mass(self):
        graph = ig.Graph(n=6, edges=[(0, 1), (2, 3), (4, 5)])
        memberships = [[0], [0], [1], [1], [2], [2]]  # three communities of equal mass, one per pair
        state = rb.build_fractional_state(graph, memberships)
        self.assertEqual(st.vertex_affine(state, 0, 2, False).labels, (0, 1))  # own label + the lighter of the other two
        self.assertEqual(st.vertex_affine(state, 4, 2, False).labels, (0, 2))

    def check_intervals(self, graph, memberships, label, step=1):
        state = rb.build_fractional_state(graph, memberships)
        top = max(len(row) for row in memberships)
        for cap in (top, top + 2):
            for allow in (False, True):
                for v in range(0, graph.vcount(), step):
                    ref = st.exact_stable_intervals(state, v, cap, allow, atol=ATOL, rtol=RTOL, prune=False)
                    self.assertEqual(st.exact_stable_intervals(state, v, cap, allow, atol=ATOL, rtol=RTOL), ref,
                                     (label, cap, allow, v, "python"))

    def test_intervals_are_identical_on_small_random_instances(self):
        for seed in range(40):
            graph, memberships = random_instance(seed)
            self.check_intervals(graph, memberships, seed)

    def test_intervals_are_identical_when_most_labels_are_filler(self):
        for seed in range(3):
            graph, memberships = many_label_instance(seed)
            self.check_intervals(graph, memberships, ("many", seed), step=5)

    def test_regrets_are_identical_when_most_labels_are_filler(self):
        gammas = [0.0, 0.01, 0.1, 0.5, 1.0]
        for seed in range(2):
            graph, memberships = many_label_instance(seed)
            state = rb.build_fractional_state(graph, memberships)
            top = max(len(row) for row in memberships)
            for cap in (top, top + 2):
                for allow in (False, True):
                    for v in range(0, graph.vcount(), 7):
                        full = st.vertex_regrets(state, v, gammas, cap, allow, atol=ATOL, rtol=RTOL, prune=False)
                        pruned = st.vertex_regrets(state, v, gammas, cap, allow, atol=ATOL, rtol=RTOL)
                        for x, y in zip(pruned[0], full[0], strict=True):
                            self.assertAlmostEqual(x, y, delta=1e-12, msg=(seed, cap, allow, v))
                        self.assertEqual(pruned[1], full[1])

    def test_regret_rows_match_the_best_response_audit_when_most_labels_are_filler(self):
        gammas = [0.0, 0.05, 0.5, 1.0]
        graph, memberships = many_label_instance(0)
        top = max(len(row) for row in memberships)
        for allow in (False, True):
            ref = reference_rows(graph, memberships, top, gammas, allow)
            for engine in st.ENGINES:
                fast = engine_rows(graph, memberships, top, gammas, allow, engine)
                for f, r in zip(fast, ref, strict=True):
                    for key in COUNT_KEYS:
                        self.assertEqual(f[key], r[key], (allow, engine, f["gamma"], key))
                    for key in REGRET_KEYS:
                        self.assertAlmostEqual(f[key], r[key], delta=1e-12, msg=(allow, engine, key))


if __name__ == "__main__":
    unittest.main()
