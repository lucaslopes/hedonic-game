"""The C core (``hedonic._core``) through ``hedonic._native``: best-response
local moving for partitions.

The pinned values are results recorded on macOS arm64 (Apple clang -O3). The
core is written to give the same bits on every platform, but the pins are
skipped unless HIG_PINNED=1 (``tools/native_tests.py`` sets it), so a
platform that differs is caught there. The C core alone is tested by
``make -C native test``; ``tests/test_equilibrium.py`` checks results against
brute-force best responses.
"""

import math
import os
import random
import unittest
from array import array

import igraph as ig

from hedonic import _native

ZACHARY_SEED0 = [0, 0, 0, 0, 1, 2, 2, 0, 3, 0, 1, 1, 0, 0, 3, 3, 2, 0, 3, 0, 3, 0, 3, 3, 4,
                 4, 3, 4, 4, 3, 3, 4, 3, 3]


# The partition pins of native/tests/test_core.c (test_pins), from the same
# seeded random.Random order: (seed, resolution or None for the density, cap,
# isolation, edge-weighted, node-weighted, max_total, exact) -> (k, quality
# hex, membership in order of first appearance). Edge e weighs
# 0.5 + 0.25 (e mod 7), vertex v 0.5 (1 + v mod 4).
PINS = [
    ((0, None, 1, False, False, False, None, None), (5, "0x1.b8f407b8f407dp-2", ZACHARY_SEED0)),
    ((1, 0.2, 1, True, True, False, None, None), (10, "0x1.96f439aec498ep-2", [
        0, 0, 0, 0, 1, 1, 1, 0, 2, 3, 1, 4, 0, 0, 5, 5, 1, 0, 6, 0, 6, 7, 5, 8, 9, 9, 8, 6, 6, 8,
        2, 6, 5, 6])),
    ((3, 0.1, 1, False, False, False, None, 3), (3, "0x1.01f81f81f81f8p-1", [
        0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 0, 0, 2, 2, 1, 0, 2, 0, 2, 0, 2, 2, 1, 1, 2, 2, 1, 2,
        2, 1, 2, 2])),
    ((7, 0.5, 1, True, False, False, None, None), (19, "0x1.20d20d20d20d2p-4", [
        0, 0, 0, 0, 1, 2, 2, 0, 3, 4, 1, 5, 6, 0, 7, 8, 2, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17,
        15, 18, 17, 3, 16, 3, 3])),
]


def pinned_run(seed, resolution, cap, isolation, weighted, node_weighted, max_total, exact):
    """(k, quality hex, canonical result) of one pinned call."""
    g = ig.Graph.Famous("Zachary")
    r = _native.community_hedonic(
        g, seed=seed, resolution=resolution, max_memberships=cap, allow_isolation=isolation,
        edge_weights=[0.5 + 0.25 * (e % 7) for e in range(78)] if weighted else None,
        node_weights=[0.5 * (1 + v % 4) for v in range(34)] if node_weighted else None,
        max_total_communities=max_total, n_communities=exact)
    names = {}
    result = [names.setdefault(c, len(names)) for c in r.membership]
    return len(r), r.quality.hex(), result


def partition_regret(graph, membership, gamma, allow_isolation, node_weights=None):
    """Largest gain of a unilateral move, from the definition of the game
    (simple graphs: the empty community is offered to every vertex, which
    can only overstate the regret of a vertex alone in its community)."""
    n = graph.vcount()
    nw = node_weights or [1.0] * n
    mass = {}
    for v in range(n):
        mass[membership[v]] = mass.get(membership[v], 0.0) + nw[v]
    worst = -math.inf
    for v in range(n):
        to = {}
        for u in graph.neighbors(v):
            to[membership[u]] = to.get(membership[u], 0.0) + 1.0
        own = to.get(membership[v], 0.0) - gamma * nw[v] * (mass[membership[v]] - nw[v])
        best = max((to.get(c, 0.0) - gamma * nw[v] * mass[c] for c in mass if c != membership[v]),
                   default=-math.inf)
        if allow_isolation:
            best = max(best, 0.0)
        worst = max(worst, best - own)
    return worst


class TestPartition(unittest.TestCase):
    def setUp(self):
        self.g = ig.Graph.Famous("Zachary")

    @unittest.skipUnless(os.environ.get("HIG_PINNED") == "1", "bit-level pins: HIG_PINNED=1")
    def test_matches_recorded_pin(self):
        p = _native.community_hedonic(self.g, seed=0)
        self.assertEqual(p.membership, ZACHARY_SEED0)
        self.assertEqual(p.quality.hex(), (0.4306184012066366).hex())

    @unittest.skipUnless(os.environ.get("HIG_PINNED") == "1", "bit-level pins: HIG_PINNED=1")
    def test_matches_the_c_pins(self):
        for arguments, expected in PINS:
            self.assertEqual(pinned_run(*arguments), expected, arguments)

    def test_equilibrium(self):
        for iso in (False, True):
            for gamma in (0.05, 0.2, -0.01):
                for seed in range(5):
                    p = _native.community_hedonic(self.g, resolution=gamma, seed=seed,
                                                         allow_isolation=iso)
                    self.assertLessEqual(partition_regret(self.g, p.membership, gamma, iso), 1e-12)

    def test_seed_and_rng_agree(self):
        a = _native.community_hedonic(self.g, seed=4)
        b = _native.community_hedonic(self.g, rng=random.Random(4))
        self.assertEqual(a.membership, b.membership)
        self.assertEqual(a.quality, b.quality)

    def test_counts(self):
        p = _native.community_hedonic(self.g, resolution=0.05, n_communities=4, seed=1)
        self.assertEqual(len(set(p.membership)), 4)
        p = _native.community_hedonic(self.g, resolution=0.05, max_total_communities=3,
                                             seed=1)
        self.assertLessEqual(len(set(p.membership)), 3)

    def test_equilibrium_is_a_fixed_point(self):
        # Started from an equilibrium, whatever the visiting order, no vertex
        # moves: the result is the start.
        edges = array("q", [x for e in self.g.get_edgelist() for x in e])
        for iso in (False, True):
            first, k, q = _native.partition(34, edges, random.Random(0), resolution=0.1,
                                            allow_isolation=iso)
            for seed in range(1, 6):
                again, k2, q2 = _native.partition(34, edges, random.Random(seed),
                                                  membership=array("q", first), resolution=0.1,
                                                  allow_isolation=iso)
                self.assertEqual(list(again), list(first))
                self.assertEqual((k2, q2), (k, q))

    def test_start_is_used(self):
        # Two cliques joined by one edge, started from the two cliques (an
        # equilibrium): the start comes back unchanged.
        g = ig.Graph.Full(5) + ig.Graph.Full(5)
        g.add_edge(0, 5)
        edges = array("q", [x for e in g.get_edgelist() for x in e])
        start = [0] * 5 + [1] * 5
        membership, k, _ = _native.partition(10, edges, random.Random(0),
                                             membership=array("q", start), resolution=0.2)
        self.assertEqual(list(membership), start)
        self.assertEqual(k, 2)

    def test_raw_calls_reject_n_iterations(self):
        edges = array("q", [x for e in self.g.get_edgelist() for x in e])
        with self.assertRaises(TypeError):
            _native.partition(34, edges, random.Random(0), n_iterations=0)

    def test_raw_call(self):
        edges = array("q", [x for e in self.g.get_edgelist() for x in e])
        membership, k, q = _native.partition(34, edges, random.Random(0),
                                                    resolution=self.g.density(),
                                                    allow_isolation=False)
        self.assertEqual(k, len(set(membership)))
        self.assertTrue(math.isfinite(q))

    def test_former_tie_cycle_terminates(self):
        """The instance of test_former_tie_cycle in native/tests/test_core.c, on
        which the previous core cycled between equal-payoff moves:
        every run stops at an equilibrium."""
        edges = [(9, 20), (20, 19), (9, 3), (8, 11), (4, 10), (10, 17), (16, 7), (0, 17),
                 (20, 5), (3, 15), (2, 8), (15, 13), (17, 13), (21, 20), (8, 4)]
        nw = [0.5, 1.3, 1.7, 0.7, 1.9, 1.3, 1.3, 0.7, 0.9, 1.7, 2.3, 1.7, 0.9, 0.7, 1.1, 1.9,
              2.1, 0.7, 1.5, 1.9, 0.9, 1.7]
        g = ig.Graph(n=22, edges=edges)
        for seed in range(20):
            p = _native.community_hedonic(g, resolution=0.342, node_weights=nw, seed=seed)
            self.assertLessEqual(partition_regret(g, p.membership, 0.342, False, nw), 1e-12)

    def test_no_diverged_error(self):
        self.assertFalse(hasattr(_native, "DivergedError"))


class TestCover(unittest.TestCase):
    def test_covers_are_refused(self):
        g = ig.Graph.Famous("Zachary")
        self.assertFalse(hasattr(_native, "cover"))
        for M in (2, 3, -1):
            with self.assertRaisesRegex(ValueError, "partitions only"):
                _native.community_hedonic(g, seed=0, max_memberships=M, resolution=0.1)


class TestRemovedOptions(unittest.TestCase):
    def test_native_call_rejects_removed_keywords(self):
        g = ig.Graph.Famous("Zachary")
        for kw in (dict(local_move_only=False), dict(local_move_only=True), dict(beta=0.3),
                   dict(n_iterations=-1), dict(n_iterations=2), dict(debug_trace=True),
                   dict(phase_policy="direct")):
            with self.assertRaises(TypeError, msg=kw):
                _native.community_hedonic(g, seed=0, **kw)


class TestGeneratorBridge(unittest.TestCase):
    """An exact random.Random is run in C from its state and handed back; any
    other generator is called. Both give the same draws, results and end state."""

    class Called(random.Random):
        pass

    def setUp(self):
        self.g = ig.Graph.Famous("Zachary")

    def test_same_results_and_end_state(self):
        rng = random.Random(4)
        g = ig.Graph(n=3000, edges=[(rng.randrange(3000), rng.randrange(3000))
                                    for _ in range(12000)]).simplify()
        runs = [dict(), dict(allow_isolation=False, resolution=0.001)]
        for kw in runs:
            for seed in range(3):
                native, called = random.Random(seed), self.Called(seed)
                native.random()
                called.random()
                if seed == 2:
                    native.gauss(0, 1)
                    called.gauss(0, 1)
                a = _native.community_hedonic(g, rng=native, **kw)
                b = _native.community_hedonic(g, rng=called, **kw)
                self.assertEqual(a.membership, b.membership, kw)
                self.assertEqual(native.getstate(), called.getstate(), kw)

    def test_methods_set_on_the_instance_are_called(self):
        rng, calls = random.Random(1), []
        draw = rng.getrandbits
        rng.getrandbits = lambda k: (calls.append(k), draw(k))[1]
        _native.community_hedonic(self.g, rng=rng)
        self.assertTrue(calls)

    def test_state_written_back_after_an_error(self):
        rng = random.Random(3)
        before = rng.getstate()
        with self.assertRaises(ValueError):
            _native.community_hedonic(self.g, rng=rng, initial_membership=[-1] + [0] * 33)
        self.assertEqual(rng.getstate(), before)
        _native.community_hedonic(self.g, rng=rng)
        self.assertNotEqual(rng.getstate(), before)


class TestErrors(unittest.TestCase):
    def setUp(self):
        self.g = ig.Graph.Famous("Zachary")

    def check(self, exc, **kw):
        with self.assertRaises(exc):
            _native.community_hedonic(self.g, seed=0, **kw)

    def test_rejections(self):
        self.check(ValueError, resolution=math.nan)
        self.check(ValueError, resolution=math.inf)
        self.check(ValueError, max_memberships=0)
        self.check(ValueError, n_communities=35)
        self.check(ValueError, initial_membership=[5 * v for v in range(34)])
        self.check(ValueError, initial_membership=[-1] + [0] * 33)
        self.check(ValueError, initial_membership=[[0, 1]] + [[0]] * 33)
        self.check(ValueError, node_weights=[-1.0] + [1.0] * 33)
        self.check(ValueError, edge_weights=[math.nan] + [1.0] * 77)
        self.check(ValueError, initial_membership=[[]] + [[0]] * 33)
        with self.assertRaises(ValueError):
            _native.community_hedonic(ig.Graph(directed=True, n=3, edges=[(0, 1)]))
        with self.assertRaises(ValueError):
            _native.community_hedonic(self.g, seed=0, rng=random.Random(0))

    def test_rng_errors_propagate(self):
        class Broken:
            def getrandbits(self, k):
                raise RuntimeError("no entropy")
        with self.assertRaises(RuntimeError):
            _native.community_hedonic(self.g, rng=Broken())


if __name__ == "__main__":
    unittest.main()
