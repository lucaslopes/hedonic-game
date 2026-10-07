"""Controls added to ``Game.community_hedonic`` for the 1.0.5 line.

Covers end-to-end seed replay, restoration of the caller's igraph generator,
node weights, the ``max_memberships=-1`` convention, global community-count
constraints, the diagnostic trace passthrough, and result provenance.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import igraph as ig

from hedonic import Game
from hedonic.Game import HEDONIC_ALGORITHM_IDENTITY, seeded_igraph_rng

HAS_COUNT_CONSTRAINTS = "max_total_communities" in (
    ig.Graph.community_leiden.__code__.co_varnames
)


def _karate() -> Game:
    return Game(ig.Graph.Famous("Zachary"))


def _occupied(result) -> int:
    rows = result.membership
    if rows and isinstance(rows[0], list):
        return len({label for row in rows for label in row})
    return len(set(rows))


class TestSeedReplay(unittest.TestCase):
    def setUp(self):
        self.addCleanup(ig.set_random_number_generator, ig.get_random_number_generator())

    def test_same_seed_same_result_for_partitions_and_covers(self):
        for max_memberships in (1, 3):
            for local_move_only in (True, False):
                a = _karate().community_hedonic(
                    initial_membership=None, max_memberships=max_memberships,
                    resolution=0.1, local_move_only=local_move_only, seed=7,
                )
                # Perturb the caller's generator between the calls.
                ig.set_random_number_generator(random.Random(999))
                b = _karate().community_hedonic(
                    initial_membership=None, max_memberships=max_memberships,
                    resolution=0.1, local_move_only=local_move_only, seed=7,
                )
                ig.set_random_number_generator(random)
                self.assertEqual(a.membership, b.membership)

    def test_seed_replays_across_processes(self):
        script = textwrap.dedent(
            """
            import json, igraph as ig
            from hedonic import Game
            g = Game(ig.Graph.Famous("Zachary"))
            r = g.community_hedonic(initial_membership=None, max_memberships=3,
                                    resolution=0.08, seed=11)
            print(json.dumps(r.membership))
            """
        )
        outputs = {
            subprocess.run([sys.executable, "-c", script], check=True,
                           capture_output=True, text=True).stdout
            for _ in range(2)
        }
        self.assertEqual(len(outputs), 1)
        in_process = _karate().community_hedonic(
            initial_membership=None, max_memberships=3, resolution=0.08, seed=11
        )
        self.assertEqual(json.loads(outputs.pop()), in_process.membership)

    def test_seed_restores_the_callers_generator(self):
        getter = getattr(ig, "get_random_number_generator", None)
        if getter is None:
            self.skipTest("binding without get_random_number_generator")
        mine = random.Random(123)
        ig.set_random_number_generator(mine)
        try:
            _karate().community_hedonic(max_memberships=2, resolution=0.1, seed=1)
            self.assertIs(getter(), mine)
        finally:
            ig.set_random_number_generator(random)

    def test_seed_must_be_an_integer(self):
        with self.assertRaises(ValueError):
            _karate().community_hedonic(seed=1.5)
        with self.assertRaises(ValueError):
            _karate().community_hedonic(seed=True)

    def test_seed_restores_generator_and_stream_after_native_error(self):
        mine = random.Random(123)
        state = mine.getstate()
        ig.set_random_number_generator(mine)
        game = _karate()
        with patch.object(Game, "community_leiden", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                game.community_hedonic(seed=1)
        self.assertIs(ig.get_random_number_generator(), mine)
        self.assertEqual(mine.getstate(), state)

    def test_seed_restores_c_default_after_native_validation_error(self):
        ig.set_random_number_generator(None)
        with self.assertRaises(ig.InternalError):
            _karate().community_hedonic(seed=1, max_memberships=2,
                                       edge_weights=[-1.0] + [1.0] * 77)
        self.assertIsNone(ig.get_random_number_generator())

    def test_seed_contexts_restore_nested_generators(self):
        previous = ig.get_random_number_generator()
        with seeded_igraph_rng(1):
            outer = ig.get_random_number_generator()
            state = outer.getstate()
            with seeded_igraph_rng(2):
                self.assertIsNot(ig.get_random_number_generator(), outer)
            self.assertIs(ig.get_random_number_generator(), outer)
            self.assertEqual(outer.getstate(), state)
        self.assertIs(ig.get_random_number_generator(), previous)

    def test_old_binding_rejects_seed_before_changing_generator(self):
        mine = random.Random(123)
        ig.set_random_number_generator(mine)
        with patch.object(ig, "get_random_number_generator", None):
            with patch.object(ig, "set_random_number_generator") as setter:
                with self.assertRaisesRegex(RuntimeError, "1.0.0.5 or later"):
                    _karate().community_hedonic(seed=1)
                setter.assert_not_called()
        self.assertIs(ig.get_random_number_generator(), mine)


class TestNodeWeights(unittest.TestCase):
    def test_unit_node_weights_match_the_default(self):
        for max_memberships in (1, 2):
            a = _karate().community_hedonic(
                initial_membership=None, max_memberships=max_memberships,
                resolution=0.1, seed=3,
            )
            b = _karate().community_hedonic(
                initial_membership=None, max_memberships=max_memberships,
                resolution=0.1, seed=3, node_weights=[1.0] * 34,
            )
            self.assertEqual(a.membership, b.membership)
            self.assertTrue(b._hedonic_provenance["node_weighted"])

    def test_zero_node_weights_remove_crowding(self):
        # With every vertex weight zero the crowding term vanishes, so even at
        # a high resolution a returned partition is a hedonic equilibrium of
        # the pure edge-count game: no vertex has more neighbours elsewhere.
        game = _karate()
        part = game.community_hedonic(
            initial_membership=None, resolution=0.9, seed=0, node_weights=[0.0] * 34
        )
        labels = part.membership
        self.assertLess(len(set(labels)), 34)
        for v in range(game.vcount()):
            counts: dict[int, int] = {}
            for u in game.neighbors(v):
                counts[labels[u]] = counts.get(labels[u], 0) + 1
            self.assertGreaterEqual(counts.get(labels[v], 0), max(counts.values()))

    def test_node_weights_from_a_vertex_attribute(self):
        game = _karate()
        game.vs["w"] = [1.0 + (v % 3) for v in range(game.vcount())]
        a = game.community_hedonic(initial_membership=None, max_memberships=2,
                                   resolution=0.05, seed=5, node_weights="w")
        b = game.community_hedonic(initial_membership=None, max_memberships=2,
                                   resolution=0.05, seed=5, node_weights=list(game.vs["w"]))
        self.assertEqual(a.membership, b.membership)

    def test_invalid_node_weights_are_rejected(self):
        for weights in ([1.0] * 33, [-1.0] + [1.0] * 33, [math.nan] + [1.0] * 33,
                        [math.inf] + [1.0] * 33, [True] * 34):
            with self.assertRaises(ValueError):
                _karate().community_hedonic(node_weights=weights)

    def test_node_weighted_quality_matches_an_independent_potential(self):
        game = _karate()
        weights = [0.5 + (v % 4) * 0.25 for v in range(game.vcount())]
        gamma = 0.07
        cover = game.community_hedonic(initial_membership=None, max_memberships=3,
                                       resolution=gamma, seed=2, node_weights=weights)
        rows = cover.membership
        mass: dict[int, float] = {}
        for v, row in enumerate(rows):
            for c in row:
                mass[c] = mass.get(c, 0.0) + weights[v] / math.sqrt(len(row))
        support = 0.0
        for u, v in game.get_edgelist():
            shared = len(set(rows[u]) & set(rows[v]))
            support += shared / math.sqrt(len(rows[u]) * len(rows[v]))
        phi = (support - gamma / 2 * sum(m * m for m in mass.values())) / game.ecount()
        self.assertAlmostEqual(cover._hedonic_native_quality, phi, places=12)


class TestCapAndCounts(unittest.TestCase):
    def test_minus_one_removes_the_practical_cap(self):
        game = _karate()
        cover = game.community_hedonic(initial_membership=None, max_memberships=-1,
                                       resolution=0.05, seed=4)
        self.assertIsInstance(cover, ig.VertexCover)
        provenance = cover._hedonic_provenance
        self.assertEqual(provenance["max_memberships"], -1)
        self.assertEqual(provenance["effective_max_memberships"], game.vcount())
        same = _karate().community_hedonic(initial_membership=None,
                                           max_memberships=game.vcount(),
                                           resolution=0.05, seed=4)
        self.assertEqual(cover.membership, same.membership)
        for value in (0, -2, 1.5, True):
            with self.assertRaises(ValueError):
                _karate().community_hedonic(max_memberships=value)

    @unittest.skipUnless(HAS_COUNT_CONSTRAINTS, "binding without count constraints")
    def test_upper_bound_and_exact_count(self):
        for max_memberships in (1, 3):
            for local_move_only in (True, False):
                bounded = _karate().community_hedonic(
                    initial_membership=None, max_memberships=max_memberships,
                    resolution=0.3, local_move_only=local_move_only, seed=1,
                    max_total_communities=4,
                )
                self.assertLessEqual(_occupied(bounded), 4)
                self.assertEqual(bounded._hedonic_provenance["count_constraint"], "at_most")
                exact = _karate().community_hedonic(
                    initial_membership=None, max_memberships=max_memberships,
                    resolution=0.05, local_move_only=local_move_only, seed=1,
                    n_communities=6,
                )
                self.assertEqual(_occupied(exact), 6)
                self.assertEqual(exact._hedonic_provenance["equilibrium_notion"],
                                 "count_constrained")

    def test_invalid_count_requests(self):
        game = _karate()
        with self.assertRaises(ValueError):
            game.community_hedonic(max_total_communities=0)
        with self.assertRaises(ValueError):
            game.community_hedonic(n_communities=True)
        with self.assertRaises(ValueError):
            game.community_hedonic(max_total_communities=2, n_communities=3)

    def test_old_binding_reports_unsupported_counts_and_restores_seed(self):
        mine = random.Random(123)
        previous = ig.get_random_number_generator()
        self.addCleanup(ig.set_random_number_generator, previous)
        ig.set_random_number_generator(mine)
        with patch.object(Game, "community_leiden", side_effect=TypeError("unexpected keyword argument")):
            with self.assertRaisesRegex(RuntimeError, "community-count.*1.0.0.5"):
                _karate().community_hedonic(n_communities=2, seed=1)
        self.assertIs(ig.get_random_number_generator(), mine)

    @unittest.skipUnless(HAS_COUNT_CONSTRAINTS, "binding without count constraints")
    def test_infeasible_counts_fail_closed(self):
        with self.assertRaises(Exception):
            _karate().community_hedonic(initial_membership=None, n_communities=35)
        with self.assertRaises(Exception):
            # A start with 34 labels violates an upper bound of 3.
            _karate().community_hedonic(initial_membership=list(range(34)),
                                        max_total_communities=3)

    @unittest.skipUnless(HAS_COUNT_CONSTRAINTS, "binding without count constraints")
    def test_exact_count_default_start_records_extra_token_incidences(self):
        game = Game(ig.Graph.Ring(4))
        cover = game.community_hedonic(
            initial_membership=None, max_memberships=2, n_communities=6,
            n_iterations=0,
        )
        self.assertEqual(_occupied(cover), 6)
        self.assertEqual(cover._hedonic_token_preflight["initial_token_count"], 6)
        self.assertEqual(sum(map(len, cover.membership)), 6)


class TestTraceAndProvenance(unittest.TestCase):
    def test_debug_trace_passthrough(self):
        cover = _karate().community_hedonic(initial_membership=None, max_memberships=2,
                                            resolution=0.1, seed=0, debug_trace=True)
        trace = cover._params["debug_trace"]
        self.assertGreater(len(trace["moves"]), 0)
        for move in trace["moves"]:
            self.assertLessEqual(move["abs_error"], move["tolerance"])
        self.assertEqual(trace["seed_context"], {"seed": 0, "rng": "random.Random(seed)"})
        self.assertEqual(trace["algorithm_identity"], HEDONIC_ALGORITHM_IDENTITY)
        self.assertEqual(cover._hedonic_provenance["debug_trace"], "full")

    def test_debug_trace_in_both_modes_and_with_count_limits(self):
        for max_memberships in (1, 2):
            for limits in ({}, {"max_total_communities": 4}, {"n_communities": 3}):
                with self.subTest(max_memberships=max_memberships, **limits):
                    plain = _karate().community_hedonic(
                        initial_membership=None, max_memberships=max_memberships,
                        resolution=0.1, seed=3, **limits,
                    )
                    traced = _karate().community_hedonic(
                        initial_membership=None, max_memberships=max_memberships,
                        resolution=0.1, seed=3, debug_trace=True, **limits,
                    )
                    self.assertEqual(plain.membership, traced.membership)
                    trace = traced._params["debug_trace"]
                    self.assertEqual(
                        trace["mode"], "cover" if max_memberships > 1 else "partition"
                    )
                    self.assertEqual(
                        len(trace["moves"]), trace["counters"]["accepted_moves"]
                    )
                    self.assertEqual(trace["seed_context"]["seed"], 3)
                    if "n_communities" in limits:
                        self.assertTrue(all(
                            move["occupied_after"] == 3 for move in trace["moves"]
                        ))
                    counters_only = _karate().community_hedonic(
                        initial_membership=None, max_memberships=max_memberships,
                        resolution=0.1, seed=3, debug_trace="counters", **limits,
                    )
                    self.assertEqual(plain.membership, counters_only.membership)
                    compact = counters_only._params["debug_trace"]
                    self.assertEqual(compact["moves"], [])
                    self.assertEqual(
                        compact["counters"]["visits"], trace["counters"]["visits"]
                    )
        with self.assertRaisesRegex(ValueError, "debug_trace"):
            _karate().community_hedonic(debug_trace="moves")

    def test_provenance_records_identity_versions_and_stopping(self):
        cover = _karate().community_hedonic(max_memberships=2, resolution=0.1, seed=0)
        provenance = cover._hedonic_provenance
        self.assertEqual(provenance["algorithm_identity"], HEDONIC_ALGORITHM_IDENTITY)
        self.assertEqual(cover._hedonic_algorithm_identity, HEDONIC_ALGORITHM_IDENTITY)
        self.assertIsNotNone(provenance["versions"]["python_igraph"])
        self.assertTrue(provenance["native_certificate_sweep"])
        self.assertEqual(provenance["equilibrium_notion"], "cap_constrained")
        self.assertEqual(provenance["rng"], "seeded")
        budget = _karate().community_hedonic(max_memberships=2, n_iterations=1)
        self.assertFalse(budget._hedonic_provenance["native_certificate_sweep"])
        self.assertEqual(budget._hedonic_provenance["rng"], "caller igraph generator")


if __name__ == "__main__":
    unittest.main()
