"""Controls of ``Game.community_hedonic``.

Covers end-to-end seed replay, isolation of the caller's igraph generator,
node weights, the ``max_memberships=-1`` convention, global community-count
constraints, and result provenance.
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

from hedonic import Game, _native


def _karate() -> Game:
    return Game(ig.Graph.Famous("Zachary"))


def _occupied(result) -> int:
    rows = result.membership
    if rows and isinstance(rows[0], list):
        return len({label for row in rows for label in row})
    return len(set(rows))


class TestSeedReplay(unittest.TestCase):
    def setUp(self):
        # Stock python-igraph has no getter; reset to its default afterwards.
        self.addCleanup(ig.set_random_number_generator, random)
        self.addCleanup(random.setstate, random.getstate())

    def test_same_seed_same_result_for_partitions_and_covers(self):
        for max_memberships in (1, 3):
            a = _karate().community_hedonic(
                initial_membership=None, max_memberships=max_memberships,
                resolution=0.1, seed=7,
            )
            # Perturb the caller's generators between the calls.
            ig.set_random_number_generator(random.Random(999))
            random.seed(999)
            b = _karate().community_hedonic(
                initial_membership=None, max_memberships=max_memberships,
                resolution=0.1, seed=7,
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

    def test_seed_leaves_the_callers_generators_alone(self):
        mine = random.Random(123)
        state, module_state = mine.getstate(), random.getstate()
        ig.set_random_number_generator(mine)
        for max_memberships in (1, 2):
            _karate().community_hedonic(max_memberships=max_memberships,
                                        resolution=0.1, seed=1)
        self.assertEqual(mine.getstate(), state)
        self.assertEqual(random.getstate(), module_state)

    def test_unseeded_call_draws_from_the_default_generator(self):
        rng = _native.default_rng()
        state = rng.getstate()
        a = _karate().community_hedonic(initial_membership=None, resolution=0.1)
        self.assertNotEqual(rng.getstate(), state)
        self.assertEqual(a._hedonic_provenance["rng"], "random module")
        # Replaying the generator's state replays the call.
        rng.setstate(state)
        b = _karate().community_hedonic(initial_membership=None, resolution=0.1)
        self.assertEqual(a.membership, b.membership)

    def test_seed_must_be_an_integer(self):
        with self.assertRaises(ValueError):
            _karate().community_hedonic(seed=1.5)
        with self.assertRaises(ValueError):
            _karate().community_hedonic(seed=True)

    def test_interrupt_propagates_and_leaves_state_untouched(self):
        module_state = random.getstate()
        game = _karate()
        game.memberships = [v % 2 for v in range(34)]
        with patch("hedonic._native.community_hedonic", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                game.community_hedonic(seed=1)
        self.assertEqual(random.getstate(), module_state)
        self.assertEqual(game.memberships, [[v % 2] for v in range(34)])

    def test_native_validation_error_leaves_the_generator_alone(self):
        rng = _native.default_rng()
        state = rng.getstate()
        game = _karate()
        game.memberships = [v % 2 for v in range(34)]
        with self.assertRaisesRegex(ValueError, "non-negative"):
            game.community_hedonic(max_memberships=2, edge_weights=[-1.0] + [1.0] * 77)
        self.assertEqual(rng.getstate(), state)
        self.assertEqual(game.memberships, [[v % 2] for v in range(34)])

    def test_seeded_call_never_sets_igraphs_generator(self):
        with patch.object(ig, "set_random_number_generator") as setter:
            a = _karate().community_hedonic(seed=1, max_memberships=2, resolution=0.1)
            b = _karate().community_hedonic(seed=1, max_memberships=2, resolution=0.1)
        setter.assert_not_called()
        self.assertEqual(a.membership, b.membership)


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

    def test_upper_bound_and_exact_count(self):
        for max_memberships in (1, 3):
            bounded = _karate().community_hedonic(
                initial_membership=None, max_memberships=max_memberships,
                resolution=0.3, seed=1,
                max_total_communities=4,
            )
            self.assertLessEqual(_occupied(bounded), 4)
            self.assertEqual(bounded._hedonic_provenance["count_constraint"], "at_most")
            exact = _karate().community_hedonic(
                initial_membership=None, max_memberships=max_memberships,
                resolution=0.05, seed=1,
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

    def test_infeasible_counts_fail_closed(self):
        with self.assertRaises(Exception):
            _karate().community_hedonic(initial_membership=None, n_communities=35)
        with self.assertRaises(Exception):
            # A start with 34 labels violates an upper bound of 3.
            _karate().community_hedonic(initial_membership=list(range(34)),
                                        max_total_communities=3)

    def test_exact_count_above_the_vertex_count_on_a_cover(self):
        # Six communities on four vertices needs extra memberships per vertex.
        game = Game(ig.Graph.Ring(4))
        cover = game.community_hedonic(
            initial_membership=None, max_memberships=2, n_communities=6, seed=0,
        )
        self.assertEqual(_occupied(cover), 6)
        self.assertEqual(cover._hedonic_provenance["occupied_communities"], 6)
        self.assertTrue(all(1 <= len(row) <= 2 for row in cover.membership))


class TestProvenance(unittest.TestCase):
    def test_provenance_records_versions_parameters_and_stopping(self):
        cover = _karate().community_hedonic(max_memberships=2, resolution=0.1, seed=0)
        provenance = cover._hedonic_provenance
        self.assertIsNotNone(provenance["versions"]["python_igraph"])
        self.assertEqual(provenance["equilibrium_notion"], "cap_constrained")
        self.assertEqual(provenance["rng"], "seeded")
        self.assertEqual(provenance["seed"], 0)
        self.assertEqual(provenance["resolution"], 0.1)
        self.assertEqual(cover._hedonic_native_quality, cover._params["quality"])
        partition = _karate().community_hedonic(resolution=0.1)
        self.assertEqual(partition._hedonic_provenance["equilibrium_notion"],
                         "unconstrained_partition")
        self.assertEqual(partition._hedonic_provenance["rng"], "random module")
        self.assertIsNone(partition._hedonic_provenance["seed"])
        self.assertEqual(partition._hedonic_native_quality, partition.quality)
        density = _karate()
        self.assertEqual(density.community_hedonic(seed=0)._hedonic_provenance["resolution"],
                         density.density())


if __name__ == "__main__":
    unittest.main()
