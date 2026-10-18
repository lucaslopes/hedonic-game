"""Option matrix of ``Game.community_hedonic`` for partitions.

Both values of ``allow_isolation``, default and supplied starts, and
unconstrained, at-most and exact community counts. Each returned state is a
fixed point (continuing from it moves nothing) and a seeded call replays.
Invalid combinations raise errors that name the conflicting option.
"""

from __future__ import annotations

import itertools
import unittest

import igraph as ig

from hedonic import Game


def _karate() -> Game:
    graph = ig.Graph.Famous("Zachary")
    graph.es["weight"] = [1.0 + (edge % 3) / 2 for edge in range(graph.ecount())]
    return Game(graph)


class TestModeParityMatrix(unittest.TestCase):
    def test_matrix(self):
        n = 34
        cells = 0
        for (
            allow_isolation, supplied, count,
        ) in itertools.product(
            (False, True), (False, True), ("none", "at_most", "exact"),
        ):
            limits = {}
            if count == "at_most":
                limits["max_total_communities"] = 5
            elif count == "exact":
                limits["n_communities"] = 4
            start = None
            if supplied:
                k = 4 if count != "none" else 6
                start = [v % k for v in range(n)]
            kwargs = dict(
                max_memberships=1, allow_isolation=allow_isolation,
                resolution=0.08, edge_weights="weight", seed=100 + cells, **limits,
            )
            with self.subTest(supplied=supplied, **kwargs):
                game = _karate()
                plain = game.community_hedonic(initial_membership=start, **kwargs)
                again = _karate().community_hedonic(initial_membership=start, **kwargs)
                self.assertEqual(plain.membership, again.membership)
                self.assertIsInstance(plain, ig.VertexClustering)

                provenance = plain._hedonic_provenance
                self.assertEqual(provenance["count_constraint"], count)
                self.assertEqual(provenance["allow_isolation"], allow_isolation)
                self.assertEqual(
                    provenance["initialization"],
                    "initial_membership" if supplied else "default",
                )
                self.assertTrue(provenance["edge_weighted"])

                occupied = len(set(plain.membership))
                self.assertEqual(provenance["occupied_communities"], occupied)
                if count == "at_most":
                    self.assertLessEqual(occupied, 5)
                elif count == "exact":
                    self.assertEqual(occupied, 4)

                # The run ends at an equilibrium, so continuing from
                # the stored state returns it unchanged.
                resumed = game.community_hedonic(game.memberships, **kwargs)
                self.assertEqual(
                    resumed._hedonic_provenance["initialization"], "initial_membership"
                )
                self.assertEqual(resumed.membership, plain.membership)
                cells += 1
        self.assertEqual(cells, 2 * 2 * 3)


class TestMessagesNameTheConflictingOption(unittest.TestCase):
    def test_invalid_combinations(self):
        directed = Game(ig.Graph([(0, 1), (1, 2)], directed=True))
        game = _karate()
        cases = [
            (lambda: directed.community_hedonic(), "undirected"),
            (lambda: game.community_hedonic(max_memberships=0), "max_memberships"),
            (lambda: game.community_hedonic(n_communities=35),
             "n_communities=35 exceeds the vertex count"),
            (lambda: game.community_hedonic(max_memberships=2), "partitions only"),
            (lambda: game.community_hedonic(max_memberships=-1), "partitions only"),
            (lambda: game.community_hedonic(max_total_communities=2, n_communities=3),
             "n_communities must not exceed max_total_communities"),
            (lambda: game.community_hedonic(max_total_communities=0),
             "max_total_communities"),
            (lambda: game.community_hedonic(
                initial_membership=list(range(34)), max_total_communities=3),
             "initial_membership.*max_total_communities=3"),
            (lambda: game.community_hedonic(
                initial_membership=[[v] for v in range(34)], n_communities=3),
             "initial_membership.*n_communities=3"),
            (lambda: game.community_hedonic(resolution="0.1"),
             "resolution must be a finite real number"),
            (lambda: game.community_hedonic(allow_isolation="False"), "allow_isolation"),
            (lambda: game.community_hedonic(edge_weights="missing"),
             "no edge attribute 'missing'"),
            (lambda: game.community_hedonic(resolution=float("nan")), "resolution"),
            (lambda: game.community_hedonic(node_weights=[-1.0] * 34), "node_weights"),
        ]
        for call, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, pattern):
                    call()

    def test_max_communities_effect_is_recorded(self):
        game = _karate()
        disjoint = game.community_hedonic(initial_membership=None, max_communities=3, seed=1)
        self.assertEqual(
            disjoint._hedonic_provenance["max_communities_effect"], "random disjoint start"
        )
        supplied = game.community_hedonic(
            initial_membership=[v % 2 for v in range(34)], max_communities=3
        )
        self.assertEqual(
            supplied._hedonic_provenance["max_communities_effect"],
            "none: a start state was supplied",
        )


class TestWarmStart(unittest.TestCase):
    def test_flat_and_row_starts_agree(self):
        for limits in ({}, {"max_total_communities": 6}, {"n_communities": 4}):
            with self.subTest(**limits):
                common = dict(
                    resolution=0.08, edge_weights="weight", seed=7,
                    allow_isolation=False, max_memberships=1, **limits,
                )
                game = _karate()
                partition = game.community_hedonic(initial_membership=None, **common)
                flat = _karate().community_hedonic(
                    initial_membership=list(partition.membership), **common,
                )
                nested = _karate().community_hedonic(
                    initial_membership=[[c] for c in partition.membership], **common,
                )
                # Continuing from the Game's stored partition.
                loaded = game.community_hedonic(game.memberships, **common)
                self.assertEqual(flat.membership, nested.membership)
                self.assertEqual(flat.membership, loaded.membership)
                self.assertEqual(
                    flat._hedonic_provenance["initialization"], "initial_membership"
                )
                if "n_communities" in limits:
                    self.assertEqual(len(set(flat.membership)), 4)

    def test_default_start_is_recorded(self):
        part = _karate().community_hedonic(
            initial_membership=None, resolution=0.08, seed=1
        )
        self.assertEqual(part._hedonic_provenance["initialization"], "default")
        self.assertEqual(
            part._hedonic_provenance["max_communities_effect"],
            "none: disjoint singleton start",
        )


if __name__ == "__main__":
    unittest.main()
