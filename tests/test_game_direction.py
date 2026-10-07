"""Regression tests for preserving ``igraph.Graph`` directionality."""

import unittest

import igraph as ig

from hedonic import Game


class TestGameDirection(unittest.TestCase):
    @staticmethod
    def _directed_path() -> ig.Graph:
        graph = ig.Graph(
            n=3,
            edges=[(0, 1), (1, 2)],
            directed=True,
        )
        graph.vs["name"] = ["source", "middle", "sink"]
        graph.es["weight"] = [2.0, 3.0]
        graph["fixture"] = "directed-path"
        return graph

    def test_copy_constructor_preserves_direction_and_attributes(self):
        source = self._directed_path()
        game = Game(source)

        self.assertTrue(game.is_directed())
        self.assertEqual(game.get_edgelist(), source.get_edgelist())
        self.assertEqual(game.neighbors(1, mode="in"), [0])
        self.assertEqual(game.neighbors(1, mode="out"), [2])
        self.assertFalse(game.is_mutual(0, 1))
        self.assertEqual(game.vs["name"], source.vs["name"])
        self.assertEqual(game.es["weight"], source.es["weight"])
        self.assertEqual(game["fixture"], source["fixture"])

    def test_to_igraph_preserves_direction(self):
        game = Game(self._directed_path())
        round_trip = game.to_igraph()

        self.assertTrue(round_trip.is_directed())
        self.assertEqual(round_trip.get_edgelist(), [(0, 1), (1, 2)])
        self.assertEqual(round_trip.neighbors(1, mode="in"), [0])
        self.assertEqual(round_trip.neighbors(1, mode="out"), [2])
        self.assertEqual(round_trip.es["weight"], [2.0, 3.0])

    def test_copy_constructor_rejects_constructor_overrides(self):
        with self.assertRaisesRegex(TypeError, r"Game\(graph\) copies"):
            Game(self._directed_path(), directed=False)

    def test_directed_copy_rejects_all_leiden_detectors(self):
        game = Game(self._directed_path())

        for cap in (1, 2):
            with self.subTest(max_memberships=cap):
                with self.assertRaisesRegex(
                    ValueError, "community_hedonic requires an undirected graph"
                ):
                    game.community_hedonic(
                        max_memberships=cap,
                        resolution=0.1,
                        n_iterations=1,
                    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
