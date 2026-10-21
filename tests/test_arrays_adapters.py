"""``hedonic.arrays.from_networkx`` and ``from_scipy`` (skipped without the library)."""

from __future__ import annotations

import importlib.util
import unittest

import igraph as ig
import numpy as np

from hedonic import Game, arrays

HAVE_NETWORKX = importlib.util.find_spec("networkx") is not None
HAVE_SCIPY = importlib.util.find_spec("scipy") is not None


def _bits(result) -> tuple[str, str]:
    """The quality and slack bits of an arrays Result or a Game result."""
    if isinstance(result, arrays.Result):
        return result.quality.hex(), result.equilibrium_slack.hex()
    return (result._hedonic_native_quality.hex(),
            result._hedonic_provenance["equilibrium_slack"].hex())


@unittest.skipUnless(HAVE_NETWORKX, "networkx is not installed")
class TestFromNetworkx(unittest.TestCase):
    def test_nodes_edges_and_weights(self):
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(["c", "a", "b", "d"])
        graph.add_edge("a", "b", weight=2.0)
        graph.add_edge("c", "a")
        graph.add_edge("b", "d", weight=0.5)
        graph.nodes["a"]["mass"] = 3.0
        mine = arrays.from_networkx(graph, node_weight="mass")
        self.assertEqual(mine.nodes, ["c", "a", "b", "d"])
        self.assertEqual(mine.n, 4)
        expected = [(list(graph.nodes).index(u), list(graph.nodes).index(v)) for u, v in graph.edges]
        self.assertEqual(mine.edges.tolist(), [list(pair) for pair in expected])
        weights = [d.get("weight", 1.0) for _, _, d in graph.edges(data=True)]
        self.assertEqual(mine.edge_weights.tolist(), weights)
        self.assertEqual(mine.node_weights.tolist(), [1.0, 3.0, 1.0, 1.0])
        self.assertIsNone(arrays.from_networkx(graph, weight=None).edge_weights)
        self.assertIsNone(arrays.from_networkx(graph, weight="missing").edge_weights)
        self.assertIsNone(mine._replace(node_weights=None).node_weights)

    def test_detect_matches_game_on_the_same_edges(self):
        import networkx as nx

        graph = nx.karate_club_graph()
        mine = arrays.from_networkx(graph)
        for cap in (1, 3):
            result = mine.detect(max_memberships=cap, resolution=0.05, seed=4)
            game = Game(ig.Graph(mine.n, mine.edges.tolist()))
            expected = game.community_hedonic(max_memberships=cap, resolution=0.05, seed=4,
                                              edge_weights=mine.edge_weights)
            self.assertEqual(result.rows(), game.memberships)
            self.assertEqual(_bits(result), _bits(expected))
        # The last result is a cover of game, stored in game.memberships.
        self.assertEqual(mine.stable_intervals(result),
                         game.stable_intervals(edge_weights=mine.edge_weights))

    def test_a_multigraph_keeps_parallel_edges(self):
        import networkx as nx

        graph = nx.MultiGraph([(0, 1), (0, 1), (1, 2), (2, 0)])
        self.assertEqual(arrays.from_networkx(graph).edges.tolist(),
                         [[0, 1], [0, 1], [0, 2], [1, 2]])

    def test_refusals(self):
        import networkx as nx

        with self.assertRaisesRegex(ValueError, "undirected"):
            arrays.from_networkx(nx.DiGraph([(0, 1)]))
        graph = nx.Graph([(0, 1)])
        graph.edges[0, 1]["weight"] = "heavy"
        with self.assertRaisesRegex(ValueError, "edge attribute 'weight' at edge 0"):
            arrays.from_networkx(graph)
        with self.assertRaisesRegex(ValueError, "itself"):
            arrays.from_networkx(nx.Graph([(0, 1), (1, 1)])).detect()


@unittest.skipUnless(HAVE_SCIPY, "scipy is not installed")
class TestFromScipy(unittest.TestCase):
    def matrix(self, seed=0, n=40):
        import scipy.sparse as sparse

        rnd = np.random.default_rng(seed)
        upper = sparse.random(n, n, density=0.15, random_state=rnd, format="coo")
        upper = sparse.triu(upper, k=1)
        upper.data = np.round(upper.data * 4 + 0.25, 2)
        return (upper + upper.T).tocsr()

    def test_the_edges_are_the_upper_triangle(self):
        matrix = self.matrix()
        mine = arrays.from_scipy(matrix)
        coo = matrix.tocoo()
        keep = coo.row < coo.col
        order = np.lexsort((coo.col[keep], coo.row[keep]))
        self.assertEqual(mine.edges.tolist(),
                         np.column_stack([coo.row[keep], coo.col[keep]])[order].tolist())
        self.assertEqual(mine.edge_weights.tolist(), coo.data[keep][order].tolist())
        self.assertIsNone(arrays.from_scipy(matrix, weighted=False).edge_weights)

    def test_detect_matches_the_csr_tuple_and_game(self):
        for seed in range(5):
            matrix = self.matrix(seed)
            mine = arrays.from_scipy(matrix)
            for cap in (1, 2):
                with self.subTest(seed=seed, cap=cap):
                    a = mine.detect(max_memberships=cap, seed=seed)
                    b = arrays.detect(matrix.shape[0], (matrix.indptr, matrix.indices, matrix.data),
                                      max_memberships=cap, seed=seed)
                    game = Game(ig.Graph(mine.n, mine.edges.tolist()))
                    c = game.community_hedonic(max_memberships=cap, seed=seed,
                                               edge_weights=mine.edge_weights)
                    self.assertEqual(a.rows(), b.rows())
                    self.assertEqual(a.rows(), game.memberships)
                    self.assertEqual(_bits(a), _bits(c))

    def test_sparse_arrays_and_booleans(self):
        import scipy.sparse as sparse

        matrix = self.matrix(3)
        boolean = sparse.csr_array(matrix.astype(bool))
        mine = arrays.from_scipy(boolean)
        self.assertEqual(set(mine.edge_weights.tolist()), {1.0})
        self.assertEqual(mine.edges.tolist(), arrays.from_scipy(matrix).edges.tolist())

    def test_refusals(self):
        import scipy.sparse as sparse

        with self.assertRaisesRegex(TypeError, "sparse"):
            arrays.from_scipy(np.eye(3))
        with self.assertRaisesRegex(ValueError, "square"):
            arrays.from_scipy(sparse.csr_matrix((2, 3)))
        with self.assertRaisesRegex(ValueError, "symmetric"):
            arrays.from_scipy(sparse.triu(self.matrix(), k=1))


if __name__ == "__main__":
    unittest.main()
