"""The clique-percolation baseline computes the covers NetworkX defines, without NetworkX's pairwise Python step."""

from __future__ import annotations

import random
import unittest

import igraph as ig
import networkx as nx
from networkx.algorithms.community.kclique import k_clique_communities as networkx_communities

from hedonic.experiments.overlapping import methods as om


class TestKCliqueCommunities(unittest.TestCase):
    def test_same_covers_as_networkx_on_random_graphs_and_any_block_size(self):
        rng = random.Random(7)
        for trial in range(24):
            n = rng.randint(8, 60)
            graph = ig.Graph.Erdos_Renyi(n=n, p=rng.choice([0.1, 0.2, 0.35])) if trial % 2 else ig.Graph.Barabasi(n, rng.randint(2, 5))
            graph.simplify()
            reference = om._networkx_graph(graph)
            for k in (3, 4, 5):
                ours = sorted(map(tuple, om.k_clique_communities(graph, k, block_cost=rng.choice([1, 40, 3e7]))))
                theirs = sorted(tuple(sorted(c)) for c in networkx_communities(reference, k))
                self.assertEqual(ours, theirs, (trial, k))

    def test_known_example_and_edge_cases(self):
        g = ig.Graph.Full(5)
        g.add_vertices(2)
        g.add_edges([(4, 5), (3, 5), (5, 6)])
        self.assertEqual(om.k_clique_communities(g, 4), [[0, 1, 2, 3, 4]])
        self.assertEqual(om.k_clique_communities(g, 3), [[0, 1, 2, 3, 4, 5]])  # the triangle 3-4-5 shares an edge with the K5
        self.assertEqual(om.k_clique_communities(ig.Graph(4), 3), [])
        with self.assertRaises(ValueError):
            om.k_clique_communities(g, 1)

    def test_the_adapter_applies_the_community_cap(self):
        g = ig.Graph.Full(3) + ig.Graph.Full(3)
        self.assertEqual(len(om._clique_percolation(g, 0, 0.0, 0, {"clique_size": 3})), 2)
        self.assertEqual(len(om._clique_percolation(g, 0, 0.0, 0, {"clique_size": 3, "max_communities": 1})), 1)


if __name__ == "__main__":
    unittest.main()
