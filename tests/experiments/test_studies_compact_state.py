"""The compact fractional state is bit-identical to the dict-per-vertex construction it replaced."""
from __future__ import annotations

import math
import random
import unittest
from collections import defaultdict

import igraph as ig

from hedonic.experiments.overlapping import robustness as rb


def reference_state(graph, memberships, edge_weights=None):
    """The construction ``build_fractional_state`` used before the compact state (kept as the oracle)."""
    n = graph.vcount()
    normalized = [tuple(sorted({int(label) for label in labels})) for labels in memberships]
    max_label = max(row[-1] for row in normalized)
    intensities, masses = [], [0.0] * (max_label + 1)
    for vertex_labels in normalized:
        intensity = 1.0 / math.sqrt(len(vertex_labels))
        intensities.append({label: intensity for label in vertex_labels})
        for label in vertex_labels:
            masses[label] += intensity
    weights = rb._edge_weights(graph, edge_weights)
    support = [defaultdict(float) for _ in range(n)]
    for edge_index, (first, second) in enumerate(graph.get_edgelist()):
        weight = weights[edge_index]
        if weight == 0:
            continue
        for label, intensity in intensities[first].items():
            support[second][label] += weight * intensity
        for label, intensity in intensities[second].items():
            support[first][label] += weight * intensity
    return normalized, masses, [dict(row) for row in support], intensities


def random_case(seed: int, n: int = 60, labels: int = 12, extra_edges: int = 150, loops: bool = True):
    rng = random.Random(seed)
    edges = {(min(a, b), max(a, b)) for a, b in ((rng.randrange(n), rng.randrange(n)) for _ in range(extra_edges)) if a != b}
    graph = ig.Graph(n=n, edges=sorted(edges))
    if loops:
        graph.add_edges([(0, 0), (3, 3)])
    memberships = []
    for _ in range(n):
        memberships.append(rng.sample(range(labels), rng.randint(1, 4)))
    used = {label for row in memberships for label in row}
    remap = {label: i for i, label in enumerate(sorted(used))}
    return graph, [[remap[label] for label in row] for row in memberships]


class TestCompactState(unittest.TestCase):
    def check(self, graph, memberships, **kwargs):
        state = rb.build_fractional_state(graph, memberships, **kwargs)
        weights = kwargs.get("edge_weights")
        normalized, masses, support, intensities = reference_state(graph, memberships, weights)
        self.assertEqual(state.memberships, normalized)
        self.assertEqual(state.masses, masses)  # exact
        self.assertEqual(len(state.support), len(support))
        for vertex in range(graph.vcount()):
            self.assertEqual(state.support[vertex], support[vertex], vertex)  # exact float equality, same labels
            self.assertEqual(state.intensities[vertex], intensities[vertex])
        return state

    def test_support_is_bit_identical_on_random_covers(self):
        for seed in range(6):
            graph, memberships = random_case(seed)
            self.check(graph, memberships)

    def test_small_blocks_do_not_change_the_sums(self):
        graph, memberships = random_case(11)
        self.check(graph, memberships, block=7)  # one block per few vertices
        self.check(graph, memberships, block=1)

    def test_weighted_edges_and_zero_weights(self):
        graph, memberships = random_case(5, loops=False)
        rng = random.Random(3)
        weights = [rng.choice([0.0, 0.1, 0.3, 1.0, 2.5, 1e-9]) for _ in range(graph.ecount())]
        self.check(graph, memberships, edge_weights=weights)

    def test_isolated_vertices_have_empty_rows(self):
        graph = ig.Graph(n=4, edges=[(0, 1)])
        state = self.check(graph, [[0], [0], [1], [1]])
        self.assertEqual(state.support[2], {})

    def test_best_response_is_unchanged(self):
        graph, memberships = random_case(2)
        state = rb.build_fractional_state(graph, memberships)
        cap = max(len(row) for row in memberships)
        regrets = [rb.best_response(state, v, 0.05, cap, True)["regret"] for v in range(graph.vcount())]
        self.assertTrue(all(r >= 0 for r in regrets))

    def test_the_arrays_are_much_smaller_than_the_dicts(self):
        graph, memberships = random_case(1, n=400, extra_edges=3000)
        state = rb.build_fractional_state(graph, memberships)
        self.assertEqual(state.support.nbytes, 8 * (graph.vcount() + 1) + 12 * state.support.entries)


if __name__ == "__main__":
    unittest.main()
