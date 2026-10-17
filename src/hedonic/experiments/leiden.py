"""Full Leiden (local moving, refinement, aggregation) for the experiments.

``Game.community_hedonic`` runs local moving only: Leiden's refinement and
aggregation phases are not part of hedonic. Experiments that compare against
full Leiden, or use it as a phase, call stock igraph's
``Graph.community_leiden`` with the CPM objective through :func:`leiden`,
which seeds it the way ``community_hedonic(seed=...)`` seeds the core.

Results are igraph's own (the stock python-igraph wheel), so they need not
equal what the hedonic 1.x Leiden path returned; records made with it are not
reused.
"""

from __future__ import annotations

import contextlib
import random

import igraph as ig

# The name records and provenance use for this detector.
LEIDEN_IDENTITY = f"igraph.community_leiden(CPM)/igraph-{ig.__version__}"


def leiden(
    graph,
    *,
    resolution: float | None = None,
    initial_membership=None,
    n_iterations: int = -1,
    beta: float = 0.01,
    edge_weights=None,
    node_weights=None,
    seed: int | None = None,
) -> ig.VertexClustering:
    """CPM Leiden on ``graph`` (an undirected ``igraph.Graph`` or ``Game``).

    resolution: CPM gamma; the graph density if None (as ``community_hedonic``).
    initial_membership: a flat start partition, or None for singletons.
    n_iterations: Leiden iterations; negative runs until one changes nothing.
    edge_weights, node_weights: sequences or attribute names; None for ones.
    seed: run under ``random.Random(seed)`` as igraph's generator, then set
        igraph's default (the ``random`` module) back; stock igraph cannot
        report a generator set earlier. None keeps the current generator.
    """
    if graph.is_directed():
        raise ValueError("leiden requires an undirected graph")
    if resolution is None:
        resolution = graph.density()
    if initial_membership is not None:
        initial_membership = [int(label) for label in initial_membership]

    def run():
        return ig.Graph.community_leiden(
            graph,
            objective_function="CPM",
            weights=edge_weights,
            resolution=float(resolution),
            beta=float(beta),
            initial_membership=initial_membership,
            n_iterations=int(n_iterations),
            node_weights=node_weights,
        )

    if seed is None:
        return run()
    with seeded_igraph_rng(seed):
        return run()


@contextlib.contextmanager
def seeded_igraph_rng(seed: int):
    """Run the block with ``random.Random(seed)`` as igraph's generator.

    Stock python-igraph cannot report the generator set before, so igraph's
    default (the ``random`` module) is set back afterwards.
    """
    ig.set_random_number_generator(random.Random(seed))
    try:
        yield
    finally:
        ig.set_random_number_generator(random)
