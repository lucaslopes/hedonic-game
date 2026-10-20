"""Archive-free smoke checks for the documented overlapping-method adapters.

These checks deliberately distinguish a runnable implementation from an
optional provider that is not installed.  In particular, an unavailable
AGMfit/MMSB adapter must fail closed rather than silently substituting the
Hedonic or clique_percolation detector. Infomap runs through stock igraph when its
standalone binding is absent. The 1.x multi-phase Hedonic variants are not
methods any more: naming one is a clear error, never a fallback.
"""

from __future__ import annotations

import importlib.util

import igraph as ig
import pytest

from hedonic.experiments.overlapping.methods import (
    LITERATURE_METHODS,
    METHODS,
    REMOVED_METHODS,
    MethodUnavailable,
    canonical_method_name,
    method_availability,
    resolve_methods,
    run_method,
    run_method_by_name,
)


@pytest.fixture()
def tiny_graph() -> ig.Graph:
    # A small graph with triangles and an overlap-friendly bridge.
    return ig.Graph(
        n=8,
        edges=[
            (0, 1), (1, 2), (2, 0),
            (2, 3),
            (3, 4), (4, 5), (5, 3),
            (1, 6), (6, 7), (7, 1),
        ],
        directed=False,
    )


@pytest.mark.parametrize(
    "name",
    [
        "hedonic_local",
        "clique_percolation",
        "demon",
        "infomap",
    ],
)
def test_installed_adapters_run_on_tiny_graph(tiny_graph: ig.Graph, name: str) -> None:
    if name == "demon" and importlib.util.find_spec("demon") is None:
        pytest.skip("the optional demon package is not installed in this interpreter")
    cover, metadata = run_method(
        METHODS[name],
        tiny_graph,
        max_memberships=3,
        resolution=tiny_graph.density(),
        seed=7,
    )
    assert cover
    assert all(0 <= vertex < tiny_graph.vcount() for community in cover for vertex in community)
    assert metadata["method"] == name


def test_isolated_link_clustering_adapter_runs_when_locked_worker_is_ready(
    tiny_graph: ig.Graph,
) -> None:
    """CDlib HLC is exercised through its isolated pinned environment."""
    availability = method_availability()["link_clustering"]
    if not availability["available"]:
        pytest.skip(availability.get("reason", "isolated CDlib worker unavailable"))
    cover, metadata = run_method(
        METHODS["link_clustering"],
        tiny_graph,
        max_memberships=3,
        resolution=tiny_graph.density(),
        seed=7,
    )
    assert cover
    assert metadata["method"] == "link_clustering"
    assert metadata["dependency"]["isolated_environment_ready"] is True
    assert metadata["dependency"]["implementation_attribute"] == (
        "hierarchical_link_community"
    )


def test_isolated_angel_adapter_runs_when_locked_worker_is_ready(
    tiny_graph: ig.Graph,
) -> None:
    availability = method_availability()["angel"]
    if not availability["available"]:
        pytest.skip(availability.get("reason", "isolated CDlib worker unavailable"))
    # ANGEL can legitimately emit no communities on a very sparse tiny graph;
    # the bounded DBLP integration test exercises the non-empty path. This
    # assertion checks the worker contract without turning an empty detector
    # result into a substitute algorithm.
    try:
        cover, metadata = run_method(
            METHODS["angel"],
            tiny_graph,
            max_memberships=3,
            resolution=tiny_graph.density(),
            seed=7,
            parameters={"threshold": 0.6},
        )
    except RuntimeError as exc:
        if "produced no valid communities" in str(exc):
            pytest.skip("ANGEL produced an empty cover on this tiny fixture")
        raise
    assert cover
    assert metadata["method"] == "angel"


@pytest.mark.parametrize("name", ["agmfit", "mmsb"])
def test_unavailable_historical_adapters_fail_closed(tiny_graph: ig.Graph, name: str) -> None:
    availability = method_availability()[name]
    if availability["available"]:
        pytest.skip(f"optional provider became available: {name}")
    with pytest.raises(MethodUnavailable):
        run_method(
            METHODS[name],
            tiny_graph,
            max_memberships=3,
            resolution=tiny_graph.density(),
            seed=7,
        )


@pytest.mark.parametrize("name", sorted(REMOVED_METHODS))
def test_removed_hedonic_variants_are_a_clear_error(tiny_graph: ig.Graph, name: str) -> None:
    assert name not in METHODS and name not in LITERATURE_METHODS
    with pytest.raises(ValueError, match="not available in hedonic 2.0"):
        resolve_methods(name)
    with pytest.raises(ValueError, match="not available in hedonic 2.0"):
        run_method_by_name(name, tiny_graph, max_memberships=2, resolution=0.1, seed=0)


def test_the_hedonic_key_names_the_local_adapter() -> None:
    assert canonical_method_name("hedonic") == "hedonic_local"
    assert canonical_method_name("Hedonic") == "hedonic_local"


def test_hedonic_local_is_seeded_and_returns_its_labelled_rows(tiny_graph: ig.Graph) -> None:
    runs = [
        run_method(METHODS["hedonic_local"], tiny_graph, max_memberships=2,
                   resolution=tiny_graph.density(), seed=11)
        for _ in range(2)
    ]
    assert runs[0][0] == runs[1][0]
    assert all(0 <= v < tiny_graph.vcount() for community in runs[0][0] for v in community)


def test_literature_registry_exposes_all_benchmark_names() -> None:
    # The 25 names of 1.x less community_hedonic and the three multi-phase variants.
    assert len(LITERATURE_METHODS) == 21
    assert not set(LITERATURE_METHODS) & REMOVED_METHODS
    assert set(LITERATURE_METHODS).issubset(METHODS)
    assert canonical_method_name("Neo-K-Means") == "neo_kmeans"
    assert canonical_method_name("Link Communities") == "link_clustering"
    assert [adapter.name for adapter in resolve_methods("all")] == list(LITERATURE_METHODS)
    assert [adapter.name for adapter in resolve_methods("CoDeSEG,Neo-K-Means") ] == [
        "codeseg",
        "neo_kmeans",
    ]


def test_dispatcher_adapter_requires_explicit_ground_truth(tiny_graph: ig.Graph) -> None:
    with pytest.raises(MethodUnavailable, match="ground_truth"):
        run_method_by_name(
            "codeseg",
            tiny_graph,
            max_memberships=3,
            resolution=tiny_graph.density(),
            seed=7,
        )


def test_dispatcher_adapter_runs_methods_without_ground_truth(tiny_graph: ig.Graph) -> None:
    cover, metadata = run_method_by_name(
        "Louvain",
        tiny_graph,
        max_memberships=3,
        resolution=tiny_graph.density(),
        seed=7,
    )
    assert cover
    assert metadata["method"] == "louvain"

