"""Run selected CDlib algorithms in the pinned isolated environment.

This process is deliberately tiny: the main hedonic runtime uses lucas-igraph,
while CDlib depends on the separate python-igraph distribution.  Requests and
results are JSON so a bounded benchmark can record the exact external method
without importing the conflicting dependency in-process.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import inspect
import io
import json
import platform
import sys

import networkx as nx


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _ego_splitting(graph: nx.Graph, resolution: float, seed: int) -> list[list[int]]:
    """Faithful small adapter to the public EgoSplitting reference algorithm.

    The reference implementation creates connected components in every
    ego-minus-ego graph, builds the persona graph, and partitions it with
    python-louvain.  Keeping the transformation here avoids importing the
    old repository (which only contains a Python 3.5 script) into the main
    runtime while preserving its algorithmic steps.
    """
    import community.community_louvain as community_louvain

    components: dict[int, dict[int, int]] = {}
    personalities: dict[int, list[int]] = {}
    next_persona = 0
    for node in graph.nodes:
        ego_minus = graph.subgraph(list(graph.neighbors(node)))
        mapping: dict[int, int] = {}
        personas: list[int] = []
        for component in nx.connected_components(ego_minus):
            persona = next_persona
            next_persona += 1
            personas.append(persona)
            for neighbor in component:
                mapping[int(neighbor)] = persona
        components[int(node)] = mapping
        personalities[int(node)] = personas
    persona_edges = []
    for left, right in graph.edges:
        left_persona = components[int(left)].get(int(right))
        right_persona = components[int(right)].get(int(left))
        if left_persona is not None and right_persona is not None:
            persona_edges.append((left_persona, right_persona))
    persona_graph = nx.Graph()
    persona_graph.add_nodes_from(range(next_persona))
    persona_graph.add_edges_from(persona_edges)
    partition = community_louvain.best_partition(
        persona_graph, resolution=float(resolution), random_state=int(seed)
    )
    cover: dict[int, set[int]] = {int(node): set() for node in graph.nodes}
    persona_owner = {
        persona: int(node)
        for node, personas in personalities.items()
        for persona in personas
    }
    for persona, label in partition.items():
        owner = persona_owner.get(int(persona))
        if owner is not None:
            cover[owner].add(int(label))
    labels: dict[int, list[int]] = {}
    for owner, memberships in cover.items():
        for label in memberships:
            labels.setdefault(label, []).append(owner)
    return [sorted(nodes) for nodes in labels.values() if nodes]


def main() -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        from cdlib import algorithms

    request = json.load(sys.stdin)
    if not isinstance(request, dict):
        raise ValueError("CDlib request must be a JSON object")
    method = str(request.get("method", "")).strip().lower()
    n = int(request["n"])
    if n < 0:
        raise ValueError("invalid vertex count")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(
        (int(left), int(right)) for left, right in request.get("edges", [])
    )
    if method == "angel":
        function = algorithms.angel
        result = function(
            graph,
            threshold=float(request.get("threshold", 0.8)),
            min_community_size=int(request.get("min_community_size", 3)),
        )
    elif method == "nise":
        function = algorithms.overlapping_seed_set_expansion
        seed_count = max(1, min(int(request.get("seed_count", 32)), n))
        seeds = [
            int(node)
            for node, _degree in sorted(
                graph.degree(), key=lambda item: (-int(item[1]), int(item[0]))
            )[:seed_count]
        ]
        result = function(
            graph,
            seeds=seeds,
            ninf=bool(request.get("ninf", True)),
            expansion=str(request.get("expansion", "ppr")),
            stopping=str(request.get("stopping", "cond")),
            nworkers=int(request.get("nworkers", 1)),
            nruns=int(request.get("nruns", 3)),
            alpha=float(request.get("alpha", 0.99)),
            delta=float(request.get("delta", 0.2)),
        )
    elif method == "ego_splitting":
        function = _ego_splitting
        communities = _ego_splitting(
            graph,
            resolution=float(request.get("resolution", 1.0)),
            seed=int(request.get("seed", 0)),
        )
        metadata = {
            "method": method,
            "cdlib_version": _version("cdlib"),
            "networkx_version": _version("networkx"),
            "numpy_version": _version("numpy"),
            "python_igraph_version": _version("python-igraph"),
            "python_louvain_version": _version("python-louvain"),
            "python": platform.python_version(),
            "api": "EgoSplitting persona graph + community_louvain.best_partition",
            "api_signature": str(inspect.signature(function)),
            "seed_support": True,
            "output_kind": "overlapping node cover",
        }
        print(json.dumps({"communities": communities, "metadata": metadata}, sort_keys=True))
        return 0
    elif method == "infomap":
        function = algorithms.infomap
        result = function(graph, flags=str(request.get("flags", "--two-level --silent")))
    else:
        raise ValueError(f"unsupported CDlib method: {method}")
    communities = [sorted(int(vertex) for vertex in community) for community in result.communities]
    print(
        json.dumps(
            {
                "communities": communities,
                "metadata": {
                    "method": method,
                    "cdlib_version": _version("cdlib"),
                    "networkx_version": _version("networkx"),
                    "numpy_version": _version("numpy"),
                    "python_igraph_version": _version("python-igraph"),
                    "python_louvain_version": _version("python-louvain"),
                    "python": platform.python_version(),
                    "api": f"cdlib.algorithms.{method}(g_original)",
                    "api_signature": str(inspect.signature(function)),
                    "seed_support": False,
                    "output_kind": "overlapping node cover",
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
