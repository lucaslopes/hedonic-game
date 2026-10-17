"""Run CDlib's hierarchical Link Communities implementation in isolation.

The main project intentionally does not import CDlib because its dependency
closure includes the unrelated ``python-igraph`` distribution.  This worker
shares the locked ``tools/slpa_env`` project with the SLPA worker and emits a
node cover converted from CDlib's edge-community output.
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


def main() -> int:
    # CDlib emits optional-method notes while importing its algorithm module;
    # keep stdout reserved for one machine-readable terminal result.
    with contextlib.redirect_stdout(io.StringIO()):
        from cdlib import algorithms

    request = json.load(sys.stdin)
    if not isinstance(request, dict):
        raise ValueError("Link Clustering request must be a JSON object")
    n = int(request["n"])
    edges = request.get("edges", [])
    if n < 0:
        raise ValueError("invalid Link Clustering vertex count")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from((int(left), int(right)) for left, right in edges)
    result = algorithms.hierarchical_link_community(graph)

    communities = []
    for edge_community in result.communities:
        vertices = {
            int(vertex)
            for edge in edge_community
            for vertex in edge
        }
        if vertices:
            communities.append(sorted(vertices))
    print(
        json.dumps(
            {
                "communities": communities,
                "metadata": {
                    "cdlib_version": _version("cdlib"),
                    "networkx_version": _version("networkx"),
                    "python": platform.python_version(),
                    "api": "cdlib.algorithms.hierarchical_link_community(g_original)",
                    "api_signature": str(
                        inspect.signature(algorithms.hierarchical_link_community)
                    ),
                    "output_conversion": "edge communities projected to unique node sets",
                    "seed_support": False,
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
