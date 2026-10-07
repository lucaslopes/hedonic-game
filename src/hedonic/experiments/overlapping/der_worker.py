"""Standalone CDlib DER worker materialized by ``codeseg-setup``."""

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
    with contextlib.redirect_stdout(io.StringIO()):
        from cdlib import algorithms
    request = json.load(sys.stdin)
    if not isinstance(request, dict):
        raise ValueError("DER request must be a JSON object")
    n = int(request["n"])
    walk_len = int(request.get("walk_len", 3))
    threshold = float(request.get("threshold", 1e-5))
    iter_bound = int(request.get("iter_bound", 50))
    if n < 0 or walk_len < 1 or threshold < 0 or iter_bound < 1:
        raise ValueError("invalid DER request bounds")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from((int(a), int(b)) for a, b in request.get("edges", []))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = algorithms.der(graph, walk_len, threshold, iter_bound)
    communities = [sorted(int(vertex) for vertex in body) for body in result.communities]
    print(json.dumps({
        "communities": communities,
        "metadata": {
            "cdlib_version": _version("cdlib"),
            "networkx_version": _version("networkx"),
            "numpy_version": _version("numpy"),
            "python_igraph_version": _version("python-igraph"),
            "python": platform.python_version(),
            "api": "cdlib.algorithms.der(g_original, walk_len, threshold, iter_bound)",
            "api_signature": str(inspect.signature(algorithms.der)),
            "walk_len": walk_len,
            "threshold": threshold,
            "iter_bound": iter_bound,
            "seed_support": False,
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
