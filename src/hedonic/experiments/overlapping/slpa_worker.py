"""Standalone CDlib SLPA worker materialized by ``codeseg-setup``.

This file intentionally has no import from ``hedonic`` so it can run inside a
dedicated environment whose ``python-igraph`` dependency cannot contaminate
the main lucas-igraph process.
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
import numpy as np


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
        raise ValueError("SLPA request must be a JSON object")
    n = int(request["n"])
    iterations = int(request["iterations"])
    threshold = float(request["threshold"])
    seed = int(request["seed"])
    if n < 0 or iterations < 1 or not 0.0 <= threshold <= 1.0:
        raise ValueError("invalid SLPA request bounds")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from((int(a), int(b)) for a, b in request.get("edges", []))
    np.random.seed(seed)
    result = algorithms.slpa(graph, t=iterations, r=threshold)
    communities = [sorted(int(vertex) for vertex in body) for body in result.communities]
    print(json.dumps({
        "communities": communities,
        "metadata": {
            "cdlib_version": _version("cdlib"),
            "networkx_version": _version("networkx"),
            "numpy_version": _version("numpy"),
            "python_igraph_version": _version("python-igraph"),
            "python": platform.python_version(),
            "api": "cdlib.algorithms.slpa(g_original, t, r)",
            "api_signature": str(inspect.signature(algorithms.slpa)),
            "seed_support": False,
            "rng_semantics": "numpy.random.seed(seed) in isolated worker",
            "iterations": iterations,
            "threshold": threshold,
            "seed": seed,
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
