"""Run the pinned CDlib SLPA implementation in an isolated worker.

The main project uses ``lucas-igraph``.  CDlib 0.4.0 declares the unrelated
``python-igraph`` distribution, so importing it in the project environment
could overwrite the native ``igraph`` module.  This worker is intentionally
standalone and is invoked through ``uv run --project tools/slpa_env --locked``.

Input is one JSON object on stdin::

    {"n": 4, "edges": [[0, 1]], "iterations": 20,
     "threshold": 0.1, "seed": 0}

Output is one JSON object on stdout.  CDlib 0.4.0 exposes no seed argument;
the adapter therefore seeds NumPy's legacy global RNG immediately before the
call and reports that semantic explicitly in the returned receipt.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import inspect
import io
import json
import platform
import sys
from typing import Any

import networkx as nx
import numpy as np


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    # CDlib prints optional-method availability notes while importing its
    # algorithms package.  Keep stdout a single machine-readable JSON object.
    with contextlib.redirect_stdout(io.StringIO()):
        from cdlib import algorithms

    request = json.load(sys.stdin)
    if not isinstance(request, dict):
        raise ValueError("SLPA request must be a JSON object")
    n = int(request["n"])
    edges = request.get("edges", [])
    iterations = int(request["iterations"])
    threshold = float(request["threshold"])
    seed = int(request["seed"])
    if n < 0 or iterations < 1 or not 0.0 <= threshold <= 1.0:
        raise ValueError("invalid SLPA request bounds")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from((int(left), int(right)) for left, right in edges)

    # CDlib 0.4.0 delegates to numpy.random.shuffle/multinomial and has no
    # seed parameter.  Seeding in this isolated process gives deterministic
    # results without mutating the caller's RNG state.
    np.random.seed(seed)
    result = algorithms.slpa(graph, t=iterations, r=threshold)
    communities = [sorted(int(vertex) for vertex in body) for body in result.communities]
    print(
        json.dumps(
            {
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
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
