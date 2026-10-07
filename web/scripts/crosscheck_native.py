#!/usr/bin/env python3
"""Record what the native detector returns on the explainer's four-vertex graph.

The browser model enumerates every partition and marks the sinks of the
better-response dynamics. This script runs ``Game.community_hedonic`` (the
native lucas-igraph implementation) on the same graph and writes the returned
partitions to a JSON fixture. ``src/model/__tests__/nativeCrosscheck.test.ts``
then checks that every native result with ``allow_isolation=True`` is a sink
of the browser model at the same resolution.

It needs the Python package and its native dependency, so it is *not* part of
``npm run docs:generate``. Re-run it manually after a native release::

    uv run python web/scripts/crosscheck_native.py
"""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import igraph as ig

from hedonic import Game

OUTPUT = Path(__file__).resolve().parents[1] / "src/model/__tests__/fixtures/native-crosscheck.json"
GAMMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
STARTS = {"singletons": None, "grand": [0, 0, 0, 0]}
# The native visit order is random; a fixed seed makes the fixture
# reproducible. Several (gamma, start) cells have more than one reachable sink.
SEED = 0


def canonical(labels: list[int]) -> str:
    relabel: dict[int, int] = {}
    return "".join(str(relabel.setdefault(label, len(relabel))) for label in labels)


def main() -> int:
    graph = Game(ig.Graph(n=4, edges=[(0, 1), (0, 2), (0, 3), (1, 2)]))
    runs = []
    for gamma in GAMMAS:
        for allow_isolation in (False, True):
            for start_name, start in STARTS.items():
                result = graph.community_hedonic(
                    initial_membership=start,
                    resolution=gamma,
                    max_memberships=1,
                    n_iterations=-1,
                    allow_isolation=allow_isolation,
                    local_move_only=True,
                    seed=SEED,
                )
                runs.append(
                    {
                        "gamma": gamma,
                        "allowIsolation": allow_isolation,
                        "start": start_name,
                        "partition": canonical(list(result.membership)),
                    }
                )
    payload = {
        "graph": {"n": 4, "edges": [[0, 1], [0, 2], [0, 3], [1, 2]]},
        "versions": {"hedonic": version("hedonic"), "lucas-igraph": version("lucas-igraph")},
        "call": f"Game.community_hedonic(initial_membership=start, resolution=gamma, max_memberships=1, n_iterations=-1, allow_isolation=..., local_move_only=True, seed={SEED})",
        "runs": runs,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
