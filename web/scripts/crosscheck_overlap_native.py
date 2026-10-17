#!/usr/bin/env python3
"""Record what the native core returns on the overlap explainer's graph.

The browser model in ``src/model/overlap/`` ports the C core's cover local
moving (``native/core/cover/``) and its stability audit
(``native/core/check/audit.c``). This script runs the package on the same
graph (two cliques of four sharing the edge 2-3) and writes a JSON fixture
that ``src/model/__tests__/overlapNative.test.ts`` compares against:

* ``runs``: ``Game.community_hedonic(..., max_memberships=M, seed=s)`` from
  the singleton start and from the one-community start, for several
  resolutions, caps and both isolation policies. Each run also records the
  accepted moves of the same call traced by ``hedonic._core._trace_cover``
  (the package's test entry point: the same core, the same
  ``random.Random(seed)``), in the core's internal labels, and the stable
  intervals of the returned cover from ``Game.stable_intervals``.
* ``audits``: ``Game.stable_intervals`` and ``Game.regrets`` of a few fixed
  covers.

It needs the Python package with its compiled core, so it is *not* part of
``npm run docs:generate``. Re-run it after a change to the core::

    uv run python web/scripts/crosscheck_overlap_native.py
"""

from __future__ import annotations

import json
import random
from array import array
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import igraph as ig

from hedonic import Game, _core

OUTPUT = Path(__file__).resolve().parents[1] / "src/model/__tests__/fixtures/overlap-native.json"
N = 6
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
GAMMAS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.8, 1.0]
CAPS = [1, 2, 3]
SEEDS = [0, 1, 2]
STARTS = {"singletons": None, "together": [[0]] * N}
AUDIT_COVERS = {
    "natural": [[0], [0], [0, 1], [0, 1], [1], [1]],
    "halves": [[0], [0], [0], [1], [1], [1]],
    "together": [[0]] * N,
    "singletons": [[v] for v in range(N)],
    "doubled": [[0, 1], [0, 1], [0, 1], [0, 1], [1, 2], [1, 2]],
}
GRID = [i / 10 for i in range(11)]


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def intervals(rows) -> list[list[list[float]]]:
    return [[[lo, hi] for lo, hi in row] for row in rows]


def main() -> int:
    game = Game(ig.Graph(n=N, edges=EDGES))
    flat = array("q", [x for edge in EDGES for x in edge])
    runs = []
    for start_name, start in STARTS.items():
        for gamma in GAMMAS:
            for cap in CAPS:
                for isolation in (False, True):
                    for seed in SEEDS:
                        cover = game.community_hedonic(
                            initial_membership=start, resolution=gamma, max_memberships=cap,
                            allow_isolation=isolation, seed=seed)
                        rows = [sorted(row) for row in game.memberships]
                        traced, _, quality, record = _core._trace_cover(
                            N, flat, random.Random(seed), memberships=start, max_memberships=cap,
                            resolution=gamma, allow_isolation=isolation)
                        assert [list(row) for row in traced] == rows, "trace and Game disagree"
                        audit = game.stable_intervals(rows, max_memberships=cap,
                                                      allow_isolation=isolation)
                        runs.append({
                            "start": start_name, "gamma": gamma, "cap": cap,
                            "allowIsolation": isolation, "seed": seed,
                            "cover": rows,
                            "quality": cover._hedonic_native_quality,
                            "moves": [[int(move[0]), [int(c) for c in move[1]]]
                                      for move in record["moves"]],
                            "intervals": intervals(audit),
                        })
    audits = []
    for name, rows in AUDIT_COVERS.items():
        largest = max(len(row) for row in rows)
        for cap in sorted({largest, 2, 3}):
            if cap < largest:
                continue
            for isolation in (False, True):
                found = game.stable_intervals(rows, max_memberships=cap, allow_isolation=isolation)
                grid = game.regrets(GRID, rows, max_memberships=cap, allow_isolation=isolation)
                audits.append({
                    "cover": name, "rows": rows, "cap": cap, "allowIsolation": isolation,
                    "intervals": intervals(found),
                    "regret": grid.regret.tolist(), "tolerance": grid.tolerance.tolist(),
                })
    payload = {
        "graph": {"n": N, "edges": [list(edge) for edge in EDGES]},
        "versions": {"hedonic": package_version("hedonic"), "igraph": package_version("igraph")},
        "calls": {
            "run": "Game.community_hedonic(initial_membership=start, resolution=gamma, "
                   "max_memberships=cap, allow_isolation=..., seed=seed)",
            "moves": "hedonic._core._trace_cover(n, edges, random.Random(seed), memberships=start, "
                     "max_memberships=cap, resolution=gamma, allow_isolation=...)['moves']",
            "intervals": "Game.stable_intervals(cover, max_memberships=cap, allow_isolation=...)",
            "regrets": f"Game.regrets({GRID}, cover, max_memberships=cap, allow_isolation=...)",
        },
        "grid": GRID,
        "runs": runs,
        "audits": audits,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(runs)} runs, {len(audits)} audits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
