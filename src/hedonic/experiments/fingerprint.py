"""The behaviour fingerprint of the loaded Hedonic partition detector.

A record names its build twice. ``hedonic_core_source_sha256`` says exactly
which C sources computed it. The fingerprint here says what those sources
compute: a fixed grid of partition calls (:func:`partition_calls`) run on the
loaded build, hashed over every returned membership, the exact bits of every
quality and slack, and the type of every refusal. Two builds with the same
fingerprint returned the same partitions bit for bit on every call of the grid,
so their rows may share a store (``store.method_identity`` keys Hedonic on it);
rows written by another source are still replayed before a store adopts them
(``store.check_adoption``).

The grid covers what the disjoint store calls: planted partitions, density and
fixed resolutions of both signs, both isolation policies, edge and vertex
weights, count limits, random and noisy starts, small and 64-bit seeds. Its
name (``GRID``) is part of the fingerprint: change the grid only by adding a
new one under a new name. It runs in a few seconds and is cached per compiled
core (``hedonic_core_sha256``) in ``~/.cache/hedonic/fingerprints``
(``HEDONIC_FINGERPRINT_CACHE`` names another folder; empty: no file cache).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

GRID = "partitions-v1"
_CACHE: dict[str, str] = {}


def _graphs():
    import igraph as ig

    out = [ig.Graph.Famous("Zachary"), ig.Graph.Famous("Petersen")]
    rng = random.Random(7)
    for _ in range(24):
        n = rng.choice([60, 120, 200])
        k = rng.choice([2, 3, 5])
        edges = [(a, b) for a in range(n) for b in range(a + 1, n)
                 if rng.random() < (0.25 if a % k == b % k else 0.02)]
        out.append(ig.Graph(n=n, edges=edges))
    return out, rng


def partition_calls():
    """Yield ``(key, game, kwargs)`` for every call of the grid, in a fixed order."""
    from hedonic import Game

    graphs, rng = _graphs()
    for gi, graph in enumerate(graphs):
        g = Game(graph.simplify())
        if g.ecount() == 0:
            continue
        n = g.vcount()
        weights = [rng.uniform(0.5, 2) for _ in range(g.ecount())]
        node_weights = [rng.uniform(0.5, 2) for _ in range(n)]
        # A noisy start: a planted-style labelling with a fifth of the vertices relabelled.
        start = [v % 3 for v in range(n)]
        for v in rng.sample(range(n), n // 5):
            start[v] = rng.randrange(3)
        options = {
            "plain": {},
            "edge_weights": {"edge_weights": weights},
            "node_weights": {"node_weights": node_weights},
            "max_total_communities": {"max_total_communities": 5},
            "n_communities": {"n_communities": 3},
            "max_communities": {"max_communities": 4},
            "noisy_start": {"initial_membership": start},
        }
        for resolution in ("density", 0.0, 0.05, 0.5, 1.0, -0.01):
            for isolation in (False, True):
                for name, extra in options.items():
                    for seed in (0, 1, 2, 2**63 + 12345):
                        kwargs = dict(extra, allow_isolation=isolation, seed=seed, max_memberships=1,
                                      resolution=g.density() if resolution == "density" else resolution)
                        yield (gi, resolution, isolation, name, seed), g, kwargs


def compute() -> str:
    """Run the grid on the loaded build: ``"<GRID>:<sha256>"``."""
    digest = hashlib.sha256(GRID.encode())
    for key, g, kwargs in partition_calls():
        try:
            p = g.community_hedonic(**kwargs)
            out = (tuple(p.membership), p._hedonic_native_quality.hex(),
                   float(p._hedonic_provenance["equilibrium_slack"]).hex())
        except (ValueError, TypeError) as exc:
            out = ("refused", type(exc).__name__)
        digest.update(repr((key, out)).encode())
    return f"{GRID}:{digest.hexdigest()}"


def _cache_dir() -> Path | None:
    if "HEDONIC_FINGERPRINT_CACHE" in os.environ:
        return Path(os.environ["HEDONIC_FINGERPRINT_CACHE"]) if os.environ["HEDONIC_FINGERPRINT_CACHE"] else None
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "hedonic" / "fingerprints"


def partition_fingerprint() -> str:
    """The fingerprint of the loaded build, computed once per compiled core and cached."""
    from hedonic.experiments.runtime import core_sha256

    binary = core_sha256()
    if binary in _CACHE:
        return _CACHE[binary]
    folder = _cache_dir()
    path = folder / f"{binary}.json" if folder is not None else None
    try:
        if path is None:
            raise OSError("no file cache")
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("grid") == GRID and cached.get("core_sha256") == binary:
            _CACHE[binary] = cached["fingerprint"]
            return _CACHE[binary]
    except (OSError, ValueError, KeyError):
        pass
    value = compute()
    _CACHE[binary] = value
    if path is None:
        return value
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"grid": GRID, "core_sha256": binary, "fingerprint": value}), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass  # a read-only cache only costs the next process a few seconds
    return value
