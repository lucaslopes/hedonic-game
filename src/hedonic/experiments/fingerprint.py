"""The behaviour fingerprints of the loaded Hedonic core.

A record names its build twice. ``hedonic_core_source_sha256`` says exactly
which C sources computed it. A fingerprint says what those sources compute: a
fixed grid of calls run on the loaded build, hashed over every returned value
bit for bit and the type of every refusal. There are three grids, one per
thing a record can hold:

``partitions-v1`` (:func:`partition_calls`)
    partition calls (``max_memberships=1``): memberships, quality and slack.
    The disjoint store keys Hedonic's rows on it (``store.method_identity``).
``covers-v1`` (:func:`cover_calls`)
    cover calls (``max_memberships`` 2, 3 and -1): the labelled rows
    (``Game.memberships``), quality and slack.
``audit-v1`` (:func:`audit_calls`)
    the fixed-cover stability audit (``stable_intervals`` and ``regrets``) on
    fixed covers: every interval end and every regret and tolerance, bit for bit.

Two builds with the same three fingerprints returned the same values on every
call of every grid; ``runtime.CORE_CLASSES`` maps such a triple to the source
digest records name, so a core of another source (a Rust port, a later C
build) that computes the same results keeps the records' identity. Rows or
units written by another source are still replayed before they are adopted
(``store.check_adoption``, ``replay``).

The partition grid covers what the disjoint store calls: planted partitions,
density and fixed resolutions of both signs, both isolation policies, edge and
vertex weights, count limits, random and noisy starts, small and 64-bit seeds.
The cover grid adds caps, starts with more labels than the cap and the
uncapped game; the audit grid caps at, above and far above a cover's largest
row, both isolation policies, edge weights, three tolerances and the pruned and
reference label sets, on covers with many labels per vertex so the windowed
search settles whole ranges of cells. A grid's name is part of its fingerprint:
change a grid only by adding a new one under a new name. Each runs in seconds
and is cached per compiled core (``hedonic_core_sha256``) in
``~/.cache/hedonic/fingerprints`` (``HEDONIC_FINGERPRINT_CACHE`` names another
folder; empty: no file cache).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

GRID = "partitions-v1"
COVER_GRID = "covers-v1"
AUDIT_GRID = "audit-v1"
GRIDS = (GRID, COVER_GRID, AUDIT_GRID)
_CACHE: dict[tuple[str, str], str] = {}


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
    """Yield ``(key, game, kwargs)`` for every call of the partition grid, in a fixed order."""
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


def _overlapping_graphs():
    """Zachary and planted graphs whose blocks overlap: a vertex of two blocks has edges into both."""
    import igraph as ig

    out = [ig.Graph.Famous("Zachary")]
    rng = random.Random(11)
    for _ in range(7):
        n = rng.choice([40, 70, 100])
        k = rng.choice([3, 4])
        blocks = [{v % k} | ({rng.randrange(k)} if rng.random() < 0.3 else set()) for v in range(n)]
        edges = [(a, b) for a in range(n) for b in range(a + 1, n)
                 if rng.random() < (0.3 if blocks[a] & blocks[b] else 0.03)]
        out.append((ig.Graph(n=n, edges=edges), blocks))
    return [(out[0], [{v % 2} for v in range(out[0].vcount())])] + out[1:], rng


def cover_calls():
    """Yield ``(key, game, kwargs)`` for every call of the cover grid, in a fixed order."""
    from hedonic import Game

    graphs, rng = _overlapping_graphs()
    for gi, (graph, blocks) in enumerate(graphs):
        g = Game(graph.simplify())
        n = g.vcount()
        weights = [rng.uniform(0.5, 2) for _ in range(g.ecount())]
        node_weights = [rng.uniform(0.5, 2) for _ in range(n)]
        # A start of one to three labels per vertex (the planted blocks, a tenth of them disturbed), and a start
        # of five labels per vertex, more than every finite cap of the grid: a vertex keeps its preferred ones.
        start = [sorted(block | ({rng.randrange(5)} if rng.random() < 0.1 else set())) for block in blocks]
        wide = [sorted(rng.sample(range(8), 5)) for _ in range(n)]
        options = {
            "plain": {},
            "edge_weights": {"edge_weights": weights},
            "node_weights": {"node_weights": node_weights},
            "max_total_communities": {"max_total_communities": 4},
            "max_communities": {"max_communities": 3},
            "cover_start": {"initial_membership": start},
            "wide_start": {"initial_membership": wide},
            "flat_start": {"initial_membership": [min(block) for block in blocks]},
        }
        for cap in (2, 3, -1):
            for resolution in ("density", 0.0, 0.1, 0.5, 1.0, -0.01):
                for isolation in (False, True):
                    for name, extra in options.items():
                        for seed in (0, 1, 2**63 + 12345):
                            kwargs = dict(extra, allow_isolation=isolation, seed=seed, max_memberships=cap,
                                          resolution=g.density() if resolution == "density" else resolution)
                            yield (gi, cap, resolution, isolation, name, seed), g, kwargs


AUDIT_TOLERANCES = ((1e-10, 1e-9), (0.0, 0.0), (1e-6, 1e-3))
AUDIT_RESOLUTIONS = (0.0, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.25, 1 / 3, 0.5, 0.6, 0.75, 0.9, 0.99, 1.0)


def audit_calls():
    """Yield ``(key, game, kwargs)`` for every audit of the grid, in a fixed order: ``kwargs`` are those of
    ``hedonic._audit.stable_intervals`` (``regrets`` takes them too, with :data:`AUDIT_RESOLUTIONS`)."""
    from hedonic import Game

    graphs, rng = _overlapping_graphs()
    for gi, (graph, blocks) in enumerate(graphs):
        g = Game(graph.simplify())
        n = g.vcount()
        weights = [rng.uniform(0.5, 2) for _ in range(g.ecount())]
        covers = {
            "planted": [sorted(block) for block in blocks],
            "flat": [min(block) for block in blocks],
            # Many labels per vertex, on many communities: wide windows and many cells between score crossings.
            "dense": [sorted(rng.sample(range(12), rng.randint(1, 5))) for _ in range(n)],
        }
        for name, cover in covers.items():
            own = max(len(row) if isinstance(row, list) else 1 for row in cover)
            for cap_name, cap in (("own", None), ("above", own + 1), ("uncapped", n)):
                for weighted in (False, True):
                    for isolation in (False, True):
                        for atol, rtol in AUDIT_TOLERANCES:
                            for prune in (True, False):
                                kwargs = dict(memberships=cover, max_memberships=cap, allow_isolation=isolation,
                                              edge_weights=weights if weighted else None, atol=atol, rtol=rtol,
                                              prune=prune)
                                yield (gi, name, cap_name, weighted, isolation, atol, rtol, prune), g, kwargs


def compute() -> str:
    """Run the partition grid on the loaded build: ``"<GRID>:<sha256>"``."""
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


def compute_covers() -> str:
    """Run the cover grid on the loaded build: ``"<COVER_GRID>:<sha256>"``."""
    digest = hashlib.sha256(COVER_GRID.encode())
    for key, g, kwargs in cover_calls():
        try:
            c = g.community_hedonic(**kwargs)
            out = (tuple(tuple(int(label) for label in row) for row in g.memberships),
                   c._hedonic_native_quality.hex(), float(c._hedonic_provenance["equilibrium_slack"]).hex())
        except (ValueError, TypeError) as exc:
            out = ("refused", type(exc).__name__)
        digest.update(repr((key, out)).encode())
    return f"{COVER_GRID}:{digest.hexdigest()}"


def compute_audit() -> str:
    """Run the audit grid on the loaded build: ``"<AUDIT_GRID>:<sha256>"``."""
    import numpy as np

    from hedonic import _audit

    digest = hashlib.sha256(AUDIT_GRID.encode())
    for key, g, kwargs in audit_calls():
        try:
            intervals = _audit.stable_intervals(g, **kwargs)
            out = tuple(tuple((float(lo).hex(), float(hi).hex()) for lo, hi in row) for row in intervals)
        except (ValueError, TypeError) as exc:
            out = ("refused", type(exc).__name__)
        digest.update(repr((key, "intervals", out)).encode())
        try:
            regret, tolerance = _audit.regrets(g, AUDIT_RESOLUTIONS, **kwargs)
            out = (hashlib.sha256(np.ascontiguousarray(regret, dtype="<f8").tobytes()).hexdigest(),
                   hashlib.sha256(np.ascontiguousarray(tolerance, dtype="<f8").tobytes()).hexdigest())
        except (ValueError, TypeError) as exc:
            out = ("refused", type(exc).__name__)
        digest.update(repr((key, "regrets", out)).encode())
    return f"{AUDIT_GRID}:{digest.hexdigest()}"


# What computes each grid, by name (looked up at call time, so a test can patch a function).
COMPUTE = {GRID: "compute", COVER_GRID: "compute_covers", AUDIT_GRID: "compute_audit"}


def _cache_dir() -> Path | None:
    if "HEDONIC_FINGERPRINT_CACHE" in os.environ:
        return Path(os.environ["HEDONIC_FINGERPRINT_CACHE"]) if os.environ["HEDONIC_FINGERPRINT_CACHE"] else None
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "hedonic" / "fingerprints"


def _read_cached(path: Path, binary: str) -> dict[str, str]:
    """The fingerprints a cache file holds for this core, by grid. A file of the first format (one grid:
    ``grid``/``fingerprint``) is read too."""
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(cached, dict) or cached.get("core_sha256") != binary:
        return {}
    found = {grid: value for grid, value in (cached.get("fingerprints") or {}).items()
             if isinstance(value, str) and value.startswith(f"{grid}:")}
    legacy = cached.get("fingerprint")
    if cached.get("grid") in GRIDS and isinstance(legacy, str) and legacy.startswith(f"{cached['grid']}:"):
        found.setdefault(cached["grid"], legacy)
    return found


def _write_cached(path: Path, binary: str, values: dict[str, str]) -> None:
    try:
        merged = {**_read_cached(path, binary), **values}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        # ``grid``/``fingerprint`` keep the partition grid where hedonic 2.0.0rc11 and earlier read it.
        payload = {"core_sha256": binary, "fingerprints": dict(sorted(merged.items()))}
        if GRID in merged:
            payload.update(grid=GRID, fingerprint=merged[GRID])
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass  # a read-only cache only costs the next process a few seconds


def fingerprint(grid: str = GRID) -> str:
    """One grid's fingerprint of the loaded build, computed once per compiled core and cached."""
    from hedonic.experiments.runtime import core_sha256

    if grid not in COMPUTE:
        raise ValueError(f"unknown fingerprint grid {grid!r}; choose from {', '.join(GRIDS)}")
    binary = core_sha256()
    if (binary, grid) in _CACHE:
        return _CACHE[binary, grid]
    folder = _cache_dir()
    path = folder / f"{binary}.json" if folder is not None else None
    cached = _read_cached(path, binary) if path is not None else {}
    if grid in cached:
        _CACHE[binary, grid] = cached[grid]
        return cached[grid]
    value = globals()[COMPUTE[grid]]()
    _CACHE[binary, grid] = value
    if path is not None:
        _write_cached(path, binary, {grid: value})
    return value


def partition_fingerprint() -> str:
    """The partition grid's fingerprint of the loaded build (what the disjoint store keys Hedonic's rows on)."""
    return fingerprint(GRID)


def cover_fingerprint() -> str:
    return fingerprint(COVER_GRID)


def audit_fingerprint() -> str:
    return fingerprint(AUDIT_GRID)


def fingerprints() -> tuple[str, str, str]:
    """The three fingerprints of the loaded build, in the order of :data:`GRIDS` (``runtime.CORE_CLASSES``' keys)."""
    return tuple(fingerprint(grid) for grid in GRIDS)
