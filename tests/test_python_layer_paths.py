"""The Python layer's shortcuts give what the long way gives.

``normalize_rows`` reads plain ``int`` memberships without its per-label
checks (``_plain_rows``), a default start is left to the core (``start=None``
is the singletons the Game used to build), the self-loop check asks igraph's
``is_simple`` first, and large result lists are built with the cyclic
collector held (``gc_paused``). None of it may change a row, a message, a
result bit or the provenance.
"""

from __future__ import annotations

import gc
import random
from unittest.mock import patch

import igraph as ig
import numpy as np
import pytest

from hedonic import Game, _core, _native
from hedonic._game import memberships as _memberships
from hedonic._game.helpers import gc_paused
from hedonic._game.memberships import normalize_rows


def _outcome(fn):
    try:
        return "ok", fn()
    except Exception as exc:  # the same type and message on both paths
        return type(exc).__name__, str(exc)


def _both(membership, n, **kwargs):
    fast = _outcome(lambda: normalize_rows(membership, n, **kwargs))
    with patch.object(_memberships, "_plain_rows", return_value=None):
        slow = _outcome(lambda: normalize_rows(membership, n, **kwargs))
    return fast, slow


def _memberships_of(rng, n):
    """Flat and nested memberships, valid and not, of ``n`` vertices."""
    k = rng.randint(1, n)
    flat = [rng.randrange(k) for _ in range(n)]
    yield flat
    yield tuple(flat)
    yield [[label] for label in flat]
    rows = [rng.sample(range(k), rng.randint(1, min(k, 4))) for _ in range(n)]
    yield rows
    yield [tuple(row) for row in rows]
    yield [list(row) if v % 2 else tuple(row) for v, row in enumerate(rows)]
    yield [row + row[:1] for row in rows]                       # a repeated label
    yield [[label + 3 for label in row] for row in rows]        # gaps below
    yield [[2 * label for label in row] for row in rows]        # gaps between
    yield np.array(flat)
    yield [np.int64(label) for label in flat]                   # not plain ints
    yield [[np.int64(label) for label in row] for row in rows]
    bad = list(flat)
    bad[rng.randrange(n)] = rng.choice([-1, True, 1.0, "x", None, [0]])
    yield bad
    nested = [list(row) for row in rows]
    v = rng.randrange(n)
    nested[v] = rng.choice([[], [-1], [True], [0.0], ["x"], 0, None, [[0]]])
    yield nested


def test_the_plain_path_gives_the_rows_and_errors_of_the_checked_path():
    rng = random.Random(7)
    seen = set()
    for case in range(300):
        n = rng.randint(1, 12)
        for membership in _memberships_of(rng, n):
            for kwargs in ({}, {"collapse_duplicate_labels": True},
                           {"renumber_labels": True},
                           {"collapse_duplicate_labels": True, "renumber_labels": True},
                           {"max_memberships": rng.randint(1, 4)},
                           {"max_memberships": rng.randint(1, 4), "collapse_duplicate_labels": True}):
                fast, slow = _both(membership, n, **kwargs)
                assert fast == slow, (membership, kwargs)
                assert fast[0] != "ok" or all(type(row) is list for row in fast[1])
                seen.add(fast[0])
    assert {"ok", "ValueError"} <= seen


def test_the_plain_path_reads_the_memberships_it_is_for():
    calls = []
    real = _memberships._plain_rows

    def spy(*args):
        out = real(*args)
        calls.append(out is not None)
        return out

    with patch.object(_memberships, "_plain_rows", spy):
        normalize_rows([0, 1, 1, 0], 4)
        normalize_rows([[0], [0, 1], (1,), [1]], 4)
        normalize_rows(np.array([0, 1, 1, 0]), 4)
        normalize_rows([[0, 0], [1], [1], [0]], 4, collapse_duplicate_labels=True)
    assert calls == [True, True, True, True]


def test_gc_paused_restores_the_collector_as_it_found_it():
    assert gc.isenabled()
    with gc_paused():
        assert not gc.isenabled()
    assert gc.isenabled()
    with pytest.raises(RuntimeError):
        with gc_paused():
            raise RuntimeError
    assert gc.isenabled()
    gc.disable()
    try:
        with gc_paused():
            assert not gc.isenabled()
        assert not gc.isenabled()
    finally:
        gc.enable()


def _graphs():
    yield ig.Graph.Famous("Zachary")
    yield ig.Graph.Lattice([15, 15], circular=False)
    rng = random.Random(3)
    blocks = [(u, v) for u in range(200) for v in range(u + 1, 200)
              if rng.random() < (0.1 if u // 100 == v // 100 else 0.01)]
    yield ig.Graph(200, blocks)


@pytest.mark.parametrize("max_memberships", [1, 3, -1])
def test_the_core_default_start_is_the_singletons(max_memberships):
    """start=None and the start the Game built before (list(range(n)), or
    [[v] for v in range(n)] for a cover) give the same bits."""
    for graph in _graphs():
        n = graph.vcount()
        cap = n if max_memberships == -1 else max_memberships
        weights = [1.0 + (e % 5) / 4 for e in range(graph.ecount())]
        singletons = list(range(n)) if cap == 1 else [[v] for v in range(n)]
        for seed in range(3):
            for options in ({}, {"edge_weights": np.array(weights)}, {"allow_isolation": True}):
                given = _core.detect(graph, random.Random(seed), 0.05, max_memberships=cap,
                                     start=singletons, **options)
                default = _core.detect(graph, random.Random(seed), 0.05, max_memberships=cap,
                                       start=None, **options)
                assert repr(given) == repr(default)


def test_a_default_start_reaches_the_core_as_none():
    seen = []
    real = _native.community_hedonic

    def spy(game, **kwargs):
        seen.append(kwargs["initial_membership"])
        return real(game, **kwargs)

    game = Game(ig.Graph.Famous("Zachary"))
    with patch.object(_native, "community_hedonic", spy):
        game.community_hedonic(seed=1)
        game.community_hedonic(seed=1, max_memberships=2)
        game.community_hedonic(seed=1, max_communities=4)
    assert seen[:2] == [None, None]
    assert seen[2] is not None
    result = game.community_hedonic(seed=1, max_memberships=2)
    assert result._hedonic_provenance["initialization"] == "default"
    assert result._hedonic_provenance["occupied_communities"] == len(
        {label for row in game.memberships for label in row})


def test_loops_are_refused_and_multiple_edges_are_not():
    multi = Game(ig.Graph([(0, 1), (0, 1), (1, 2), (2, 0)]))
    assert not multi.is_simple()
    multi.community_hedonic(seed=0)
    for edges in ([(0, 1), (1, 1), (1, 2)], [(0, 1), (0, 1), (2, 2), (1, 2)]):
        with pytest.raises(ValueError, match="without self-loops"):
            Game(ig.Graph(edges)).community_hedonic(seed=0)
