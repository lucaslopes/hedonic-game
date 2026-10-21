"""Argument buffers for the fixed-cover C audit (no Python scoring).

:func:`state` and :func:`vertex_indices` take a vertex count and any graph the
core reads (an ``igraph.Graph``, or the edge view of :mod:`hedonic.arrays`), so
``Game`` and the array path share them.
"""
from itertools import chain
from numbers import Integral

import numpy as np

from . import _core
from ._game.memberships import normalize_rows
from ._native import _edge_weight_buffer
from .stability import RegretResult


def state(graph, n, memberships, max_memberships, read_weights):
    """The core's audit state of ``memberships`` on ``graph`` (``n`` vertices),
    and the deviation cap; ``read_weights()`` gives the edge weights (or None),
    read after the cover is checked."""
    if n == 0 and len(memberships) == 0:
        rows = []
        if max_memberships is not None and (
            isinstance(max_memberships, bool) or not isinstance(max_memberships, Integral)
            or max_memberships < 1
        ):
            raise ValueError("max_memberships must be an integer >= 1")
    else:
        rows = normalize_rows(
            memberships, n, max_memberships=max_memberships,
            collapse_duplicate_labels=True, renumber_labels=True,
        )
    cap = max((len(row) for row in rows), default=1) if max_memberships is None else int(max_memberships)
    offsets = np.empty(len(rows) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum([len(row) for row in rows], out=offsets[1:])
    labels = np.fromiter(chain.from_iterable(rows), dtype=np.int64, count=int(offsets[-1]))
    weights = read_weights()
    return _core._audit_prepare(graph, offsets, labels, weights), cap


def _prepare(game, memberships=None, max_memberships=None, edge_weights=None):
    if game.is_directed():
        raise ValueError("stability audits require an undirected graph")
    if memberships is None:
        memberships = game.memberships
    if memberships is None:
        raise ValueError("Game has no memberships; supply a cover or run community_hedonic first")
    return state(game, game.vcount(), memberships, max_memberships,
                 lambda: _edge_weight_buffer(game, edge_weights, nonnegative=True))


def vertex_indices(n, vertices):
    """The audited vertices as int64 indices of a graph with ``n`` vertices."""
    if vertices is None:
        return np.arange(n, dtype=np.int64)
    values = list(vertices)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in values):
        raise ValueError("vertices must contain integer vertex indices")
    if any(v < 0 or v >= n for v in values):
        raise ValueError("audit vertex outside the graph")
    return np.asarray(values, dtype=np.int64)


def resolution_grid(resolutions):
    """The resolutions of a regret audit as a contiguous float64 vector."""
    gammas = np.asarray(resolutions, dtype=np.float64)
    if gammas.ndim != 1:
        raise ValueError("resolutions must be a one-dimensional sequence")
    return np.ascontiguousarray(gammas)


def intervals(audit, selected, cap, allow_isolation, atol, rtol, prune):
    return _core._audit_intervals(audit, selected, cap, allow_isolation, atol, rtol, prune)


def regret_arrays(audit, selected, gammas, cap, allow_isolation, atol, rtol, prune):
    result = RegretResult(np.empty((len(selected), len(gammas))),
                          np.empty((len(selected), len(gammas))))
    _core._audit_regrets(audit, selected, gammas, *result, cap, allow_isolation, atol, rtol, prune)
    return result


def stable_intervals(game, memberships=None, *, max_memberships=None,
                     allow_isolation=False, edge_weights=None, atol=1e-10,
                     rtol=1e-9, vertices=None, prune=True):
    selected = vertex_indices(game.vcount(), vertices)
    audit, cap = _prepare(game, memberships, max_memberships, edge_weights)
    return intervals(audit, selected, cap, allow_isolation, atol, rtol, prune)


def regrets(game, resolutions, memberships=None, *, max_memberships=None,
            allow_isolation=False, edge_weights=None, atol=1e-10,
            rtol=1e-9, vertices=None, prune=True):
    gammas = resolution_grid(resolutions)
    selected = vertex_indices(game.vcount(), vertices)
    audit, cap = _prepare(game, memberships, max_memberships, edge_weights)
    return regret_arrays(audit, selected, gammas, cap, allow_isolation, atol, rtol, prune)
