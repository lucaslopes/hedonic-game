"""Argument buffers for the core's fixed-cover audit (no Python scoring)."""
from itertools import chain
from numbers import Integral

import numpy as np

from . import _core
from ._native import _edge_weight_buffer
from .stability import RegretResult


def _prepare(game, memberships=None, max_memberships=None, edge_weights=None):
    if game.is_directed():
        raise ValueError("stability audits require an undirected graph")
    if memberships is None:
        memberships = game.memberships
    if memberships is None:
        raise ValueError("Game has no memberships; supply a cover or run community_hedonic first")
    if game.vcount() == 0 and len(memberships) == 0:
        rows = []
        if max_memberships is not None and (
            isinstance(max_memberships, bool) or not isinstance(max_memberships, Integral)
            or max_memberships < 1
        ):
            raise ValueError("max_memberships must be an integer >= 1")
    else:
        rows = game._normalize_membership(
            memberships, max_memberships=max_memberships,
            collapse_duplicate_labels=True, renumber_labels=True,
        )
    cap = max((len(row) for row in rows), default=1) if max_memberships is None else int(max_memberships)
    offsets = np.empty(len(rows) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum([len(row) for row in rows], out=offsets[1:])
    labels = np.fromiter(chain.from_iterable(rows), dtype=np.int64, count=int(offsets[-1]))
    weights = _edge_weight_buffer(game, edge_weights, nonnegative=True)
    return _core._audit_prepare(game, offsets, labels, weights), cap


def _vertices(game, vertices):
    if vertices is None:
        return np.arange(game.vcount(), dtype=np.int64)
    values = list(vertices)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in values):
        raise ValueError("vertices must contain integer vertex indices")
    if any(v < 0 or v >= game.vcount() for v in values):
        raise ValueError("audit vertex outside the graph")
    return np.asarray(values, dtype=np.int64)


def stable_intervals(game, memberships=None, *, max_memberships=None,
                     allow_isolation=False, edge_weights=None, atol=1e-10,
                     rtol=1e-9, vertices=None, prune=True):
    selected = _vertices(game, vertices)
    state, cap = _prepare(game, memberships, max_memberships, edge_weights)
    return _core._audit_intervals(state, selected, cap, allow_isolation, atol, rtol, prune)


def regrets(game, resolutions, memberships=None, *, max_memberships=None,
            allow_isolation=False, edge_weights=None, atol=1e-10,
            rtol=1e-9, vertices=None, prune=True):
    gammas = np.asarray(resolutions, dtype=np.float64)
    if gammas.ndim != 1:
        raise ValueError("resolutions must be a one-dimensional sequence")
    gammas = np.ascontiguousarray(gammas)
    selected = _vertices(game, vertices)
    state, cap = _prepare(game, memberships, max_memberships, edge_weights)
    result = RegretResult(np.empty((len(selected), len(gammas))),
                          np.empty((len(selected), len(gammas))))
    _core._audit_regrets(state, selected, gammas, *result, cap, allow_isolation, atol, rtol, prune)
    return result
