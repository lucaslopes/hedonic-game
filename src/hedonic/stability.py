"""Aggregate fixed-cover stability intervals returned by :class:`hedonic.Game`.

Intervals are closed subsets of [0, 1], sorted and disjoint within each
vertex. These measures use the audit's heuristic tolerance, not the
certified equilibrium slack returned by the detector.
"""
from math import log10
from typing import NamedTuple

import numpy as np

__all__ = ["RegretResult", "stable_fraction", "stable_measures", "nash_window"]


class RegretResult(NamedTuple):
    """Per-vertex, per-resolution arrays; stability is ``regret <= tolerance``."""

    regret: np.ndarray
    tolerance: np.ndarray


def stable_fraction(intervals, resolutions):
    """Fraction of vertices stable at each resolution (closed endpoints).

    ``intervals`` is the list returned by ``Game.stable_intervals`` and
    ``resolutions`` is a one-dimensional grid in [0, 1]. An empty population
    has stable fraction NaN. Overlapping intervals count their owner once.
    """
    grid = np.asarray(resolutions, dtype=float)
    if grid.ndim != 1 or np.any(~np.isfinite(grid)) or np.any((grid < 0) | (grid > 1)):
        raise ValueError("resolutions must be a finite one-dimensional grid in [0, 1]")
    counts = np.zeros(len(grid), dtype=np.int64)
    for row in intervals:
        stable = np.zeros(len(grid), dtype=bool)
        for lo, hi in row:
            stable |= (lo <= grid) & (grid <= hi)
        counts += stable
    return counts / len(intervals) if len(intervals) else np.full(len(grid), np.nan)


def stable_measures(intervals, *, log_floor=1e-4):
    """Mean stable shares of linear and logarithmic resolution, as a pair.

    Linear measure averages the lengths in [0, 1]. Log measure averages the
    lengths in log10 resolution on [log10(log_floor), 0]; the default floor
    is 1e-4 (four decades). An empty population returns (NaN, NaN).
    """
    if not 0 < log_floor < 1:
        raise ValueError("log_floor must be strictly between 0 and 1")
    if not len(intervals):
        return float("nan"), float("nan")
    # Keep the paper's flattened vertex order, NumPy reductions and log10.
    # Sequential Python additions and math.log10 differ in the last bits.
    pairs = np.asarray([pair for row in intervals for pair in row], dtype=float).reshape(-1, 2)
    lo, hi = pairs[:, 0], pairs[:, 1]
    floor = np.maximum(lo, log_floor)
    above = hi > floor
    linear = float((hi - lo).sum()) / len(intervals)
    logarithmic = float((np.log10(hi[above]) - np.log10(floor[above])).sum())
    return linear, logarithmic / (len(intervals) * -log10(log_floor))


def nash_window(intervals):
    """Intersect every vertex's closed stable set, in vertex order.

    Returns a list of (lo, hi) pairs. An empty population imposes no
    constraints and returns [(0, 1)]; any never-stable vertex returns [].
    """
    window = [(0.0, 1.0)]
    for row in intervals:
        window = [(max(a, lo), min(b, hi)) for a, b in window for lo, hi in row
                  if max(a, lo) <= min(b, hi)]
        if not window:
            break
    return window
