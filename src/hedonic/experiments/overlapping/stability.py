"""Stability of a fixed overlapping cover over the resolution range.

Two tools that answer the same question as ``gt_spectrum.spectrum_audit`` (is
each vertex stable at resolution gamma?) without relying on a dense gamma grid
for the stability answer itself:

* ``exact_stable_intervals`` returns, per vertex, the exact set of gamma in
  [0, 1] where the vertex is stable (regret <= the audit tolerance).  Every
  candidate action has utility affine in gamma, the tolerance is affine on
  each piece cut at |U_cur| = 1, and the top-k prefix of gains changes only at
  pairwise crossings.  Between consecutive crossings every constraint is affine,
  so the stable set is an explicit union of intervals.  It is not assumed to be
  one interval, which matters because the tolerance depends on gamma.

* ``certify_interval`` is the sound endpoint rule: the regret of a fixed profile
  is convex in gamma, so its maximum on [g0, g1] is at an endpoint, and it is
  stable on the whole interval when the larger endpoint regret does not exceed
  the smallest tolerance the interval can attain.

* ``vector_spectrum_rows`` is the vectorised grid audit.  Each gamma costs one
  sort over all (vertex, candidate) entries instead of a Python loop per vertex.
  It reproduces ``spectrum_audit``'s per-gamma stability counts and regret
  statistics.

Scope: a fixed cover, unit node weights, gamma in [0, 1], exact prefix
optimisation over the gamma-free candidate set of ``robustness``.  Mean, max and
p95 regret are not determined by the stable intervals and are reported only by
the grid audit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from hedonic.experiments.overlapping import robustness as rb

Interval = tuple[float, float]
REGRET_EPS = 1e-15  # the same threshold as robustness.best_response


@dataclass(frozen=True)
class VertexAffine:
    """Gamma-affine data of one vertex's candidate actions.

    A candidate label c has score ``a[c] - gamma * b[c]`` (the support minus the
    crowding term of ``robustness._label_score``).  The current action has
    utility ``cur_a - gamma * cur_b``.
    """

    labels: tuple[int, ...]
    a: tuple[float, ...]
    b: tuple[float, ...]
    cur_a: float
    cur_b: float
    cap: int


def vertex_affine(state: rb.FractionalState, vertex: int, cap: int, allow_isolation: bool, *,
                  prune: bool = True) -> VertexAffine:
    """The gamma-affine data of ``vertex``'s candidate actions.

    ``rb._candidate_labels`` is a superset: besides the vertex's own and neighbour labels it adds the ``cap + |labels| + 1``
    lightest-mass labels (every community when the cover is small), all with zero support.  A label with zero support
    that the vertex does not hold scores ``-gamma * crowding <= 0`` and never enters an optimal prefix: once a prefix sum
    is positive, a non-positive score lowers ``sum / sqrt(k)``, and when no score is positive the best action is the single
    highest score, which among zero-support labels is the lightest one.  So with ``prune`` those labels are dropped except
    the lightest (ties by label), which keeps the optimum, the regret and the stable set exactly and leaves the kernel only
    the labels that can matter.  Without it the cost per vertex grows like ``cap`` squared: a cover with a vertex in 1,817
    communities gave every vertex ~1,900 filler labels and an audit of about a day.  ``prune=False`` is the reference the
    tests compare against.
    """
    if prune:
        labels = _relevant_labels(state, vertex, allow_isolation)
    else:
        labels = rb._candidate_labels(state, vertex, 0.0, cap, allow_isolation, dense=False)
    a, b = [], []
    for label in labels:
        if label == rb.FRESH_LABEL:
            a.append(0.0)
            b.append(0.0)
        else:
            a.append(float(state.support[vertex].get(label, 0.0)))
            b.append(float(state.masses[label] - state.intensities[vertex].get(label, 0.0)))
    current = state.memberships[vertex]
    norm = math.sqrt(len(current))
    index = {label: i for i, label in enumerate(labels)}
    cur_a = sum(a[index[label]] for label in current) / norm
    cur_b = sum(b[index[label]] for label in current) / norm
    return VertexAffine(tuple(labels), tuple(a), tuple(b), float(cur_a), float(cur_b), int(cap))


def _relevant_labels(state: rb.FractionalState, vertex: int, allow_isolation: bool) -> list[int]:
    """The candidate labels that can matter for ``vertex`` (see ``vertex_affine``), ascending.

    Its own labels, the labels with positive support among its neighbours, the fresh label when isolation is allowed,
    and the lightest of the remaining labels: those all have zero support and are not held, so their score is
    ``-gamma * mass`` and the lightest is the first of ``state.lightest_labels`` (sorted by mass, then label) that is
    none of the above.  This is the set ``rb._candidate_labels`` returns with its zero-support filler reduced to that
    one label, built without enumerating the filler.
    """
    keep = set(state.memberships[vertex])
    keep.update(label for label, value in state.support[vertex].items() if value > 0.0)
    if allow_isolation:
        keep.add(rb.FRESH_LABEL)
    for label in state.lightest_labels:
        if label not in keep:
            keep.add(label)
            break
    return sorted(keep)


def _merge(intervals: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] + 1e-15:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _cell_stable(data: VertexAffine, p: float, q: float, atol: float, rtol: float) -> Interval | None:
    """Stable sub-interval of the cell [p, q] on which all pieces are affine."""
    mid = 0.5 * (p + q)
    scores = [a_c - mid * b_c for a_c, b_c in zip(data.a, data.b)]
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    u_mid = data.cur_a - mid * data.cur_b
    if abs(u_mid) <= 1.0:
        tau_a, tau_b = atol + rtol, 0.0
    else:
        sign = 1.0 if u_mid > 0 else -1.0
        tau_a = atol + rtol * sign * data.cur_a
        tau_b = -rtol * sign * data.cur_b
    lo, hi = p, q
    sum_a = sum_b = 0.0
    for k in range(1, min(data.cap, len(order)) + 1):
        i = order[k - 1]
        sum_a += data.a[i]
        sum_b += data.b[i]
        root = math.sqrt(k)
        # regret-minus-tolerance constraint: c0 + c1 * gamma <= 0
        c0 = sum_a / root - data.cur_a - tau_a
        c1 = -sum_b / root + data.cur_b - tau_b
        if c1 > 0:
            hi = min(hi, -c0 / c1)
        elif c1 < 0:
            lo = max(lo, -c0 / c1)
        elif c0 > 0:
            return None
        if lo > hi:
            return None
    return (lo, hi) if lo <= hi else None


def exact_stable_intervals(
    state: rb.FractionalState,
    vertex: int,
    cap: int,
    allow_isolation: bool,
    *,
    atol: float,
    rtol: float,
    engine: str = "python",
    prune: bool = True,
) -> list[Interval]:
    """Exact stable set of ``vertex`` on [0, 1] as a sorted list of closed intervals.

    ``engine`` is ``"python"`` (the reference) or ``"numba"`` (the compiled kernel, which must return the
    same floats; ``TestCompiledIntervals`` checks equality, not closeness).  ``prune`` drops the candidate labels that
    cannot matter (``vertex_affine``); the intervals are the same either way.
    """
    data = vertex_affine(state, vertex, cap, allow_isolation, prune=prune)
    if engine == "numba":
        return _compiled_intervals(data, atol, rtol)
    if engine != "python":
        raise ValueError("engine must be 'python' or 'numba'")
    points = {0.0, 1.0}
    n = len(data.labels)
    for i in range(n):
        for j in range(i + 1, n):
            db = data.b[i] - data.b[j]
            if db != 0.0:
                gamma = (data.a[i] - data.a[j]) / db
                if 0.0 < gamma < 1.0:
                    points.add(gamma)
    if data.cur_b != 0.0:
        for level in (1.0, -1.0):
            gamma = (data.cur_a - level) / data.cur_b
            if 0.0 < gamma < 1.0:
                points.add(gamma)
    cuts = sorted(points)
    pieces = []
    for p, q in zip(cuts, cuts[1:]):
        if q > p:
            found = _cell_stable(data, p, q, atol, rtol)
            if found is not None:
                pieces.append(found)
    return _merge(pieces)


def in_intervals(intervals: Sequence[Interval], gamma: float, slack: float = 0.0) -> bool:
    return any(lo - slack <= gamma <= hi + slack for lo, hi in intervals)


def tolerance_floor(data: VertexAffine, g0: float, g1: float, atol: float, rtol: float) -> float:
    """Smallest value of tau(gamma) = atol + rtol * max(1, |U_cur(gamma)|) on [g0, g1]."""
    u0 = data.cur_a - g0 * data.cur_b
    u1 = data.cur_a - g1 * data.cur_b
    if u0 * u1 <= 0.0:
        min_abs = 0.0
    else:
        min_abs = min(abs(u0), abs(u1))
    return atol + rtol * max(1.0, min_abs)


def certify_interval(
    state: rb.FractionalState,
    vertex: int,
    cap: int,
    allow_isolation: bool,
    *,
    g0: float,
    g1: float,
    atol: float,
    rtol: float,
) -> bool:
    """True only if the vertex is provably stable on the whole of [g0, g1].

    Sound because regret is convex in gamma (fixed cover): its maximum on the
    interval is at an endpoint, and it is compared with the smallest tolerance
    the interval attains.  A False answer is not evidence of instability.
    """
    if not 0.0 <= g0 <= g1 <= 1.0:
        raise ValueError("need 0 <= g0 <= g1 <= 1")
    data = vertex_affine(state, vertex, cap, allow_isolation)
    worst = 0.0
    for gamma in (g0, g1):
        diagnostic = rb.best_response(state, vertex, gamma, cap, allow_isolation)
        worst = max(worst, float(diagnostic["regret"]))
    return worst <= tolerance_floor(data, g0, g1, atol, rtol)


# ---------------------------------------------------------------------------
# Vectorised grid audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entries:
    """Flat (vertex, candidate) arrays for every vertex, built once per cover."""

    n: int
    vertex: np.ndarray
    a: np.ndarray
    b: np.ndarray
    is_current: np.ndarray
    current_norm: np.ndarray
    cap: int
    starts: np.ndarray  # offsets: vertex v owns entries starts[v]:starts[v+1]


def build_entries(state: rb.FractionalState, cap: int, allow_isolation: bool,
                  vertices: range | None = None, *, prune: bool = True) -> Entries:
    """Candidate entries of ``vertices`` (default: every vertex); vertex ``vertices[i]`` is numbered ``i``."""
    vertices = range(len(state.memberships)) if vertices is None else vertices
    n = len(vertices)
    vertex, a, b, is_current = [], [], [], []
    current_norm = np.empty(n)
    for i, v in enumerate(vertices):
        data = vertex_affine(state, v, cap, allow_isolation, prune=prune)
        current = set(state.memberships[v])
        vertex.extend([i] * len(data.labels))
        a.extend(data.a)
        b.extend(data.b)
        is_current.extend(label in current for label in data.labels)
        current_norm[i] = math.sqrt(len(current))
    vertex_array = np.asarray(vertex, dtype=np.int64)
    starts = np.searchsorted(vertex_array, np.arange(n + 1))
    return Entries(
        n=n,
        vertex=vertex_array,
        a=np.asarray(a, dtype=float),
        b=np.asarray(b, dtype=float),
        is_current=np.asarray(is_current, dtype=bool),
        current_norm=current_norm,
        cap=int(cap),
        starts=starts.astype(np.int64),
    )


def vertex_regrets(entries: Entries, gamma: float, atol: float, rtol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-vertex (regret, tolerance, current utility) at one gamma, all vertices at once."""
    score = entries.a - gamma * entries.b
    order = np.lexsort((-score, entries.vertex))
    v_sorted = entries.vertex[order]
    s_sorted = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(v_sorted)) + 1]
    lengths = np.diff(np.r_[starts, len(v_sorted)])
    seg = np.repeat(np.arange(len(starts)), lengths)
    k = np.arange(len(v_sorted)) - starts[seg] + 1
    csum = np.cumsum(s_sorted)
    prefix = csum - (csum[starts] - s_sorted[starts])[seg]
    util = np.full(len(v_sorted), -np.inf)
    valid = k <= entries.cap
    util[valid] = prefix[valid] / np.sqrt(k[valid])
    best = np.maximum.reduceat(util, starts)
    current = np.bincount(entries.vertex, weights=score * entries.is_current, minlength=entries.n)
    current = current / entries.current_norm
    diff = best - current[v_sorted[starts]]
    regret = np.where(diff > REGRET_EPS, diff, 0.0)
    tolerance = atol + rtol * np.maximum(1.0, np.abs(current))
    return regret, tolerance, current


try:  # numba is optional: the numpy path is the reference and the fallback
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False


if HAVE_NUMBA:

    @njit(parallel=True)
    def _regret_kernel(starts, a, b, is_current, current_norm, cap, gamma, atol, rtol, regret, tolerance):
        n = starts.shape[0] - 1
        for v in prange(n):
            lo = starts[v]
            m = starts[v + 1] - lo
            score = np.empty(m)
            current = 0.0
            for i in range(m):
                score[i] = a[lo + i] - gamma * b[lo + i]
                if is_current[lo + i]:
                    current += score[i]
            current = current / current_norm[v]
            order = np.argsort(-score)
            best = -np.inf
            prefix = 0.0
            for k in range(min(cap, m)):
                prefix += score[order[k]]
                utility = prefix / np.sqrt(k + 1.0)
                if utility > best:
                    best = utility
            diff = best - current
            regret[v] = diff if diff > REGRET_EPS else 0.0
            tolerance[v] = atol + rtol * max(1.0, abs(current))

    @njit
    def _breakpoints_kernel(a, b, cur_a, cur_b):
        # 0, 1, every crossing of two candidate scores in (0, 1), and the crossings of the current score with +-1
        n = a.shape[0]
        points = np.empty(n * (n - 1) // 2 + 4)
        points[0] = 0.0
        points[1] = 1.0
        m = 2
        for i in range(n):
            for j in range(i + 1, n):
                db = b[i] - b[j]
                if db != 0.0:
                    gamma = (a[i] - a[j]) / db
                    if 0.0 < gamma < 1.0:
                        points[m] = gamma
                        m += 1
        if cur_b != 0.0:
            for level in range(2):
                gamma = (cur_a - (1.0 if level == 0 else -1.0)) / cur_b
                if 0.0 < gamma < 1.0:
                    points[m] = gamma
                    m += 1
        return np.sort(points[:m])

    @njit
    def _exact_intervals_kernel(a, b, cur_a, cur_b, cap, atol, rtol):
        # ``exact_stable_intervals`` cell by cell.  Scores are sorted by descending value, ties by ascending
        # index (the reference's stable sort).  Two scores cross at most once, so the order carried from one
        # cell to the next changes only a few times in total, and an insertion sort from it is linear per cell.
        n = a.shape[0]
        cuts = _breakpoints_kernel(a, b, cur_a, cur_b)
        cells = cuts.shape[0] - 1
        out_lo = np.empty(cells)
        out_hi = np.empty(cells)
        pieces = 0
        scores = np.empty(n)
        order = np.arange(n)
        top = min(cap, n)
        for c in range(cells):
            p = cuts[c]
            q = cuts[c + 1]
            if not q > p:
                continue
            mid = 0.5 * (p + q)
            for i in range(n):
                scores[i] = a[i] - mid * b[i]
            for idx in range(1, n):
                x = order[idx]
                sx = scores[x]
                j = idx - 1
                while j >= 0:
                    y = order[j]
                    sy = scores[y]
                    if sy < sx or (sy == sx and y > x):
                        order[j + 1] = y
                        j -= 1
                    else:
                        break
                order[j + 1] = x
            u_mid = cur_a - mid * cur_b
            if abs(u_mid) <= 1.0:
                tau_a = atol + rtol
                tau_b = 0.0
            else:
                sign = 1.0 if u_mid > 0 else -1.0
                tau_a = atol + rtol * sign * cur_a
                tau_b = -rtol * sign * cur_b
            lo = p
            hi = q
            sum_a = 0.0
            sum_b = 0.0
            stable = True
            for k in range(1, top + 1):
                i = order[k - 1]
                sum_a += a[i]
                sum_b += b[i]
                root = math.sqrt(k)
                c0 = sum_a / root - cur_a - tau_a
                c1 = -sum_b / root + cur_b - tau_b
                if c1 > 0:
                    hi = min(hi, -c0 / c1)
                elif c1 < 0:
                    lo = max(lo, -c0 / c1)
                elif c0 > 0:
                    stable = False
                    break
                if lo > hi:
                    stable = False
                    break
            if stable and lo <= hi:
                # cells are visited in increasing order, so the reference's sort and merge become one pass
                if pieces > 0 and lo <= out_hi[pieces - 1] + 1e-15:
                    out_hi[pieces - 1] = max(out_hi[pieces - 1], hi)
                else:
                    out_lo[pieces] = lo
                    out_hi[pieces] = hi
                    pieces += 1
        return out_lo[:pieces], out_hi[:pieces]


def compiled_regrets(entries: Entries, gamma: float, atol: float, rtol: float) -> tuple[np.ndarray, np.ndarray]:
    """Numba-compiled per-vertex (regret, tolerance) at one gamma; same semantics as ``vertex_regrets``."""
    if not HAVE_NUMBA:
        raise RuntimeError("numba is not installed; use vertex_regrets")
    starts = entries.starts
    regret = np.empty(entries.n)
    tolerance = np.empty(entries.n)
    _regret_kernel(starts, entries.a, entries.b, entries.is_current, entries.current_norm,
                   entries.cap, float(gamma), float(atol), float(rtol), regret, tolerance)
    return regret, tolerance


def default_engine() -> str:
    return "numba" if HAVE_NUMBA else "python"


def warm_up() -> None:
    """Compile the interval kernel in this process, so forked workers inherit compiled code."""
    if HAVE_NUMBA:
        _exact_intervals_kernel(np.array([1.0, 0.5]), np.array([0.0, 1.0]), 1.0, 0.5, 2, 1e-10, 1e-9)


def _compiled_intervals(data: VertexAffine, atol: float, rtol: float) -> list[Interval]:
    if not HAVE_NUMBA:
        raise RuntimeError("numba is not installed; use engine='python'")
    lo, hi = _exact_intervals_kernel(np.asarray(data.a, dtype=float), np.asarray(data.b, dtype=float),
                                     float(data.cur_a), float(data.cur_b), int(data.cap), float(atol), float(rtol))
    return list(zip(lo.tolist(), hi.tolist()))


def _percentile(values: np.ndarray, q: float) -> float:
    return float(rb._percentile(values.tolist(), q))


def vector_spectrum_rows(
    entries: Entries,
    gammas: Sequence[float],
    *,
    atol: float,
    rtol: float,
    allow_isolation: bool,
    engine: str = "numpy",
) -> list[dict[str, Any]]:
    """Per-gamma rows with the same stability fields as ``gt_spectrum.spectrum_audit``.

    ``engine`` is ``"numpy"`` (one sort per gamma) or ``"numba"`` (compiled loop over vertices).
    """
    if engine not in ("numpy", "numba"):
        raise ValueError("engine must be 'numpy' or 'numba'")
    rows = []
    for gamma in gammas:
        if engine == "numba":
            regret, tolerance = compiled_regrets(entries, float(gamma), atol, rtol)
        else:
            regret, tolerance, _ = vertex_regrets(entries, float(gamma), atol, rtol)
        rows.append(_spectrum_row(float(gamma), regret, tolerance, entries.cap, atol, rtol, allow_isolation))
    return rows


def _spectrum_row(gamma: float, regret: np.ndarray, tolerance: np.ndarray, cap: int, atol: float, rtol: float,
                  allow_isolation: bool) -> dict[str, Any]:
    n = len(regret)
    stable_count = int((regret <= tolerance).sum())
    profitable = n - stable_count
    return {
        "gamma": gamma,
        "action_policy": "open_labels" if allow_isolation else "fixed_labels",
        "allow_isolation": bool(allow_isolation),
        "cap": int(cap),
        "n_vertices_covered": n,
        "stable_vertex_count": stable_count,
        "stable_fraction": stable_count / n,
        "profitable_vertex_count": profitable,
        "profitable_fraction": profitable / n,
        "mean_positive_regret": float(regret.sum() / n),
        "max_positive_regret": float(regret.max()),
        "p95_positive_regret": _percentile(regret, 0.95),
        "max_tolerance": float(max(atol + rtol, tolerance.max())),
        "is_nash_equilibrium": profitable == 0,
        "atol": float(atol),
        "rtol": float(rtol),
    }


ENTRY_BLOCK = 5_000  # vertices whose candidate entries are materialised at once by ``chunked_spectrum_rows``


def chunked_spectrum_rows(
    state: rb.FractionalState,
    cap: int,
    gammas: Sequence[float],
    *,
    atol: float,
    rtol: float,
    allow_isolation: bool,
    engine: str = "numpy",
    block: int = ENTRY_BLOCK,
) -> list[dict[str, Any]]:
    """``vector_spectrum_rows`` without materialising the candidate entries of every vertex at once.

    The entries of all vertices are about (support labels + cap) per vertex: on the 1.1 M-vertex LiveJournal cover
    that is a billion entries, tens of GB as arrays and several times that as the Python lists they are built from.
    A vertex's regret does not depend on any other vertex, so the vertices go through in blocks, only the per-vertex
    regret and tolerance of each gamma are kept (two floats per vertex), and the rows are computed from those.
    With the compiled engine the result is bit-identical to the unchunked one; with numpy the per-block cumulative
    sums differ from the global one by rounding only.
    """
    gammas = [float(g) for g in gammas]
    n = len(state.memberships)
    regret = np.empty((len(gammas), n))
    tolerance = np.empty((len(gammas), n))
    for low in range(0, n, block):
        piece = range(low, min(n, low + block))
        entries = build_entries(state, cap, allow_isolation, piece)
        for index, gamma in enumerate(gammas):
            if engine == "numba":
                r, t = compiled_regrets(entries, gamma, atol, rtol)
            else:
                r, t, _ = vertex_regrets(entries, gamma, atol, rtol)
            regret[index, piece.start:piece.stop], tolerance[index, piece.start:piece.stop] = r, t
        del entries
    return [_spectrum_row(g, regret[i], tolerance[i], cap, atol, rtol, allow_isolation) for i, g in enumerate(gammas)]
