"""Stability of a fixed overlapping cover over the resolution range, for the experiments.

Every study that audits a cover (``cover_audit``, ``gt_posthoc``, ``gt_spectrum`` and the benchmark's robustness
stage) goes through this module:

* :func:`prepare` turns a graph and a cover into a read-only audit object. The default engine, ``"c"``, is the
  library's C audit (``native/core/check/audit.c``, the code behind ``Game.stable_intervals`` and ``Game.regrets``):
  its prepared state holds the adjacency and the rows, O(n + m + incidences), builds a vertex's supports when it
  visits it, and can be shared with forked processes. ``"python"`` is the reference implementation below, on
  ``robustness.FractionalState``; it returns the same floats, bit for bit, and exists for tests and cross-checks.
* :func:`stream_intervals` yields every vertex's exact stable set on [0, 1] (a sorted list of closed intervals),
  chunk by chunk, optionally in forked worker processes that split the vertices.
* :func:`spectrum_rows` evaluates the regret of every vertex on a resolution grid and aggregates each resolution
  into a row (stable count, regret mean, max and 95th percentile, Nash flag).
* :func:`audit_cover` is the endpoint audit a spectrum condition stores for each returned cover.

Arithmetic (see ``native/README.md``): the C audit sums with sequential binary64 additions, except the two
current-utility coefficients of a vertex's affine data, which use the compensated (Neumaier) sum of CPython 3.12's
built-in ``sum``. The Python reference spells that sum out (``_compensated_sum``) instead of calling ``sum``, so both
engines agree on any interpreter, not only on the one the C was checked against.

Scope: a fixed cover, unit node weights, gamma in [0, 1], exact prefix optimisation over a gamma-free candidate set
(the vertex's own labels, the labels with positive support, the lightest remaining label and, with isolation, a
fresh label). Mean, max and p95 regret are not determined by the stable intervals and come from the grid.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from hedonic.experiments.overlapping import robustness as rb

Interval = tuple[float, float]
REGRET_EPS = 1e-15  # regrets at most this are zero (the C audit and the reference clamp the same way)
ENGINES = ("c", "python")
DEFAULT_ENGINE = "c"
CHUNKS_PER_WORKER = 16  # vertex chunks per forked worker: small enough to balance the load, few enough to be cheap
MAX_CHUNK = 8192  # vertices per interval query at most, so progress keeps moving on large graphs
REGRET_CELLS = 16_000_000  # vertex x resolution regrets (two doubles each) held at once by ``spectrum_rows``
MAX_GAMMA_BLOCK = 32  # resolutions per regret query (a vertex's supports are built once per query)


# --------------------------------------------------------------------------- the Python reference
def _compensated_sum(values) -> float:
    """CPython 3.12's float ``sum`` (Neumaier compensation), written out so it does not depend on the interpreter."""
    high = low = 0.0
    for value in values:
        total = high + value
        low += (high - total) + value if abs(high) >= abs(value) else (value - total) + high
        high = total
    return high + low


@dataclass(frozen=True)
class VertexAffine:
    """Gamma-affine data of one vertex's candidate actions.

    A candidate label c has score ``a[c] - gamma * b[c]`` (the support minus the crowding term of
    ``robustness._label_score``).  The current action has utility ``cur_a - gamma * cur_b``.
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

    ``rb._candidate_labels`` is a superset: besides the vertex's own and neighbour labels it adds the
    ``cap + |labels| + 1`` lightest-mass labels (every community when the cover is small), all with zero support.  A
    label with zero support that the vertex does not hold scores ``-gamma * crowding <= 0`` and never enters an optimal
    prefix: once a prefix sum is positive, a non-positive score lowers ``sum / sqrt(k)``, and when no score is positive
    the best action is the single highest score, which among zero-support labels is the lightest one.  So with
    ``prune`` those labels are dropped except the lightest (ties by label), which keeps the optimum, the regret and the
    stable set exactly.  ``prune=False`` is the larger set the tests compare against.
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
    cur_a = _compensated_sum(a[index[label]] for label in current) / norm
    cur_b = _compensated_sum(b[index[label]] for label in current) / norm
    return VertexAffine(tuple(labels), tuple(a), tuple(b), float(cur_a), float(cur_b), int(cap))


def _relevant_labels(state: rb.FractionalState, vertex: int, allow_isolation: bool) -> list[int]:
    """The candidate labels that can matter for ``vertex`` (see ``vertex_affine``), ascending.

    Its own labels, the labels with positive support among its neighbours, the fresh label when isolation is allowed,
    and the lightest of the remaining labels: those all have zero support and are not held, so their score is
    ``-gamma * mass`` and the lightest is the first of ``state.lightest_labels`` (sorted by mass, then label) that is
    none of the above.
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


def exact_stable_intervals(state: rb.FractionalState, vertex: int, cap: int, allow_isolation: bool, *,
                           atol: float, rtol: float, prune: bool = True) -> list[Interval]:
    """Exact stable set of ``vertex`` on [0, 1] as a sorted list of closed intervals (the Python reference).

    Every candidate action has utility affine in gamma, the tolerance is affine on each piece cut at |U_cur| = 1, and
    the top-k prefix of gains changes only at pairwise crossings.  Between consecutive crossings every constraint is
    affine, so the stable set is an explicit union of intervals (not assumed to be one, because the tolerance depends
    on gamma).
    """
    data = vertex_affine(state, vertex, cap, allow_isolation, prune=prune)
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


def vertex_regrets(state: rb.FractionalState, vertex: int, gammas: Sequence[float], cap: int, allow_isolation: bool,
                   *, atol: float, rtol: float, prune: bool = True) -> tuple[list[float], list[float]]:
    """Regret and tolerance of ``vertex`` at each resolution (the Python reference of the C grid audit).

    Scores are evaluated first; the current utility adds the held labels' scores one after another in candidate order;
    the best action is the best prefix, of at most ``cap`` labels, of the scores in descending order.
    """
    data = vertex_affine(state, vertex, cap, allow_isolation, prune=prune)
    held = set(state.memberships[vertex])
    current_mask = [label in held for label in data.labels]
    norm = math.sqrt(len(held))
    regrets, tolerances = [], []
    for gamma in gammas:
        scores = [a - gamma * b for a, b in zip(data.a, data.b)]
        current = 0.0
        for score, is_current in zip(scores, current_mask):
            if is_current:
                current += score
        current = current / norm
        best, prefix = -math.inf, 0.0
        for k, score in enumerate(sorted(scores, reverse=True)[:cap], 1):
            prefix += score
            utility = prefix / math.sqrt(k)
            if utility > best:
                best = utility
        diff = best - current
        regrets.append(diff if diff > REGRET_EPS else 0.0)
        tolerances.append(atol + rtol * max(1.0, abs(current)))
    return regrets, tolerances


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

    Sound because regret is convex in gamma (fixed cover): its maximum on the interval is at an endpoint, and it is
    compared with the smallest tolerance the interval attains.  A False answer is not evidence of instability.
    """
    if not 0.0 <= g0 <= g1 <= 1.0:
        raise ValueError("need 0 <= g0 <= g1 <= 1")
    data = vertex_affine(state, vertex, cap, allow_isolation)
    worst = 0.0
    for gamma in (g0, g1):
        diagnostic = rb.best_response(state, vertex, gamma, cap, allow_isolation)
        worst = max(worst, float(diagnostic["regret"]))
    return worst <= tolerance_floor(data, g0, g1, atol, rtol)


# --------------------------------------------------------------------------- prepared covers (the engines)
class PreparedCover:
    """A fixed cover of a graph, ready for audit queries. Read-only: forked processes may query it concurrently."""

    engine = ""

    def __init__(self, memberships: list[tuple[int, ...]], n_labels: int):
        self.memberships = memberships
        self.n_vertices = len(memberships)
        self.n_communities = n_labels

    def intervals(self, vertices: np.ndarray, cap: int, allow_isolation: bool, *, atol: float,
                  rtol: float) -> list[list[Interval]]:
        """The stable intervals of each of ``vertices``, in that order."""
        raise NotImplementedError

    def regrets(self, vertices: np.ndarray, gammas: np.ndarray, cap: int, allow_isolation: bool, *, atol: float,
                rtol: float) -> tuple[np.ndarray, np.ndarray]:
        """``(regret, tolerance)``, each shaped ``(len(vertices), len(gammas))``."""
        raise NotImplementedError


class _CCover(PreparedCover):
    engine = "c"

    def __init__(self, graph, memberships: list[tuple[int, ...]], n_labels: int, weights: np.ndarray | None):
        from hedonic import _core

        super().__init__(memberships, n_labels)
        offsets = np.zeros(len(memberships) + 1, dtype=np.int64)
        np.cumsum([len(row) for row in memberships], out=offsets[1:])
        labels = np.fromiter((label for row in memberships for label in row), dtype=np.int64, count=int(offsets[-1]))
        self._core = _core
        self._state = _core._audit_prepare(graph, offsets, labels, weights)

    def intervals(self, vertices, cap, allow_isolation, *, atol, rtol):
        selected = np.ascontiguousarray(vertices, dtype=np.int64)
        return self._core._audit_intervals(self._state, selected, int(cap), bool(allow_isolation), float(atol),
                                           float(rtol), True)

    def regrets(self, vertices, gammas, cap, allow_isolation, *, atol, rtol):
        selected = np.ascontiguousarray(vertices, dtype=np.int64)
        grid = np.ascontiguousarray(gammas, dtype=np.float64)
        regret = np.empty((len(selected), len(grid)))
        tolerance = np.empty((len(selected), len(grid)))
        self._core._audit_regrets(self._state, selected, grid, regret, tolerance, int(cap), bool(allow_isolation),
                                  float(atol), float(rtol), True)
        return regret, tolerance


class _PythonCover(PreparedCover):
    engine = "python"

    def __init__(self, graph, memberships: list[tuple[int, ...]], n_labels: int, weights: np.ndarray | None):
        super().__init__(memberships, n_labels)
        self.state = rb.build_fractional_state(graph, memberships, edge_weights=weights)

    def intervals(self, vertices, cap, allow_isolation, *, atol, rtol):
        _check_cap(self.memberships, vertices, cap)
        return [exact_stable_intervals(self.state, int(v), int(cap), bool(allow_isolation), atol=atol, rtol=rtol)
                for v in vertices]

    def regrets(self, vertices, gammas, cap, allow_isolation, *, atol, rtol):
        _check_cap(self.memberships, vertices, cap)
        grid = [float(g) for g in gammas]
        regret = np.empty((len(vertices), len(grid)))
        tolerance = np.empty((len(vertices), len(grid)))
        for i, v in enumerate(vertices):
            regret[i], tolerance[i] = vertex_regrets(self.state, int(v), grid, int(cap), bool(allow_isolation),
                                                     atol=atol, rtol=rtol)
        return regret, tolerance


def _check_cap(memberships, vertices, cap: int) -> None:
    if any(len(memberships[int(v)]) > cap for v in vertices):
        raise ValueError("a vertex holds more labels than max_memberships")


def prepare(graph, memberships: Sequence[Sequence[int]], *, engine: str = DEFAULT_ENGINE,
            edge_weights: Sequence[float] | None = None) -> PreparedCover:
    """The audit object of a valid cover: one non-empty row per vertex, labels contiguous from zero.

    ``edge_weights`` defaults to the graph's ``weight`` attribute, else ones; weights are finite and non-negative.
    """
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {', '.join(ENGINES)}, not {engine!r}")
    if graph.is_directed():
        raise ValueError("stability audits require an undirected graph")
    normalized, max_label = rb.normalize_memberships(graph, memberships)
    weights = None
    if edge_weights is not None or "weight" in graph.edge_attributes():
        weights = np.ascontiguousarray(rb._edge_weights(graph, edge_weights), dtype=np.float64)
    kind = _CCover if engine == "c" else _PythonCover
    return kind(graph, normalized, max_label + 1, weights)


# --------------------------------------------------------------------------- forked queries
_FORKED: dict[str, Any] = {}  # what forked workers read: set before the pool forks, so they inherit it


def _forked_intervals(chunk: np.ndarray) -> list[list[Interval]]:
    cover, cap, allow, atol, rtol = _FORKED["intervals"]
    return cover.intervals(chunk, cap, allow, atol=atol, rtol=rtol)


def _forked_regrets(task: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    cover, cap, allow, atol, rtol = _FORKED["regrets"]
    vertices, gammas = task
    return cover.regrets(vertices, gammas, cap, allow, atol=atol, rtol=rtol)


def _pool(workers: int):
    return mp.get_context("fork").Pool(workers)


def chunk_size(n_vertices: int, workers: int) -> int:
    """Vertices per query: about ``CHUNKS_PER_WORKER`` chunks per worker, at most ``MAX_CHUNK``."""
    return max(1, min(MAX_CHUNK, -(-n_vertices // (max(1, workers) * CHUNKS_PER_WORKER))))


def stream_intervals(cover: PreparedCover, cap: int, allow_isolation: bool, *, atol: float, rtol: float,
                     workers: int = 1, order: Sequence[int] | None = None,
                     chunk: int | None = None) -> Iterator[tuple[np.ndarray, list[list[Interval]]]]:
    """``(vertices, intervals)`` chunk by chunk, in ``order`` (default: index order).

    With ``workers > 1`` forked processes take the chunks; the prepared cover is inherited, never copied or pickled.
    The results do not depend on the worker count or the chunk size: each vertex's intervals are its own.
    """
    vertices = np.arange(cover.n_vertices, dtype=np.int64) if order is None else np.asarray(order, dtype=np.int64)
    size = chunk or chunk_size(len(vertices), workers)
    chunks = [vertices[low:low + size] for low in range(0, len(vertices), size)]
    if workers > 1 and len(chunks) > 1:
        _FORKED["intervals"] = (cover, int(cap), bool(allow_isolation), float(atol), float(rtol))
        try:
            with _pool(min(workers, len(chunks))) as pool:
                yield from zip(chunks, pool.imap(_forked_intervals, chunks))
        finally:
            _FORKED.pop("intervals", None)
    else:
        for part in chunks:
            yield part, cover.intervals(part, cap, allow_isolation, atol=atol, rtol=rtol)


def stable_intervals(cover: PreparedCover, cap: int, allow_isolation: bool, *, atol: float, rtol: float,
                     workers: int = 1) -> list[list[Interval]]:
    """Every vertex's stable intervals, in vertex order."""
    out: list[list[Interval]] = [[] for _ in range(cover.n_vertices)]
    for vertices, intervals in stream_intervals(cover, cap, allow_isolation, atol=atol, rtol=rtol, workers=workers):
        for v, found in zip(vertices.tolist(), intervals):
            out[v] = found
    return out


def regret_blocks(cover: PreparedCover, gammas: Sequence[float], cap: int, allow_isolation: bool, *, atol: float,
                  rtol: float, workers: int = 1, block: int | None = None
                  ) -> Iterator[tuple[list[float], np.ndarray, np.ndarray]]:
    """``(resolutions of the block, regret, tolerance)``, the arrays shaped ``(n_vertices, len(block))``.

    The resolutions go through in blocks, so that at most ``REGRET_CELLS`` vertex x resolution values are held; with
    ``workers > 1`` forked processes split the vertices of each block.
    """
    grid = [float(g) for g in gammas]
    n = cover.n_vertices
    size = block or max(1, min(MAX_GAMMA_BLOCK, REGRET_CELLS // max(1, n)))
    blocks = [grid[low:low + size] for low in range(0, len(grid), size)]
    if not blocks:
        return
    vertices = np.arange(n, dtype=np.int64)
    if workers > 1 and n > 1:
        step = chunk_size(n, workers)
        parts = [vertices[low:low + step] for low in range(0, n, step)]
        _FORKED["regrets"] = (cover, int(cap), bool(allow_isolation), float(atol), float(rtol))
        try:
            with _pool(min(workers, len(parts))) as pool:
                for piece in blocks:
                    values = np.asarray(piece, dtype=np.float64)
                    results = pool.map(_forked_regrets, [(part, values) for part in parts])
                    yield piece, np.concatenate([r for r, _ in results]), np.concatenate([t for _, t in results])
        finally:
            _FORKED.pop("regrets", None)
    else:
        for piece in blocks:
            regret, tolerance = cover.regrets(vertices, np.asarray(piece, dtype=np.float64), cap, allow_isolation,
                                              atol=atol, rtol=rtol)
            yield piece, regret, tolerance


def _percentile(values: np.ndarray, q: float) -> float:
    return float(rb._percentile(values.tolist(), q))


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


def spectrum_rows(cover: PreparedCover, cap: int, gammas: Sequence[float], *, atol: float, rtol: float,
                  allow_isolation: bool, workers: int = 1,
                  progress: Callable[[int, float], None] | None = None) -> list[dict[str, Any]]:
    """One row per resolution: how many vertices are stable there and the distribution of their regrets.

    ``progress(done, gamma)`` is called after each resolution. Each resolution's regrets are summed as one contiguous
    array in vertex order, so the rows do not depend on ``workers``.
    """
    rows: list[dict[str, Any]] = []
    for piece, regret, tolerance in regret_blocks(cover, gammas, cap, allow_isolation, atol=atol, rtol=rtol,
                                                  workers=workers):
        for j, gamma in enumerate(piece):
            rows.append(_spectrum_row(gamma, np.ascontiguousarray(regret[:, j]), np.ascontiguousarray(tolerance[:, j]),
                                      cap, atol, rtol, allow_isolation))
            if progress is not None:
                progress(len(rows), gamma)
    return rows


# --------------------------------------------------------------------------- the endpoint audit of one cover
def audit_cover(graph, memberships: Sequence[Sequence[int]], *, max_memberships: int, allow_isolation: bool,
                gamma: float | None = None, interval: tuple[float, float] = (0.0, 1.0), atol: float = 1e-10,
                rtol: float = 1e-9, engine: str = DEFAULT_ENGINE) -> dict[str, Any]:
    """Endpoint robustness and the equilibrium residual at ``gamma`` of a fixed cover: the fields of
    ``robustness.audit_cover`` (without its optional per-vertex intervals), from one grid query of the engine.

    A vertex is robust on ``interval`` when it is stable at both endpoints (its regret is convex in gamma).
    """
    lower, upper = (float(interval[0]), float(interval[1]))
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError("interval must satisfy 0 <= lower <= upper <= 1")
    cover = prepare(graph, memberships, engine=engine)
    selected_gamma = graph.density() if gamma is None else float(gamma)
    if selected_gamma < 0 or selected_gamma > 1:
        raise ValueError("gamma must lie in [0, 1] for robustness auditing")
    n = cover.n_vertices
    cap = int(max_memberships)
    if cap < 1:
        raise ValueError("max_memberships must be positive")
    if any(len(row) > cap for row in cover.memberships):
        raise ValueError("current membership exceeds max_memberships")
    regret, tolerance = cover.regrets(np.arange(n, dtype=np.int64), np.array([lower, upper, selected_gamma]), cap,
                                      allow_isolation, atol=atol, rtol=rtol)
    stable = regret <= tolerance
    robust = stable[:, 0] & stable[:, 1]
    at = np.ascontiguousarray(regret[:, 2])
    tol = np.ascontiguousarray(tolerance[:, 2])
    saturated = sum(len(row) >= cap for row in cover.memberships)
    return {
        "robust_fraction_gamma_0_1": int(robust.sum()) / n,
        "robust_vertex_count_gamma_0_1": int(robust.sum()),
        "stable_fraction_at_resolution": int(stable[:, 2].sum()) / n,
        "stable_vertex_count_at_resolution": int(stable[:, 2].sum()),
        "mean_positive_regret_at_resolution": float(at.sum() / n),
        "max_positive_regret_at_resolution": float(at.max()),
        "max_stability_tolerance_at_resolution": float(tol.max()),
        "p95_positive_regret_at_resolution": _percentile(at, 0.95),
        "profitable_vertex_count_at_resolution": int((at > tol).sum()),
        "is_local_equilibrium_at_resolution": bool(stable[:, 2].all()),
        "near_tolerance_vertex_count": int((at <= 10.0 * tol).sum()),
        "n_vertices_scored": n,
        "n_vertices_graph": n,
        "n_communities": cover.n_communities,
        "max_memberships": cap,
        "cap_saturated_fraction": saturated / n,
        "cap_saturated_vertex_count": int(saturated),
        "allow_isolation": bool(allow_isolation),
        "gamma": selected_gamma,
        "interval": [lower, upper],
        "robustness_atol": float(atol),
        "robustness_rtol": float(rtol),
        "equilibrium_interval_lower": None,
        "equilibrium_interval_upper": None,
        "equilibrium_interval_width": None,
        "vertex_diagnostics": None,
    }
