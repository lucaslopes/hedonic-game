"""Independent paper audit oracle, pure Python and NumPy (no numba).

Ported from experiments/overlapping/{robustness,stability}.py. The state
builder uses the original dictionary contribution loop, so it independently
checks the C adjacency and sparse accumulator. Never imported by the package.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np

FRESH_LABEL = -1
REGRET_EPS = 1e-15

def _current_sum(values):
    """Freeze the supplied reference's CPython 3.12 sum on older Pythons too."""
    high = low = 0.0
    for value in values:
        total = high + value
        low += ((high - total) + value if abs(high) >= abs(value)
                else (value - total) + high)
        high = total
    return high + low


@dataclass
class FractionalState:
    memberships: list
    masses: list
    support: list
    intensities: list
    lightest_labels: list

    @property
    def n_vertices(self):
        return len(self.memberships)

    @property
    def n_communities(self):
        return len(self.masses)


def build_fractional_state(graph, memberships, *, edge_weights=None):
    rows = [tuple(sorted(set(row))) for row in memberships]
    if len(rows) != graph.vcount() or any(not row for row in rows):
        raise ValueError("non-empty row required for every vertex")
    labels = {c for row in rows for c in row}
    if labels != set(range(len(labels))):
        raise ValueError("contiguous labels required")
    masses = [0.0] * len(labels)
    intensity = [1.0 / math.sqrt(len(row)) for row in rows]
    for v, row in enumerate(rows):
        for c in row:
            masses[c] += intensity[v]
    weights = edge_weights
    if weights is None:
        weights = graph.es["weight"] if "weight" in graph.edge_attributes() else [1.0] * graph.ecount()
    supports = [{} for _ in rows]
    for (first, second), weight in zip(graph.get_edgelist(), weights):
        if weight == 0:
            continue
        for source, target in ((first, second), (second, first)):
            term = float(weight) * intensity[source]
            for c in rows[source]:
                supports[target][c] = supports[target].get(c, 0.0) + term
    return FractionalState(rows, masses, supports,
                           [{c: intensity[v] for c in row} for v, row in enumerate(rows)],
                           sorted(labels, key=lambda c: (masses[c], c)))

def _candidate_labels(
    state: FractionalState,
    vertex: int,
    gamma: float,
    max_memberships: int,
    allow_isolation: bool,
    *,
    dense: bool,
) -> list[int]:
    """Return an exact candidate-label superset for prefix optimization.

    In dense mode this is every active label.  In sparse mode, neighbor/current
    labels plus the lightest-mass labels suffice: unseen labels have no support
    and therefore non-positive score for gamma >= 0.  If all scores are
    non-positive, the best action has cardinality one, so the lightest unseen
    label is enough.
    """
    n_communities = state.n_communities
    if dense or n_communities <= max(256, max_memberships * 8):
        labels = set(range(n_communities))
    else:
        labels = set(state.memberships[vertex])
        labels.update(state.support[vertex])
        labels.update(state.lightest_labels[: max_memberships + len(labels) + 1])
    if allow_isolation:
        labels.add(FRESH_LABEL)
    return sorted(labels)


def _label_score(
    state: FractionalState,
    vertex: int,
    label: int,
    gamma: float,
) -> float:
    if label == FRESH_LABEL:
        return 0.0
    support = state.support[vertex].get(label, 0.0)
    own_intensity = state.intensities[vertex].get(label, 0.0)
    crowding = state.masses[label] - own_intensity
    return support - float(gamma) * crowding


def _utility_for_labels(
    state: FractionalState,
    vertex: int,
    labels: Sequence[int],
    gamma: float,
) -> float:
    if not labels:
        raise ValueError("a membership action must be non-empty")
    return sum(_label_score(state, vertex, label, gamma) for label in labels) / math.sqrt(
        len(labels)
    )


def best_response(
    state: FractionalState,
    vertex: int,
    gamma: float,
    max_memberships: int,
    allow_isolation: bool,
    *,
    dense: bool = False,
) -> dict[str, Any]:
    """Return the exact best prefix response and regret for one vertex."""
    if not 0 <= vertex < state.n_vertices:
        raise IndexError("vertex outside state")
    if max_memberships < 1:
        raise ValueError("max_memberships must be positive")
    current = state.memberships[vertex]
    if len(current) > max_memberships:
        raise ValueError("current membership exceeds max_memberships")
    labels = _candidate_labels(
        state,
        vertex,
        gamma,
        max_memberships,
        allow_isolation,
        dense=dense,
    )
    scored = sorted(
        ((label, _label_score(state, vertex, label, gamma)) for label in labels),
        key=lambda item: (-item[1], item[0]),
    )
    best_utility = -math.inf
    best_labels: tuple[int, ...] = ()
    prefix = 0.0
    for cardinality, (label, score) in enumerate(scored[:max_memberships], 1):
        prefix += score
        utility = prefix / math.sqrt(cardinality)
        if utility > best_utility + 1e-15:
            best_utility = utility
            best_labels = tuple(item[0] for item in scored[:cardinality])
    if not best_labels:
        raise ValueError("no admissible membership action")
    current_utility = _utility_for_labels(state, vertex, current, gamma)
    if current_utility + 1e-15 >= best_utility:
        best_utility = current_utility
        best_labels = current
        regret = 0.0
    else:
        regret = best_utility - current_utility
    return {
        "vertex": int(vertex),
        "gamma": float(gamma),
        "current_memberships": list(current),
        "best_memberships": list(best_labels),
        "best_utility": float(best_utility),
        "current_utility": float(current_utility),
        "regret": float(regret),
        "candidate_label_count": len(labels),
        "dense_candidates": bool(dense or state.n_communities <= max(256, max_memberships * 8)),
    }


def _tolerance(value: float, atol: float, rtol: float) -> float:
    return float(atol + rtol * max(1.0, abs(value)))


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


def vertex_affine(state: FractionalState, vertex: int, cap: int, allow_isolation: bool, *,
                  prune: bool = True) -> VertexAffine:
    """The gamma-affine data of ``vertex``'s candidate actions.

    ``_candidate_labels`` is a superset: besides the vertex's own and neighbour labels it adds the ``cap + |labels| + 1``
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
        labels = _candidate_labels(state, vertex, 0.0, cap, allow_isolation, dense=False)
    a, b = [], []
    for label in labels:
        if label == FRESH_LABEL:
            a.append(0.0)
            b.append(0.0)
        else:
            a.append(float(state.support[vertex].get(label, 0.0)))
            b.append(float(state.masses[label] - state.intensities[vertex].get(label, 0.0)))
    current = state.memberships[vertex]
    norm = math.sqrt(len(current))
    index = {label: i for i, label in enumerate(labels)}
    cur_a = _current_sum(a[index[label]] for label in current) / norm
    cur_b = _current_sum(b[index[label]] for label in current) / norm
    return VertexAffine(tuple(labels), tuple(a), tuple(b), float(cur_a), float(cur_b), int(cap))


def _relevant_labels(state: FractionalState, vertex: int, allow_isolation: bool) -> list[int]:
    """The candidate labels that can matter for ``vertex`` (see ``vertex_affine``), ascending.

    Its own labels, the labels with positive support among its neighbours, the fresh label when isolation is allowed,
    and the lightest of the remaining labels: those all have zero support and are not held, so their score is
    ``-gamma * mass`` and the lightest is the first of ``state.lightest_labels`` (sorted by mass, then label) that is
    none of the above.  This is the set ``_candidate_labels`` returns with its zero-support filler reduced to that
    one label, built without enumerating the filler.
    """
    keep = set(state.memberships[vertex])
    keep.update(label for label, value in state.support[vertex].items() if value > 0.0)
    if allow_isolation:
        keep.add(FRESH_LABEL)
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
    state: FractionalState,
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


def build_entries(state: FractionalState, cap: int, allow_isolation: bool,
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


def compiled_regrets(entries, gamma, atol=1e-10, rtol=1e-9):
    """The numba _regret_kernel's scalar loops, without compilation."""
    regret = np.empty(entries.n)
    tolerance = np.empty(entries.n)
    for v in range(entries.n):
        lo = entries.starts[v]
        m = entries.starts[v + 1] - lo
        score = np.empty(m)
        current = 0.0
        for i in range(m):
            score[i] = entries.a[lo + i] - gamma * entries.b[lo + i]
            if entries.is_current[lo + i]:
                current += score[i]
        current = current / entries.current_norm[v]
        order = np.argsort(-score)
        best = -np.inf
        prefix = 0.0
        for k in range(min(entries.cap, m)):
            prefix += score[order[k]]
            utility = prefix / np.sqrt(k + 1.0)
            if utility > best:
                best = utility
        diff = best - current
        regret[v] = diff if diff > REGRET_EPS else 0.0
        tolerance[v] = atol + rtol * max(1.0, abs(current))
    return regret, tolerance
