"""Every call returns a certified equilibrium: an exact-arithmetic oracle.

Random small games (unit, mixed, wide-range 1e-300..1e300, tie-heavy and,
for partitions, signed and cancellation-heavy edge weights; unit, mixed,
wide, huge and dyadic node weights; the odd self-loop and multi-edge for
partitions; resolutions near ties and resolutions whose products are exact;
isolation on and off; no count limit, at most K or exactly K communities;
caps 1, 2, 3 and -1; default and random starts) run through the core's
traced entry points (``hedonic._core._trace_partition`` / ``_trace_cover``),
which also return the start state, every accepted move and the raw
equilibrium slack. In exact arithmetic the oracle checks:

(a) every accepted move strictly raises the exact potential (Fractions for
    partitions; for covers, whose payoffs involve 1/sqrt(k), Decimal with at
    least 80 significant digits, more when the terms span a wide range, and
    a rigorous margin for the Decimal roundings);
(b) at return, no feasible unilateral deviation raises a vertex's exact
    payoff by more than the slack, and the slack is exactly 0 for unit-weight
    partitions at resolutions whose products are exact (0, 1/2, 1, ...);
(c) the independent C verifier (``_core._verify_*``, native/core/check/check.c)
    certifies no gain above the slack.

Partition deviations are scored exactly. Cover deviations are every set of
at most ``max_memberships`` occupied labels, with fresh labels (payoff 0)
when isolation and the count limits allow them: all 2^L subsets are
enumerated with numpy, screened in floating point with a rigorous margin,
and the near-best ones decided exactly. The generator keeps n * cap <=
MAX_LABELS, so no game is ever skipped. The prefix lemma of
native/README.md (the best set is a prefix of the gains sorted in decreasing
order) is checked exhaustively in exact arithmetic.

``run_case`` / ``run_cases`` are also the engine of a longer sweep run
outside the suite.
"""

from __future__ import annotations

import itertools
import math
import random
import unittest
from array import array
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np

from hedonic import _core

MAX_LABELS = 20  # occupied labels of a cover game are at most n * cap
RESOLUTIONS = (-0.7, -0.05, 0.0, 0.02, 0.1, 0.35, 0.8, 2.0, 0.1462, 1 / 3, 2 / 3, 0.3)
EXACT_RESOLUTIONS = (0.0, 0.5, 0.25, 1.0, 2.0, -0.5, -1.0, 0.125)
CAPS = (1, 2, 3, -1)
LARGEST = {1: 14, 2: 10, 3: 6, -1: 4}  # vertices per cap: n * cap <= MAX_LABELS
LIMIT = 2.0 ** 1000  # |gamma| N^2 cap stays far inside the core's arithmetic bound


# ---- Random games -------------------------------------------------------------


def _edge_weights(rng, count, overlapping):
    mode = rng.choice(("unit", "unit", "mixed", "wide", "ties") +
                      (() if overlapping else ("signed", "cancel")))
    if mode == "unit":
        return None, mode
    if mode == "mixed":
        return [float(rng.randint(1, 3)) if rng.random() < 0.2 else rng.uniform(0.1, 2.1)
                for _ in range(count)], mode
    if mode == "wide":
        return [10.0 ** rng.uniform(-300, 300) for _ in range(count)], mode
    if mode == "ties":
        return [rng.choice((0.5, 1.0, 2.0, 3.0)) for _ in range(count)], mode
    if mode == "signed":
        return [rng.uniform(-1.0, 2.0) for _ in range(count)], mode
    big = 2.0 ** rng.randint(20, 60)  # "cancel": large opposite weights and small ones
    return [rng.choice((big, -big, big + 1.0, -(big - 1.0), 1e-3, rng.uniform(-1, 1) * big))
            for _ in range(count)], mode


def _node_weights(rng, n):
    mode = rng.choice(("unit", "unit", "mixed", "wide", "huge", "dyadic"))
    if mode == "unit":
        return None, mode
    if mode == "mixed":
        return [0.0 if rng.random() < 0.15 else rng.uniform(0.2, 2.2) for _ in range(n)], mode
    if mode == "wide":
        return [10.0 ** rng.uniform(-300, 150) for _ in range(n)], mode
    if mode == "huge":
        return [10.0 ** rng.uniform(100, 150) for _ in range(n)], mode
    return [rng.choice((0.0, 0.5, 1.0, 2.0)) for _ in range(n)], mode


def random_game(case: int) -> dict:
    """A random small game; n * cap <= MAX_LABELS for covers."""
    rng = random.Random(case)
    cap = rng.choice(CAPS)
    overlapping = cap != 1
    n = rng.randint(2, LARGEST[cap])
    effective = n if cap == -1 else min(cap, n)
    p = rng.uniform(0.15, 0.85)
    while True:
        edges = []
        for a in range(n):
            for b in range(a, n):
                if a == b and (overlapping or rng.random() > 0.1):
                    continue
                if rng.random() < p:
                    edges.append((a, b))
        if edges and rng.random() < 0.2:
            edges.append(edges[0])  # a multi-edge
        if not edges:
            continue
        weights, edge_mode = _edge_weights(rng, len(edges), overlapping)
        if weights is None or sum(weights) > 0:
            break
    node_weights, node_mode = _node_weights(rng, n)
    exact_arithmetic = (not overlapping and weights is None and node_mode in ("unit", "dyadic")
                        and rng.random() < 0.6)
    if exact_arithmetic:
        gamma = rng.choice(EXACT_RESOLUTIONS)
    elif rng.random() < 0.2:
        gamma = rng.randint(1, 5) / rng.randint(1, 7)  # near a tie of small integer gains
    else:
        gamma = rng.choice(RESOLUTIONS)
    total = sum(node_weights) if node_weights else float(n)
    while abs(gamma) * max(total, 1.0) ** 2 * effective > LIMIT:
        gamma /= 2.0 ** 64
    isolation = rng.random() < 0.5
    limit = rng.random()
    max_total = exact = None
    if limit < 0.25:
        max_total = rng.randint(1, n)
    elif limit < 0.5:
        exact = rng.randint(1, n * effective if overlapping else n)
    start = None
    if rng.random() < 0.35:
        target = exact or max_total or n
        rows = []
        for v in range(n):
            row = [v % target] + list(range(v + n, target, n))
            size = rng.randint(len(row), max(len(row), effective))
            while len(row) < size:
                row.append(rng.randrange(target))
            rows.append(sorted(set(row)))
        if overlapping:
            start = rows
        else:
            flat = [row[0] for row in rows]
            relabel = {c: i for i, c in enumerate(sorted(set(flat)))}
            start = [relabel[c] for c in flat]
            if exact is not None and len(relabel) != exact:
                start = None
    return dict(case=case, n=n, edges=edges, weights=weights, node_weights=node_weights,
                gamma=gamma, cap=effective, overlapping=overlapping, isolation=isolation,
                max_total=max_total, exact=exact, start=start, edge_mode=edge_mode,
                node_mode=node_mode, exact_arithmetic=exact_arithmetic)


def core_arguments(game):
    flat = array("q", [x for edge in game["edges"] for x in edge])
    kwargs = dict(
        edge_weights=None if game["weights"] is None else array("d", game["weights"]),
        node_weights=None if game["node_weights"] is None else array("d", game["node_weights"]),
        resolution=game["gamma"], max_total_communities=game["max_total"],
        n_communities=game["exact"], allow_isolation=game["isolation"])
    return flat, kwargs


def run_core(game, seed):
    """The traced run: (rows, quality, record)."""
    flat, kwargs = core_arguments(game)
    rng = random.Random(seed)
    if game["overlapping"]:
        rows, _, quality, record = _core._trace_cover(
            game["n"], flat, rng, memberships=game["start"], max_memberships=game["cap"],
            **kwargs)
        rows = [list(row) for row in rows]
    else:
        membership, _, quality, record = _core._trace_partition(
            game["n"], flat, rng, membership=game["start"], **kwargs)
        rows = [[c] for c in membership]
    return rows, quality, record


def _node(game, v):
    return game["node_weights"][v] if game["node_weights"] else 1.0


def _weight(game, e):
    return game["weights"][e] if game["weights"] else 1.0


def float_quality(game, rows):
    """The documented quality, in floating point (loops count twice in q)."""
    total = sum(game["weights"]) if game["weights"] else float(len(game["edges"]))
    q, mass = 0.0, Counter()
    for e, (a, b) in enumerate(game["edges"]):
        shared = len(set(rows[a]) & set(rows[b]))
        q += 2.0 * _weight(game, e) * shared / math.sqrt(len(rows[a]) * len(rows[b]))
    for v in range(game["n"]):
        for c in rows[v]:
            mass[c] += _node(game, v) / math.sqrt(len(rows[v]))
    return (q - game["gamma"] * sum(m * m for m in mass.values())) / (2.0 * total)


# ---- Partitions: Fractions ------------------------------------------------------


def partition_potential(game, labels):
    """sum of internal edge weights - gamma / 2 sum_C N(C)^2, exactly (loops left out)."""
    total = Fraction(0)
    for e, (a, b) in enumerate(game["edges"]):
        if a != b and labels[a] == labels[b]:
            total += Fraction(_weight(game, e))
    mass = Counter()
    for v in range(game["n"]):
        mass[labels[v]] += Fraction(_node(game, v))
    return total - Fraction(game["gamma"]) / 2 * sum(m * m for m in mass.values())


def partition_gains(game, labels):
    """The exact gain of each vertex's best feasible deviation (None: it has none)."""
    n, gamma = game["n"], Fraction(game["gamma"])
    mass, size = Counter(), Counter(labels)
    for v in range(n):
        mass[labels[v]] += Fraction(_node(game, v))
    k = len(size)
    gains = []
    for v in range(n):
        own, alone = labels[v], size[labels[v]] == 1
        if game["exact"] is not None and alone:
            gains.append(None)  # leaving would empty a community
            continue
        to = Counter()
        for e, (a, b) in enumerate(game["edges"]):
            if a == v and b != v:
                to[labels[b]] += Fraction(_weight(game, e))
            elif b == v and a != v:
                to[labels[a]] += Fraction(_weight(game, e))
        nv = Fraction(_node(game, v))
        current = to[own] - gamma * nv * (mass[own] - nv)
        best = None
        for c in size:
            if c != own:
                value = to[c] - gamma * nv * mass[c]
                best = value if best is None or value > best else best
        if (game["isolation"] and game["exact"] is None and not alone
                and (game["max_total"] is None or k < game["max_total"])):
            best = Fraction(0) if best is None or best < 0 else best
        gains.append(None if best is None else best - current)
    return gains


# ---- Covers: Decimal --------------------------------------------------------------


def precision(game):
    """Significant digits: 100 plus the decimal range of the game's terms."""
    logs = [math.log10(abs(w)) for w in (game["weights"] or [1.0]) if w != 0.0]
    nodes = [w for w in (game["node_weights"] or [1.0]) if w > 0.0] or [1.0]
    gamma = abs(game["gamma"])
    if gamma > 0.0:
        logs += [math.log10(gamma) + 2 * math.log10(min(nodes)),
                 math.log10(gamma) + 2 * math.log10(sum(nodes)) + math.log10(game["cap"])]
    return 100 + int(max(logs) - min(logs))


class CoverOracle:
    """The cover game in Decimal at `digits` significant digits. Every value
    comes with a tolerance that rigorously (and generously) bounds the
    Decimal rounding: each operation errs by at most half a unit in the last
    place, and a value is a sum of at most `terms` products of a few factors."""

    def __init__(self, game, digits=None):
        self.game = game
        self.digits = digits or precision(game)
        self.unit = Decimal(10) ** (6 - self.digits)

    def potential(self, rows):
        """(potential, tolerance); self-loops are absent from covers."""
        g = self.game
        with localcontext() as ctx:
            ctx.prec = self.digits
            inv = {k: 1 / Decimal(k).sqrt() for k in {len(r) for r in rows}}
            total, scale = Decimal(0), Decimal(0)
            for e, (a, b) in enumerate(g["edges"]):
                w = Decimal(_weight(g, e))
                shared = len(set(rows[a]) & set(rows[b]))
                total += w * shared * inv[len(rows[a])] * inv[len(rows[b])]
                scale += abs(w) * shared
            mass = {}
            for v in range(g["n"]):
                term = Decimal(_node(g, v)) * inv[len(rows[v])]
                for c in rows[v]:
                    mass[c] = mass.get(c, Decimal(0)) + term
            gamma = Decimal(g["gamma"])
            crowd = sum((m * m for m in mass.values()), Decimal(0))
            total -= gamma / 2 * crowd
            scale += abs(gamma) * crowd
            terms = len(g["edges"]) + g["n"] * g["cap"] + 50
            return total, terms * self.unit * scale

    def vertex(self, rows, v):
        """(labels, gains, current, sole, tolerance): every occupied label's
        gain g_c (v's own contribution left out), v's current payoff, which
        labels v holds alone, and the tolerance of every value."""
        g = self.game
        with localcontext() as ctx:
            ctx.prec = self.digits
            labels = sorted({c for row in rows for c in row})
            holders = {c: [u for u in range(g["n"]) if c in rows[u]] for c in labels}
            inv = {k: 1 / Decimal(k).sqrt() for k in range(1, g["cap"] + 1)}
            nv, gamma = Decimal(_node(g, v)), Decimal(g["gamma"])
            support = dict.fromkeys(labels, Decimal(0))
            scale = Decimal(0)
            for e, (a, b) in enumerate(g["edges"]):
                u = b if a == v else a if b == v else None
                if u is None or u == v:
                    continue
                w = Decimal(_weight(g, e))
                scale += abs(w)
                for c in rows[u]:
                    support[c] += w * inv[len(rows[u])]
            scale += abs(gamma) * nv * sum((Decimal(_node(g, u)) for u in range(g["n"])),
                                           Decimal(0))
            gains = []
            for c in labels:
                crowd = sum((Decimal(_node(g, u)) * inv[len(rows[u])] for u in holders[c]
                             if u != v), Decimal(0))
                gains.append(support[c] - gamma * nv * crowd)
            own = set(rows[v])
            current = sum((x for c, x in zip(labels, gains) if c in own), Decimal(0))
            current *= inv[len(rows[v])]
            sole = [c in own and holders[c] == [v] for c in labels]
            terms = (len(g["edges"]) + g["n"]) * g["cap"] + 50
            return labels, gains, current, sole, terms * self.unit * scale

    def check_vertex(self, rows, v, slack):
        """(ok, exact sets decided): ok when no feasible set of labels beats
        v's row by more than `slack`, by exhaustive enumeration."""
        g = self.game
        labels, gains, current, sole, tolerance = self.vertex(rows, v)
        L, cap = len(labels), g["cap"]
        k, n_sole = len(labels), sum(sole)
        values = np.array([float(x) for x in gains])
        sums = np.zeros(1)
        pop = np.zeros(1, dtype=np.int64)
        kept = np.zeros(1, dtype=np.int64)
        for i in range(L):  # subset sums by doubling: bit i of the index is label i
            sums = np.concatenate([sums, sums + values[i]])
            pop = np.concatenate([pop, pop + 1])
            kept = np.concatenate([kept, kept + int(sole[i])])
        feasible = (pop >= 1) & (pop <= cap)
        if g["exact"] is not None:
            feasible &= kept == n_sole  # labels v holds alone must stay occupied
        if g["max_total"] is not None:
            feasible &= k - (n_sole - kept) <= g["max_total"]
        value0 = np.where(feasible, sums / np.sqrt(np.maximum(pop, 1)), -np.inf)
        # Fresh labels score 0: the set with f of them is worth sum / sqrt(|S| + f),
        # best at f = 0 or at the largest f allowed.
        fresh = np.zeros_like(pop)
        if g["isolation"] and g["exact"] is None:
            fresh = cap - pop
            if g["max_total"] is not None:
                fresh = np.minimum(fresh, g["max_total"] - (k - (n_sole - kept)))
        ok_fresh = fresh >= 1
        value_fresh = np.where(ok_fresh, sums / np.sqrt(np.maximum(pop + fresh, 1)), -np.inf)
        value = np.maximum(value0, value_fresh)
        best_float = float(value.max())
        # Float error of a subset value: L additions, a sqrt and a division.
        margin = 4.0 * (L + 8) * 2.0 ** -52 * float(np.abs(values).sum()) + 1e-300
        with localcontext() as ctx:
            ctx.prec = self.digits
            limit = current + Decimal(slack) + 4 * tolerance
            if best_float == -math.inf or Decimal(best_float) + Decimal(margin) <= limit:
                return True, 0
            # Decide the near-best sets exactly; sets with equal gain
            # multisets have equal values, so one of each.
            near = np.nonzero(value >= best_float - 2.0 * margin)[0]
            classes = {}
            label_class = [classes.setdefault(x, len(classes)) for x in gains]
            seen, checked = set(), 0
            for index in near.tolist():
                members = [i for i in range(L) if index >> i & 1]
                options = []
                if value0[index] > -np.inf:
                    options.append(0)
                if ok_fresh[index]:
                    options.append(int(fresh[index]))
                for f in options:
                    key = (tuple(sorted(label_class[i] for i in members)), f)
                    if key in seen:
                        continue
                    seen.add(key)
                    checked += 1
                    exact = (sum((gains[i] for i in members), Decimal(0)) /
                             Decimal(len(members) + f).sqrt())
                    if exact > limit:
                        return False, checked
            return True, checked


# ---- One case -----------------------------------------------------------------------


class OracleFailure(AssertionError):
    pass


def structure(rows):
    """The cover as a multiset of member sets (labels do not matter)."""
    members = {}
    for v, row in enumerate(rows):
        for c in row:
            members.setdefault(c, set()).add(v)
    return Counter(frozenset(s) for s in members.values())


def run_case(case: int, game: dict | None = None) -> Counter:
    """Runs one game through the oracle; returns counts, raises OracleFailure."""
    game = game or random_game(case)
    context = {k: game.get(k) for k in ("case", "n", "cap", "gamma", "isolation", "max_total",
                                        "exact", "edge_mode", "node_mode")}

    def fail(*what):
        raise OracleFailure((context,) + what)

    rows, quality, record = run_core(game, seed=game.get("case", case))
    slack = record["raw_slack"]
    counts = Counter(kind="cover" if game["overlapping"] else "partition")
    if not slack >= 0.0:
        fail("slack", slack)
    used = {c for row in rows for c in row}
    if game["exact"] is not None and len(used) != game["exact"]:
        fail("exact count")
    if game["max_total"] is not None and len(used) > game["max_total"]:
        fail("count limit")
    if not all(1 <= len(row) <= game["cap"] for row in rows):
        fail("cap")
    if game["edge_mode"] in ("unit", "mixed", "ties", "signed") and game["node_mode"] in (
            "unit", "mixed", "dyadic"):
        expected = float_quality(game, rows)
        if not math.isclose(quality, expected, rel_tol=1e-9, abs_tol=1e-9):
            fail("quality", quality, expected)

    state = [list(row) for row in record["start"]]
    flat, kwargs = core_arguments(game)
    if not game["overlapping"]:
        before = partition_potential(game, [r[0] for r in state])
        for v, row, improvement, bound in record["moves"]:
            if not improvement > bound:
                fail("uncertified move", improvement, bound)
            state[v] = list(row)
            after = partition_potential(game, [r[0] for r in state])
            if not after > before:  # (a)
                fail("a move did not raise the potential", v, after - before)
            before = after
            counts["moves"] += 1
        if structure(state) != structure(rows):
            fail("the trace does not end at the result")
        membership = [row[0] for row in rows]
        for gain in partition_gains(game, membership):  # (b)
            if gain is not None:
                counts["vertices"] += 1
                if gain > Fraction(slack):
                    fail("a deviation beats the slack", gain, slack)
        if game["exact_arithmetic"]:
            counts["exact_games"] += 1
            if slack != 0.0:
                fail("nonzero slack in exact arithmetic", slack)
        report = _core._verify_partition(game["n"], flat, membership, **kwargs)
    else:
        oracle = CoverOracle(game)
        before, tol_before = oracle.potential(state)
        for v, row, improvement, bound in record["moves"]:
            if not improvement > bound:
                fail("uncertified move", improvement, bound)
            state[v] = list(row)
            after, tol_after = oracle.potential(state)
            if not after - before > tol_before + tol_after:  # (a)
                fail("a move did not raise the potential", v, after - before)
            before, tol_before = after, tol_after
            counts["moves"] += 1
        if structure(state) != structure(rows):
            fail("the trace does not end at the result")
        for v in range(game["n"]):  # (b)
            ok, decided = oracle.check_vertex(rows, v, slack)
            counts["vertices"] += 1
            counts["exact_sets"] += decided
            if not ok:
                fail("a deviation beats the slack", v, slack)
        report = _core._verify_cover(game["n"], flat, rows, game["cap"], **kwargs)
    if report["max_certified"] > slack:  # (c)
        fail("the verifier certifies a gain above the slack", report, slack)
    counts["verified_vertices"] += report["vertices"]
    counts["improvable"] += report["improvable"]
    if slack > 0.0:
        counts["positive_slack"] += 1
    return counts


def run_cases(cases, first=0):
    """Totals over cases first .. first + cases - 1: games per kind, accepted
    moves checked, vertices checked, exact set decisions, ..."""
    totals = Counter()
    for case in range(first, first + cases):
        counts = run_case(case)
        totals[counts.pop("kind")] += 1
        totals.update(counts)
    return totals


# ---- The prefix lemma ---------------------------------------------------------------


def at_least(a, j, b, k):
    """a / sqrt(j) >= b / sqrt(k), exactly (a, b rational; j, k >= 1)."""
    if (a >= 0) != (b >= 0):
        return a >= 0
    if a >= 0:
        return a * a * k >= b * b * j
    return a * a * k <= b * b * j


def best_set(gains, cap):
    """max over non-empty sets S, |S| <= cap, of sum(S) / sqrt(|S|), as (sum, |S|)."""
    best = None
    for size in range(1, cap + 1):
        for subset in itertools.combinations(gains, size):
            candidate = (sum(subset, Fraction(0)), size)
            if best is None or not at_least(*best, *candidate):
                best = candidate
    return best


def best_prefix(gains, cap):
    """max over k <= cap of (sum of the k largest gains) / sqrt(k)."""
    ordered = sorted(gains, reverse=True)
    best = None
    for size in range(1, min(cap, len(ordered)) + 1):
        candidate = (sum(ordered[:size], Fraction(0)), size)
        if best is None or not at_least(*best, *candidate):
            best = candidate
    return best


def lemma_holds(gains, cap):
    a, b = best_set(gains, cap), best_prefix(gains, cap)
    return at_least(*a, *b) and at_least(*b, *a)


# ---- Tests ----------------------------------------------------------------------------


class TestEquilibrium(unittest.TestCase):
    def test_random_games(self):
        totals = run_cases(400)
        self.assertGreater(totals["partition"], 50)
        self.assertGreater(totals["cover"], 150)
        self.assertGreater(totals["moves"], 1000)
        self.assertGreater(totals["exact_games"], 10)

    def test_generator_never_skips(self):
        for case in range(400):
            game = random_game(case)
            self.assertLessEqual(game["n"] * game["cap"] if game["overlapping"] else game["n"],
                                 MAX_LABELS)

    def test_exact_arithmetic_has_zero_slack(self):
        import igraph as ig
        from hedonic import Game

        for gamma in EXACT_RESOLUTIONS:
            for seed in range(3):
                result = Game(ig.Graph.Famous("Zachary")).community_hedonic(
                    resolution=gamma, seed=seed, allow_isolation=bool(seed % 2))
                self.assertEqual(result._hedonic_provenance["equilibrium_slack"], 0.0)

    def test_prefix_lemma_exhaustive(self):
        values = [Fraction(x) for x in (-3, -2, -1, Fraction(-1, 2), 0, Fraction(1, 2), 1, 2, 3)]
        checked = 0
        for size in range(1, 7):
            for gains in itertools.combinations_with_replacement(values, size):
                for cap in range(1, size + 1):
                    self.assertTrue(lemma_holds(gains, cap), (gains, cap))
                    checked += 1
        self.assertEqual(checked, 27_027)  # sum over sizes s of C(s + 8, s) * s

    def test_prefix_lemma_random_rationals(self):
        rng = random.Random(7)
        for _ in range(400):
            size = rng.randint(1, 9)
            gains = [Fraction(rng.randint(-40, 40), rng.randint(1, 9)) for _ in range(size)]
            cap = rng.randint(1, size)
            self.assertTrue(lemma_holds(gains, cap), (gains, cap))

    def test_oracle_detects_a_non_equilibrium(self):
        # Two triangles joined by an edge: splitting a triangle is not stable.
        game = dict(n=6, edges=[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)],
                    weights=None, node_weights=None, gamma=0.1, cap=2, isolation=False,
                    max_total=None, exact=None)
        gains = partition_gains(game, [0, 0, 1, 1, 1, 1])
        self.assertGreater(max(g for g in gains if g is not None), 0)
        oracle = CoverOracle(game, 80)
        self.assertFalse(oracle.check_vertex([[0], [0], [1], [1], [1], [1]], 2, 1e-9)[0])
        self.assertTrue(oracle.check_vertex([[0], [0], [0], [1], [1], [1]], 2, 1e-9)[0])


if __name__ == "__main__":
    unittest.main()
