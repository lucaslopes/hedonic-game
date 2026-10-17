"""Every call returns a certified equilibrium: an exact-arithmetic oracle.

Random small games (unit, mixed, wide-range 1e-300..1e300, tie-heavy,
signed and cancellation-heavy edge weights; unit, mixed, wide, huge and
dyadic node weights; the odd self-loop and multi-edge; resolutions near ties
and resolutions whose products are exact; isolation on and off; no count
limit, at most K or exactly K communities; default and random starts) run
through the core's traced entry point (``hedonic._core._trace_partition``),
which also returns the start state, every accepted move and the raw
equilibrium slack. In exact arithmetic (Fractions) the oracle checks:

(a) every accepted move strictly raises the exact potential;
(b) at return, no feasible unilateral deviation raises a vertex's exact
    payoff by more than the slack, and the slack is exactly 0 for unit-weight
    games at resolutions whose products are exact (0, 1/2, 1, ...);
(c) the independent C verifier (``_core._verify_partition``,
    native/core/check.c) certifies no gain above the slack.

``run_case`` / ``run_cases`` are also the engine of a longer sweep run
outside the suite.
"""

from __future__ import annotations

import math
import random
import unittest
from array import array
from collections import Counter
from fractions import Fraction

from hedonic import _core

LARGEST = 14  # vertices of a random game at most
RESOLUTIONS = (-0.7, -0.05, 0.0, 0.02, 0.1, 0.35, 0.8, 2.0, 0.1462, 1 / 3, 2 / 3, 0.3)
EXACT_RESOLUTIONS = (0.0, 0.5, 0.25, 1.0, 2.0, -0.5, -1.0, 0.125)
LIMIT = 2.0 ** 1000  # |gamma| N^2 stays far inside the core's arithmetic bound


# ---- Random games -------------------------------------------------------------


def _edge_weights(rng, count):
    mode = rng.choice(("unit", "unit", "mixed", "wide", "ties", "signed", "cancel"))
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
    """A random small game of at most LARGEST vertices."""
    rng = random.Random(case)
    n = rng.randint(2, LARGEST)
    p = rng.uniform(0.15, 0.85)
    while True:
        edges = []
        for a in range(n):
            for b in range(a, n):
                if a == b and rng.random() > 0.1:
                    continue
                if rng.random() < p:
                    edges.append((a, b))
        if edges and rng.random() < 0.2:
            edges.append(edges[0])  # a multi-edge
        if not edges:
            continue
        weights, edge_mode = _edge_weights(rng, len(edges))
        if weights is None or sum(weights) > 0:
            break
    node_weights, node_mode = _node_weights(rng, n)
    exact_arithmetic = (weights is None and node_mode in ("unit", "dyadic")
                        and rng.random() < 0.6)
    if exact_arithmetic:
        gamma = rng.choice(EXACT_RESOLUTIONS)
    elif rng.random() < 0.2:
        gamma = rng.randint(1, 5) / rng.randint(1, 7)  # near a tie of small integer gains
    else:
        gamma = rng.choice(RESOLUTIONS)
    total = sum(node_weights) if node_weights else float(n)
    while abs(gamma) * max(total, 1.0) ** 2 > LIMIT:
        gamma /= 2.0 ** 64
    isolation = rng.random() < 0.5
    limit = rng.random()
    max_total = exact = None
    if limit < 0.25:
        max_total = rng.randint(1, n)
    elif limit < 0.5:
        exact = rng.randint(1, n)
    start = None
    if rng.random() < 0.35:
        target = exact or max_total or n
        flat = [v % target for v in range(n)]
        for v in range(target, n):
            flat[v] = rng.randrange(target)
        relabel = {c: i for i, c in enumerate(sorted(set(flat)))}
        start = [relabel[c] for c in flat]
    return dict(case=case, n=n, edges=edges, weights=weights, node_weights=node_weights,
                gamma=gamma, isolation=isolation, max_total=max_total, exact=exact,
                start=start, edge_mode=edge_mode, node_mode=node_mode,
                exact_arithmetic=exact_arithmetic)


def core_arguments(game):
    flat = array("q", [x for edge in game["edges"] for x in edge])
    kwargs = dict(
        edge_weights=None if game["weights"] is None else array("d", game["weights"]),
        node_weights=None if game["node_weights"] is None else array("d", game["node_weights"]),
        resolution=game["gamma"], max_total_communities=game["max_total"],
        n_communities=game["exact"], allow_isolation=game["isolation"])
    return flat, kwargs


def run_core(game, seed):
    """The traced run: (membership, quality, record)."""
    flat, kwargs = core_arguments(game)
    membership, _, quality, record = _core._trace_partition(
        game["n"], flat, random.Random(seed), membership=game["start"], **kwargs)
    return list(membership), quality, record


def _node(game, v):
    return game["node_weights"][v] if game["node_weights"] else 1.0


def _weight(game, e):
    return game["weights"][e] if game["weights"] else 1.0


def float_quality(game, labels):
    """The documented quality, in floating point (loops count twice in q)."""
    total = sum(game["weights"]) if game["weights"] else float(len(game["edges"]))
    q, mass = 0.0, Counter()
    for e, (a, b) in enumerate(game["edges"]):
        if labels[a] == labels[b]:
            q += 2.0 * _weight(game, e)
    for v in range(game["n"]):
        mass[labels[v]] += _node(game, v)
    return (q - game["gamma"] * sum(m * m for m in mass.values())) / (2.0 * total)


# ---- The exact game: Fractions --------------------------------------------------


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


# ---- One case -----------------------------------------------------------------------


class OracleFailure(AssertionError):
    pass


def structure(labels):
    """The partition as a multiset of member sets (labels do not matter)."""
    members = {}
    for v, c in enumerate(labels):
        members.setdefault(c, set()).add(v)
    return Counter(frozenset(s) for s in members.values())


def run_case(case: int, game: dict | None = None) -> Counter:
    """Runs one game through the oracle; returns counts, raises OracleFailure."""
    game = game or random_game(case)
    context = {k: game.get(k) for k in ("case", "n", "gamma", "isolation", "max_total",
                                        "exact", "edge_mode", "node_mode")}

    def fail(*what):
        raise OracleFailure((context,) + what)

    membership, quality, record = run_core(game, seed=game.get("case", case))
    slack = record["raw_slack"]
    counts = Counter(games=1)
    if not slack >= 0.0:
        fail("slack", slack)
    used = set(membership)
    if game["exact"] is not None and len(used) != game["exact"]:
        fail("exact count")
    if game["max_total"] is not None and len(used) > game["max_total"]:
        fail("count limit")
    if game["edge_mode"] in ("unit", "mixed", "ties", "signed") and game["node_mode"] in (
            "unit", "mixed", "dyadic"):
        expected = float_quality(game, membership)
        if not math.isclose(quality, expected, rel_tol=1e-9, abs_tol=1e-9):
            fail("quality", quality, expected)

    state = [row[0] for row in record["start"]]
    flat, kwargs = core_arguments(game)
    before = partition_potential(game, state)
    for v, row, improvement, bound in record["moves"]:
        if not improvement > bound:
            fail("uncertified move", improvement, bound)
        state[v] = row[0]
        after = partition_potential(game, state)
        if not after > before:  # (a)
            fail("a move did not raise the potential", v, after - before)
        before = after
        counts["moves"] += 1
    if structure(state) != structure(membership):
        fail("the trace does not end at the result")
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
    if report["max_certified"] > slack:  # (c)
        fail("the verifier certifies a gain above the slack", report, slack)
    counts["verified_vertices"] += report["vertices"]
    counts["improvable"] += report["improvable"]
    if slack > 0.0:
        counts["positive_slack"] += 1
    return counts


def run_cases(cases, first=0):
    """Totals over cases first .. first + cases - 1: games, accepted moves
    checked, vertices checked, games in exact arithmetic, ..."""
    totals = Counter()
    for case in range(first, first + cases):
        totals.update(run_case(case))
    return totals


# ---- Tests ----------------------------------------------------------------------------


class TestEquilibrium(unittest.TestCase):
    def test_random_games(self):
        totals = run_cases(400)
        self.assertEqual(totals["games"], 400)
        self.assertGreater(totals["moves"], 1000)
        self.assertGreater(totals["exact_games"], 10)
        self.assertEqual(totals["improvable"], 0)

    def test_exact_arithmetic_has_zero_slack(self):
        import igraph as ig
        from hedonic import Game

        for gamma in EXACT_RESOLUTIONS:
            for seed in range(3):
                result = Game(ig.Graph.Famous("Zachary")).community_hedonic(
                    resolution=gamma, seed=seed, allow_isolation=bool(seed % 2))
                self.assertEqual(result._hedonic_provenance["equilibrium_slack"], 0.0)

    def test_oracle_detects_a_non_equilibrium(self):
        # Two triangles joined by an edge: splitting a triangle is not stable.
        game = dict(n=6, edges=[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)],
                    weights=None, node_weights=None, gamma=0.1, isolation=False,
                    max_total=None, exact=None)
        gains = partition_gains(game, [0, 0, 1, 1, 1, 1])
        self.assertGreater(max(g for g in gains if g is not None), 0)
        gains = partition_gains(game, [0, 0, 0, 1, 1, 1])
        self.assertLessEqual(max(g for g in gains if g is not None), 0)


if __name__ == "__main__":
    unittest.main()
