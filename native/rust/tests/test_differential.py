"""The Rust core against the C core, bit for bit, on small random games.

Both extensions run in one process: the Rust one as hedonic._core, the C one
as hedonic._core_c built beside it (without it, the test is skipped). Every
result, trace, verifier report, audit answer, error (type and message) and the
caller's generator state after the call must be the same. Fast and fail-fast:

    python -m pytest -x -q native/rust/tests/test_differential.py
"""

import math
import os
import random
from array import array

import pytest

if os.environ.get("HEDONIC_CORE"):
    pytest.skip("compares hedonic._core with hedonic._core_c: unset HEDONIC_CORE", allow_module_level=True)

igraph = pytest.importorskip("igraph")
np = pytest.importorskip("numpy")
from hedonic import _core as rs  # noqa: E402

try:
    import hedonic._core_c as c
except ModuleNotFoundError:
    pytest.skip("the C core is not installed: no hedonic._core_c beside hedonic._core", allow_module_level=True)

CASES = int(os.environ.get("HIG_DIFF_CASES", "150"))


def bits(x):
    """Floats compared by their bits (NaN equal to NaN), recursively."""
    if isinstance(x, float):
        return ("f", x.hex() if not math.isnan(x) else "nan")
    if isinstance(x, (list, tuple)):
        return type(x).__name__, [bits(v) for v in x]
    if isinstance(x, dict):
        return {k: bits(v) for k, v in sorted(x.items())}
    return x


def outcome(fn, *args, **kwargs):
    """(result, None) or (None, (exception type, message))."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:  # noqa: BLE001
        return None, (type(e).__name__, str(e))


def same(name, call):
    """call(module) on both cores, with a fresh generator each, compared."""
    got = {}
    for label, module in (("c", c), ("rs", rs)):
        got[label] = call(module)
    assert bits(got["c"]) == bits(got["rs"]), f"{name}: C {got['c']!r}\n  Rust {got['rs']!r}"
    return got["c"]


def random_graph(rnd, n):
    p = rnd.choice([0.08, 0.15, 0.3, 0.6])
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rnd.random() < p]
    if rnd.random() < 0.2 and edges:  # parallel edges add up
        edges += rnd.sample(edges, max(1, len(edges) // 5))
    return edges


def weights(rnd, count, kind):
    if kind == 0:
        return None
    if kind == 1:  # dyadic
        return array("d", [0.25 * rnd.randint(1, 12) for _ in range(count)])
    return array("d", [rnd.uniform(0.05, 3.0) for _ in range(count)])


def rows_start(rnd, n, cap, k):
    return [sorted(rnd.sample(range(k), rnd.randint(1, min(cap, k)))) for _ in range(n)]


class Odd(random.Random):
    """Not an exact random.Random: the cores call getrandbits(32)."""


def test_plain_calls_agree():
    rnd = random.Random(20261020)
    for case in range(CASES):
        n = rnd.randint(1, 28)
        edges = random_graph(rnd, n)
        cover_mode = rnd.random() < 0.5 and edges and n >= 2
        flat = array("q", [x for e in edges for x in e])
        ew = weights(rnd, len(edges), rnd.randint(0, 2))
        nw = weights(rnd, n, rnd.choice([0, 0, 1, 2]))
        gamma = rnd.choice([0.0, 0.05, 0.1, 0.3, 0.7, 1.5, -0.02, len(edges) / max(1, n * (n - 1) / 2)])
        iso = rnd.random() < 0.5
        limits = {}
        if rnd.random() < 0.25:
            limits["max_total_communities"] = rnd.randint(1, max(1, n))
        if rnd.random() < 0.2:
            limits["n_communities"] = rnd.randint(1, max(1, n))
        seed = rnd.randrange(2**32)
        gen = Odd if rnd.random() < 0.2 else random.Random

        def run(module, fn, **kw):
            r = gen(seed)
            res, err = outcome(getattr(module, fn), n, flat, r, edge_weights=ew, node_weights=nw,
                               resolution=gamma, allow_isolation=iso, **limits, **kw)
            return res, err, r.getstate()

        if cover_mode:
            cap = rnd.randint(2, min(5, n))
            start = None
            if rnd.random() < 0.3:
                start = rows_start(rnd, n, cap, rnd.randint(1, n))
            res, err, _ = same(f"case {case} trace_cover", lambda m: run(m, "_trace_cover", memberships=start,
                                                                     max_memberships=cap))
            same(f"case {case} cover", lambda m: run(m, "cover", memberships=start, max_memberships=cap))
            if err is None:
                same(f"case {case} verify_cover", lambda m: outcome(
                    m._verify_cover, n, flat, res[0], cap, edge_weights=ew, node_weights=nw, resolution=gamma,
                    allow_isolation=iso, **limits))
        else:
            start = None
            if rnd.random() < 0.3:
                start = [rnd.randrange(max(1, n // 2)) for _ in range(n)]
            res, err, _ = same(f"case {case} trace_partition",
                               lambda m: run(m, "_trace_partition", membership=start))
            same(f"case {case} partition", lambda m: run(m, "partition", membership=start))
            if err is None:
                same(f"case {case} verify_partition", lambda m: outcome(
                    m._verify_partition, n, flat, res[0], edge_weights=ew, node_weights=nw, resolution=gamma,
                    allow_isolation=iso, **limits))


def test_detect_agrees_on_igraph_graphs():
    rnd = random.Random(7)
    for case in range(CASES // 2):
        n = rnd.randint(2, 40)
        g = igraph.Graph(n=n, edges=random_graph(rnd, n))
        if g.ecount() == 0:
            g.add_edge(0, 1)
        cap = rnd.choice([1, 1, 2, 3])
        start = None
        if rnd.random() < 0.3:
            k = rnd.randint(1, n)
            start = [rnd.randrange(k) for _ in range(n)] if cap == 1 else rows_start(rnd, n, cap, k)
            used = sorted({x for r in start for x in (r if isinstance(r, list) else [r])})
            relabel = {x: i for i, x in enumerate(used)}
            start = [relabel[r] for r in start] if cap == 1 else [[relabel[x] for x in r] for r in start]
        ew = weights(rnd, g.ecount(), rnd.randint(0, 2))
        gamma = rnd.choice([0.05, 0.2, 0.5, -0.01])
        seed = rnd.randrange(2**32)

        def run(module):
            r = random.Random(seed)
            res, err = outcome(module.detect, g, r, gamma, max_memberships=cap, start=start,
                               allow_isolation=rnd_iso, edge_weights=ew)
            return res, err, r.getstate()

        rnd_iso = rnd.random() < 0.5
        same(f"detect case {case}", run)


def test_audit_agrees():
    rnd = random.Random(11)
    for case in range(CASES // 3):
        n = rnd.randint(2, 30)
        g = igraph.Graph(n=n, edges=random_graph(rnd, n))
        if g.ecount() == 0:
            g.add_edge(0, 1)
        k = rnd.randint(1, n)
        rows = rows_start(rnd, n, 3, k)
        used = sorted({x for r in rows for x in r})
        relabel = {x: i for i, x in enumerate(used)}
        rows = [[relabel[x] for x in r] for r in rows]
        offsets = np.cumsum([0] + [len(r) for r in rows]).astype(np.int64)
        labels = np.array([x for r in rows for x in r], dtype=np.int64)
        ew = None if rnd.random() < 0.5 else np.array([rnd.uniform(0.1, 2) for _ in range(g.ecount())])
        vertices = np.array(rnd.sample(range(n), rnd.randint(1, n)), dtype=np.int64)
        grid = np.array(sorted(rnd.random() for _ in range(5)))
        cap = rnd.choice([0, 3, 4])
        iso, prune = rnd.random() < 0.5, rnd.random() < 0.7

        def run(module):
            state, err = outcome(module._audit_prepare, g, offsets, labels, ew)
            if err:
                return err
            intervals = outcome(module._audit_intervals, state, vertices, cap, iso, 1e-10, 1e-9, prune)
            regret = np.zeros(len(vertices) * len(grid))
            tolerance = np.zeros(len(vertices) * len(grid))
            done = outcome(module._audit_regrets, state, vertices, grid, regret, tolerance, cap, iso, 1e-10,
                           1e-9, prune)
            return intervals, done, regret.tolist(), tolerance.tolist()

        same(f"audit case {case}", run)


def test_audit_windowed_agrees():
    """Larger audits, where the windowed search excludes, certifies and
    finds a certified core (vertices with 128 candidates or more)."""
    rnd = random.Random(12)
    for case in range(max(4, CASES // 25)):
        n = rnd.randint(60, 240)
        g = igraph.Graph(n=n, edges=random_graph(rnd, n))
        if g.ecount() == 0:
            g.add_edge(0, 1)
        k = rnd.randint(n // 2, n)
        rows = rows_start(rnd, n, 3, k)
        used = sorted({x for r in rows for x in r})
        relabel = {x: i for i, x in enumerate(used)}
        rows = [[relabel[x] for x in r] for r in rows]
        offsets = np.cumsum([0] + [len(r) for r in rows]).astype(np.int64)
        labels = np.array([x for r in rows for x in r], dtype=np.int64)
        ew = None if rnd.random() < 0.5 else np.array([float(rnd.randint(1, 3)) for _ in range(g.ecount())])
        vertices = np.arange(n, dtype=np.int64)
        cap = rnd.choice([0, 4, 64])
        iso, prune = rnd.random() < 0.5, rnd.random() < 0.3
        atol, rtol = rnd.choice([(1e-10, 1e-9), (0.0, 0.0), (1e-3, 1e-2), (0.05, 0.0)])

        def run(module):
            state, err = outcome(module._audit_prepare, g, offsets, labels, ew)
            return err or outcome(module._audit_intervals, state, vertices, cap, iso, atol, rtol, prune)

        same(f"windowed audit case {case}", run)


def test_errors_agree():
    """The same exception type and message for bad arguments."""
    edges = array("q", [0, 1, 1, 2])
    bad = [
        ("partition", (3, array("q", [0, 1, 1]), random.Random(0)), {}),
        ("partition", (3, [0, 1, 1, 2], random.Random(0)), {}),
        ("partition", (3, array("d", [0, 1, 1, 2]), random.Random(0)), {}),
        ("partition", (-1, edges, random.Random(0)), {}),
        ("partition", (3, edges, random.Random(0)), {"edge_weights": array("d", [1.0])}),
        ("partition", (3, edges, random.Random(0)), {"membership": [0, 1]}),
        ("partition", (3, edges, random.Random(0)), {"max_total_communities": 0}),
        ("partition", (3, edges, random.Random(0)), {"resolution": float("nan")}),
        ("partition", (3, edges, random.Random(0)), {"membership": [0, -1, 2]}),
        ("partition", (3, edges, random.Random(0)), {"n_communities": 5, "max_total_communities": 2}),
        ("cover", (3, array("q", [0, 0]), random.Random(0)), {}),
        ("cover", (3, edges, random.Random(0)), {"max_memberships": 4}),
        ("cover", (3, edges, random.Random(0)), {"memberships": [[0], [], [1]]}),
        ("cover", (3, edges, random.Random(0)), {"memberships": [[0], [0]]}),
        ("cover", (3, edges, random.Random(0)), {"memberships": [[0], 5, [1]]}),
        ("cover", (3, edges, random.Random(0)), {"memberships": [[0, 0, 1], [0], [1]]}),
        ("_verify_cover", (3, edges, [[1, 0], [0], [0]], 2), {}),
        ("_verify_partition", (3, edges, [0, 0]), {}),
    ]
    for fn, args, kw in bad:
        same(f"{fn}{args[:1]} {kw}", lambda m: outcome(getattr(m, fn), *args, **kw))

    g = igraph.Graph(n=3, edges=[(0, 1), (1, 2)])
    starts = [[0, 1], [0, True, 1], [0, 1.0, 2], [0, -2, 1], [[0], 1, [1]], [[0], [], [1]], [0, 2, 2],
              [[0, 1, 2], [0], [1]], [[0, 0], [1], [0]], [0, 2**70, 1]]
    for start in starts:
        for cap in (1, 2):
            same(f"detect start {start} cap {cap}", lambda m: outcome(
                m.detect, g, random.Random(0), 0.1, max_memberships=cap, start=start))
    same("detect cap 0", lambda m: outcome(m.detect, g, random.Random(0), 0.1, max_memberships=0))
    same("detect weights", lambda m: outcome(m.detect, g, random.Random(0), 0.1,
                                             edge_weights=array("d", [1.0])))

    class Broken(random.Random):
        def getrandbits(self, k):
            raise ZeroDivisionError("no bits")

    same("broken generator", lambda m: outcome(m.partition, 3, edges, Broken(0)))
    with pytest.raises(ZeroDivisionError, match="no bits"):
        rs.partition(3, edges, Broken(0))


def test_generator_state_and_shortcut():
    """The exact random.Random runs natively and ends where its own draws
    would; an overridden one is called; both give the same partition."""
    g = igraph.Graph.Famous("Zachary")
    for module in (c, rs):
        a, b = random.Random(5), Odd(5)
        ra = module.detect(g, a, 0.1)
        rb = module.detect(g, b, 0.1)
        assert bits(ra) == bits(rb)
        assert a.getstate() == b.getstate()
        a.random()  # still usable
