"""How the core reads an igraph.Graph's edges.

The binding borrows the edges of the igraph_t behind
Graph.__graph_as_capsule() when everything read from it agrees with the
public API (vcount, ecount, undirected, every end a vertex, three edges as
graph.es reports them), and copies get_edgelist() otherwise. These tests
force each fallback through a subclass whose capsule method misbehaves and
check that the results are those of the plain graph, bit for bit.
"""

import random
import sys

import pytest

igraph = pytest.importorskip("igraph")

from hedonic import Game, _core  # noqa: E402

RUST = getattr(_core, "backend", "c") == "rust"
CAPSULE = "__graph_as_capsule"


def base_graph():
    random.seed(4)
    return igraph.Graph.Barabasi(400, 3)


def other_capsule(kind):
    """A __graph_as_capsule that returns something else than this graph's."""
    def raises(self):
        raise RuntimeError("no capsule")

    def not_a_capsule(self):
        return object()

    def directed(self):
        g = igraph.Graph(n=self.vcount(), edges=self.get_edgelist(), directed=True)
        directed._keep = g  # the capsule points into g: keep it alive
        return getattr(igraph.GraphBase, CAPSULE)(g)

    def other_edges(self):
        # The same counts, other edges: a cycle with chords.
        n, m = self.vcount(), self.ecount()
        edges, k = [], 1
        while len(edges) < m:
            edges += [(i, (i + k) % n) for i in range(n)][: m - len(edges)]
            k += 1
        g = igraph.Graph(n=n, edges=edges)
        other_edges._keep = g
        return getattr(igraph.GraphBase, CAPSULE)(g)

    def fewer_vertices(self):
        g = igraph.Graph(n=self.vcount() - 1)
        fewer_vertices._keep = g
        return getattr(igraph.GraphBase, CAPSULE)(g)

    return {"raises": raises, "not a capsule": not_a_capsule, "directed": directed,
            "other edges": other_edges, "fewer vertices": fewer_vertices}[kind]


KINDS = ["raises", "not a capsule", "directed", "other edges", "fewer vertices"]


def subclass(kind):
    cls = type("Fallback", (Game,), {})
    setattr(cls, CAPSULE, other_capsule(kind))
    return cls


def results(game):
    out = []
    for cap, gamma in ((1, 0.02), (3, 0.06)):
        found = game.community_hedonic(max_memberships=cap, resolution=gamma, seed=3)
        params = found._params if cap > 1 else {"quality": found.quality, "equilibrium_slack": found.equilibrium_slack}
        out.append((list(found.membership), params["quality"].hex(), params["equilibrium_slack"].hex()))
    rows = [[v % 7, (v * 3) % 7] for v in range(game.vcount())]
    intervals = game.stable_intervals(rows, max_memberships=2)
    regrets = game.regrets([0.0, 0.01, 0.1, 0.5], rows, max_memberships=2)
    out.append([[(a.hex(), b.hex()) for a, b in row] for row in intervals])
    out.append([[float(x).hex() for x in a.ravel()] for a in regrets])
    return out


def test_capsule_has_the_documented_name():
    assert callable(getattr(igraph.GraphBase, CAPSULE, None))


@pytest.mark.parametrize("kind", KINDS)
def test_fallback_gives_the_same_results(kind):
    g = base_graph()
    plain = Game(g)
    forced = subclass(kind)(g)
    assert forced.get_edgelist() == plain.get_edgelist()
    assert results(forced) == results(plain)
    if RUST and getattr(sys, "_is_gil_enabled", lambda: True)():
        assert _core._edges_borrowed(plain)
        assert not _core._edges_borrowed(forced)
        # Sanity: an override that returns the graph's own capsule is read.
        own = type("Own", (Game,), {CAPSULE: lambda self: getattr(igraph.GraphBase, CAPSULE)(self)})
        assert _core._edges_borrowed(own(g))


@pytest.mark.parametrize("kind", KINDS)
def test_detect_on_fallback_matches_borrowed(kind):
    """detect() over copied edges: the same result and generator state."""
    g = base_graph()
    for cap in (1, 2):
        a, b = random.Random(9), random.Random(9)
        got = _core.detect(subclass(kind)(g), a, 0.03, max_memberships=cap)
        want = _core.detect(g, b, 0.03, max_memberships=cap)
        assert repr(got) == repr(want) and [x.hex() for x in got if isinstance(x, float)] == \
            [x.hex() for x in want if isinstance(x, float)]
        assert a.getstate() == b.getstate()


@pytest.mark.skipif(not RUST, reason="the C binding keeps the borrowed edges across Python code")
def test_graph_changed_during_call_is_refused():
    """A generator that is called (not copied) may run any Python code; one
    that adds an edge makes the edges read at the start stale, and the call
    refuses to finish on them."""
    g = Game(base_graph())

    class Adding(random.Random):
        calls = 0

        def getrandbits(self, k):
            Adding.calls += 1
            if Adding.calls == 50:
                g.add_edges([(0, g.vcount() - 1)])
            return super().getrandbits(k)

    with pytest.raises(RuntimeError, match="The graph changed during the call."):
        _core.detect(g, Adding(1), 0.02)
