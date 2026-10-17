# Hedonic

`hedonic` detects communities in [`igraph`](https://igraph.org/) graphs as a
hedonic game: wrap a graph in `Game` and call `community_hedonic()` to get a
partition of its vertices into communities.

The detector is a small C core compiled into the package (`hedonic._core`).
The only dependencies are the standard [`igraph`](https://pypi.org/project/igraph/)
package and NumPy, so `hedonic` installs next to any igraph 1.x you already
use. Python 3.12 or newer.

## TL;DR

```bash
pip install hedonic
```

```python
from hedonic import Game

h = Game(g)  # g: an igraph graph (convert NetworkX, edge lists, ... through igraph)
partition = h.community_hedonic()
```

- `from hedonic import Game` is the public API.
- `community_hedonic()` puts each vertex in exactly one community and returns
  an `igraph.VertexClustering`.
- Every call runs until no vertex can certainly improve by moving: the result
  is an equilibrium of the game, with a certified bound on what any vertex
  could still gain (`equilibrium_slack`, 0 when the arithmetic is exact).
  There is nothing to tune: no iteration budget, no round cap, no tolerance.
- A `Game` remembers its last result and, when `initial_membership` is
  omitted, starts the next call from it. Pass `initial_membership=None` for a
  fresh start (see [Continuing from the last result](#continuing-from-the-last-result)).

## What this is

A **hedonic game** is a coalition-formation game: each vertex is a player,
each community is a coalition, and a player prefers the coalition that
improves its payoff. Here the payoff is the Constant Potts Model (CPM) score
at a resolution γ (by default, the graph density): a vertex gains from its
neighbours inside a community and pays γ for every member.

`community_hedonic()` visits the vertices in random order and lets each one
take its **best response**: move to the community it prefers. After a move it
revisits the vertex's neighbours, and it stops only when every vertex has been
visited, without moving, since the last move. That is a Nash equilibrium: no
vertex can improve its payoff alone.

The computation is in floating point, so the core proves each step instead
of trusting it: every payoff it compares carries a rigorous bound of its
rounding error, and a vertex moves only when its computed gain exceeds that
bound. Every move therefore raises the exact CPM potential, and the process
always stops. The result reports `equilibrium_slack` in
`result._hedonic_provenance`: no vertex can raise its exact payoff by more
than the slack times the total edge weight (normalized like the quality).
It is exactly 0 when the arithmetic is exact (for instance unit weights at
resolution 0, 1/2 or 1), so the result is then an exact Nash equilibrium; on
complete DBLP and Amazon at other resolutions it is about `1e-21`.
[`native/README.md`](native/README.md) gives the bounds and the termination
argument.

`Game` is an `igraph.Graph` subclass: after wrapping, the usual igraph API
(`h.vcount()`, `h.density()`, shortest paths, and so on) is still available.

## Installation

```bash
python -m pip install hedonic
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv add hedonic
```

Wheels are built for Linux (x86_64, aarch64), macOS (x86_64, arm64) and
Windows (AMD64, ARM64). Elsewhere pip builds the C core from the source
distribution, which needs a C11 compiler.

`Game.evaluate_against(..., method="one_to_one_f1")` uses SciPy; install it
with the `metrics` extra: `pip install "hedonic[metrics]"`.

## Tutorial

### 1. Wrap a graph as a `Game`

`Game` copies an existing `igraph.Graph`, or is constructed the same way as an
igraph graph. Other libraries convert through igraph first.

```python
import igraph as ig
from hedonic import Game

# Famous example graph (Zachary's karate club)
h = Game(ig.Graph.Famous("Zachary"))

# Equivalent shortcut: Graph class methods return a Game
h = Game.Famous("Zachary")

# Edge list
h = Game(n=4, edges=[(0, 1), (1, 2), (2, 3), (3, 0)])

# Adjacency matrix
h = Game(ig.Graph.Adjacency([[0, 1, 1], [1, 0, 1], [1, 1, 0]], mode="undirected"))

# NetworkX (networkx is not a dependency)
import networkx as nx

h = Game(ig.Graph.from_networkx(nx.karate_club_graph()))
```

Wrapping preserves the graph's directionality and its vertex, edge and graph
attributes, and `to_igraph()` preserves them on the way back. The detector
requires an undirected graph, so a directed input is rejected explicitly
rather than silently converted.

### 2. Detect communities

```python
h = Game.Famous("Zachary")

# Each vertex has one community id
partition = h.community_hedonic(seed=0)
print(partition.membership[:8])
print(partition.sizes())         # community sizes
```

Iterate communities as lists of vertex ids:

```python
for community in partition:
    print(list(community))
```

To start a run from a known partition, pass it as the start (one label per
vertex, contiguous from 0):

```python
start = [0] * 17 + [1] * 17
partition = h.community_hedonic(initial_membership=start, seed=0)
```

### Continuing from the last result

Every call stores its result in `h.memberships`, one single-label list per
vertex (`[[c0], [c1], ...]`). When `initial_membership` is omitted, the next
call starts from that state, so repeated calls continue from the previous
equilibrium:

| `initial_membership` | Start state |
|---|---|
| omitted | the stored `h.memberships`, or singletons if nothing is stored |
| `None` | a fresh default start (singletons, or a feasible start under count limits) |
| a list | that partition |

You can also assign `h.memberships` yourself, as flat labels or as one-label
lists.

### 3. Parameters

| Parameter | Default | Role |
|---|---|---|
| `resolution` | graph density | CPM γ. Smaller γ gives larger communities |
| `allow_isolation` | `False` | Allow a vertex to open a new, empty community |
| `initial_membership` | stored state | See [Continuing from the last result](#continuing-from-the-last-result) |
| `max_communities` | `None` | Random start over `[0, max_communities)` when no start is given; not an output limit |
| `edge_weights` | unweighted | Edge weights (sequence or edge-attribute name) |
| `node_weights` | unweighted | Vertex weights (sequence or vertex-attribute name) in the crowding term `γ w_u w_v` |
| `max_total_communities` | `None` | At most `K` occupied communities, in the result and in every state visited |
| `n_communities` | `None` | Exactly `K` occupied communities |
| `seed` | `None` | Seed of the visiting order (see below) |

Everything after `allow_isolation` is keyword-only.

**Reproducible runs.** With `seed=`, the visiting order is drawn from
`random.Random(seed)`, so the same graph, parameters, start and seed give the
same result:

```python
a = h.community_hedonic(initial_membership=None, seed=7)
b = h.community_hedonic(initial_membership=None, seed=7)
assert a.membership == b.membership
```

Without a seed, the order is drawn from Python's `random` module, which is
also python-igraph's default generator, so `random.seed(...)` before the call
makes it repeatable too.

**Community-count limits.** `max_total_communities=K` keeps the number of
occupied communities at most `K`, and `n_communities=K` keeps it exactly `K`:

```python
partition = h.community_hedonic(initial_membership=None, n_communities=2)
bounded = h.community_hedonic(initial_membership=None, max_total_communities=4)
```

Both must be positive integers, and `n_communities` must not exceed
`max_total_communities`. Without a start state, a deterministic feasible start
is built; a supplied start that violates a limit is rejected, never silently
repaired. The result is then an equilibrium over the moves the limit allows.

**What the result carries.** Besides the communities, a result has
`_hedonic_provenance` (versions, effective parameters, count mode, and
`equilibrium_slack`) and `_hedonic_native_quality`, the potential of the
returned state.

### 4. Scoring against a reference

`evaluate_against` scores the stored partition against a known partition or
cover (a reference may let communities overlap):

```python
h = Game.Famous("Zachary")
h.community_hedonic(seed=0)
truth = [0] * 17 + [1] * 17          # any flat partition, cover or VertexClustering
print(h.evaluate_against(truth, method="f1"))
```

Methods: `f1` (symmetric best match), `one_to_one_f1` (needs SciPy),
`jaccard`, `omega`, `node_micro_f1` and `size_weighted_community_f1`. The
underlying functions are in `hedonic.metrics`.

### 5. Worked example

```python
import igraph as ig
from hedonic import Game

h = Game(ig.Graph.Famous("Zachary"))

partition = h.community_hedonic(seed=0)
print(f"{h.vcount()} vertices, {h.ecount()} edges")
print(f"communities: {len(partition)}, sizes {partition.sizes()}")
print(f"quality: {partition.quality:.4f}")

truth = [0] * 17 + [1] * 17
print(f"F1 against a two-block reference: {h.evaluate_against(truth):.3f}")
```

## Interactive explainer

The `web/` directory contains an educational website that explains the
hedonic game visually: friends and strangers on a balance scale, the
resolution parameter, the Familiarity Index, the decision tree, the metagraph
of all partitions of a four-vertex graph, and better-response walks to a
stable equilibrium. It also hosts documentation generated from this package's
source. It is a small browser-side teaching model, not a replacement for the
detector. From the repository root:

```bash
npm install
npm run dev
```

The GitHub Actions workflow `.github/workflows/pages.yml` tests, builds and
publishes it to GitHub Pages; see `web/README.md` for details.

## Development

```bash
git clone https://github.com/lucaslopes/hedonic-game.git
cd hedonic-game
python -m pip install -e ".[metrics]" pytest
python -m pytest
make -C native test            # the C core's own tests (sanitizers)
python tools/native_tests.py   # bit-level pins, as the wheel builds run them
```

The C core lives in `native/` (`include/hedonic_core.h` is its API,
[`native/README.md`](native/README.md) its design) and is compiled by
`setup.py` into `hedonic._core`. Rounding is fixed by the source (explicit
fused multiply-adds, computed by the instruction or the core's own correctly
rounded soft fma, never the C library's; no contraction), so results are
identical across platforms; the wheel workflow
(`.github/workflows/wheels.yml`) checks this on every platform it builds for.

## License

This project is distributed under the GNU General Public License, version 3 or
later. See [LICENSE](LICENSE). The C core was written for this package and
contains no code from igraph or other libraries; the binding implements the
MT19937 generator step (Matsumoto and Nishimura) as CPython's `random` module
does, so that seeded runs draw exactly what `random.Random` would.
