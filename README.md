# Hedonic Game for Community Detection

Community detection (Constant Potts Model optimized by the Leiden algorithm) as a hedonic game: wrap a graph in `Game` and call `community_hedonic()`. One method covers both **disjoint** partitions and
**overlapping** covers.

Requires Python 3.12+.

## TL;DR

```bash
pip install hedonic
```

```python
from hedonic import Game

h = Game(g)  # g: an igraph graph, or convert from NetworkX / edge list / adjacency
partition = h.community_hedonic(max_memberships=1)  # disjoint
cover = h.community_hedonic(max_memberships=3)      # overlapping (cap > 1)
```

- `from hedonic import Game` is the public API.
- Wrap the graph once: `h = Game(g)`.
- `max_memberships=1` → each vertex belongs to one community (`VertexClustering`).
- `max_memberships>1` → each vertex may belong to up to that many communities (`VertexCover`).

## What this is

A **hedonic game** is a coalition-formation game: each vertex is a player, each
community is a coalition, and a player prefers the coalition that improves its
payoff. Here the payoff is the constant-potts (CPM) score at a resolution
parameter γ (by default, the graph density). Players repeatedly take a
**best-response** move — join, leave, or (when overlapping is allowed)
add/substitute a community — until no player wants to change.

That local-moving process is the same model as Leiden's first phase. This
package exposes it as `Game.community_hedonic()`, implemented by a pinned
native Leiden binding (`lucas-igraph`) so disjoint and overlapping detection
share one entry point.

Use it when you want:

- a **partition** of the vertex set (`max_memberships=1`), or
- a **cover**, where hubs and bridges can sit in several communities
  (`max_memberships>1`).

`Game` is an `igraph.Graph` subclass, so after wrapping you still have the
usual igraph API (`h.vcount()`, `h.density()`, shortest paths, and so on).

## Installation

```bash
pip install hedonic
```

From a local checkout, prefer `uv`:

```bash
uv sync
```

The optional experiment CLI and adapters need:

```bash
uv sync --extra experiments
```

## Tutorial

### 1. Wrap a graph as a `Game`

`Game` copies an existing `igraph.Graph`, or you can construct one the same
way you would construct an igraph graph. Other libraries convert through
igraph first.

```python
import igraph as ig
from hedonic import Game

# Famous example graph (Zachary's karate club)
g = ig.Graph.Famous("Zachary")
h = Game(g)

# Equivalent shortcut: Graph class methods return a Game
h = Game.Famous("Zachary")

# Edge list
h = Game(n=4, edges=[(0, 1), (1, 2), (2, 3), (3, 0)])

# Adjacency matrix
h = Game(ig.Graph.Adjacency([[0, 1, 1], [1, 0, 1], [1, 1, 0]], mode="undirected"))

# NetworkX (networkx is not a core dependency)
import networkx as nx

h = Game(ig.Graph.from_networkx(nx.karate_club_graph()))
```

### 2. Detect communities

```python
h = Game.Famous("Zachary")

# Disjoint: each vertex has one community id
partition = h.community_hedonic(max_memberships=1)
print(partition.summary())
print(partition.membership[:8])  # e.g. [0, 0, 0, 0, 1, 1, 1, 0]
print(partition.sizes())         # community sizes

# Overlapping: each vertex has a list of community ids (length ≤ max_memberships)
cover = h.community_hedonic(max_memberships=3)
print(cover.summary())
print(cover.membership[:4])      # e.g. [[0, 1], [0, 1], [0], [2]]
```

Iterate communities as lists of vertex ids:

```python
for community in partition:          # disjoint clusters
    print(list(community))

for community in cover:              # overlapping clusters
    print(list(community))
```

A common pattern is to seed the overlapping run from a disjoint partition:

```python
seed = h.community_hedonic(max_memberships=1)
cover = h.community_hedonic(
    max_memberships=3,
    initial_membership=list(seed.membership),
)
```

### 3. Parameters that matter

| Parameter | Default | Role |
|---|---|---|
| `max_memberships` | `1` | `1` = disjoint partition; `>1` = overlapping cover, cap per vertex |
| `resolution` | graph density | CPM γ. Smaller γ → larger communities |
| `n_iterations` | `-1` | Leiden outer iterations. Negative means “repeat until the native mover reports no change” |
| `local_move_only` | `True` | `True`: hedonic best-response only. `False`: full Leiden (refine + aggregate) |
| `allow_isolation` | `False` | Allow a vertex to open a new (empty) community |
| `initial_membership` | singletons | Disjoint: length-`n` list of ints. Overlapping: same, or a list of id-lists per vertex |
| `max_communities` | `n` | Random disjoint init over `[0, max_communities)` when no membership is given |
| `ensure_equilibrium` | `False` | Force `n_iterations=-1` on the native call. Still audit if you need a certificate |
| `edge_weights` | unweighted | Optional edge weights |
| `seed` | `42` when sampling | RNG seed for random disjoint initialization |
| `beta` | `0.01` | Leiden refinement randomness (`local_move_only=False` only) |

Minimal call with the usual research defaults:

```python
result = h.community_hedonic(
    max_memberships=1,          # or >1 for a cover
    resolution=h.density(),     # explicit; this is already the default
    n_iterations=-1,
    local_move_only=True,
)
```

`n_iterations=-1` is the binding's “no further change” stop, not a standalone
proof that every admissible move has non-positive regret. If you need a
game-theoretic certificate, audit the returned membership separately (the
experiment extras include such audits).

### 4. Worked example

```python
import igraph as ig
from hedonic import Game

h = Game(ig.Graph.Famous("Zachary"))

partition = h.community_hedonic(max_memberships=1)
print(f"{h.vcount()} vertices, {h.ecount()} edges")
print(f"disjoint communities: {len(partition)}")
print(f"sizes: {partition.sizes()}")

cover = h.community_hedonic(max_memberships=3, n_iterations=-1)
print(f"overlapping communities: {len(cover)}")
print(f"memberships per vertex (first 5): {cover.membership[:5]}")
```

## Experiments

Reproducible adapters live under `hedonic.experiments` and the `hedonic-exp`
CLI. They are optional (`pip install 'hedonic[experiments]'` or
`uv sync --extra experiments`) and write generated output under `artifacts/`
or an external path you pass in.

```bash
uv run hedonic-exp --help
uv run hedonic-exp list
uv run hedonic-exp overlapping-small
```

See [docs/reproduce_disjoint.md](docs/reproduce_disjoint.md),
[docs/reproduce_overlapping.md](docs/reproduce_overlapping.md), and
[docs/snap_overlapping_benchmark.md](docs/snap_overlapping_benchmark.md).

## Development

```bash
uv sync --extra experiments
uv run pytest -q
uv build --no-sources
```

Release tagging and PyPI publishing are documented in
[docs/OIDC_SETUP.md](docs/OIDC_SETUP.md). The helper `./scripts/release.sh patch`
(or `minor` / `major`) bumps the version locally; it does not push unless you
pass `--push`.

## License

[GPL-3.0](LICENSE)
