# Hedonic

`hedonic` detects communities in [`igraph`](https://igraph.org/) graphs as a
hedonic game: wrap a graph in `Game` and call `community_hedonic()`, which
returns a **disjoint** partition at a Nash equilibrium of the game.

The detector is a small C core compiled into the package (`hedonic._core`).
The only dependencies are the standard [`igraph`](https://pypi.org/project/igraph/)
package and NumPy, so `hedonic` installs next to any igraph 1.x you already
use. Python 3.12 or newer.

Overlapping communities (covers) are hedonic 2.

## TL;DR

```bash
pip install hedonic              # the library
pip install "hedonic[cli]"       # plus the command line that reproduces the paper
```

```python
from hedonic import Game

h = Game(g)  # g: an igraph graph (convert NetworkX, edge lists, ... through igraph)
partition = h.community_hedonic(seed=0)  # an igraph.VertexClustering
```

- `from hedonic import Game` is the public API.
- Each vertex ends in exactly one community; the result is an
  `igraph.VertexClustering`.
- Every call runs until no vertex can certainly improve by moving: the result
  is an equilibrium of the game, with a certified bound on what any vertex
  could still gain (`equilibrium_slack`). There is nothing to tune: no
  iteration budget, no round cap, no tolerance.
- Every call starts afresh: from singletons, unless you pass
  `initial_membership` (or ask for a random start with `max_communities`, or
  set a community-count limit). The result is also stored in
  `h.memberships`; pass `initial_membership=h.memberships` to continue from
  it (see [Start states](#start-states)).

## What this is

A **hedonic game** is a coalition-formation game: each vertex is a player,
each community is a coalition, and a player prefers the coalition that
improves its payoff. Here the payoff is the Constant Potts Model (CPM) score
at a resolution γ (by default, the weighted density: the total edge weight
over the number of vertex pairs): a vertex gains from its neighbours inside a
community and pays γ for every member.

`community_hedonic()` visits the vertices in random order and lets each one
take its **best response**: move to the community it prefers. After a move
it revisits the vertex's neighbours, and it stops only when every vertex has
been visited, without moving, since the last move. That is a Nash
equilibrium: no vertex can improve its payoff alone. It is the equilibrium
this process reaches from its start, not necessarily the partition of highest
potential, and another seed can reach another one.

The computation is in floating point, so the core proves each step instead
of trusting it: every payoff it compares carries a rigorous bound of its
rounding error, and a vertex moves only when its computed gain exceeds that
bound. Every move therefore raises the exact CPM potential, and the process
always stops. The result reports `equilibrium_slack` in
`result._hedonic_provenance`: no vertex can raise its exact payoff by more
than the slack times the total edge weight (normalized like the quality). It
is exactly 0 whenever no deviation's exact gain can be positive: always when
the arithmetic is exact (for instance unit weights at resolution 0, 1/2 or
1), and often otherwise (with integer weights, for instance, when every
vertex's best deviation is certainly worse). The result is then an exact Nash
equilibrium. On complete DBLP and Amazon it is about `1e-21`.
[`native/README.md`](https://github.com/lucaslopes/hedonic/blob/main/native/README.md)
gives the bounds and the termination argument.

`Game` is an `igraph.Graph` subclass: after wrapping, the usual igraph API
(`h.vcount()`, `h.density()`, shortest paths, and so on) is still available.

## Installation

```bash
python -m pip install hedonic
```

That is the library alone (igraph and NumPy). Two extras add more:

- `python -m pip install "hedonic[metrics]"` adds SciPy, which the scores
  built on a one-to-one matching need (see
  [Scoring against a reference](#4-scoring-against-a-reference)).
- `python -m pip install "hedonic[cli]"` adds the command line (`hedonic` and
  `hedonic-exp`) and what its experiments need (SciPy, pandas, Matplotlib,
  PyArrow, DuckDB and tqdm). The command line runs on macOS and Linux; on
  Windows the package installs the library alone.

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv add hedonic
```

Wheels are built for Linux (x86_64, aarch64), macOS (x86_64, arm64) and
Windows (AMD64, ARM64). Elsewhere pip builds the C core from the source
distribution, which needs a C11 compiler.

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
requires an undirected graph without self-loops, so a directed input or a
loop is rejected explicitly rather than silently converted
(`h.simplify(multiple=False, loops=True)` removes the loops). Parallel edges
are accepted and count as one edge whose weight is their sum.

### 2. Detect communities

```python
h = Game.Famous("Zachary")

partition = h.community_hedonic(seed=0)
print(partition.membership[:8])  # one community id per vertex
print(partition.sizes())         # community sizes

for community in partition:      # communities as lists of vertex ids
    print(list(community))
```

### Start states

| `initial_membership` | Start state |
|---|---|
| `None` (default) | singletons; a random start with `max_communities`; a feasible start under count limits |
| one label per vertex | that partition |
| one list of labels per vertex | each vertex keeps the label it prefers (below) |

Labels are non-negative integers; a start whose labels have gaps
(`[5, 9, 5]`) is renumbered in label order (`[0, 1, 0]`). An igraph
`VertexClustering` (or `VertexCover`) of a graph with the same vertices is
accepted too.

Every call stores its result in `h.memberships`, one list of labels per
vertex, and never reads it back by itself. To continue from the last
equilibrium, pass it:

```python
h.community_hedonic(seed=0)
again = h.community_hedonic(initial_membership=h.memberships, seed=1)
```

When a start gives a vertex more than one label, the vertex keeps the one it
prefers (ranked by the payoff of each community on the start, ties to the
larger community, then the smaller label) and the game runs to an
equilibrium from there. `_hedonic_provenance["projected_vertices"]` counts
the vertices that had to choose.

### 3. Parameters

| Parameter | Default | Role |
|---|---|---|
| `initial_membership` | `None` | The start; see [Start states](#start-states) |
| `max_memberships` | `1` | Communities per vertex; `1` (a partition) is the only value hedonic 1 accepts |
| `resolution` | weighted density | CPM γ: total edge weight over the number of vertex pairs (the density of an unweighted graph). Smaller γ gives larger communities |
| `allow_isolation` | `False` | Allow a vertex to open a new, empty community |
| `max_communities` | `None` | Random start over `[0, max_communities)` when no start is given; without isolation the result then has at most that many communities |
| `edge_weights` | unweighted | Edge weights (sequence or edge-attribute name; an attribute called `weight` is used only when named) |
| `node_weights` | unweighted | Vertex weights (sequence or vertex-attribute name) in the crowding term `γ w_u w_v` |
| `max_total_communities` | `None` | At most `K` occupied communities, in the result and in every state visited |
| `n_communities` | `None` | Exactly `K` occupied communities |
| `seed` | `None` | Seed of the visiting order (see below) |

Everything after `initial_membership` is keyword-only.

**Reproducible runs.** With `seed=`, the visiting order is drawn from
`random.Random(seed)`, so the same graph, parameters, start and seed give the
same result:

```python
a = h.community_hedonic(seed=7)
b = h.community_hedonic(seed=7)
assert a.membership == b.membership
```

Without a seed, the order is drawn from Python's `random` module, which is
also python-igraph's default generator, so `random.seed(...)` before the call
makes it repeatable too.

**Community-count limits.** `max_total_communities=K` keeps the number of
occupied communities at most `K`, and `n_communities=K` keeps it exactly `K`:

```python
two = h.community_hedonic(n_communities=2)
few = h.community_hedonic(max_total_communities=4)
```

Both must be positive integers, and `n_communities` must not exceed
`max_total_communities`. Without a start state, a deterministic feasible start
is built; a supplied start that violates a limit (after each vertex keeps one
label) is rejected, never silently repaired. The result is then an
equilibrium over the moves the limit allows.

**What the result carries.** Besides the communities, a result has
`_hedonic_provenance` (versions, effective parameters, count mode, and
`equilibrium_slack`) and `_hedonic_native_quality`, the quality of the
returned state: `(Φ − γ Σ_v n_v² / 2) / W` for its exact potential `Φ` (a
sum over pairs of vertices), the vertex weights `n_v` and the total edge
weight `W`; the constant term is each vertex's own crowding, the same for
every state.

### 4. Scoring against a reference

`evaluate_against` scores the stored result against a known partition or
list of communities:

```python
h = Game.Famous("Zachary")
h.community_hedonic(seed=0)
truth = [0] * 17 + [1] * 17          # a flat partition, a list of communities or a VertexClustering
print(h.evaluate_against(truth, method="f1"))
```

Methods: `f1` (symmetric best match), `one_to_one_f1`, `jaccard`, `omega`,
`node_micro_f1` and `size_weighted_community_f1`; the two built on a
one-to-one matching (`one_to_one_f1`, `node_micro_f1`) need SciPy (the
`metrics` extra). `omega` is exact when the graph has at most 100,000 vertex
pairs and otherwise estimated from that many sampled pairs
(`omega_sample_size`, `omega_seed`). A flat reference may use any hashable
labels, but every vertex needs one (`None` and NaN are refused); a list of
lists is read as communities, each a list of vertices of the graph, and the
communities may overlap. Communities with the same members count once, on
both sides.

A reference that labels only part of the graph is scored against the whole
result by default. `restrict_to_reference=True` first drops the vertices the
reference does not contain from every detected community
(`hedonic.metrics.restrict_cover`), the usual convention for partial ground
truth:

```python
h.evaluate_against([[0, 1, 2], [33, 32]], method="f1", restrict_to_reference=True)
```

The underlying functions are in `hedonic.metrics` (also reachable as
`hedonic.metrics` after `import hedonic`).

### 5. Worked example

```python
import igraph as ig
from hedonic import Game

h = Game(ig.Graph.Famous("Zachary"))

partition = h.community_hedonic(seed=0)
print(f"{h.vcount()} vertices, {h.ecount()} edges")
print(f"communities: {len(partition)}, sizes {partition.sizes()}")
print(f"equilibrium slack: {partition._hedonic_provenance['equilibrium_slack']}")
```

## Command line: reproducing the disjoint paper

`pip install "hedonic[cli]"` installs two commands. `hedonic` is the front
door for runs; `hedonic-exp list` names the analysis and utility commands.

```bash
hedonic guide                    # where you are, what to do next, and the topics
hedonic show disjoint            # the presets: grids, methods, time and disk estimates
```

The paper's experiment is the stochastic block model (SBM) benchmark: 5
community counts × 10 within-community edge probabilities × 10 difficulties ×
100 network seeds = 50,000 graphs of 1,020 vertices, each started from noisy
partitions at 5 noise levels × 10 seeds, with 10 seeded restarts of Leiden
and Hedonic per start.

```bash
hedonic run disjoint --preset smoke             # the whole pipeline in seconds
hedonic run disjoint --preset paper --dry-run   # the paper's plan, time and disk
hedonic run disjoint --preset paper             # durable and resumable, about a day on 8 workers
hedonic run queue disjoint --dry-run            # the queue's plan and what is already done
hedonic run queue disjoint                      # the same as a queue: the smoke run, then the paper's grid
```

Every grid value is also a flag (`--nodes`, `--communities`, `--p-in`,
`--difficulty`, `--seeds`, `--noises`, `--partition-seeds`, `--methods`,
`--restarts`, `--workers`, `--partitions`); `hedonic run disjoint --help`
lists them, and `hedonic run disjoint` alone opens a wizard in a terminal.

Its methods are Hedonic (this package's detector), stock igraph's CPM Leiden
(`community_leiden`, the paper's full Leiden), and four references (ground
truth, Mirror, One Pass, spectral). Graphs are regenerated from their seeds
and never stored, so nothing is downloaded.

**The store, the statistics and the figures.** A run writes a Parquet result
store (one shard per graph, one row per detector run), then the logical-cell
table (`plot_table.parquet`), the paper's five figures (`figures/`:
`gt_robustness`, `noise`, `n_communities`, `acc_robustness`,
`acc_efficiency`) and every statistic of the paper's Empirical Analysis
(`stats/`: `summary.md`, `paper_stats.json`, `tables.tex` and CSV tables).
The same steps are available one at a time:

```bash
hedonic-exp disjoint-store STORE --progress      # shards, rate, ETA, disk
hedonic-exp disjoint-store STORE --sql "SELECT method, noise, avg(adjusted_rand) FROM cells GROUP BY ALL ORDER BY ALL"
hedonic-exp disjoint-store STORE --verify        # regenerate graphs and replay sampled rows from their seeds
hedonic-exp disjoint-stats --data STORE          # the paper's statistics
hedonic-exp plots --data STORE/plot_table.parquet --output_dir FIGURES
```

Each method's results carry their own identity (its call, parameters,
restarts and the build that computes it), so a store that already holds some
methods keeps them and runs only the missing or changed ones; a store made by
another build is refused before anything is written, and `--rerun METHOD` or
`--reuse-from STORE` (which first replays a sample with this build) says
what to do with it.

**Runs.** Runs execute detached (tmux when available, else a background
process; on macOS under `caffeinate -i`), so closing the terminal or pressing
Ctrl-C does not stop them: `hedonic run list`, `status`, `attach`, `logs`,
`stop NAME` and `resume NAME` manage them, and `hedonic run check NAME`
replays a sample of a run's rows from their seeds and compares them. A queue
skips what its files show as finished and resumes the rest, so running it
again continues it. Runs go to `~/Databases/Hedonic/Disjoint`
(`HEDONIC_DISJOINT_OUTPUT_DIR`, or `hedonic config set disjoint_output_root
PATH`), a queue to `<disjoint_output_root>/disjoint-queue` (`--root DIR`),
and the registry of runs to `~/.cache/hedonic/runs` (`HEDONIC_RUNS_DIR`).
`hedonic config example` prints every setting, and `hedonic update` compares
the installed version with PyPI (it upgrades only with `--yes`).

## Interactive explainer

The `web/` directory contains an educational website that explains the
disjoint hedonic game visually: friends and strangers on a balance scale, the
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
publishes it to GitHub Pages; see
[`web/README.md`](https://github.com/lucaslopes/hedonic/blob/main/web/README.md)
for details.

## Development

```bash
git clone https://github.com/lucaslopes/hedonic.git
cd hedonic
python -m pip install -e ".[cli]" pytest
python -m pytest
make -C native test            # the C core's own tests (sanitizers)
python tools/native_tests.py   # bit-level pins, as the wheel builds run them
```

The C core lives in `native/` (`include/hedonic_core.h` is its API,
[`native/README.md`](https://github.com/lucaslopes/hedonic/blob/main/native/README.md)
its design) and is compiled by `setup.py` into `hedonic._core`. Rounding is
fixed by the source (explicit fused multiply-adds, computed by the
instruction or the core's own correctly rounded soft fma, never the C
library's; no contraction), so results are identical across platforms; the
wheel workflow (`.github/workflows/wheels.yml`) checks this on every platform
it builds for.

## Citation

The game and the SBM benchmark are those of:

> Lucas Lopes Felipe, Konstantin Avrachenkov and Daniel Sadoc Menasché.
> From Leiden to Pleasure Island: The Constant Potts Model for community
> detection as a hedonic game. *Physica A: Statistical Mechanics and its
> Applications* 680, 130989 (2025). Preprint:
> [arXiv:2509.03834](https://arxiv.org/abs/2509.03834).

```bibtex
@article{felipe2025leiden,
  title   = {From Leiden to Pleasure Island: The Constant Potts Model for community detection as a hedonic game},
  author  = {Felipe, Lucas Lopes and Avrachenkov, Konstantin and Menasch{\'e}, Daniel Sadoc},
  journal = {Physica A: Statistical Mechanics and its Applications},
  volume  = {680},
  pages   = {130989},
  year    = {2025}
}
```

## License

This project is distributed under the GNU General Public License, version 3 or
later. See [LICENSE](https://github.com/lucaslopes/hedonic/blob/main/LICENSE).
The C core was written for this package and contains no code from igraph or
other libraries; the binding implements the MT19937 generator step (Matsumoto
and Nishimura) as CPython's `random` module does, so that seeded runs draw
exactly what `random.Random` would.
