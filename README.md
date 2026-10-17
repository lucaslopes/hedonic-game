# Hedonic

`hedonic` detects communities in [`igraph`](https://igraph.org/) graphs as a
hedonic game: wrap a graph in `Game` and call `community_hedonic()`. One
method returns either a **disjoint** partition or an **overlapping** cover.

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
partition = h.community_hedonic(max_memberships=1)  # disjoint
cover = h.community_hedonic(max_memberships=3)      # overlapping
```

- `from hedonic import Game` is the public API.
- `max_memberships=1` puts each vertex in exactly one community and returns an
  `igraph.VertexClustering`.
- `max_memberships>1` lets each vertex join up to that many communities and
  returns an `igraph.VertexCover`.
- Every call runs until no vertex can certainly improve by moving: the result
  is an equilibrium of the game, with a certified bound on what any vertex
  could still gain (`equilibrium_slack`). There is nothing to tune: no
  iteration budget, no round cap, no tolerance.
- Every call starts afresh: from singletons, unless you pass
  `initial_membership` (or ask for a random start with `max_communities`, or
  set a community-count limit). The result is also stored in
  `h.memberships`; pass
  `initial_membership=h.memberships` to continue from it (see
  [Start states](#start-states)).

## What this is

A **hedonic game** is a coalition-formation game: each vertex is a player,
each community is a coalition, and a player prefers the coalition that
improves its payoff. Here the payoff is the Constant Potts Model (CPM) score
at a resolution γ (by default, the weighted density: the total edge weight
over the number of vertex pairs): a vertex gains from its neighbours inside a
community and pays γ for every member.

`community_hedonic()` visits the vertices in random order and lets each one
take its **best response**: move to the community it prefers (disjoint), or
choose the set of at most `max_memberships` communities it prefers
(overlapping, where a vertex in k communities takes part in each with
intensity 1/√k).
After a move it revisits the vertex's neighbours, and it stops only when
every vertex has been visited, without moving, since the last move. That is a
Nash equilibrium: no vertex can improve its payoff alone. It is the
equilibrium this process reaches from its start, not necessarily the
partition of highest potential, and another seed can reach another one.

The computation is in floating point, so the core proves each step instead
of trusting it: every payoff it compares carries a rigorous bound of its
rounding error, and a vertex moves only when its computed gain exceeds that
bound. Every move therefore raises the exact CPM potential, and the process
always stops. The result reports `equilibrium_slack` in
`result._hedonic_provenance`: no vertex can raise its exact payoff by more
than the slack times the total edge weight (normalized like the quality).
For a partition it is exactly 0 whenever no deviation's exact gain can be
positive: always when the arithmetic is exact (for instance unit weights at
resolution 0, 1/2 or 1), and often otherwise (with integer weights, for
instance, when every vertex's best deviation is certainly worse). The result
is then an exact Nash equilibrium. A cover's intensities 1/√k are rounded,
so its slack stays a tiny positive number, smaller on larger graphs: about
`1e-15` on Zachary's karate club and `1e-17` on complete DBLP and Amazon,
where partitions get about `1e-21`.
[`native/README.md`](https://github.com/lucaslopes/hedonic/blob/main/native/README.md) gives the bounds, the termination
argument and the proof that the best set of communities is a prefix of the
sorted gains.

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

The package also installs the experiment command line (`hedonic` and
`hedonic-exp`, see [Command line](#command-line-reproducing-the-experiments))
and its dependencies (SciPy, pandas, Matplotlib, NetworkX, PyArrow, DuckDB and
tqdm). The experiments' robustness audits run on the same C audit as
`Game.stable_intervals` and `Game.regrets`, so they need nothing more. The
command line runs on macOS and Linux; on Windows the package installs the
library alone.

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

# Disjoint: each vertex has one community id
partition = h.community_hedonic(max_memberships=1, seed=0)
print(partition.membership[:8])
print(partition.sizes())         # community sizes

# Overlapping: each vertex has a list of community ids (length <= max_memberships)
cover = h.community_hedonic(max_memberships=3, seed=0)
print(cover.membership[:4])
```

Iterate communities as lists of vertex ids:

```python
for community in partition:          # disjoint clusters
    print(list(community))

for community in cover:              # overlapping clusters
    print(list(community))
```

### Start states

| `initial_membership` | Start state |
|---|---|
| `None` (default) | singletons; a random start with `max_communities`; a feasible start under count limits |
| one label per vertex | that partition |
| one list of labels per vertex | that cover |

Labels are non-negative integers; a start whose labels have gaps
(`[5, 9, 5]`) is renumbered in label order (`[0, 1, 0]`).

Every call stores its result in `h.memberships`, one list of labels per
vertex, and never reads it back by itself. To continue from the last
equilibrium, pass it:

```python
h.community_hedonic(max_memberships=1, seed=0)
cover = h.community_hedonic(max_memberships=3, initial_membership=h.memberships)
```

A partition is a valid start for a cover. The reverse works too: when a start
gives a vertex more labels than `max_memberships`, the vertex keeps the ones
it prefers (ranked by the payoff of each community on the start, ties to the
larger community, then the smaller label) and the game runs to an equilibrium
from there. A cover passed to `max_memberships=1` therefore starts from the
partition in which every vertex keeps its preferred community.
`_hedonic_provenance["projected_vertices"]` counts the vertices that had to
choose.

### 3. Parameters

| Parameter | Default | Role |
|---|---|---|
| `initial_membership` | `None` | The start; see [Start states](#start-states) |
| `max_memberships` | `1` | `1` = partition; `>1` = cover with that per-vertex cap; `-1` = no practical per-vertex cap |
| `resolution` | weighted density | CPM γ: total edge weight over the number of vertex pairs (the density of an unweighted graph). Smaller γ gives larger communities |
| `allow_isolation` | `False` | Allow a vertex to open a new, empty community |
| `max_communities` | `None` | Random disjoint start over `[0, max_communities)` when no start is given; without isolation the result then has at most that many communities |
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
partition = h.community_hedonic(n_communities=2)
cover = h.community_hedonic(max_memberships=3, max_total_communities=4)
```

Both must be positive integers, and `n_communities` must not exceed
`max_total_communities`. Without a start state, a deterministic feasible start
is built; a supplied start that violates a limit (after its projection to
`max_memberships`) is rejected, never silently repaired. The result is then
an equilibrium over the moves the limit allows.

**What the result carries.** Besides the communities, a result has
`_hedonic_provenance` (versions, effective parameters, count mode, and
`equilibrium_slack`) and `_hedonic_native_quality`, the quality of the
returned state: `(Φ − γ Σ_v n_v² / 2) / W` for its exact potential `Φ` (a
sum over pairs of vertices), the vertex weights `n_v` and the total edge
weight `W`; the constant term is each vertex's own crowding, the same for
every state.

### 4. Scoring against a reference

`evaluate_against` scores the stored result against a known cover or
partition:

```python
h = Game.Famous("Zachary")
h.community_hedonic(seed=0)
truth = [0] * 17 + [1] * 17          # any flat partition, cover or VertexClustering
print(h.evaluate_against(truth, method="f1"))
```

Methods: `f1` (symmetric best match), `one_to_one_f1`, `jaccard`, `omega`,
`node_micro_f1` and `size_weighted_community_f1`; the two built on a
one-to-one matching (`one_to_one_f1`, `node_micro_f1`) need SciPy. `omega`
is exact when the graph has at most 100,000 vertex pairs and otherwise
estimated from that many sampled pairs (`omega_sample_size`, `omega_seed`). A
flat reference may use any hashable labels, but every vertex needs one (`None`
and NaN are refused); a list of lists is read as communities, each a list of
vertices of the graph. Communities with the same members count once, on both
sides.

A reference that labels only part of the graph (the SNAP ground-truth covers,
for instance) is scored against the whole result by default.
`restrict_to_reference=True` first drops the vertices the reference does not
contain from every detected community (`hedonic.metrics.restrict_cover`), the
usual convention for partial ground truth:

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

partition = h.community_hedonic(max_memberships=1, seed=0)
print(f"{h.vcount()} vertices, {h.ecount()} edges")
print(f"disjoint communities: {len(partition)}, sizes {partition.sizes()}")

cover = h.community_hedonic(max_memberships=3, seed=0)
print(f"overlapping communities: {len(cover)}")
print(f"memberships of the first five vertices: {cover.membership[:5]}")
```

## Command line: reproducing the experiments

`pip install hedonic` installs two commands. `hedonic` is the front door for
runs; `hedonic-exp list` names the analysis and utility commands.

```bash
hedonic guide                    # where you are, what to do next, and the topics
hedonic show methods             # the nine overlapping methods and whether each can run here
hedonic run exp                  # a one-minute benchmark (a wizard in a terminal; --dry-run plans)
```

The disjoint (stochastic block model) paper:

```bash
hedonic run disjoint --preset smoke             # the whole pipeline in seconds
hedonic run disjoint --preset paper --dry-run   # the paper's plan, time and disk
hedonic run disjoint --preset paper             # durable and resumable, about a day on 8 cores
```

The overlapping-communities paper is one queue of six runs: the Hedonic
calibration, the baseline calibration, the synthetic networks, then, pair by
pair, the ground-truth start curve, the robustness spectrum and the benchmark,
on eight SNAP graph/cover pairs (DBLP, Amazon, YouTube and LiveJournal, each
with its `top5000` and `all` covers):

```bash
hedonic setup --all --dry-run    # the pairs, the eight baselines and the system tools they need
hedonic setup --all              # download the pairs; build and install the baselines
hedonic run queue paper --dry-run
hedonic run queue paper          # detached and resumable; days on a laptop
hedonic run attach paper-queue   # live view (q or Ctrl-C closes the view, never the run)
hedonic run queue paper --report # what is done, run by run and pair by pair
```

`hedonic setup` needs git, a C and C++ compiler, CMake and make, a C++
compiler with OpenMP for FOX (Homebrew `g++` on macOS) and
[`uv`](https://docs.astral.sh/uv/) for the locked CDlib environment of ANGEL
and Ego-Splitting; `--dry-run` says which are missing. Starting the queue
again continues it: every run decides from its files what is already done.
The queue writes under `--root DIR` (default `~/hedonic-runs/paper-queue`).

Runs execute detached (tmux when available), so closing the terminal does not
stop them: `hedonic run list`, `status`, `attach`, `logs`, `stop NAME` and
`resume NAME` manage them. Data goes to `~/hedonic-data` (`HEDONIC_DATA_DIR`;
SNAP files under `networks/`, or `HEDONIC_NETWORKS_DIR`), runs to
`~/hedonic-runs` (`HEDONIC_OUTPUT_DIR`), and built method runtimes to
`~/.cache/hedonic/codeseg` (`HEDONIC_CODESEG_CACHE`, or `hedonic config set
cache_dir PATH`). `hedonic config set output_dir PATH` and
`hedonic config set network_root PATH` change them for every run;
`hedonic config example` prints every setting.

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
publishes it to GitHub Pages; see `web/README.md` for details.

## Stability of a fixed cover

```python
from hedonic.stability import stable_fraction, stable_measures, nash_window

intervals = h.stable_intervals(max_memberships=3, allow_isolation=True)
regret, tolerance = h.regrets([0.0, h.density(), 0.1, 1.0], max_memberships=3,
                            allow_isolation=True)
stable = regret <= tolerance
linear_share, log_share = stable_measures(intervals)  # log floor: 1e-4
fractions = stable_fraction(intervals, [0.0, 0.1, 1.0])
window = nash_window(intervals)
```

Both methods default to `h.memberships`, preserving its community labels.
Pass `memberships=` to audit another cover, or `vertices=range(start, stop)`
to split work across processes. Rows are copied, repeated labels count once,
and label gaps are renumbered in numeric order. Every vertex needs a label;
complete uncovered vertices explicitly with private labels. The cap defaults
to the largest row and cannot be smaller.

The C audit finds closed stable intervals over resolution [0, 1] using the
paper's tolerance `1e-10 + 1e-9 * max(1, abs(current_utility))`. This differs
from the detector's rigorous equilibrium certificate. Audits support unit
node weights only, finite non-negative edge weights, and undirected graphs
without self-loops. `edge_weights=None` uses the graph's `weight` attribute
if present, otherwise ones. The native implementation computes each vertex's
supports on demand, avoiding a global vertex-by-label support table.
See [the core arithmetic contract](native/README.md#fixed-cover-stability-audit).

## Development

```bash
git clone https://github.com/lucaslopes/hedonic.git
cd hedonic
python -m pip install -e . pytest
python -m pytest
make -C native test            # the C core's own tests (sanitizers)
python tools/native_tests.py   # bit-level pins, as the wheel builds run them
```

The C core lives in `native/` (`include/hedonic_core.h` is its API,
[`native/README.md`](https://github.com/lucaslopes/hedonic/blob/main/native/README.md) its design) and is compiled by
`setup.py` into `hedonic._core`. Rounding is fixed by the source (explicit
fused multiply-adds, computed by the instruction or the core's own correctly
rounded soft fma, never the C library's; no contraction), so results are
identical across platforms; the wheel workflow
(`.github/workflows/wheels.yml`) checks this on every platform it builds for.

## License

This project is distributed under the GNU General Public License, version 3 or
later. See [LICENSE](https://github.com/lucaslopes/hedonic/blob/main/LICENSE). The C core was written for this package and
contains no code from igraph or other libraries; the binding implements the
MT19937 generator step (Matsumoto and Nishimura) as CPython's `random` module
does, so that seeded runs draw exactly what `random.Random` would.
