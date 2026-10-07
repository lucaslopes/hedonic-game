# Hedonic

`hedonic` detects communities in [`igraph`](https://igraph.org/) graphs as a
hedonic game: wrap a graph in `Game` and call `community_hedonic()`. One
method returns either a **disjoint** partition or an **overlapping** cover.

The native implementation comes from the `lucas-igraph` dependency, which is
installed automatically (Hedonic 1.0.5 pins `lucas-igraph==1.0.0.5`).
Hedonic requires Python 3.12 or newer.

## TL;DR

```bash
pip install hedonic
```

```python
from hedonic import Game

h = Game(g)  # g: an igraph graph (convert NetworkX, edge lists, ... through igraph)
partition = h.community_hedonic(max_memberships=1)                        # disjoint
cover = h.community_hedonic(max_memberships=3, initial_membership=None)  # overlapping
```

- `from hedonic import Game` is the public API.
- `max_memberships=1` puts each vertex in exactly one community and returns an
  `igraph.VertexClustering`.
- `max_memberships>1` lets each vertex join up to that many communities and
  returns an `igraph.VertexCover`.
- A `Game` remembers its last result and, when `initial_membership` is
  omitted, starts the next call from it. Pass `initial_membership=None` for a
  fresh start (see [Continuing from the last result](#continuing-from-the-last-result)).

## What this is

A **hedonic game** is a coalition-formation game: each vertex is a player,
each community is a coalition, and a player prefers the coalition that
improves its payoff. Here the payoff is the Constant Potts Model (CPM) score
at a resolution γ (by default, the graph density). Players repeatedly take a
**best-response** move (join, leave, or, when overlap is allowed, add or swap
a community) until no player wants to change.

That local-moving process is the first phase of the Leiden algorithm. Hedonic
exposes it as `Game.community_hedonic()`, backed by the native Leiden
implementation in `lucas-igraph`, so disjoint and overlapping detection share
one entry point.

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

The optional experiment CLI and adapters need the `experiments` extra
(`pip install "hedonic[experiments]"`, or `uv sync --extra experiments` in a
checkout).

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

# NetworkX (networkx is not a core dependency)
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

# Disjoint: each vertex has one community id
partition = h.community_hedonic(max_memberships=1, seed=0)
print(partition.membership[:8])  # e.g. [0, 0, 0, 0, 1, 2, 2, 0]
print(partition.sizes())         # community sizes

# Overlapping: each vertex has a list of community ids (length <= max_memberships)
cover = h.community_hedonic(max_memberships=3, initial_membership=None, seed=0)
print(cover.membership[:4])      # e.g. [[0, 1], [0, 1], [0, 1, 2], [0, 1]]
```

Iterate communities as lists of vertex ids:

```python
for community in partition:          # disjoint clusters
    print(list(community))

for community in cover:              # overlapping clusters
    print(list(community))
```

To seed an overlapping run from a disjoint partition, pass the partition as
the start:

```python
partition = h.community_hedonic(max_memberships=1, initial_membership=None)
cover = h.community_hedonic(
    max_memberships=3,
    initial_membership=list(partition.membership),
)
```

`phase_policy="disjoint_then_overlap"` does the same in one call (see
[New in 1.0.5](#new-in-105)).

### Continuing from the last result

Every call stores its result in `h.memberships`, one list of labels per
vertex. When `initial_membership` is omitted, the next call starts from that
state, so repeated calls continue from the previous equilibrium:

| `initial_membership` | Start state |
|---|---|
| omitted | the stored `h.memberships`, or singletons if nothing is stored |
| `None` | a fresh default start (singletons, or a feasible start under count limits) |
| a list | that partition or cover |

Because the stored state is reused, switching from an overlapping call back to
`max_memberships=1` without `initial_membership=None` fails: the stored cover
has vertices in several communities. You can also assign `h.memberships`
yourself, and score it with `h.evaluate_against(ground_truth, method="f1")`
(needs the `experiments` extra).

### 3. Parameters that matter

| Parameter | Default | Role |
|---|---|---|
| `max_memberships` | `1` | `1` = partition; `>1` = cover with that per-vertex cap; `-1` = no practical per-vertex cap |
| `resolution` | graph density | CPM γ. Smaller γ gives larger communities |
| `n_iterations` | `-1` | Negative: repeat until the native mover finds no improving move. Positive: a finite budget that may stop earlier |
| `local_move_only` | `True` | `True`: hedonic best response only. `False`: full Leiden (refine and aggregate) |
| `allow_isolation` | `False` | Allow a vertex to open a new, empty community |
| `initial_membership` | stored state | See [Continuing from the last result](#continuing-from-the-last-result) |
| `max_communities` | `None` | Random disjoint start over `[0, max_communities)` when no start is given; not an output limit |
| `edge_weights` | unweighted | Optional edge weights (sequence or edge-attribute name) |
| `seed` | `None` | Seed for the whole call (see below) |
| `beta` | `0.01` | Leiden refinement randomness (`local_move_only=False` only) |

A negative `n_iterations` ends with a native sweep in which no vertex has an
improving move; that is a strong stopping rule, not an independent proof. If
you need a game-theoretic certificate, audit the returned memberships
separately (the experiment extras include such audits).

### 4. Worked example

```python
import igraph as ig
from hedonic import Game

h = Game(ig.Graph.Famous("Zachary"))

partition = h.community_hedonic(max_memberships=1, seed=0)
print(f"{h.vcount()} vertices, {h.ecount()} edges")
print(f"disjoint communities: {len(partition)}, sizes {partition.sizes()}")

cover = h.community_hedonic(max_memberships=3, initial_membership=None, seed=0)
print(f"overlapping communities: {len(cover)}")
print(f"memberships of the first five vertices: {cover.membership[:5]}")
```

## New in 1.0.5

1.0.5 adds these keyword arguments to `community_hedonic`. Existing calls keep
working unchanged.

**Reproducible runs.** With `seed=`, the whole call runs with
`random.Random(seed)` as igraph's random generator, and the previous generator
is restored afterwards. The same graph, parameters, start and seed then give
the same result on a given build:

```python
a = h.community_hedonic(initial_membership=None, seed=7)
b = h.community_hedonic(initial_membership=None, seed=7)
assert a.membership == b.membership
```

**Community-count limits.** `max_total_communities=K` keeps the number of
occupied communities at most `K`, and `n_communities=K` keeps it exactly `K`,
for the result and for every state visited during optimization:

```python
partition = h.community_hedonic(initial_membership=None, n_communities=2)
cover = h.community_hedonic(initial_membership=None, max_memberships=3,
                            max_total_communities=4, local_move_only=False)
cover = h.community_hedonic(initial_membership=None, max_memberships=-1,
                            n_communities=5)
```

Both must be positive integers, and `n_communities` must not exceed
`max_total_communities`. Without a start state, a deterministic feasible start
is built; a supplied start that violates a limit is rejected, never silently
repaired. Under a count limit a stable result is stable among the moves the
limit allows (for example, a vertex may not empty the last occupied slot of a
community under `n_communities`), so it need not be an unconstrained
equilibrium.

**Unlimited overlap.** `max_memberships=-1` removes the practical per-vertex
cap: the effective cap is the vertex count.

**Vertex weights.** `node_weights=` (a sequence or vertex-attribute name,
finite and non-negative) weights the crowding term of the CPM score; unit
weights are the default.

**Two-stage overlap.** `phase_policy="disjoint_then_overlap"` first runs the
disjoint game with the same settings, then starts the overlapping run from
that partition. The default, `"direct"`, runs overlapping detection from its
own start. The two-stage policy imposes a partition structure that the overlap
can only extend: on complete com-DBLP it was faster but less accurate
(symmetric F1 0.380 versus 0.425 direct), so it is meant for warm starts,
ablations and very large graphs. The disjoint result is attached as
`result._hedonic_disjoint_stage`.

```python
cover = h.community_hedonic(max_memberships=3, seed=1,
                            phase_policy="disjoint_then_overlap")
```

**Diagnostics.** `debug_trace="counters"` records move, visit, iteration and
sweep counters; `debug_trace=True` (or `"full"`) also records every accepted
move with a recomputed potential change. It is expensive, so use it on small
graphs. The trace is in `result._params["debug_trace"]`, for partitions and
covers alike.

**Provenance.** Every result carries `result._hedonic_provenance`: the
algorithm identity, the installed `hedonic`, python-igraph and igraph C
versions, the effective parameters (including the effective per-vertex cap),
the count mode, the seed, how the run was started, the number of occupied
communities, and the stopping rule.

Interrupting a run (for example with Ctrl-C) raises through the native layer,
which releases its memory; no partial result is returned.

## Try it in one command

```bash
pip install hedonic
hedonic run exp
```

In a terminal, `hedonic run exp` opens a wizard that you drive with the
arrow keys, space and enter. Its first choice runs the default experiment:
one seed on a 5,000-vertex SNAP com-DBLP subgraph, comparing HOC-local
(`Game.community_hedonic`) with CoDeSEG. **Customize** walks through the
networks (all seven SNAP ground-truth networks), the methods (the ten that
passed the staged screening), the graph size, seeds, settings, threads,
timeout, output folder and cache folder. At the end it shows the equivalent
command line, so the same run can be scripted:

```bash
hedonic run exp --yes                      # the default, no prompts
hedonic run exp --yes --json > results.json  # wait for the run, then print the summary as JSON
hedonic run exp --networks dblp,amazon --methods all --seeds 0-4 --nodes 20000
hedonic run exp --networks youtube --full --methods hoc_local,fox,cpm
hedonic run exp --list                     # networks and methods
hedonic run exp --help                     # every option
```

**Reproducing the paper.** A single command runs the paper's experiment:

```bash
hedonic run paper --dry-run   # show the 200-run plan and a time estimate
hedonic run paper             # = hedonic run exp --config paper
```

It covers the two smallest SNAP networks (DBLP and Amazon, as full graphs),
all ten methods at their screened settings, and seeds 0–9, one run at a
time. The order is seed → method → network: seed 0 runs method 1 on DBLP and
then Amazon, then method 2 on both networks, and so on through all ten
methods; then seed 1, and so on. Each run has a one-hour timeout. The plan
takes several hours, so it runs as a durable, resumable run like any other. Records go to
`hedonic-paper-reproduction/<name>/`.

Other named configs are listed by `hedonic run exp --list-configs`. Any
config can be adjusted with flags, for example `--config paper --seeds 0-2`,
or saved to a file and passed back:

```bash
hedonic run exp --config paper --save-config my.json   # or my.toml
hedonic run exp --config my.json
```

Config files can be JSON or TOML (a `[hedonic_run]` table, or top-level keys as
written by `--save-config my.toml`). `--order`
changes the loop nesting of any run, for example `--order seed,method,network`.

**Your own defaults.** Per-user settings live in
`~/.config/hedonic/config.toml`; set `HEDONIC_CONFIG` to use another file.
Machine settings (`output_dir`, `cache_dir`, `network_root`, `threads`)
apply to every run, including named configs, and `run` names what a bare
`hedonic run` starts. That start asks for confirmation first; add `--yes` to
skip it.

```bash
hedonic config set output_dir ~/hedonic-runs    # every run is saved there
hedonic config set run paper                    # `hedonic run` = the paper reproduction
hedonic config                                  # show the file
```

The file can also define your own named configs in `[profiles.NAME]` tables;
see [`configs/hedonic-run.example.toml`](configs/hedonic-run.example.toml).

**Runs are durable.** Each benchmark runs in a detached tmux session, or a
detached background process when tmux is not installed. Your terminal shows
a live view with a progress bar, the current run, elapsed time, a calibrated
ETA, and the results table so far. Closing the terminal or pressing Ctrl-C
closes only the view; stopping a run is explicit:

```bash
hedonic run list            # all runs, status, progress
hedonic run attach [NAME]   # reopen the live view
hedonic run status [NAME]   # one snapshot
hedonic run logs [NAME]     # raw tmux pane or log file
hedonic run stop NAME       # stop (explicit)
hedonic run resume [NAME]   # continue after a reboot, a stop or a failure; finished runs are reused
```

A run is `running`, `completed`, `completed_with_failures` (some method failed
or was unavailable), `stopped` (you stopped it), `interrupted` (the worker
disappeared, for example after a reboot) or `failed` (an unexpected error, with
its reason; see `hedonic run logs NAME`). `NAME` may be any unique fragment of a
run name, such as its timestamp. Run names use letters, digits, `-` and `_`.

Results are printed as one table per network, with every accuracy metric
(symmetric/matching/micro/size-weighted F1, LFK-ONMI, Omega) and the
detection time. With several seeds each cell is mean ± sd. The best value
in each column is highlighted. Records and `summary.json` are saved under
the output folder.

To see what is available on this machine:

```bash
hedonic show methods             # ready / installed automatically on first use / missing a system tool
hedonic show networks --remote   # vertices, edges, on-disk copy, download size
hedonic show metrics             # what each accuracy metric measures
hedonic show covers              # graph/cover pairs found in the local SNAP cache
```

**Guide and updates.** `hedonic guide` prints suggested next steps from your local
state (data on disk, unfinished runs) and a short tour by topic (`hedonic guide
runs`, `spectrum`, `config`, …; an arrow-key menu in a terminal).
`hedonic update` compares the installed version with PyPI (one HTTPS request) and
installs nothing unless you add `--yes`; a development checkout is never changed,
and a build newer than PyPI is reported as unreleased.

**Ground-truth robustness spectrum.** `hedonic run spectrum` (or the last
wizard choice) audits the supplied overlapping covers found in your local SNAP
cache: for every available graph/cover pair it plots the fraction of vertices
with no profitable unilateral action against the resolution, for fixed and open
labels, and runs seeded local moving from the exact ground-truth cover, with
every returned cover independently audited and scored against its start. It
never downloads data unless you add `--provision`, and it is durable and
resumable like the benchmark runs (`hedonic run attach|stop|resume`). Details:
[`docs/overlapping_gt_spectrum.md`](docs/overlapping_gt_spectrum.md).

```bash
hedonic run spectrum --dry-run --inspect   # cohort, plan and sizes; runs nothing
hedonic run spectrum                       # wizard, then a durable run
```

**Networks** (`--networks`, or `all`): the seven SNAP networks with ground
truth. The default subgraph size is 5,000 vertices, set with `--nodes`; use
`--full` for the whole graph.

| key | network | vertices | edges |
|---|---|---:|---:|
| `dblp` | DBLP co-authorship | 317,080 | 1,049,866 |
| `amazon` | Amazon co-purchasing | 334,863 | 925,872 |
| `youtube` | YouTube | 1,134,890 | 2,987,624 |
| `wikipedia` | Wikipedia (top categories) | 1,791,489 | 28,511,807 |
| `livejournal` | LiveJournal | 3,997,962 | 34,681,189 |
| `orkut` | Orkut | 3,072,441 | 117,185,083 |
| `friendster` | Friendster | 65,608,366 | 1,806,067,135 |

**Methods** (`--methods`, or `all`): the ten methods that finished the staged
screening on full DBLP within ten minutes. The scores and times below are for
the `screened` setting on full com-DBLP (Apple M1 Max, 10 threads where
supported). `--preset registry` uses the registry (paper) defaults instead.

| key | method | F1 | ONMI | time [s] | needs |
|---|---|---:|---:|---:|---|
| `fox` | FOX (LazyFox) | 0.428 | 0.525 | 184 | CMake, OpenMP C++ compiler |
| `hoc_local` | HOC local moving (`Game.community_hedonic`) | 0.425 | 0.519 | 40 | nothing |
| `cpm` | clique percolation | 0.419 | 0.524 | 9 | nothing |
| `codeseg` | CoDeSEG | 0.406 | 0.518 | 6 | C++ compiler |
| `angel` | ANGEL | 0.403 | 0.518 | 58 | `uv` (isolated CDlib environment) |
| `bigclam` | BigCLAM | 0.395 | 0.472 | 582 | git, make, C++ compiler |
| `ego_splitting` | Ego-Splitting | 0.333 | 0.500 | 96 | `uv` (isolated CDlib environment) |
| `ncgame` | NcGame | 0.326 | 0.484 | 15 | git |
| `neo_kmeans` | NEO-K-Means | 0.334 | 0.398 | 475 | make, C/C++ compiler |
| `hoc_multilevel` | HOC multilevel | 0.171 | 0.392 | 63 | nothing |

Output goes to `OUTPUT_DIR/NAME/` (default `hedonic-exp-output/exp-<date>-<time>/`):
- one runner record per network, seed and method, in
  `<network>-n<nodes>/seed<k>/runs/<network>/<method>.json`;
- `progress.json`, which the live view reads;
- `summary.json`, with the table values and the exact command line;
- `worker.log`.

On first use the tool downloads the SNAP archives and prepares external
runtimes, then caches everything under `~/.cache/hedonic`:
- CoDeSEG's C++ source, pinned to the reproduced commit, is compiled with
  the system compiler. The repository has no licence file, so its code is
  not bundled.
- The other external methods are set up through `hedonic-exp codeseg-setup`.

A method whose runtime cannot be prepared is reported as unavailable, with
the reason, and the others still run. `--foreground` runs in the current
terminal instead of a detached session. tmux is optional but recommended,
because it lets `hedonic run logs` show the raw pane. The list of runs is
kept in `~/.cache/hedonic/runs/`; set `HEDONIC_RUNS_DIR` to move it.

## Experiments

Experiment drivers are optional so the core installation stays small:

```bash
uv sync --extra experiments
hedonic-exp list
hedonic-exp smoke
```

The `hedonic-exp` commands cover small smoke checks, synthetic disjoint
experiments, overlapping diagnostics, and reproducible benchmark pipelines.
Experiment data paths can be configured with the TOML files under `configs/`
or with the documented `HEDONIC_*_DIR` environment variables.

For a turnkey run on a fresh machine, install the reproduction extra and use
`overlapping-reproduce`. The smoke profile is archive-free but still exercises
the complete method registry on a deterministic ~1,000-node AGMfit-like
overlapping fixture. The full profile bootstraps the requested SNAP archives
and external runtimes, then runs all registered methods. Every method gets an
explicit `completed`, `unavailable`, or `failed` row; an unavailable external
implementation is never silently replaced by another detector.

```bash
python -m pip install "hedonic[reproduce]"
# Archive-free qualification (~1,000 vertices; all registered methods)
hedonic-exp overlapping-reproduce --profile smoke --resume

# Complete full-DBLP SNAP run (the current benchmark target)
hedonic-exp overlapping-reproduce --profile full --resume
# Check downloads/runtimes without launching detectors
hedonic-exp overlapping-reproduce --profile full --preflight
# Opt in to the complete seven-network catalogue when desired
hedonic-exp overlapping-reproduce --profile full --datasets all --resume
```

The turnkey command records the setup manifest, method provenance, accuracy
metrics (symmetric best-match F1/Jaccard, one-to-one matching, node-membership
F1, size-weighted F1, sampled Omega, LFK-ONMI, coverage/overlap diagnostics),
wall-clock time, vertex/edge throughput, CPU time, and peak RSS. Results are resumable under
`artifacts/overlapping/turnkey/<profile>/`. CoDeSEG, LazyFox, and BigCLAM
still require a working C/C++ toolchain and CMake; Octave/R-based methods are
marked unavailable when those system runtimes are absent. Use
`--require-all` when an incomplete environment should fail instead of
producing an explicit partial ledger.
The top-level `turnkey_manifest.json` marks a run with method failures or
unavailable runtimes as `partial`, even when the process itself exits zero.

The full DBLP run is intentionally a large, long-running experiment; use
`--preflight` first and adjust `--timeout` or `--methods` when qualifying a
new machine. `--datasets all` expands the same command to the seven supported
SNAP archives.

For the lower-level DBLP-only protocol, `codeseg-setup` and
`overlapping-codeseg` remain available. The setup command downloads SNAP files
lazily and records external source checkouts, executable hashes, and the
isolated CDlib worker in a cache manifest. `--max-nodes 5000` is a bounded
qualification fixture; remove it for a full `com-DBLP` detector run. The full
archive preflight is 317,080 vertices, 1,049,866 edges, and 13,477 SNAP
communities.

The same 25 names are available from the experiment adapter registry, so
scripts do not need to know which backend implements a method:

```python
from hedonic.experiments.overlapping.methods import run_method_by_name

cover, metadata = run_method_by_name(
    "Neo-K-Means", graph,
    max_memberships=4,
    resolution=graph.density(),
    seed=0,
    ground_truth=snap_cover,
)
```

The same registry can be paired with the shared accuracy implementation:

```python
from hedonic.experiments.overlapping.metrics import evaluate_cover

scores = evaluate_cover(
    cover,
    snap_cover,
    graph.vcount(),
    compute_omega=True,
)
print(scores["f1"], scores["matching_f1"], scores["node_micro_f1"], scores["omega"])
```

Thus a benchmark caller selects a detector by its registered name and scores
the returned cover through one common metric contract; the turnkey command
does exactly this for every selected method.

Names are case-insensitive and common spellings such as `Neo-K-Means` and
`Link Communities` are normalized. Optional external runtimes fail closed
with `MethodUnavailable`; they are never silently replaced by another method.

OSLOM is retained in the complete registry but may be unavailable because its
legacy executable is not bundled. NISE and SSE use the recovered official
seed-expansion source family and are recorded as separate rows. DSC, Stad, and
PaNDEMON remain literature candidates without a certified local runtime
adapter.

The complete method/PDF/code audit is maintained in
[`docs/papers/overlapping_communities/references/DBLP Overlapping Community Detection - consolidated.md`](docs/papers/overlapping_communities/references/DBLP%20Overlapping%20Community%20Detection%20-%20consolidated.md),
and the reproduction protocol is documented in
[`docs/reproduce_codeseg.md`](docs/reproduce_codeseg.md).

## Interactive explainer

The `web/` directory contains an educational website that explains the
disjoint hedonic game visually: friends and strangers on a balance scale, the
resolution parameter, the Familiarity Index, the decision tree, the metagraph
of all partitions of a four-vertex graph, and better-response walks to a
stable equilibrium. It also hosts documentation generated from this package's
source. It is a small browser-side teaching model, not a replacement for the
native detector. From the repository root:

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
uv sync --extra experiments
uv run pytest
uv build
```

The public API is intentionally small:

```python
from hedonic import Game
```

Research manuscripts, private evidence, and manuscript-only configuration are
kept outside the public `main` publication boundary.

## Releases and publishing

Hedonic versions are independent from the four-component release identity of
the native `lucas-igraph` dependency. A dependency update can therefore be a
normal Hedonic patch release without changing the `community_hedonic` call
pattern.

Package builds are produced with `uv build`. Publishing is a maintainer-only
release step performed after the tagged source and hosted checks have been
verified. Credentials must be supplied through the configured secret or a
hidden interactive environment variable; never put a PyPI token directly in a
shell command, README, commit, or issue.

After publication, the `Validate released wheels` workflow can be dispatched
with the immutable version (for example `1.0.5`). It installs that exact PyPI
release on Ubuntu, Windows, and macOS runners under Python 3.12 and 3.13,
verifies the `lucas-igraph` pin, and exercises the installed-wheel copy and
directionality contract.

## License

This project is distributed under the GNU General Public License, version 3 or
later. See [LICENSE](LICENSE) for the complete terms.
