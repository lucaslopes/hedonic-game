# SNAP overlapping benchmark

Run the complete saved-SNAP benchmark through the experiment CLI:

```bash
uv sync --extra experiments

hedonic-exp overlapping-benchmark --profile smoke \
  --output_dir /tmp/hedonic-snap-smoke

hedonic-exp overlapping-benchmark \
  --datasets amazon,dblp,livejournal,youtube,wikipedia \
  --cover top5000 --profile standard \
  --output_dir ~/Databases/Hedonic/Networks/SNAP_BENCHMARK_CLI
```

The input root defaults to `~/Databases/Hedonic/Networks`; set
`HEDONIC_NETWORKS_DIR` or pass `--data_root` to override it. Raw inputs are
never modified. Validated normalized graph/cover caches live under
`~/.cache/hedonic/snap` by default (`HEDONIC_SNAP_CACHE_DIR` overrides it).

The five valid overlapping benchmarks are Amazon, DBLP, LiveJournal, YouTube,
and Wikipedia category lists. The first four have supplied `all` and
`top5000` covers. Wikipedia's ground truth is
`wiki-topcats-categories` and only has an `all` variant; a global `top5000`
run records Wikipedia as an intentional skip. email-Eu-core, Cora, PubMed, and
`DBLP_CLI` are deliberately excluded because they are respectively disjoint
labels or prior output artifacts.

## Profiles and controls

- `smoke` uses built-in small overlapping graphs and runs without any archive.
- `standard` runs every selected saved network on a deterministic,
  ground-truth-informed induced subgraph capped at 3,000 nodes.
- `full` removes the node cap. Use it with explicit `--timeout_per_run` and
  `--resume` for the largest social graphs.

All profiles select the three multi-phase hedonic density variants plus CPM and
DEMON by default. `hedonic_local` remains available as an explicit `--methods`
choice.

`--resolutions auto` evaluates each graph density; comma lists and inclusive
`start:stop:count` grids are also accepted. `--seeds 0-4`, `--omega`, and
`--omega_sample_size` configure repeat runs and the memory-safe sampled Omega
index. `--dry-run` validates selected data without launching detectors.

## One-command paper reproduction

The full protocol for
[`main.tex`](papers/overlapping_communities/main.tex) is declared in
`[overlapping_paper]` in [`configs/hedonic.toml`](../configs/hedonic.toml).
It compares `hedonic_multiphase`, `hedonic_multiphase_x10`, and
`hedonic_multiphase_x100` with CPM and DEMON; the paper-specific runner rejects
`hedonic_local`. The three hedonic variants are full multi-phase runs with
`allow_isolation=True` and resolutions `min(density × 1, 1)`,
`min(density × 10, 1)`, and `min(density × 100, 1)`, respectively.

```bash
uv sync --extra experiments
uv run hedonic-exp reproduce-overlapping-paper --dry-run
uv run hedonic-exp reproduce-overlapping-paper
tmux attach -t hedonic-overlapping-paper
```

Before launching tmux, the launcher measures each graph and supplied cover,
derives `max_memberships` from the maximum ground-truth memberships of any
single node, and estimates parent/child igraph plus method-specific NetworkX
and historical peak memory. The default 64-GiB Mac plan reserves 24 GiB for
macOS and treats the remaining 40 GiB as a hard budget. LiveJournal is never
co-scheduled with another dataset; uncertain estimates force serial workers.
The detector monitor sums RSS for the root and all descendants at high
frequency, records the observed peak and enforcement reason, and classifies
`memory_limit`, `timeout`, `oom`/`exit=-9`, and `failed` separately. A coordinator merges all cache records into
`docs/papers/overlapping_communities/artifacts/full/`, regenerates plots and
the paper table fragment, and runs `latexmk` only after every planned full-run
record is present and completed. Adjust all normal settings in
TOML—not an ad-hoc shell command—and use `--no-tmux` only for smoke/debug
configs.

Large detector covers are returned through a private temporary pickle artifact,
not a `multiprocessing.Queue`. This avoids bounded-pipe deadlocks when a child
finishes detection with a cover too large to flush before process exit; the
temporary artifact is removed immediately after the parent reads it.
The transport and shared-warm-start changes increment the run protocol, so
legacy singleton-start and affected timeout records are retained as
incompatible backups and recalculated on `--resume`.

## Methods

`hedonic_local` and the three `hedonic_multiphase*` variants call
`Game.community_hedonic(max_memberships=K, n_iterations=-1)`, where `K` is the
maximum number of supplied ground-truth communities containing any one node
(at least two), not the total number of communities. The local method uses only
the local-moving phase; the multi-phase variants enable full Leiden refinement
and aggregation. They set `allow_isolation=True` and cache the effective
density multiplier resolution in each per-run record. For a given dataset and
seed, every hedonic variant receives the same seeded random disjoint warm start,
requesting one initial label per supplied ground-truth community. CPM and DEMON
do not accept an initial cover.

Two independent default overlapping baselines are included in the experiments
extra:

| Method | Family and implementation | Key parameters | Scale/output |
|---|---|---|---|
| `cpm` | Clique percolation via NetworkX `k_clique_communities` | `clique_size=3` | Direct node cover; can be exponential in dense clique-rich graphs, so it is timeout-protected. |
| `demon` | Local expansion via the external `demon.Demon` package | `epsilon=0.25`, `min_community_size=2` | Direct node cover; local ego-network expansion can be expensive on high-degree graphs. |

Use `hedonic-exp overlapping-benchmark --list-methods` for the exact installed
availability and requirements. The archive loader preserves Wikipedia's raw
directed graph for provenance, then the benchmark applies the tracked
`common_undirected_simple_v1` projection before computing density or passing
the graph to any detector or metric. Hedonic, CPM, and DEMON therefore consume
the same loop-free undirected simple graph. Historical records without this
analysis-graph identity are inadmissible under the locked paper protocol.

## Results and interpretation

Each load records source paths, graph directionality, normalized ID mapping,
dropped/missing original IDs, community-size statistics, and overlap statistics.
DBLP's cached `vs["label"]` mapping is used instead of mistaking original IDs
for igraph indices.

Results are incremental and resume-safe:

```text
<output>/
├── manifest.json
├── method_availability.json
├── runs/<dataset>/<cover>/<method>/seed_<seed>/resolution_<gamma>.json
├── results.jsonl
├── results.csv.gz
├── summary.json
├── summary.csv
├── logs/benchmark.log
└── plots/
```

Per-run records distinguish `completed`, `timeout`, `memory_limit`, `oom`,
`failed`, `skipped_unsupported`, and `skipped_not_scalable`. Each isolated
run records wall-clock time, configured timeout and memory limit, aggregate
detector-tree RSS peak, process count, termination reason, and compact RSS
samples. `--resume` reuses only compatible `completed` records (or an explicit
unsupported-dependency decision); it never prints `[resume]` for a failure.
Protocol/method/dataset/seed/resolution/membership/timeout/memory/Omega changes
make a record incompatible and retain its JSON beside the replacement.
CPM/DEMON resource exhaustion is an explicit `skipped_not_scalable` policy
decision rather than a metric or a repeated detector invocation. Hedonic
resource failures remain explicitly classified as timeout/OOM/memory-limit,
but are likewise not automatically launched again under the same protocol.
Retries default to zero; timeout, memory-limit, and OOM records are never retried.
Summary and plots separate
recovery scores (best-match, one-to-one, node-membership and weighted F1),
structural overlap behavior (inclusion, coverage, overlap, distribution,
coverage/size/membership diagnostics), CPM graph quality, and runtime. Omega
is opt-in and sampled; no dense vertex-pair matrix is allocated.

## Safe paper reproduction

Inspect the complete plan without starting detectors or a tmux session:

```bash
uv run hedonic-exp reproduce-overlapping-paper --dry-run
```

The dry-run prints the hard budget, reserve, worker count, every wave, and the
reason for serialization. It writes `orchestration/plan.json`; previous plans
are retained as timestamped `plan.previous.*.json` files. The coordinator waits
for every worker wave and writes `failure_report.json`/`.csv`. It never switches
the manuscript to full results or compiles the paper when any condition is
missing or non-completed. `--compile-partial` is an explicit opt-in and leaves
an `EXPLICIT PARTIAL COMPILE` warning in `paper_status.tex`.

The paper artifact directory also contains `condition_summary.csv` and
`condition_summary.json`, plus `coverage_report.csv`/`.json` with expected,
completed, not-scalable, timeout, OOM, memory-limit, and missing counts for
each planned dataset/method/seed condition.

The exact full launch, after reviewing the dry-run, is:

```bash
uv run hedonic-exp reproduce-overlapping-paper
```

Do not launch this command automatically from a correction or CI smoke run.

### Locked evidence audit

Before launching or interpreting a full-paper run, reconcile its evidence with
the tracked protocol lock:

```bash
hedonic-exp overlapping-audit \
  --output docs/papers/overlapping_communities/evidence/protocol_audit.json
```

The command is read-only with respect to experiment artifacts: it never loads
a graph or launches a detector. It inventories all 125 requested conditions
and records the exact reason each present record is accepted or rejected. The
JSON lock pins the modified `lucas-igraph` Git revision and SHA-256 hashes of
the authoritative config and detector/orchestration sources. New manifests and
run records embed that identity plus a hash of the loader's dataset metadata.
Changing a resource envelope, protocol field, code hash, or native dependency
therefore cannot silently reuse evidence from another condition.
