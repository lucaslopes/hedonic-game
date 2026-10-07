# AGENTS.md — Working on `hedonic`

Guidance for humans and coding agents editing this repo. Keep the **core library light** and put experiment I/O, sweeps, and metrics in `hedonic.experiments`.

---

## Public `main` and private `paper` workflow

This local repository has an asymmetric publication boundary:

- `main` tracks `origin/main`, which is the public GitHub repository. Only
  public-ready code, documentation, and explicitly reviewed artifacts belong
  there.
- `paper` is the local private research branch. It contains manuscripts,
  unpublished figures, evidence, and manuscript-only protocol material. It
  must never be pushed to `origin` or merged wholesale into `main`.
- The different `.gitignore` rules are intentional. On `main`,
  `docs/papers/`, private evidence, and manuscript-only configuration are
  ignored. On `paper`, those files are tracked. Ignore rules do not make an
  already tracked file private.

### Daily workflow

1. Do research, write manuscripts, and commit private progress on `paper`.
2. Bring public updates into research only in the safe direction:
   `git switch paper`, `git fetch origin`, then `git merge origin/main`.
3. Do **not** run `git merge paper` while on `main`. A merge would make the
   private branch history reachable from the public branch.
4. When a change developed on `paper` is ready to publish, either recreate it
   on `main` or cherry-pick one commit only after confirming it contains solely
   public code and documentation. Inspect it with `git show --stat <commit>`
   before cherry-picking.
5. Push only `main` to `origin`. A local pre-push hook rejects attempts to
   push `paper` to `origin`.

If a private remote is added later, name it `private` and set `paper` to track
`private/paper`; never change `origin` away from the public repository.

### Repository skills

This repository has two complementary skills with different scopes:

- **`sibling-release-pipeline`** (`.skills/sibling-release-pipeline/SKILL.md`)
  is explicit-only. Invoke it for a coordinated `1.0.0.N` release across the
  C `igraph` repository, `python-igraph`, and this project. It integrates the
  detailed work into a local `lucas-dev` line, creates one squashed sibling
  release commit per upstream, verifies the exact vendored C pointer, waits
  for push-triggered GitHub Actions, and only then moves `lucas`/
  `physica-a` and creates immutable tags. It does not open pull requests by
  default and it must stop before PyPI unless publication is separately
  authorized. Upstream release messages use the preceding sibling as a
  repository-specific template, while rewriting every claim to match the new
  diff and checks.
- **`clean-worktree-commits`** is the cleanup path for a dirty checkout. Use it
  before a release when uncommitted work must be preserved in reviewable local
  commits. It inspects every staged, unstaged, and untracked path, runs
  proportionate checks, refuses destructive cleanup/history rewriting, and
  never pushes. Review README/package metadata as part of the public cohort
  when the result may be published to PyPI.

For a coordinated release that reaches this repository, the Hedonic package
version is `1.N.x`: `lucas-igraph 1.0.0.N` maps to Hedonic `1.N.0` (for example
`1.0.0.4` -> `1.4.0`), never an independent patch number. A Hedonic-only change
with no upstream release increments `x` (`1.4.1`) with the `lucas-igraph` pin
unchanged; a Python-only upstream change reuses the C-core commit through an
alias tag (see the skill's "Partial releases"). Update a private `paper` worktree unless
the user explicitly selects the public `main` release boundary. Never replace
historical `.3` locks or evidence merely because the runtime dependency moves
forward.

---

## Package layout

```
src/hedonic/
├── __init__.py          # public API: Game only
├── Game.py              # Game(igraph.Graph) + community_hedonic
├── utils.py             # small helpers (e.g. seeded sampling)
├── py.typed
└── experiments/         # optional extra: replication & sweeps
    ├── config.py        # data path defaults + env overrides
    ├── CLI.py           # hedonic-exp + hedonic entrypoints (capital C, intentional)
    ├── plots/           # paper figures (matplotlib; optional extra)
    │   └── paper_figures.py  # V1020 PHYSA figures
    ├── disjoint/
    │   ├── sbm_sweep.py     # synthetic SBM parameter sweeps
    │   ├── data_loader.py   # JSON → CSV result pipeline
    │   └── reproduce.py     # end-to-end: sweep → CSV → figures
    └── overlapping/
        ├── metrics.py           # F1/Jaccard/Omega, cover helpers
        ├── small_graphs.py      # smoke tests (no DBLP)
        ├── dblp_full.py         # full DBLP graph
        ├── dblp_subgraph.py     # L-hop GT subgraphs
        ├── complexity_scale.py  # size vs wallclock (local-moving vs multi-phase)
        ├── resolution_f1.py     # full-DBLP F1 vs γ (multi-seed CI + line plot)
        ├── snap.py              # shared, ID-safe SNAP graph/cover loader
        ├── methods.py           # hedonic + external overlap-method adapters
        ├── benchmark.py         # five-network resumable SNAP benchmark
        ├── codeseg_reproduction.py  # DBLP SNAP protocol (25 default methods)
        ├── codeseg_setup.py     # data/runtime bootstrap + doctor manifest
        ├── reproduce_paper.py   # TOML/tmux full-paper orchestration + aggregation
        ├── quickstart.py        # `hedonic run exp`: catalogue, flags, wizard, provisioning, runs, tables
        ├── runmanager.py        # durable runs: tmux/process launch, worker, live viewer, list/stop/resume
        ├── tui.py               # dependency-free arrow-key prompts (numbered fallback off a TTY)
        ├── info.py              # `hedonic show methods|networks|metrics|covers`
        ├── guide.py             # `hedonic guide [TOPIC]`: state-aware tour; keep TOPICS in step with the CLI
        ├── selfupdate.py        # `hedonic update [--yes]`: PyPI check, pip upgrade only when asked, never a dev checkout
        ├── userconfig.py        # ~/.config/hedonic/config.toml: machine defaults, favourite run, user profiles
        ├── extra_methods.py     # opt-in runner methods (ego_splitting) outside the 25-method registry
        ├── gt_spectrum.py       # SNAP ground-truth robustness spectrum: discovery, audit, GT-seeded local moving, front door
        └── gt_spectrum_report.py  # its figures, pair table and generated TeX
```

| Layer | What belongs here | What does **not** |
|-------|-------------------|-------------------|
| **Core** (`Game`, `utils`) | Graph model, `community_hedonic`, membership validation | Pandas, file walks, DBLP loaders, plotting |
| **Experiments** | Argparse, paths, sweeps, metrics vs GT, JSON dumps | Re-implementing Leiden / hedonic local moves |

Public import:

```python
from hedonic import Game
```

Do **not** reintroduce `OverlappingGame` or pure-Python Leiden ports. Overlapping lives in **lucas-igraph** via `max_memberships`.

---

## Core model: `Game`

`Game` subclasses `igraph.Graph` (from **lucas-igraph**). Wrap any graph:

Finite negative resolution is an explicit executable extension. Keep every
mathematical claim from a non-negative-resolution protocol scoped to that
domain.

```python
import igraph as ig
from hedonic import Game

g = Game(ig.Graph.Famous("Petersen"))
# or: Game(ig.Graph.SBM(pref_matrix, block_sizes, directed=False))
```

### Primary method: `community_hedonic`

All exploratory community detection (disjoint **and** overlapping) should go through **`Game.community_hedonic`**, which calls `community_leiden` with the right flags.

| Goal | Call pattern | Return type |
|------|----------------|-------------|
| Disjoint partition | `max_memberships=1` (default) | `VertexClustering` |
| Overlapping cover | `max_memberships > 1` | `VertexCover` |
| Hedonic local-moving only | `local_move_only=True` (default) | same as above |
| Full Leiden (refine + aggregate) | `local_move_only=False` | same as above |

```python
# Disjoint — main exploratory method
part = g.community_hedonic(
    resolution=g.density(),   # CPM γ; default is density if omitted
    max_memberships=1,
    local_move_only=True,
    n_iterations=-1,
    initial_membership=None,  # default: singleton partition
)

# Overlapping — same API
cover = g.community_hedonic(
    resolution=g.density(),
    max_memberships=4,
    local_move_only=True,
    n_iterations=-1,
)

# Warm-start overlapping from a disjoint partition (flat labels OK)
cover = g.community_hedonic(
    resolution=g.density(),
    max_memberships=4,
    initial_membership=list(part.membership),  # expanded to [[c], ...] internally
)
```

**Useful parameters**

- `initial_membership` — disjoint: `list[int]`; overlapping: flat `list[int]` or per-vertex `list[list[int]]`
- `max_communities` — only when `initial_membership is None` and disjoint: random labels in `[0, K)` (initialization only, never an output constraint; `_hedonic_provenance["max_communities_effect"]` records whether it applied)
- `max_memberships` — per-vertex cap; `-1` removes the practical cap (effective cap = vertex count)
- `max_total_communities` / `n_communities` — keyword-only global limits on
  occupied communities (at most / exactly K), enforced at every local-moving,
  aggregate and token level by lucas-igraph 1.0.0.5; infeasible requests and
  violating start states are rejected, never repaired
- `allow_isolation` — allow empty-community moves
- `n_iterations` — use a negative value (the default is `-1`) for the native
  no-change run and final hedonic Nash certificate sweep. The candidate set is
  complete for both resolution signs and both isolation policies (and for the
  feasible actions under a count limit). Still audit the returned cover when a
  mathematical certificate is needed. Returns are stamped with
  `HEDONIC_ALGORITHM_IDENTITY = "community_hedonic/lucas-igraph-1.0.0.5"` and
  carry `_hedonic_provenance` (runtime versions, effective parameters, count
  mode, stopping rule). The frozen 1.0.0.3 producer string stays in
  `HISTORICAL_ALGORITHM_IDENTITIES` for Gate H. Interruption (`KeyboardInterrupt`)
  propagates through the native error path.
- `seed` — when given, the whole call runs under `random.Random(seed)` as
  igraph's generator (restored afterwards), so seed + inputs replay the
  trajectory on a given build; `None` keeps the caller's generator
- `edge_weights`, `node_weights` (keyword-only, finite and non-negative), `beta` — weights and Leiden refinement noise
- `debug_trace` — keyword-only, for partitions and covers, with or without
  count limits: `True`/`"full"` records every accepted local move (with a
  direct recomputation of its potential change; bounded fixtures only), every
  overlapping multilevel proposal and the counters; `"counters"` records only
  the counters. The envelope in `result._params["debug_trace"]` names the
  schema version, runtime versions, `seed_context`, the count policy, and
  what is recorded and omitted (rejected candidates are only counted)
- `phase_policy` — keyword-only, covers only: `"direct"` (default) or the
  opt-in `"disjoint_then_overlap"` warm start (disjoint game with the same
  options, its partition passed as `initial_membership`; stage-1 result in
  `_hedonic_disjoint_stage` and provenance). On complete DBLP it was faster
  but lost symmetric F1 (0.380 vs 0.425), so use it for ablations, warm
  starts and very large graphs, not as a default

**Do not** call raw `community_leiden` in experiment code unless you are debugging the binding itself. Prefer `community_hedonic` so defaults stay consistent (`local_move_only=True`, density resolution, membership normalization).

---

## Dependencies

```toml
# Core
lucas-igraph==1.0.0.3   # released native dependency
numpy

# Optional: pip install "hedonic[experiments]"  or  uv sync --extra experiments
pandas, scipy, tqdm, stopwatch-py, matplotlib, seaborn, networkx, demon

# CoDeSEG/SNAP reproduction: pip install "hedonic[reproduce]"
# (the SNAP downloader is built into experiments; no data is fetched at install)
```

Setup:

```bash
uv sync --extra experiments
# For the complete CoDeSEG setup:
uv sync --extra reproduce
hedonic-exp codeseg-setup --all --build-native
hedonic-exp codeseg-doctor --require-all
# If lucas-igraph fails to rebuild on macOS, set SDKROOT to the active Xcode SDK first.
# Fox's OpenMP build auto-detects Homebrew g++; override with --fox-cxx if needed.
```

CLI after install (console scripts from `pyproject.toml`: `hedonic-exp` → `hedonic.experiments.CLI:main`,
`hedonic` → `hedonic.experiments.CLI:hedonic_main`):

```bash
hedonic-exp --help
hedonic-exp list
# or without install:
python -m hedonic.experiments.CLI --help
```

---

## Experiments package

### `hedonic run` front door (`quickstart`, `runmanager`)

`hedonic run exp` is the path users (and long benchmarks) take. Keep these
contracts when changing it:

- **Same runner.** Every detector call goes through
  `codeseg_reproduction.main` (one call per network × seed × method, with
  `--resume`), so records, bounds, the paper filter, and metrics are those of
  `overlapping-codeseg`. Do not score or load data in a second way.
- **Catalogue.** `quickstart.NETWORKS` lists the seven SNAP ground-truth
  networks. `quickstart.METHODS` lists the ten methods that passed the staged
  screening
  (`docs/papers/overlapping_communities/screening/screening_report.md`).
  Each entry carries its Stage 3 `screened` setting, its full-DBLP detection
  time (the ETA basis), and F1/ONMI hints. Update these values only from
  saved screening records. Community-count parameters (`scaled`) shrink with
  the subgraph. HOC-local uses `absolute_resolution` 0.1462 (= 7000 × DBLP
  density), because a density multiple does not transfer to denser
  subgraphs.
- **Named configs.** `quickstart.PROFILES` holds the named configs:
  `default`, and `paper`, the paper's reproducible experiment (DBLP and
  Amazon as full graphs × the 10 methods × seeds 0–9, screened settings,
  1-hour timeout, `order="seed,method,network"`). YouTube is excluded
  because ANGEL did not finish on full YouTube within an hour.
  - Selected with `--config NAME|FILE.json|FILE.toml`; explicit flags
    override the config. `hedonic run paper` is shorthand for
    `hedonic run exp --config paper`.
  - `--dry-run` prints the plan and a time estimate, and `--save-config`
    writes a resolved config.
  - Changing `PROFILES["paper"]` changes the paper's experiment; record why
    in the manuscript.
  - The time estimate uses `MEASURED_FULL`, the full-Amazon/YouTube detection
    times from saved records, and otherwise a linear-in-edges extrapolation
    from full DBLP.
- **User config.** `overlapping.userconfig` reads
  `~/.config/hedonic/config.toml` (`HEDONIC_CONFIG` overrides the path;
  `XDG_CONFIG_HOME` is respected).
  - `[defaults]` holds machine keys only (`output_dir`, `cache_dir`,
    `network_root`, `threads`). They are applied on top of any named config
    and under explicit flags.
  - `run` is what a bare `hedonic run` starts. It always confirms first, and
    a non-TTY session never starts without `--yes`.
  - `[profiles.NAME]` tables are user-defined configs; `config = "paper"`
    inside one names its base.
  - Keep machine paths out of the repository; the shareable template is
    `configs/hedonic-run.example.toml`.
  - Managed with `hedonic config [show|path|set|unset]`.
- **Modes.** In a terminal, no arguments means the wizard (`tui.py`). Flags,
  `--yes`, or a non-TTY session run directly. The wizard always prints the
  equivalent flags.
- **Durability.** Runs execute detached in tmux session `hedonic-<name>`, or,
  without tmux, in a detached process. The worker ignores SIGINT and writes
  `OUTPUT_DIR/NAME/progress.json` after every step. The terminal shows only a
  viewer, whose Ctrl-C never stops the run. `stop` is explicit, and `resume`
  relaunches the same config, reusing finished records. The registry of runs
  is `~/.cache/hedonic/runs/` (`HEDONIC_RUNS_DIR`). `--foreground` keeps the
  old in-terminal behaviour.
  - Liveness is the worker **pid** (checked against its command line so a
    recycled pid is never trusted or signalled), never the tmux session: the
    pane deliberately stays open after the worker ends. An unexpected worker
    error is recorded as `failed` with its reason; a vanished worker is
    `interrupted`. `launch` records absolute paths, resets `progress.json`,
    forwards the build/config environment (`SDKROOT`, `XDG_*`, `UV_*`, ...)
    into the tmux pane, and replaces the idle pane of a finished run.
  - `--json` waits for the run and prints only the summary rows on stdout
    (progress goes to stderr); `--detach` returns immediately instead.
  - The viewer repaints in place, fits every line to the terminal width, and
    hides the results table while the terminal is too short for it (the full
    table is printed at the end). `render` drops `± sd` when it would not fit.
  - Run names are `[A-Za-z0-9_-]{1,64}` (they become directory and tmux
    session names); `NAME` arguments accept an exact name or a unique fragment.
- **Provisioning.** Data and runtimes are fetched lazily into `--cache-dir`
  (default `~/.cache/hedonic`: `snap/`, `codeseg/`).
  - CoDeSEG is compiled from its pinned source files with the system C++
    compiler. It has no licence file, so it must never be vendored.
  - Other runtimes come from `codeseg-setup` (FOX: CMake plus OpenMP g++;
    BigCLAM/NEO-K-Means: make; NcGame: git; ANGEL/Ego-Splitting: the
    isolated CDlib environment).
  - An unprepared runtime is an explicit `unavailable` row with its reason.
- **`hedonic guide` / `hedonic update`.** `guide.TOPICS` is user-facing documentation: change it in the same
  commit as any flag or verb it mentions (a test checks the `hedonic run` verbs it names). `update` makes a
  single GET to pypi.org, never installs without `--yes`, and never touches an editable/development install.
- **Ground-truth robustness spectrum.** `hedonic run spectrum` (also the wizard's
  fourth start-menu choice) launches `overlapping-gt-spectrum` through the same
  durable machinery: `runmanager.launch_spectrum` registers an entry with
  `kind: "spectrum"` and the study's argv (always with `--resume`); the worker
  mirrors the study's progress events into `progress.json`, `snapshot_spectrum`
  renders per-pair denominators, and `stop`/`resume`/`attach`/`--json` behave as
  for benchmark runs. `hedonic show covers` prints the discovery report. The
  study never edits the v3 ledger; its own lock is
  `configs/overlapping-gt-spectrum-protocol.lock.json` (re-freeze with
  `--write-lock` only after review; any change to a tracked file invalidates
  stored records and refuses locked runs).
- **Tests.** `TestCLI` covers the front door, parsing/catalogue, the result
  table, and the `run` verbs. Keep them passing when changing flags.

```bash
hedonic run exp                      # wizard; first choice = default run
hedonic run exp --yes --json         # default run, no prompts, JSON summary
hedonic run paper --dry-run          # the paper's 200-run plan (seed -> method -> network) and ETA
hedonic run paper                    # = hedonic run exp --config paper (durable, resumable)
hedonic run exp --networks all --methods all --seeds 0-9 --full --detach --name full-bench
hedonic run attach full-bench        # live view; Ctrl-C closes only the view
hedonic run stop full-bench && hedonic run resume full-bench
```

### Paths (`experiments/config.py`)

| Variable | Env override | Default (expanded from `~/…`) |
|----------|--------------|--------------------------------|
| `DBLP_DIR` | `HEDONIC_DBLP_DIR` | `~/Databases/Hedonic/Networks/DBLP` |
| `NETWORKS_DIR` | `HEDONIC_NETWORKS_DIR` | `~/Databases/Hedonic/Networks` |
| `SYNTHETIC_DIR` | `HEDONIC_SYNTHETIC_DIR` | `~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020` |
| `OUTPUT_DIR` | `HEDONIC_OUTPUT_DIR` | `artifacts` |

**TOML configs live under [`configs/`](configs/)** (default load: `configs/hedonic.toml` from cwd). Example template: [`configs/hedonic.example.toml`](configs/hedonic.example.toml). Override with `--config path.toml` or `HEDONIC_CONFIG`. Priority: **CLI flag > env > TOML > defaults**. Use `~/…` paths in TOML (expanded at load); do not hard-code `/Users/<name>`.

```toml
[paths]
dblp_dir = "~/Databases/Hedonic/Networks/DBLP"
networks_dir = "~/Databases/Hedonic/Networks"
synthetic_dir = "~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020"
output_dir = "artifacts"

[overlapping_resolution]
output_dir = "artifacts/overlapping/resolution_f1"
resolutions = "0:1:11"
seeds = "0-4"

[overlapping_paper]
# `hedonic-exp reproduce-overlapping-paper` reads the complete registered
# protocol here: methods, seeds, timeouts, Omega, worker RAM budget, tmux
# session, paper/output paths, and per-dataset cover shards.
methods = ["hedonic_multiphase", "hedonic_multiphase_x10", "hedonic_multiphase_x100", "cpm", "demon"]
```

Always import paths from config (never hardcode absolute home paths in new modules):

```python
from hedonic.experiments.config import DBLP_DIR, SYNTHETIC_DIR, OUTPUT_DIR
```

### CLI (`experiments/CLI.py`) — **keep this in sync**

Entries: `hedonic-exp` → `hedonic.experiments.CLI:main` (every registered
subcommand) and `hedonic` → `hedonic.experiments.CLI:hedonic_main` (the
user-facing front door: `hedonic run exp` maps onto the `quickstart`
subcommand; `hedonic run list|status|attach|logs|stop|resume` onto
`overlapping.runmanager.dispatch`). Both live in `CLI.py`.

Subcommands are registered in **`COMMANDS`** (the single source of truth in `CLI.py`):

| Subcommand | Module | Purpose | Data |
|------------|--------|---------|------|
| `smoke` | (CLI built-in) | Isolated check: small graphs + tiny SBM sweep | none |
| `disjoint` | `disjoint.sbm_sweep` | SBM sweeps → JSON under `SYNTHETIC_DIR` (presets: `v1020`, `v1020-smoke`) | synthetic (or `--output_root`) |
| `disjoint-load` | `disjoint.data_loader` | JSON results → gzipped CSV (`--simple` for smoke) | synthetic results |
| `plots` | `plots.paper_figures` | PHYSA V1020 paper figures from CSV (`gt_robustness`, `noise`, …) | synthetic CSV |
| `reproduce-disjoint` | `disjoint.reproduce` | End-to-end: sweep → CSV → figures | synthetic |
| `overlapping-small` | `overlapping.small_graphs` | Smoke + metrics on small graphs | none |
| `overlapping-dnn` | `overlapping.dnn_certificate` | Locked tiny graphs: exact valid-cover optimum + DNN SDP outer certificate | none |
| `overlapping-dnn-rational` | `overlapping.dnn_rational_certificate` | Versioned exact rational diagonally-dominant DNN dual certificates plus optional numerical calibration; `--require-debug-trace` audits projection label counts and the tie budget on 1.0.0.5+ | none |
| `overlapping-oracle` | `overlapping.unit_l2_oracle` | Independent unit-ℓ₂ prefix oracle, CE1–CE7, CE5 token normalization, collision lists, restricted-balance token identity, slack-variable coefficient bound; `--proof-check` reconstructs numbered propositions; `--fee-check` verifies the separate `unit-l2-cpm-fee-v1` objective; optional tiny native differential | none |
| `overlapping-integrity` | `overlapping.integrity_grid` | Checkpointed exhaustive tiny-graph native/oracle integrity grid (`smoke` or resumable `standard`) | none |
| `overlapping-certificate-reconcile` | `overlapping.certificate_reconcile` | Detector-free 3840/864/125/tiny ledger classification (`status_coverage.csv`, `certificate_manifest.json`, displayed TeX tables/macros including `analysis_units_table.tex`); optional `--replay-dnn` / `--replay-gt` / `--replay-paper`; `--gate-h-check` verifies frozen locks, independent DNN dual witnesses, short proofs, wrapper/metric contracts, and TeX/PDF snapshot hashes; `--no-tex-fragments` / `--tex-fragment-dir` control generated tables | saved artifacts (SNAP graphs only if `--replay-gt`; paper replay uses local shards) |
| `overlapping-controlled` | `overlapping.controlled_overlap` | LFR-derived controlled overlap with cap, initialization, phase, and resolution ablations (not canonical overlapping LFR) | none |
| `overlapping-lfr` | `overlapping.overlap_lfr` | Planted-overlap study: compatibility fixture for smoke; fail-closed official LFR binary for pilot/standard (28-condition prospective grid) | none |
| `overlapping-tracking` | `overlapping.tracking` | Mirror/no-change, one-sweep, local/full noisy-cover tracking and graph-level robustness–recovery triangle | none |
| `overlapping-baselines` | `overlapping.baselines` | Resumable graph-ledger-bound SLPA/DEMON/KCP adapters, scoped Chen-style replica, and trivial/disjoint controls; external implementations required for standard | TKT-11 graph ledger (standard) |
| `overlapping-resource-envelope` | `overlapping.resource_envelope` | Original/token/audit incidence, token, projection, RSS, stage-timing, and capacity-envelope measurements | none |
| `overlapping-subgraph` | `overlapping.dblp_subgraph` | L-hop around GT communities | DBLP |
| `overlapping-full` | `overlapping.dblp_full` | Full DBLP + optional resolution sweep | DBLP |
| `overlapping-scale` | `overlapping.complexity_scale` | Wallclock scaling as subnetworks grow (local-moving T/F, timeout stop + plot) | DBLP (or `--smoke`) |
| `overlapping-resolution` | `overlapping.resolution_f1` | Full-DBLP overlap metrics vs resolution [0,1] (cached-cover rescoring, singleton modes, optional sampled Omega) | DBLP (or `--smoke`) |
| `overlapping-benchmark` | `overlapping.benchmark` | Resumable Amazon/DBLP/LiveJournal/YouTube/Wikipedia overlapping-cover benchmark; `--methods` accepts the shared literature registry (25 names) plus historical AGMfit/MMSB adapters; `--profile agmfit-replication` records 500 AGMfit-style overlap-centered induced windows per dataset | saved SNAP networks (or `--profile smoke`) |
| `overlapping-codeseg` | `overlapping.codeseg_reproduction` | DBLP SNAP overlap protocol: 25 default methods (CoDeSEG catalogue, Hedonic variants, ANGEL, Infomap, DEMON, CPM, Link Communities, NEO-K-Means, NISE, SSE, QOCE, SVI, ESSC) scored with the shared cover-metric vector plus LFK-ONMI; external adapters fail closed | saved SNAP networks (or `--smoke`) |
| `overlapping-reproduce` | `overlapping.turnkey` | One-command bootstrap plus complete registered SNAP method benchmark; smoke uses a deterministic ~1,000-node AGMfit-like fixture, while full defaults to DBLP (pass `--datasets all` for all seven) and writes accuracy, throughput, runtime, CPU, and peak-RSS ledgers | synthetic smoke or saved/downloaded SNAP networks |
| `info` | `overlapping.info` | `hedonic show methods|networks|metrics`: method readiness (ready / installable / missing tools, from the setup manifest and `shutil.which`), SNAP network sizes, on-disk copies and optional `--remote` HEAD download sizes, metric definitions; `--json` | none (reads local state) |
| `quickstart` | `overlapping.quickstart` (+ `runmanager`, `tui`, `extra_methods`) | Benchmark front door behind `hedonic run exp`: any of the 7 SNAP networks × the 10 screened methods × seeds × subgraph size, `screened`/`registry` presets, named configs (`--config paper`), wizard, durable tmux runs with live viewer/ETA, list/status/attach/logs/stop/resume; lazily downloads SNAP data, compiles pinned CoDeSEG, and provisions other runtimes via `codeseg-setup` (cache `~/.cache/hedonic`) | SNAP networks (auto-downloaded) |
| `codeseg-setup` | `overlapping.codeseg_setup` | DBLP-only default bootstrap for SNAP data and all registered runtimes; prepares native, isolated, Octave, SVI, and ESSC implementations and writes a resumable manifest | SNAP catalogue/cache (or `--dry-run --offline`) |
| `codeseg-doctor` | `overlapping.codeseg_setup` | Read-only readiness audit for the DBLP dataset and registered runtimes; `--require-all` fails closed | local archives/cache |
| `overlapping-gt-robustness` | `overlapping.ground_truth_robustness` | Supplied-cover [0,1] robustness plus local/multi-phase equilibria seeded from the complete overlapping GT cover | saved SNAP networks (or `--smoke`) |
| `overlapping-gt-spectrum` | `overlapping.gt_spectrum` | SNAP ground-truth robustness spectrum (separately versioned, v3 untouched): discovers present/missing/unsupported graph/cover pairs in the local cache (no download without `--provision`), audits each supplied cover over a resolution grid (stable-vertex fraction, regret, Nash status; fixed and open labels), runs seeded local moving from the exact GT cover, audits and scores every return against its start; `--discover`, `--dry-run [--inspect]`, `--audit-only`, `--resume`, `--rescore-only`, `--replot`, `--write-lock`; see `docs/overlapping_gt_spectrum.md` | saved SNAP networks (or `--profile smoke`) |
| `overlapping-audit` | `overlapping.protocol` | Read-only locked-protocol audit of all 125 overlapping-paper shard records | saved artifacts only |
| `reproduce-overlapping-paper` | `overlapping.reproduce_paper` | TOML-driven paper protocol: RAM-bounded tmux shards → merged records/plots/tables → guarded `main.tex` compilation | saved SNAP networks |
| `list` | meta | List subcommands | — |

```bash
# One-command front door (console script `hedonic`, same registry).
# `hedonic run exp` without flags opens an arrow-key wizard; flags or --yes run
# directly. Runs execute detached in tmux (fallback: detached process) and
# survive Ctrl-C / closed terminals; stop is explicit. Registry of runs:
# ~/.cache/hedonic/runs (override HEDONIC_RUNS_DIR).
hedonic run exp --yes                # == hedonic-exp quickstart --yes
hedonic run exp --networks dblp,amazon --methods all --seeds 0-2 --nodes 20000
hedonic run exp --foreground --methods hoc_local,codeseg --nodes 1000
hedonic run list | status | attach | logs | stop NAME | resume NAME
hedonic show methods | networks [--remote] | metrics [--json]

# Isolated / CI-friendly (no databases)
hedonic-exp smoke
hedonic-exp overlapping-small
hedonic-exp overlapping-dnn --output artifacts/overlapping/dnn_certificate/dnn_certificate.json
hedonic-exp overlapping-dnn --list-instances
hedonic-exp overlapping-oracle --list-examples
hedonic-exp overlapping-oracle --proof-check
hedonic-exp overlapping-oracle --fee-check
hedonic-exp overlapping-oracle --random-cases 300 --token-cases 100
hedonic-exp overlapping-oracle --native-differential
hedonic-exp overlapping-dnn-rational \
  --instances collision_path3,duplicate_cap4 --exact-only
hedonic-exp overlapping-dnn-rational --exact-only --require-debug-trace \
  --output artifacts/overlapping/dnn_rational_certificate/candidate-1.0.0.5.json
hedonic-exp overlapping-integrity --profile smoke
hedonic-exp overlapping-integrity --profile standard --resume \
  --require-igraph-version 1.0.0.4 \
  --output-dir artifacts/overlapping/integrity_grid/v1
hedonic-exp overlapping-certificate-reconcile \
  --output-dir artifacts/evidence/overlapping_communities
hedonic-exp overlapping-certificate-reconcile \
  --output-dir artifacts/evidence/overlapping_communities \
  --replay-dnn --replay-gt --replay-paper
hedonic-exp overlapping-certificate-reconcile \
  --output-dir artifacts/evidence/overlapping_communities \
  --gate-h-check
hedonic-exp overlapping-controlled --smoke \
  --output artifacts/overlapping/controlled_overlap/controlled_overlap.json
hedonic-exp overlapping-lfr --profile smoke \
  --output-dir artifacts/overlapping/overlap_lfr/smoke
hedonic-exp overlapping-lfr --profile standard --preflight \
  --official-format lfrbenchmarks \
  --official-generator /path/to/LFRbenchmarks/unweighted_undirected/benchmark \
  --official-config configs/astra-lfrbenchmarks.example.json \
  --output-dir artifacts/overlapping/overlap_lfr/standard-preflight
hedonic-exp overlapping-lfr --profile pilot --dry-run \
  --output-dir artifacts/overlapping/overlap_lfr/pilot
hedonic-exp overlapping-lfr --profile standard --dry-run \
  --output-dir artifacts/overlapping/overlap_lfr/standard
hedonic-exp overlapping-tracking --profile smoke \
  --output-dir artifacts/overlapping/tracking_triangle/smoke
hedonic-exp overlapping-baselines --profile smoke \
  --output artifacts/overlapping/baselines/smoke.json
hedonic-exp overlapping-baselines --preflight \
  --preflight-output artifacts/evidence/overlapping_communities/tkt13_slpa_preflight.json
hedonic-exp overlapping-resource-envelope --profile smoke \
  --output-dir artifacts/overlapping/resource_envelope/smoke
hedonic-exp overlapping-gt-robustness --smoke \
  --output-dir artifacts/overlapping/ground_truth_robustness_v3/smoke
hedonic-exp disjoint --smoke --output_root artifacts/disjoint/smoke

# Reproduce disjoint SBM sweep (subset of methods)
hedonic-exp disjoint --folder_name smoke --max_n_nodes 40 --n_communities 2 \
  --seeds 1 --p_in 0.2 --difficulty 0.3 --noises 0.01 --partition_seeds 0 \
  --methods Hedonic,Leiden --output_root artifacts/disjoint/sbm

# PHYSA V1020 structural smoke (same layout/methods as archived grid; safe root)
hedonic-exp disjoint --preset v1020-smoke \
  --output_root artifacts/disjoint/v1020

# Full V1020 grid via CLI (very large — never write into archived V1020)
hedonic-exp disjoint --preset v1020 \
  --output_root artifacts/disjoint/v1020

# Combine prior disjoint JSON dumps
hedonic-exp disjoint-load --results_folder /path/to/resultados --output artifacts/disjoint/resultados.csv.gzip --simple

# Paper figures (same five stems as archived V1020/figures/)
hedonic-exp plots --smoke --output_dir artifacts/disjoint/figures_smoke
hedonic-exp plots --data /path/to/resultados_ari.csv.gzip \
  --output_dir artifacts/disjoint/v1020/figures

# Complete pipeline: sweep → CSV → figures (never write into archived V1020)
hedonic-exp reproduce-disjoint --preset v1020-smoke \
  --output_root artifacts/disjoint/v1020
# Figures only from archived CSV (read-only) into the repository artifact tree:
hedonic-exp reproduce-disjoint --plots-only \
  --data ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/resultados_ari.csv.gzip \
  --output_root artifacts/disjoint/v1020 --max_rows 50000

# Overlapping (point HEDONIC_DBLP_DIR if needed)
# Defaults: n_iterations=-1 (native no-change stop); independently audit
# returned memberships when a mathematical certificate is required.
hedonic-exp overlapping-subgraph --levels 1 --n_communities 5 \
  --methods leiden,hedonic_v1 --output artifacts/overlapping/dblp_subgraph/smoke.json
hedonic-exp overlapping-full --resolution 1e-4 --seed 0 --output artifacts/overlapping/dblp_full/results.json

# Complexity scale: network size vs wallclock to equilibrium
# Two lines: local_move_only True vs False; density γ; K = #GT in window
# Growth stops per line when --timeout is hit (full graph not required)
hedonic-exp overlapping-scale --smoke --output_dir artifacts/overlapping/complexity_scale/smoke
hedonic-exp overlapping-scale --timeout 30 --max-levels 6 \
  --output_dir artifacts/overlapping/complexity_scale/dblp
# Full multi-phase only (local_move_only=False), 10 min budget per size
hedonic-exp overlapping-scale --variant full --timeout 600 --max-levels 6 \
  --community_idx 1004 --output_dir artifacts/overlapping/complexity_scale/full

# Full-DBLP F1 vs resolution γ ∈ [0,1]: multi-phase, allow_isolation,
# max_memberships = #GT with size>1; multi-seed F1 CI band on the plot.
# Per-(γ,seed) covers + metadata cached under <output_dir>/runs/ (resume-safe).
# Paths from configs/hedonic.toml by default (~/… expanded).
# Re-score all metrics from covers without re-detection: --rescore-only
hedonic-exp overlapping-resolution --smoke --output_dir artifacts/overlapping/resolution_f1/smoke
hedonic-exp overlapping-resolution --config configs/hedonic.toml \
  --resolutions 0:1:11 --seeds 0-4
# Resume after interrupt (same --output_dir; skips completed runs/ files):
hedonic-exp overlapping-resolution --resolutions 0:1:11 --seeds 0-4 \
  --output_dir artifacts/overlapping/resolution_f1
# Recompute matching, node-membership, weighted F1, and diagnostics from
# cached covers only; report both consistent singleton modes:
hedonic-exp overlapping-resolution --rescore-only --singleton-mode both \
  --resolutions 0:1:11 --seeds 0-4 \
  --output_dir artifacts/overlapping/resolution_f1
# Optional sampled Omega (never allocates a dense vertex-pair matrix):
hedonic-exp overlapping-resolution --rescore-only --omega \
  --omega-sample-size 100000 --singleton-mode size_ge_2 \
  --output_dir artifacts/overlapping/resolution_f1

# Reproducible five-network SNAP overlap benchmark. The smoke profile uses
# built-in tiny covers; standard bounds each graph to a deterministic
# GT-informed induced subgraph. Wikipedia has only an `all` category cover,
# so `--cover top5000` records it explicitly as skipped.
hedonic-exp overlapping-benchmark --profile smoke \
  --output_dir artifacts/overlapping/snap_benchmark/smoke
hedonic-exp overlapping-benchmark --datasets amazon,dblp,livejournal,youtube,wikipedia \
  --cover top5000 --profile standard \
  --output_dir artifacts/overlapping/snap_benchmark
hedonic-exp overlapping-benchmark --list-networks
hedonic-exp overlapping-benchmark --list-methods

# The legacy benchmark and Python adapter share the literature names.
# Aliases such as "Neo-K-Means" and "Link Communities" are normalized.
hedonic-exp overlapping-benchmark --datasets dblp \
  --methods codeseg,neo-k-means,link-communities \
  --profile full --max_nodes 5000

# DBLP SNAP overlap reproduction (25 default methods) plus the local Hedonic
# comparators. Use --max-nodes for bounded qualification; omit it for the
# full com-DBLP detector run.
hedonic-exp overlapping-codeseg --datasets dblp --max-nodes 5000 \
  --require-all --resume \
  --output-dir artifacts/overlapping/codeseg/dblp-local
hedonic-exp overlapping-codeseg --datasets dblp --preflight --require-all
# Per-method parameter overrides (merged over the defaults; recorded in the
# run config).  BigCLAM/SVI/FOX/CoDeSEG accept "threads"; NISE/SSE accept
# "seeding" (sphub | hrc_graclus), "ego", "expansion", "nworkers";
# hedonic_local accepts "max_memberships", "resolution_multiplier" (x density) and
# "absolute_resolution" (a fixed CPM gamma).
hedonic-exp overlapping-codeseg --datasets dblp --methods fox \
  --method-params '{"fox": {"threads": 10, "wcc_threshold": 0.05}}' \
  --output-dir artifacts/overlapping/codeseg/fox-threads
# The hedonic_local/hedonic_multiphase* rows use the pre-registered,
# reference-free cap M_0=8 unless --hedonic-max-memberships is given. The value
# "reference-multiplicity" (largest per-vertex reference multiplicity) is a
# reference-informed ablation; every record carries max_memberships_source.
hedonic-exp overlapping-codeseg --datasets dblp --methods hedonic_local \
  --hedonic-max-memberships 64 --output-dir artifacts/overlapping/codeseg/dblp-m64

# Turnkey benchmark: all registered methods on an archive-free ~1,000-node
# AGMfit-like fixture. Missing external runtimes remain explicit unavailable
# rows; no substitute detector is used.
hedonic-exp overlapping-reproduce --profile smoke --resume

# Full run: bootstrap/cache/build what is available, then benchmark full DBLP.
hedonic-exp overlapping-reproduce --profile full --resume
# Preflight only (loads/checks data and dependencies; no detectors).
hedonic-exp overlapping-reproduce --profile full --preflight
# Explicitly request all seven SNAP networks when that larger run is wanted.
hedonic-exp overlapping-reproduce --profile full --datasets all --resume

# Bootstrap the optional SNAP catalogue, isolated CDlib worker, upstream
# checkouts, and a machine-readable runtime manifest. Native CMake builds use
# Release flags; a normal C/C++ toolchain and CMake are still required.
hedonic-exp codeseg-setup --all --build-native
hedonic-exp codeseg-doctor --require-all

# Supplied overlapping-cover robustness and GT-seeded equilibria.  This is a
# separate experiment and does not alter the locked paper benchmark.
hedonic-exp overlapping-gt-robustness \
  --config configs/overlapping-ground-truth.toml \
  --datasets amazon,dblp --seeds 0-4

# SNAP ground-truth robustness spectrum (docs/overlapping_gt_spectrum.md).
# Cache discovery and a no-run plan first; nothing is downloaded without --provision.
hedonic-exp overlapping-gt-spectrum --discover
hedonic-exp overlapping-gt-spectrum --dry-run --inspect
hedonic-exp overlapping-gt-spectrum --profile smoke \
  --output-dir artifacts/overlapping/gt_spectrum_smoke
hedonic run spectrum --name gt-spectrum-registered \
  --output-dir artifacts/overlapping/gt_spectrum_v1        # durable, resumable, every available pair
hedonic-exp overlapping-gt-spectrum --replot               # rebuild figures/summaries from the ledger

# Reconcile existing evidence without loading graphs or rerunning detectors.
# JSON output also produces a sibling CSV with one row per condition.
hedonic-exp overlapping-audit \
  --output artifacts/evidence/overlapping_communities/protocol_audit.json
hedonic-exp overlapping-certificate-reconcile \
  --output-dir artifacts/evidence/overlapping_communities
# Optional read-only membership replay (does not rewrite frozen 1.0.0.2 ledgers):
hedonic-exp overlapping-certificate-reconcile \
  --output-dir artifacts/evidence/overlapping_communities \
  --replay-dnn --replay-gt --replay-paper
# Independent dual-witness / freeze / ledger check (does not rewrite locks):
hedonic-exp overlapping-certificate-reconcile \
  --output-dir artifacts/evidence/overlapping_communities \
  --gate-h-check

# Complete overlapping SNAP paper reproduction. All run parameters are in
# configs/hedonic.toml [overlapping_paper]; raw SNAP archives stay read-only.
uv run hedonic-exp reproduce-overlapping-paper --dry-run
uv run hedonic-exp reproduce-overlapping-paper
tmux attach -t hedonic-overlapping-paper

# Corrected bounded equilibrium comparison used by the current manuscript
uv run hedonic-exp reproduce-overlapping-paper \
  --config configs/hedonic-equilibrium-v2.toml --no-tmux
```

The integrity smoke profile is data-free. The standard profile enumerates all
nonempty labelled simple graphs through five vertices, all wrapper-valid
two-label cap-two starts, endpoint resolutions, negative local/multilevel
policies, positive multilevel budgets, and exhaustive cap-one controls. It
writes atomic graph-batch shards, `progress.json`, and a reloaded
`integrity_manifest.json`; the standard profile also writes
`projection_trace.jsonl` from an opt-in native accepted-move/projection
fixture. Use `--resume` after interruption. Positive-budget and diagnostic
checks default to requiring lucas-igraph 1.0.0.4 or later
(`--require-igraph-version X` is an exact pin). Projection checkpoints must
follow the post-local guard of lucas-igraph 1.0.0.5: a token proposal is kept
if it beats the iteration's local-moving cover by the native margin, or ties
it within the margin while occupying fewer labels (merged duplicate bodies);
otherwise the local-moving cover is restored.

The 1.0.0.5 projection trace appends `labels_local` and `labels_proposed`
(21 raw columns, schema 2); the first 19 columns are unchanged. Counts refer
to the post-local cover and the proposal before any rollback. The Python
decoder still accepts 19-column native traces as schema 1 without inventing
counts. Integrity and DNN candidate checks require counts on a 1.0.0.5+
binding, verify that accepted ties reduce occupied labels, and reconstruct
the per-call budget of at most `n` accepted ties. Legacy traces support only
the quality checks and report that label counts were not verified. Frozen
artifacts remain unchanged; write new candidates to new paths.

Integrity preflight and resume fingerprint the resolved native provider
loaded by the process (dynamic `libigraph` or the extension for an embedded
build), using `dladdr`. An unresolved provider fails closed. The provider
signature is checked again before the final manifest is written so a library
replacement cannot silently reuse or certify another build's shards.

`overlapping-dnn-rational` uses a new protocol namespace so the byte-pinned
historical `overlapping-dnn` producer remains unchanged. It reuses the three
historical instance identities, adds collision and cap-equals-bank boundaries,
and gives every instance an exact rational diagonally-dominant DNN dual
certificate. SCS remains a separately qualified numerical calibration unless
`--exact-only` is used.

Gate H verifies the frozen numerical DNN witnesses and the versioned rational
companion independently; use `--rational-dnn` to select a non-default artifact.

Args after the subcommand are forwarded to that module’s `main(argv)`.

#### Agent rule: **always update the CLI when experiments change**

Any agent (or human) that changes experiment surface area **must** update the CLI in the **same** change. Incomplete PRs that leave `hedonic-exp` stale are not acceptable.

Update the CLI when you:

| Change | Required CLI / docs updates |
|--------|----------------------------|
| **Add** a runnable experiment module with `main()` | Register it in `CLI.COMMANDS`; extend `SUBCOMMANDS` help / `list`; document in this AGENTS.md table |
| **Remove** or rename an experiment | Drop / rename the registry entry; fix help examples; update this table |
| **Change argparse** flags, defaults, or semantics on an existing module | Ensure `hedonic-exp <cmd> --help` still works; update examples here and in `CLI.py` epilog if flags users rely on changed |
| **Add a preset** (e.g. smoke / isolated) | Prefer a flag on the module (`--smoke`) **and** wire it through the CLI (dedicated subcommand and/or documented example) |
| **Move** `main` / module path | Fix `Command.module` / `Command.attr` imports |
| **Change** `pyproject.toml` entry point | Keep `[project.scripts] hedonic-exp = "hedonic.experiments.CLI:main"` and `hedonic = "hedonic.experiments.CLI:hedonic_main"` accurate |

Checklist (copy into PR notes when touching experiments):

1. [ ] Module exposes `main(argv=None)` (import-safe, no work at import time).
2. [ ] Registered in `src/hedonic/experiments/CLI.py` → `COMMANDS`.
3. [ ] `hedonic-exp <name> --help` works (or custom help for no-argparse cmds).
4. [ ] This AGENTS.md CLI table + examples updated if the public surface changed.
5. [ ] Test added/updated under `tests/` (`TestCLI` or module-specific).
6. [ ] Prefer calling via CLI in docs/scripts over ad-hoc `python path/to/script.py`.

Do **not** add a second entrypoint module or a parallel `cli.py`; extend `CLI.py` only.

### Disjoint pattern (`disjoint/sbm_sweep.py`)

1. Build an SBM with `generate_graph` → `Game`.
2. Ground truth via block labels; optional noisy `initial_membership`.
3. Methods table maps **name → `method_call_name` + parameters**.
4. **Hedonic** and **Leiden** both call `community_hedonic` with `max_memberships=1` (Leiden uses `local_move_only=False`). Spectral sets `clusters = n_communities`.
5. Output layouts:
   - **`v1020`** (preset default): `resultados/{n}C_{size}N/Noise = …/P_in = …/Difficulty = …/Network (NNN)/partition_MMM.json` — list of method result dicts (matches archived PHYSA V1020).
   - **`legacy`**: `{folder}/{n} Communities of {size} nodes/.../Partition (MMM)/{Method}.json`.
6. Leiden/Hedonic use **`--n_runs`** stochastic restarts with unique-partition dedup (V1020 used 10).
   The sweep accepts `--resume` to skip schema-valid `v1020` cells and writes
   result JSON atomically, so interrupted cells are safe to rerun.
7. **Presets:**
   - `--preset v1020` — full archived grid (1020 nodes, communities 2–6, full p_in/difficulty/noise, 5 nets, 10 partitions, all methods). Refuses to write into the archived `.../V1020` folder.
   - `--preset v1020-smoke` — tiny structural clone for CLI checks.
   - `--smoke` — smaller legacy mini-run (CI-friendly).
8. **Isolated run:** `--smoke` / `--preset v1020-smoke` and `--output_root` (prefer `artifacts/disjoint/v1020`, never overwrite archived V1020).
9. **Figures:** `hedonic-exp plots` (or `reproduce-disjoint`) writes the five paper stems: `gt_robustness`, `noise`, `n_communities`, `acc_robustness`, `acc_efficiency`.

Use `hedonic-exp disjoint --preflight` for a no-write check of parameter
divisibility, probability bounds, generated graph sizes, and pending output
cells. It never invokes detectors. Archive protection rejects the historical
V1020 directory and all descendants, including when `folder_name` would place
output there indirectly.

When adding a method to the sweep, extend `METHODS` and implement the callable on `Game` (or a local helper in the experiment module if it is not core). If the public CLI documents method names, refresh help/examples.

### Complete V1020 disjoint reproduction (via CLI)

| Step | Command | Output |
|------|---------|--------|
| 1. Sweep | `hedonic-exp disjoint --preset v1020 --output_root artifacts/disjoint/v1020` | `resultados/{n}C_{size}N/…/partition_*.json` |
| 2. Combine | `hedonic-exp disjoint-load --results_folder …/resultados --simple` | `resultados.csv.gzip` |
| 3. Figures | `hedonic-exp plots --data …/resultados.csv.gzip --output_dir …/figures` | five PDFs matching archived names |

Or one shot: `hedonic-exp reproduce-disjoint --preset v1020 --output_root artifacts/disjoint/v1020`.

For an interrupted run, add `--resume`; for a no-write readiness check, add
`--preflight`. The audited five-seed archive's 50 missing raw cells can be
targeted without launching the full grid:

```bash
hedonic-exp disjoint --preset v1020 \
  --max_n_nodes 1020 --n_communities 5 --seeds 3 \
  --p_in 0.01 --difficulty 0.30 \
  --noises 0.10 0.25 0.50 0.75 1.00 --partition_seeds 10 \
  --output_root artifacts/disjoint/v1020-recovery --preflight
```

**Do not** write into archived `…/Synthetic_Networks/V1020` (CLI guards refuse).

### Overlapping pattern

- Detection: **`Game.community_hedonic(..., max_memberships=K)`**.
- **Always** pass **`n_iterations=-1`** (or any negative value) to request the
  native no-change stopping condition and final hedonic Nash certificate sweep.
  Positive budgets may stop early; CLI defaults are `-1`. Use an independent
  regret audit when a mathematical certificate is required.
- **SNAP benchmark `max_memberships`** defaults to the maximum number of
  ground-truth communities containing any one node (at least 2), not `len(gt)`.
  This keeps overlap capacity semantically meaningful and bounds the native
  `n × max_memberships` workspace. Override only when intentionally capping it.
- Metrics vs covers: **`experiments.overlapping.metrics`** (`evaluate_cover`, `partition_to_cover_lists`, `cover_quality`, baselines, optional Nash check).
- DBLP load: `dblp_full.load_dblp` (cache `dblp.pkl`, or `pkl/`, or `raw/*.gz`).
- Subgraph methods:
  - `leiden` — `community_hedonic(max_memberships=1, local_move_only=False, n_iterations=-1, allow_isolation=True)`
  - `hedonic_v1` — overlapping local-moving (`local_move_only=True, n_iterations=-1`), warm-started from Leiden
  - `hedonic_v2` — overlapping full multi-phase (`local_move_only=False, n_iterations=-1`)
  - `singleton` / `grand_coalition` / `total_overlap` — deterministic control baselines

### LFR-derived controlled overlap

`hedonic-exp overlapping-controlled` starts from NetworkX's **disjoint** LFR
graph and primary partition, then deterministically assigns a requested
fraction of vertices to secondary communities and adds seeded reinforcing
edges into those communities. It is deliberately named **LFR-derived
controlled overlap**, never canonical overlapping LFR. The JSON/CSV output
records requested and realized overlap, memberships per overlapping vertex,
base LFR mixing `mu`, secondary-edge probability, realized edge mixing, and
the effective retry seed used by the LFR generator.

The detector grid compares local moving and multi-phase execution,
`max_memberships` caps (`1`, numeric, or GT-informed `gt`), singleton,
data-derived neutral-disjoint, and GT-primary initializations, and density
resolution multipliers. Every hedonic call uses
`Game.community_hedonic(..., n_iterations=-1, allow_isolation=True)`, and all
cells for one generated graph share its graph seed as the detector seed so
phase comparisons are paired. Keep GT-informed cap and
initialization results visibly labeled as supervised ablations.
Each cell runs in a forked child with a five-second hard wall-clock timeout
while retaining `n_iterations=-1`; summaries preserve expected/completed/
timeout/failed coverage and use completed observations only.

### Astra E1 experimental program (TKT-11–14)

`hedonic-exp overlapping-lfr` implements the separate metadata-free planted
overlap benchmark. Its `standard` profile is prospective (28 structural cells,
30 independent graph seeds per cell); `pilot` uses five graphs per cell;
`smoke` is a bounded compatibility fixture. Pilot and standard fail closed
unless the pinned `unweighted_undirected` LFRbenchmarks executable is selected
with `--official-format lfrbenchmarks`; the generic JSON adapter is smoke-only
and cannot produce canonical evidence. The self-contained
`overlapping_lfr_compatibility_smoke_v1` generator records tau values but does
not implement the official degree sampler and is never canonical evidence.
The runner records requested/achieved mixing, overlap, multiplicity, degree,
edge-capacity, retry, graph/cover hashes, detector cap, worker RSS and every
status. Use `--resume` to skip stable graph/method/seed keys. The planted cover
is held out from detection. This experiment is explicitly distinct from the
post-hoc `LFR-derived controlled overlap` diagnostic above.

`hedonic-exp overlapping-tracking` runs the supervised tracking arm on shared
graph/perturbation units: Mirror/no-change, synchronous one-sweep, sequential
full one-sweep, local-to-convergence, and multiphase cleanup. Incidence
double-edge switches preserve vertex multiplicities and community sizes where
feasible; requested and achieved distances plus failed/shortfall attempts are
recorded. `robustness_recovery_analysis.json` reports raw and
condition-stratified graph-level associations and keeps negative associations.
Pilot and standard require the completed official TKT-11 `graphs.json` ledger
and verify every graph/cover hash before detector work. The registered
standard arm selects eight predeclared structural cells × 30 graphs (240
graphs), with an atomic compressed result shard and `progress.json` checkpoint
per graph. Use `--preflight` to write a machine-readable receipt before
launch, and `--resume` after interruption; compact `results.jsonl` is an
idempotent index while full covers remain in `result_shards/`.

`hedonic-exp overlapping-baselines` provides graph-only adapters for SLPA,
DEMON, KCP, a native cap-one CPM/Leiden control (`cpm`) plus a local-moving
disjoint control, a scoped Chen-style set-valued replica with declared self/empty-
label conventions, and singleton/grand-coalition/component controls. CDlib
SLPA and `demon.Demon` are required for standard; local
`*_compatibility_replica_v1` implementations are smoke-only and explicitly
labelled. Chen explores only current, singleton, and current-plus-one-label
actions, so it is not a complete best-response reproduction. Standard
requires the completed official TKT-11 `graphs.json` ledger and validates
every referenced graph/archive and planted-cover hash before detector work;
it never falls back to the compatibility generator. The registered standard
plan is three optimizer seeds for stochastic methods and one execution for
deterministic controls, with parameters frozen in
[`configs/astra-baselines.example.json`](configs/astra-baselines.example.json)
before scoring. Rows and graph completions are fsynced to append-only
`*.rows.jsonl`/`*.graphs.jsonl` sidecars and are idempotently reloaded on
`--resume`; malformed ledgers/checkpoints fail closed. Parameters, dependency
identities, coverage handling, stochastic seeds, information budgets, timeout
and peak-RSS limits are serialized; method failures are retained and no
hidden oracle cap/start or singleton completion is used.

`hedonic-exp overlapping-resource-envelope` measures original `n`, `m`,
incidences `T`, predicted and native token quantities, projection/accepted-move
events, per-level work when the binding exposes it, detector/cleanup/audit/scoring/end-to-end timings, subprocess peak RSS,
and typed integer-capacity/timeout/memory outcomes. Sparse scaling and dense
overlap stress are separate axes. The standard profile is prospective and may
be bounded with `--max-cases`; no censored run is removed from the ledger.
Use `--preflight` before a launch to write the dependency/source/config-bound
receipt and disposable end-to-end fixture result. The runner writes atomic
case shards and `progress.json`; `--resume` validates shard schema/protocol/plan
identity, reloads stable case keys, and emits an idempotent terminal manifest.
The registered 72-case grid and exact pilot / standard commands are bound in [`docs/plans/astra_tkt14_resource_envelope.md`](docs/plans/astra_tkt14_resource_envelope.md)
and [`configs/astra-resource-envelope.toml`](configs/astra-resource-envelope.toml).

```bash
hedonic-exp overlapping-lfr --profile smoke \
  --output-dir artifacts/overlapping/overlap_lfr/smoke
hedonic-exp overlapping-lfr --profile standard --preflight \
  --official-format lfrbenchmarks \
  --official-generator /path/to/LFRbenchmarks/unweighted_undirected/benchmark \
  --official-config configs/astra-lfrbenchmarks.example.json \
  --output-dir artifacts/overlapping/overlap_lfr/standard-preflight
hedonic-exp overlapping-lfr --profile pilot --dry-run \
  --output-dir artifacts/overlapping/overlap_lfr/pilot
hedonic-exp overlapping-lfr --profile standard --dry-run \
  --output-dir artifacts/overlapping/overlap_lfr/standard
hedonic-exp overlapping-tracking --profile smoke \
  --output-dir artifacts/overlapping/tracking_triangle/smoke
# Canonical TKT-12 launch gate and resumable 20-graph implementation pilot.
hedonic-exp overlapping-tracking --profile standard --preflight \
  --graph-ledger artifacts/overlapping/overlap_lfr/standard/graphs.json \
  --output-dir artifacts/overlapping/tracking_triangle/standard
hedonic-exp overlapping-tracking --profile pilot --resume \
  --graph-ledger artifacts/overlapping/overlap_lfr/standard/graphs.json \
  --output-dir artifacts/overlapping/tracking_triangle/pilot
# Full registered TKT-12 run (launch only after reviewing preflight/pilot).
hedonic-exp overlapping-tracking --profile standard --resume \
  --graph-ledger artifacts/overlapping/overlap_lfr/standard/graphs.json \
  --output-dir artifacts/overlapping/tracking_triangle/standard
hedonic-exp overlapping-baselines --profile smoke \
  --output artifacts/overlapping/baselines/smoke.json
# Canonical standard preflight/input binding (bounded qualification only):
hedonic-exp overlapping-baselines --profile standard \
  --graph-ledger artifacts/overlapping/overlap_lfr/standard/graphs.json \
  --max-graphs 1 --methods singleton,chen \
  --tuning-config configs/astra-baselines.example.json \
  --output artifacts/overlapping/baselines/standard-pilot/baselines.json
# Full 840-graph baseline (launch only after reviewing the bounded receipt):
hedonic-exp overlapping-baselines --profile standard \
  --graph-ledger artifacts/overlapping/overlap_lfr/standard/graphs.json \
  --tuning-config configs/astra-baselines.example.json \
  --output artifacts/overlapping/baselines/standard/baselines.json \
  --resume
hedonic-exp overlapping-resource-envelope --profile smoke \
  --output-dir artifacts/overlapping/resource_envelope/smoke
```

```bash
# Tiny 16-run structural check; writes JSON plus a sibling CSV.
hedonic-exp overlapping-controlled --smoke \
  --output artifacts/overlapping/controlled_overlap/controlled_overlap.json

# Example controlled grid (still LFR-derived, not canonical overlapping LFR).
hedonic-exp overlapping-controlled --n 80 --mus 0.2,0.4 \
  --overlap-fractions 0.1,0.3 --overlap-memberships 2,3 \
  --secondary-edge-probabilities 0.1,0.3 --graph-seeds 0,1,2 \
  --max-memberships 1,2,gt \
  --starts singleton,neutral-disjoint,gt-primary \
  --resolution-multipliers 1,10 --output artifacts/overlapping/controlled_overlap/controlled_overlap.json
```

Guide: [`docs/controlled_overlap.md`](docs/controlled_overlap.md). The tracked
16-condition software reference is
[`artifacts/evidence/overlapping_communities/controlled_overlap_smoke.json`](artifacts/evidence/overlapping_communities/controlled_overlap_smoke.json)
with a sibling flat CSV; it is not substantive multi-seed paper evidence.
The corrected multi-seed v2 ledger is
`artifacts/evidence/overlapping_communities/controlled_overlap_v2.json` and
contains 864 completed cells with isolation-enabled cleanup. The `n=80` scale
is still a sensitivity diagnostic, not scalability evidence; an `n=200` run
was abandoned after a native multi-phase call exceeded 23 minutes.

### Small-instance DNN certificate diagnostic

`hedonic-exp overlapping-dnn` implements the Chapter 5 Equation (5.11)
diagnostic on literal, SHA-256-identified graphs small enough for complete
enumeration. It enumerates all labelled, nonempty, equal-intensity membership
assignments under each instance's label and membership caps, runs
`Game.community_hedonic` from recorded seeded warm starts, and solves the
doubly-nonnegative SDP with CVXPY/SCS. The strict JSON output records instances,
seeds, covers, factors, Gram matrices, exact and SDP objectives, raw solver
status/version/tolerances, primal residual checks, and a numerically repaired
feasible-dual upper bound. It reports the algorithm-to-exact gap separately
from the combined valid-cover-to-DNN outer gap; it does not infer a separate
completely-positive representation gap.

```bash
uv sync --extra experiments
hedonic-exp overlapping-dnn --output artifacts/overlapping/dnn_certificate/dnn_certificate.json
hedonic-exp overlapping-dnn --instances path4,bow_tie5 --seeds 0,1,2 \
  --eps 1e-8 --max-iters 200000 \
  --output artifacts/overlapping/dnn_certificate/subset.json
hedonic-exp overlapping-dnn --list-instances
```

The tracked reference run is
[`artifacts/evidence/overlapping_communities/dnn_certificate_v2.json`](artifacts/evidence/overlapping_communities/dnn_certificate_v2.json).
It is a small-instance certificate calibration only, not evidence that the DNN
relaxation is generally tight or that the detector scales.

The byte-pinned numerical producer above remains frozen. The versioned
`overlapping-dnn-rational` companion adds exact rational symmetric-diagonal-
dominance dual certificates and two deliberate boundary instances without
changing that producer. Its tracked candidate artifact is
`artifacts/evidence/overlapping_communities/dual_witnesses_verified.json`.
`--require-debug-trace` additionally requires the collision/projection
diagnostic (1.0.0.4 or later) and that every recorded guard decision follows
the post-local rule of 1.0.0.5; on `collision_path3` the tied token proposal
merges the two duplicate labels and is kept. Accept/restore branches are
recorded as observations, not required. The tracked candidate was produced
under the 1.0.0.4 guard. Exact rational bounds may be looser than the separately qualified
SCS repair; do not relabel the rational fallback as a tight interval solve.

Independent complete-set identities, CE1, and CE5 token/collision checks are
checked by
`hedonic-exp overlapping-oracle`. Run `--proof-check` to reconstruct the
numbered prefix, endpoint, restricted-balance, and coefficient-counting
propositions (machine check, not a named coauthor referee report). That
command does not call the native detector unless `--native-differential` is
passed. `--fee-check` writes the separately versioned
`unit_l2_cpm_fee_v1.json`: it checks the fee potential/delta identity, complete
prefix response, fee units, the (R10) private-label incidence bound, and the
anonymous-label path bound on a graph-only K4 ablation. It does not reuse or
reinterpret the manuscript model's frozen certificates. Detector-free 3840/864/125/tiny
classification is `hedonic-exp overlapping-certificate-reconcile`. Add
`--replay-dnn`, `--replay-gt`, and `--replay-paper` to recompute tiny exact
covers, re-audit persisted GT-v3 memberships, and re-audit the 75 Hedonic
equilibrium-v2 memberships on local shard graphs without rewriting the
frozen ledgers. Displayed manuscript tables/macros are generated unless
`--no-tex-fragments` is passed; `--tex-fragment-dir` selects the write
location. `--gate-h-check` independently re-checks stored DNN dual witnesses,
frozen protocol-lock bytes, ledger counts, generated TeX, short proofs,
wrapper/metric contracts, and the TeX/PDF snapshot hashes. Do not rewrite
`configs/overlapping-paper-protocol.lock.json` or the GT-v3 lock to chase the
live wrapper: `Game.py` and related files postdate those producer identities.

### SNAP overlapping benchmark

`hedonic-exp overlapping-benchmark` is the single CLI for the saved
overlapping SNAP covers: Amazon, DBLP, LiveJournal, YouTube, and Wikipedia
categories. It uses `~/Databases/Hedonic/Networks` by default, accepts
`--data_root`, and honors `HEDONIC_NETWORKS_DIR`. The shared loader in
`overlapping.snap` prefers validated normalized caches outside the archive,
then trusted local pickles, then streamed `*.txt.gz` files. It records the
original-ID mapping, validation/dropped-ID counts, directionality, and cover
statistics for every load. DBLP cached graphs use `vs["label"]` to remap
original community IDs correctly.

The benchmark writes resumable JSON per method/dataset/seed/resolution under
`runs/`, plus `manifest.json`, `results.jsonl`, `results.csv.gz`, summaries,
method availability, and plots. `--resume` reuses only compatible completed
records (or an explicit unsupported dependency), never a timeout/OOM/memory
limit/failed record.
For a release-sensitive rerun whose change is confined to native Leiden,
`--skip-methods cpm,demon` records explicit
`skipped_external_unchanged` conditions without invoking those external
baselines; the paper TOML exposes the same policy as
`not_rerun_external_methods`. It must not be used for native Hedonic methods.
`--timeout_per_run` plus optional per-method/per-dataset maps terminates an
isolated detector process tree; unavailable optional methods, timeouts, OOM,
memory limits, and unsupported variants are explicit records rather than
silently omitted. Use `--omega --omega_sample_size …` for the
memory-safe sampled Omega metric.
Large detector covers return through a private temporary pickle artifact rather
than `multiprocessing.Queue`; this prevents a bounded-pipe deadlock while the
parent is waiting for process exit. The artifact is deleted after loading.

The raw Wikipedia loader preserves SNAP edge direction for provenance, but the
benchmark applies the recorded `common_undirected_simple_v1` projection before
density, detection, quality, or metric calculations. Every method therefore
receives the same undirected simple analysis graph; legacy mixed-direction
records are not compatible paper evidence.

Methods are adapters, not new core algorithms: `hedonic_local` and the three
`hedonic_multiphase*` variants call `Game.community_hedonic` with
`n_iterations=-1` and the maximum per-node ground-truth membership count as
`max_memberships`.
The multi-phase variants set `allow_isolation=True` and fix resolution to
`min(density × {1,10,100}, 1)`; CPM uses NetworkX clique percolation; DEMON
uses its maintained external Python package. Install all baselines with
`uv sync --extra experiments`; `--list-methods` reports availability,
parameters, algorithm family, conversion behavior, and expected scalability.

### Ground-truth-seeded overlapping robustness

`hedonic-exp overlapping-gt-robustness` is a separate reverse-engineering
experiment. It does not modify the locked 125-condition paper protocol. The
primary audit uses the unit-\(\ell_2\) membership utility from
`docs/papers/overlapping_communities/main.tex`: a cover is robust on
\([0,1]\) when each current membership set is a best response at both
\(\gamma=0\) and \(\gamma=1\). Every result names its action policy
(`fixed_labels` or `open_labels`) and stores endpoint regrets, point stability,
equilibrium status, potential change, and one-to-one-aligned cover metrics.

The detector receives the complete nested GT membership as
`initial_membership` and always uses `n_iterations=-1`. Local moving is the
primary phase; multi-phase is a paired sensitivity. Partial SNAP covers use a
covered-induced graph by default, while `--uncovered-policy singleton` is an
explicit synthetic-completion sensitivity. Incidence-switch perturbations
preserve every vertex membership count and the community-size multiset. The
canonical v3 grid targets comparable labeled-incidence distances of 0%, 0.5%,
2%, and 5%; it records the dataset-scaled requested/successful switch counts
and realized distance.
`--seeds` controls detector restarts; `--perturbation-seeds` names the
independent incidence-switch seeds.

```bash
hedonic-exp overlapping-gt-robustness --smoke \
  --output-dir artifacts/overlapping/ground_truth_robustness_v3/smoke
hedonic-exp overlapping-gt-robustness \
  --config configs/overlapping-ground-truth.toml
# Re-score persisted final covers without invoking native detection:
hedonic-exp overlapping-gt-robustness --rescore-only \
  --config configs/overlapping-ground-truth.toml
```

Artifacts are written under `ground_truth/`, `covers/`, `runs/`, `plots/`, and
the top-level `manifest.json`, `results.jsonl`, `results.csv`, and
`coverage_report.json`. The subprocess timeout is a hard wall-clock bound;
the untouched GT reference has no detector runtime. Wall-clock is preserved
for every native return, but a non-equilibrium return is never interpreted as
time to equilibrium.

Guide: [`docs/overlapping_ground_truth_robustness.md`](docs/overlapping_ground_truth_robustness.md).

Reproduction guide: [`docs/reproduce_overlapping.md`](docs/reproduce_overlapping.md).
SNAP benchmark guide: [`docs/snap_overlapping_benchmark.md`](docs/snap_overlapping_benchmark.md).

### Full overlapping-paper reproduction

`hedonic-exp reproduce-overlapping-paper` is the one-command protocol for
[`docs/papers/overlapping_communities/main.tex`](docs/papers/overlapping_communities/main.tex).
It reads every normal experiment argument from `[overlapping_paper]` in
`configs/hedonic.toml`, including data/output/paper paths, methods, seeds,
resolutions, timeout, sampled Omega, retry count, tmux name, worker cap, RAM
budget, and the five dataset/cover jobs. The paper protocol intentionally uses
the three density-scaled multi-phase variants (`hedonic_multiphase`, `_x10`,
`_x100`) with `cpm,demon`; it rejects `hedonic_local`. All three pass
`allow_isolation=True`. For each dataset and seed, all hedonic variants receive
the same seeded random disjoint warm start with one requested label per supplied
ground-truth community; CPM and DEMON do not expose an initial-cover API.

The command measures graph/cover sizes before launch, records conservative
method-aware peak-memory estimates, descendant-RSS detector limits, and the
24-GiB Mac reserve in `plan.json`, then opens tmux workers in memory-budgeted
waves. LiveJournal is always a solo wave and uncertain estimates force one
worker. Shards remain resumable under
`artifacts/papers/overlapping_communities/full/shards/`; the coordinator creates the merged `results.*`,
`summary.*`, `paper_summary.csv`, regenerated `plots/`, TeX fragments, and a
per-condition `coverage_report.*`.
`main.tex` stays on its smoke branch until all expected full-protocol records
are present and completed. Only then does the coordinator
enable the generated results branch and run `latexmk`.

The manuscript currently points to the bounded equilibrium-v2 rerun in
`configs/hedonic-equilibrium-v2.toml` (3,000-vertex induced graphs). Its 125
conditions all completed; the uncapped `configs/hedonic.toml` ledger remains a
separate historical/resource audit and must not be mixed with the v2 tables.

Run `hedonic-exp reproduce-overlapping-paper --dry-run` to inspect the CPU/RAM
bounded assignment first. Use `--compile-partial` only for an explicitly
warning-labeled partial paper. Use `--no-tmux` only for small smoke/debug configs;
the normal command is the overnight tmux workflow.

---

## How to run an existing experiment

1. `uv sync --extra experiments`
2. Point data dirs if needed:
   ```bash
   export HEDONIC_DBLP_DIR=/path/to/DBLP
   export HEDONIC_SYNTHETIC_DIR=/path/to/synthetic
   ```
3. **Prefer CLI** over ad-hoc scripts:
   ```bash
   hedonic-exp list
   hedonic-exp smoke          # no DBs
   hedonic-exp disjoint --help
   ```
4. Cheap sanity check (no databases):  
   `hedonic-exp smoke` or `hedonic-exp overlapping-small`
5. Keep full DBLP / multi-seed SBM sweeps off the critical path of small PRs unless the task explicitly requires them.

---

## How to write a new experiment (checklist)

1. **Put it under** `src/hedonic/experiments/`  
   - Disjoint / synthetic → `disjoint/`  
   - Overlapping / DBLP-style → `overlapping/`

2. **Use `Game` + `community_hedonic`** for the method under test.  
   Baselines may also go through `community_hedonic` with different flags, or thin local helpers (mirror, one-pass) if they are not core API.

3. **Use `config.DBLP_DIR` / `SYNTHETIC_DIR`** (or env overrides). Accept `--data_dir` / `--output_root` CLI flags when I/O is involved.

4. **Share metrics** from `overlapping.metrics` (or small pure helpers in the module). Do not copy Omega/F1 logic into a fourth place.

5. **Expose `main(argv=None)`** that uses `argparse` and is import-safe (no work at import time). Prefer returning `0`/`1` (or `True`/`False`) so the CLI can map exit codes.

6. **Wire the CLI** in `experiments/CLI.py` (**mandatory** — see [Agent rule](#agent-rule-always-update-the-cli-when-experiments-change)):
   - add a `Command(...)` entry to `COMMANDS`
   - set `module`, `attr`, `summary`, `needs_data`, `has_argparse_help`
   - update root epilog examples if users will invoke new flags
   - update the CLI table + examples in **this** `AGENTS.md`

7. **Optional dependency rule**: if you need pandas/scipy/tqdm/stopwatch, they belong under the `experiments` extra only.

8. **Tests**: add cases in `tests/` that call the real shipped functions (`community_hedonic`, CLI subcommand, or pure helpers). Prefer small graphs; do not require full DBLP in CI-style checks. At minimum, assert the new subcommand appears in `CLI.COMMANDS` and `hedonic-exp <cmd> --help` returns 0.

9. **Do not**:
   - add a parallel overlapping class in core
   - depend on `tmp/` clones for runtime (those are research archives only)
   - hardcode absolute `/Users/<name>/...` paths (use `~/…` + `expanduser`, or `configs/hedonic.toml`)
   - leave a new experiment runnable only as `python -m ...` without a `hedonic-exp` subcommand

### Minimal new experiment skeleton

```python
# src/hedonic/experiments/overlapping/my_exp.py
from __future__ import annotations
import argparse
from hedonic import Game
from hedonic.experiments.config import DBLP_DIR  # or SYNTHETIC_DIR
from hedonic.experiments.overlapping.metrics import (
    evaluate_cover,
    partition_to_cover_lists,
)

def run(...):
    g = Game(...)  # load or generate
    part = g.community_hedonic(resolution=g.density(), max_memberships=1)
    cover = g.community_hedonic(
        resolution=g.density(),
        max_memberships=4,
        initial_membership=list(part.membership),
    )
    cover_lists = partition_to_cover_lists(cover)
    # metrics, print, optional json.dump
    return cover_lists

def main(argv=None):
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--data_dir", default=str(DBLP_DIR))
    # ...
    args = parser.parse_args(argv)
    run(...)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

Then register in `CLI.py`:

```python
"my-exp": Command(
    name="my-exp",
    module="hedonic.experiments.overlapping.my_exp",
    attr="main",
    summary="One-line description for hedonic-exp list",
    needs_data="dblp",  # or "synthetic" or None
),
```

…and add the row + an example to the CLI section of this file.

---

## Tests

```bash
.venv/bin/python -m unittest tests.test_overlapping_and_experiments -v
```

Cover: `community_hedonic` disjoint/overlapping, config env overrides, CLI help/`list`/`smoke`/`overlapping-small`, SBM helpers + `--smoke`, and that core does not export a removed overlapping module.

---

## Web explainer (`web/`)

An educational React + TypeScript + Vite site for the **disjoint** model
(scrollytelling story + `/docs/`), deployed by `.github/workflows/pages.yml`.
The root `package.json` is an npm workspace; run everything from the root:

```bash
npm install
npm run dev            # story at /, documentation at /docs/
npm run check          # docs:generate + typecheck + lint + test + build
```

- `web/src/model/` is the mathematics (partitions as restricted growth
  strings, single-vertex moves, Familiarity Index, CPM potential, sinks,
  deterministic better-response rules). Keep it faithful to the SRC abstract
  and arXiv:2509.03834, keep it DOM-free, and extend its Vitest suite with any
  change. It is a teaching model of a four-vertex graph, never a substitute
  for `community_hedonic`.
- `npm run docs:generate` must be re-run (and its `web/src/generated/*.json`
  committed) when `Game`, `hedonic.utils`, the CLI `COMMANDS` registry or the
  disjoint sweep presets change. It parses sources with `ast` only.
- Documentation pages are `web/content/docs/*.md`; citation data is
  `web/content/references.bib` (public bibliographic metadata copied from the
  research bibliography). The public site must never read `docs/papers/`.
- `web/scripts/crosscheck_native.py` (needs the Python package) refreshes the
  fixture that checks native results against the browser model's sinks.

---

## Design invariants (do not break)

1. **One public detector API**: `Game.community_hedonic` for both modes (`max_memberships`).
2. **Overlapping algorithm** lives in **lucas-igraph** (`community_leiden` + `max_memberships`), not in this package.
3. **Experiments optional**: core installs without pandas/scipy.
4. **CLI module name is `CLI.py`** (both entrypoint strings in `pyproject.toml` must match).
5. **CLI registry stays complete**: every user-facing experiment `main()` is reachable via `hedonic-exp` (see agent rule above).
6. **`tmp/`** is local research history (often git-excluded); never a runtime import path for the library.
## Complete-network SNAP track

`overlapping-full-snap` → `hedonic.experiments.overlapping.full_snap` is the
separately versioned `full-snap-v1` complete-raw-graph protocol. It preserves all
endpoint vertices, uses no metadata-derived cap or initialization, and records
resource boundaries instead of silently selecting a subgraph. See
[the full protocol](docs/full-snap-v1.md) for metric conventions, native build
receipts, fixed/unlimited variants, baseline parameters, limits and resume.

```bash
hedonic-exp overlapping-full-snap --preflight --seeds 0 \
  --output-dir artifacts/overlapping/full-snap-v1/preflight
hedonic-exp overlapping-full-snap --smoke --datasets amazon --seeds 0 \
  --output-dir artifacts/overlapping/full-snap-v1/smoke
```

Real dataset attempts require verified native/Python source and binary
provenance via `--native-source`, `--python-source`, `--build-receipt`. The
independent unit-l2 objective/audit in `full_snap_audit` must remain distinct
from the historical binary-overlap metric helpers. Do not present sampled
stationarity checks or native no-change termination as exact Nash proofs.
