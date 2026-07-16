# AGENTS.md — Working on `hedonic`

Guidance for humans and coding agents editing this repo. Keep the **core library light** and put experiment I/O, sweeps, and metrics in `hedonic.experiments`.

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
    ├── CLI.py           # hedonic-exp entrypoint (capital C, intentional)
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
        └── reproduce_paper.py   # TOML/tmux full-paper orchestration + aggregation
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
| Hedonic local-moving only | `only_local_moving=True` (default) | same as above |
| Full Leiden (refine + aggregate) | `only_local_moving=False` | same as above |

```python
# Disjoint — main exploratory method
part = g.community_hedonic(
    resolution=g.density(),   # CPM γ; default is density if omitted
    max_memberships=1,
    only_local_moving=True,
    n_iterations=-1,
    initial_membership=None,  # default: singleton partition
)

# Overlapping — same API
cover = g.community_hedonic(
    resolution=g.density(),
    max_memberships=4,
    only_local_moving=True,
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
- `max_communities` — only when `initial_membership is None` and disjoint: random labels in `[0, K)`
- `allow_isolation` — allow empty-community moves
- `edge_weights`, `seed`, `beta` — weights, RNG for random init, Leiden refinement noise

**Do not** call raw `community_leiden` in experiment code unless you are debugging the binding itself. Prefer `community_hedonic` so defaults stay consistent (`only_local_moving=True`, density resolution, membership normalization).

---

## Dependencies

```toml
# Core
lucas-igraph   # path editable: ../python-igraph (see pyproject [tool.uv.sources])
numpy

# Optional: pip install "hedonic[experiments]"  or  uv sync --extra experiments
pandas, scipy, tqdm, stopwatch-py, matplotlib, seaborn, networkx, demon
```

Setup:

```bash
uv sync --extra experiments
# If lucas-igraph fails to rebuild on macOS, set SDKROOT to the active Xcode SDK first.
```

CLI after install (console script from `pyproject.toml` → `hedonic.experiments.CLI:main`):

```bash
hedonic-exp --help
hedonic-exp list
# or without install:
python -m hedonic.experiments.CLI --help
```

---

## Experiments package

### Paths (`experiments/config.py`)

| Variable | Env override | Default (expanded from `~/…`) |
|----------|--------------|--------------------------------|
| `DBLP_DIR` | `HEDONIC_DBLP_DIR` | `~/Databases/Hedonic/Networks/DBLP` |
| `NETWORKS_DIR` | `HEDONIC_NETWORKS_DIR` | `~/Databases/Hedonic/Networks` |
| `SYNTHETIC_DIR` | `HEDONIC_SYNTHETIC_DIR` | `~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020` |
| `OUTPUT_DIR` | `HEDONIC_OUTPUT_DIR` | `~/Databases/Hedonic/experiments` |

**TOML configs live under [`configs/`](configs/)** (default load: `configs/hedonic.toml` from cwd). Example template: [`configs/hedonic.example.toml`](configs/hedonic.example.toml). Override with `--config path.toml` or `HEDONIC_CONFIG`. Priority: **CLI flag > env > TOML > defaults**. Use `~/…` paths in TOML (expanded at load); do not hard-code `/Users/<name>`.

```toml
[paths]
dblp_dir = "~/Databases/Hedonic/Networks/DBLP"
networks_dir = "~/Databases/Hedonic/Networks"
synthetic_dir = "~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020"
output_dir = "~/Databases/Hedonic/experiments"

[overlapping_resolution]
output_dir = "~/Databases/Hedonic/Networks/DBLP_CLI/resolution_f1"
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

Entry: `hedonic-exp` → `hedonic.experiments.CLI:main`.

Subcommands are registered in **`COMMANDS`** (the single source of truth in `CLI.py`):

| Subcommand | Module | Purpose | Data |
|------------|--------|---------|------|
| `smoke` | (CLI built-in) | Isolated check: small graphs + tiny SBM sweep | none |
| `disjoint` | `disjoint.sbm_sweep` | SBM sweeps → JSON under `SYNTHETIC_DIR` (presets: `v1020`, `v1020-smoke`) | synthetic (or `--output_root`) |
| `disjoint-load` | `disjoint.data_loader` | JSON results → gzipped CSV (`--simple` for smoke) | synthetic results |
| `plots` | `plots.paper_figures` | PHYSA V1020 paper figures from CSV (`gt_robustness`, `noise`, …) | synthetic CSV |
| `reproduce-disjoint` | `disjoint.reproduce` | End-to-end: sweep → CSV → figures | synthetic |
| `overlapping-small` | `overlapping.small_graphs` | Smoke + metrics on small graphs | none |
| `overlapping-subgraph` | `overlapping.dblp_subgraph` | L-hop around GT communities | DBLP |
| `overlapping-full` | `overlapping.dblp_full` | Full DBLP + optional resolution sweep | DBLP |
| `overlapping-scale` | `overlapping.complexity_scale` | Wallclock scaling as subnetworks grow (local-moving T/F, timeout stop + plot) | DBLP (or `--smoke`) |
| `overlapping-resolution` | `overlapping.resolution_f1` | Full-DBLP overlap metrics vs resolution [0,1] (cached-cover rescoring, singleton modes, optional sampled Omega) | DBLP (or `--smoke`) |
| `overlapping-benchmark` | `overlapping.benchmark` | Resumable Amazon/DBLP/LiveJournal/YouTube/Wikipedia overlapping-cover benchmark with hedonic, CPM, and DEMON | saved SNAP networks (or `--profile smoke`) |
| `reproduce-overlapping-paper` | `overlapping.reproduce_paper` | TOML-driven paper protocol: RAM-bounded tmux shards → merged records/plots/tables → guarded `main.tex` compilation | saved SNAP networks |
| `list` | meta | List subcommands | — |

```bash
# Isolated / CI-friendly (no databases)
hedonic-exp smoke
hedonic-exp overlapping-small
hedonic-exp disjoint --smoke --output_root /tmp/hedonic-smoke

# Reproduce disjoint SBM sweep (subset of methods)
hedonic-exp disjoint --folder_name smoke --max_n_nodes 40 --n_communities 2 \
  --seeds 1 --p_in 0.2 --difficulty 0.3 --noises 0.01 --partition_seeds 0 \
  --methods Hedonic,Leiden --output_root /tmp/sbm

# PHYSA V1020 structural smoke (same layout/methods as archived grid; safe root)
hedonic-exp disjoint --preset v1020-smoke \
  --output_root ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI

# Full V1020 grid via CLI (very large — never write into archived V1020)
hedonic-exp disjoint --preset v1020 \
  --output_root ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI

# Combine prior disjoint JSON dumps
hedonic-exp disjoint-load --results_folder /path/to/resultados --output /tmp/out.csv.gzip --simple

# Paper figures (same five stems as archived V1020/figures/)
hedonic-exp plots --smoke --output_dir /tmp/hedonic-figs
hedonic-exp plots --data /path/to/resultados_ari.csv.gzip \
  --output_dir ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI/figures

# Complete pipeline: sweep → CSV → figures (never write into archived V1020)
hedonic-exp reproduce-disjoint --preset v1020-smoke \
  --output_root ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI
# Figures only from archived CSV (read-only) into V1020_CLI:
hedonic-exp reproduce-disjoint --plots-only \
  --data ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/resultados_ari.csv.gzip \
  --output_root ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI --max_rows 50000

# Overlapping (point HEDONIC_DBLP_DIR if needed)
# Defaults: n_iterations=-1 (to equilibrium), max_memberships=n_GT communities
hedonic-exp overlapping-subgraph --levels 1 --n_communities 5 \
  --methods leiden,hedonic_v1 --output /tmp/subgraph_smoke.json
hedonic-exp overlapping-full --resolution 1e-4 --output /tmp/dblp_full.json

# Complexity scale: network size vs wallclock to equilibrium
# Two lines: only_local_moving True vs False; density γ; K = #GT in window
# Growth stops per line when --timeout is hit (full graph not required)
hedonic-exp overlapping-scale --smoke --output_dir /tmp/hedonic-scale
hedonic-exp overlapping-scale --timeout 30 --max-levels 6 \
  --output_dir /tmp/hedonic-scale-dblp
# Full multi-phase only (only_local_moving=False), 10 min budget per size
hedonic-exp overlapping-scale --variant full --timeout 600 --max-levels 6 \
  --community_idx 1004 --output_dir /tmp/hedonic-scale-dblp-full

# Full-DBLP F1 vs resolution γ ∈ [0,1]: multi-phase, allow_isolation,
# max_memberships = #GT with size>1; multi-seed F1 CI band on the plot.
# Per-(γ,seed) covers + metadata cached under <output_dir>/runs/ (resume-safe).
# Paths from configs/hedonic.toml by default (~/… expanded).
# Re-score all metrics from covers without re-detection: --rescore-only
hedonic-exp overlapping-resolution --smoke --output_dir /tmp/hedonic-res-f1
hedonic-exp overlapping-resolution --config configs/hedonic.toml \
  --resolutions 0:1:11 --seeds 0-4
# Resume after interrupt (same --output_dir; skips completed runs/ files):
hedonic-exp overlapping-resolution --resolutions 0:1:11 --seeds 0-4 \
  --output_dir ~/Databases/Hedonic/Networks/DBLP_CLI/resolution_f1
# Recompute matching, node-membership, weighted F1, and diagnostics from
# cached covers only; report both consistent singleton modes:
hedonic-exp overlapping-resolution --rescore-only --singleton-mode both \
  --resolutions 0:1:11 --seeds 0-4 \
  --output_dir ~/Databases/Hedonic/Networks/DBLP_CLI/resolution_f1
# Optional sampled Omega (never allocates a dense vertex-pair matrix):
hedonic-exp overlapping-resolution --rescore-only --omega \
  --omega-sample-size 100000 --singleton-mode size_ge_2 \
  --output_dir ~/Databases/Hedonic/Networks/DBLP_CLI/resolution_f1

# Reproducible five-network SNAP overlap benchmark. The smoke profile uses
# built-in tiny covers; standard bounds each graph to a deterministic
# GT-informed induced subgraph. Wikipedia has only an `all` category cover,
# so `--cover top5000` records it explicitly as skipped.
hedonic-exp overlapping-benchmark --profile smoke \
  --output_dir /tmp/hedonic-snap-smoke
hedonic-exp overlapping-benchmark --datasets amazon,dblp,livejournal,youtube,wikipedia \
  --cover top5000 --profile standard \
  --output_dir ~/Databases/Hedonic/Networks/SNAP_BENCHMARK_CLI
hedonic-exp overlapping-benchmark --list-networks
hedonic-exp overlapping-benchmark --list-methods

# Complete overlapping SNAP paper reproduction. All run parameters are in
# configs/hedonic.toml [overlapping_paper]; raw SNAP archives stay read-only.
uv run hedonic-exp reproduce-overlapping-paper --dry-run
uv run hedonic-exp reproduce-overlapping-paper
tmux attach -t hedonic-overlapping-paper
```

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
| **Change** `pyproject.toml` entry point | Keep `[project.scripts] hedonic-exp = "hedonic.experiments.CLI:main"` accurate |

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
4. **Hedonic** and **Leiden** both call `community_hedonic` with `max_memberships=1` (Leiden uses `only_local_moving=False`). Spectral sets `clusters = n_communities`.
5. Output layouts:
   - **`v1020`** (preset default): `resultados/{n}C_{size}N/Noise = …/P_in = …/Difficulty = …/Network (NNN)/partition_MMM.json` — list of method result dicts (matches archived PHYSA V1020).
   - **`legacy`**: `{folder}/{n} Communities of {size} nodes/.../Partition (MMM)/{Method}.json`.
6. Leiden/Hedonic use **`--n_runs`** stochastic restarts with unique-partition dedup (V1020 used 10).
7. **Presets:**
   - `--preset v1020` — full archived grid (1020 nodes, communities 2–6, full p_in/difficulty/noise, 5 nets, 10 partitions, all methods). Refuses to write into the archived `.../V1020` folder.
   - `--preset v1020-smoke` — tiny structural clone for CLI checks.
   - `--smoke` — smaller legacy mini-run (CI-friendly).
8. **Isolated run:** `--smoke` / `--preset v1020-smoke` and `--output_root` (prefer `.../V1020_CLI`, never overwrite archived V1020).
9. **Figures:** `hedonic-exp plots` (or `reproduce-disjoint`) writes the five paper stems: `gt_robustness`, `noise`, `n_communities`, `acc_robustness`, `acc_efficiency`.

When adding a method to the sweep, extend `METHODS` and implement the callable on `Game` (or a local helper in the experiment module if it is not core). If the public CLI documents method names, refresh help/examples.

### Complete V1020 disjoint reproduction (via CLI)

| Step | Command | Output |
|------|---------|--------|
| 1. Sweep | `hedonic-exp disjoint --preset v1020 --output_root …/V1020_CLI` | `resultados/{n}C_{size}N/…/partition_*.json` |
| 2. Combine | `hedonic-exp disjoint-load --results_folder …/resultados --simple` | `resultados.csv.gzip` |
| 3. Figures | `hedonic-exp plots --data …/resultados.csv.gzip --output_dir …/figures` | five PDFs matching archived names |

Or one shot: `hedonic-exp reproduce-disjoint --preset v1020 --output_root …/V1020_CLI`.

**Do not** write into archived `…/Synthetic_Networks/V1020` (CLI guards refuse).

### Overlapping pattern

- Detection: **`Game.community_hedonic(..., max_memberships=K)`**.
- **Always** pass **`n_iterations=-1`** (or any negative value) so local moving runs until equilibrium. Positive budgets may stop early; CLI defaults are `-1`.
- **SNAP benchmark `max_memberships`** defaults to the maximum number of
  ground-truth communities containing any one node (at least 2), not `len(gt)`.
  This keeps overlap capacity semantically meaningful and bounds the native
  `n × max_memberships` workspace. Override only when intentionally capping it.
- Metrics vs covers: **`experiments.overlapping.metrics`** (`evaluate_cover`, `partition_to_cover_lists`, `cover_quality`, baselines, optional Nash check).
- DBLP load: `dblp_full.load_dblp` (cache `dblp.pkl`, or `pkl/`, or `raw/*.gz`).
- Subgraph methods:
  - `leiden` — `community_hedonic(max_memberships=1, only_local_moving=False)`
  - `hedonic_v1` — overlapping local-moving (`only_local_moving=True`), warm-started from Leiden
  - `hedonic_v2` — overlapping full multi-phase (`only_local_moving=False`)
  - `singleton` / `grand_coalition` / `total_overlap` — deterministic control baselines

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
method availability, and plots. `--resume` skips existing run records.
`--timeout_per_run` terminates an isolated detector subprocess; unavailable
optional methods, timeouts, and unsupported variants are explicit records
rather than silently omitted. Use `--omega --omega_sample_size …` for the
memory-safe sampled Omega metric.

Methods are adapters, not new core algorithms: `hedonic_local` and the three
`hedonic_multiphase*` variants call `Game.community_hedonic` with
`n_iterations=-1` and the maximum per-node ground-truth membership count as
`max_memberships`.
The multi-phase variants enable `allow_isolation=True` and fix resolution to
`min(density × {1,10,100}, 1)`; CPM uses NetworkX clique percolation; DEMON
uses its maintained external Python package. Install all baselines with
`uv sync --extra experiments`; `--list-methods` reports availability,
parameters, algorithm family, conversion behavior, and expected scalability.

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
`allow_isolation=True`.

The command measures graph/cover sizes before launch, records conservative
per-job peak-memory estimates and parent-RSS detector limits in `plan.json`,
then opens tmux workers in memory-budgeted waves so incompatible large graphs
cannot overlap. Shards remain resumable under
`artifacts/full/shards/`; the coordinator creates the merged `results.*`,
`summary.*`, `paper_summary.csv`, regenerated `plots/`, and TeX fragments.
`main.tex` stays on its smoke branch until all expected full-protocol records
are present and completed. Only then does the coordinator
enable the generated results branch and run `latexmk`.

Run `hedonic-exp reproduce-overlapping-paper --dry-run` to inspect the CPU/RAM
bounded assignment first. Use `--no-tmux` only for small smoke/debug configs;
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

## Design invariants (do not break)

1. **One public detector API**: `Game.community_hedonic` for both modes (`max_memberships`).
2. **Overlapping algorithm** lives in **lucas-igraph** (`community_leiden` + `max_memberships`), not in this package.
3. **Experiments optional**: core installs without pandas/scipy.
4. **CLI module name is `CLI.py`** (entrypoint string in `pyproject.toml` must match).
5. **CLI registry stays complete**: every user-facing experiment `main()` is reachable via `hedonic-exp` (see agent rule above).
6. **`tmp/`** is local research history (often git-excluded); never a runtime import path for the library.
