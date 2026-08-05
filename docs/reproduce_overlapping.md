# Reproducing the Overlapping DBLP Experiments

This guide explains how to **reproduce the overlapping community-detection experiments** on DBLP (and small-graph smokes) using the current `hedonic` library and the `hedonic-exp` CLI.

The reference research pipeline lives in the old repo (local archive, git-excluded):

```text
tmp/hedonic-overlapping/
```

Historical results and write-ups from that pipeline:

```text
tmp/hedonic-overlapping/results/RESULTADOS.md
tmp/hedonic-overlapping/results/subgraph_L1_n*.json
```

**Do not treat `tmp/` as a runtime dependency.** All runs go through `hedonic-exp` and write under a path you choose (for example under your DBLP results tree or `/tmp`).

---

## What “full reproduction” means

A complete overlapping reproduction covers the suite documented in the old `RESULTADOS.md`, mapped onto the current CLI:

| Stage | What | CLI command |
|-------|------|-------------|
| 0. Smoke | Small graphs, no DBLP | `hedonic-exp overlapping-small` |
| 1. Subgraph suite | L-hop windows around GT communities | `hedonic-exp overlapping-subgraph` |
| 2. Full graph (optional) | Whole DBLP + optional γ sweep | `hedonic-exp overlapping-full` |
| 3. Complexity scale | Size vs wallclock (local-moving T/F); timeout stop | `hedonic-exp overlapping-scale` |
| 4. Resolution evaluation | Cached-cover overlap metrics vs γ with multi-seed summaries | `hedonic-exp overlapping-resolution` |

There is no separate figure pipeline for overlapping (unlike disjoint PHYSA V1020). Artifacts are **JSON metrics**, per-run JSON cover caches for the resolution experiment, plus an optional **covers pickle** for subgraph re-scoring.

---

## Prerequisites

```bash
# From the hedonic-game repo root
uv sync --extra experiments
```

Optional dependencies used here: `pandas`, `scipy`, `tqdm`, `stopwatch-py` (and whatever `lucas-igraph` needs for overlapping Leiden).

Confirm the CLI:

```bash
hedonic-exp list
hedonic-exp overlapping-small --help
hedonic-exp overlapping-subgraph --help
hedonic-exp overlapping-full --help
```

You can also run without installing the console script:

```bash
python -m hedonic.experiments.CLI list
```

### Paths

| Variable | Env override | Default |
|----------|--------------|---------|
| DBLP root | `HEDONIC_DBLP_DIR` | `~/Databases/Hedonic/Networks/DBLP` |

Loader order in `dblp_full.load_dblp` (first hit wins):

1. `{data_dir}/dblp.pkl` — combined cache `(g, gt, node_map)`
2. `{data_dir}/pkl/com-dblp.ungraph.pkl` + `com-dblp.all.cmty.pkl`
3. `{data_dir}/raw/*.gz` or `{data_dir}/*.gz` (SNAP files; builds and saves `dblp.pkl`)

If your tree only has `pkl/` / `raw/`, the first subgraph/full run builds `dblp.pkl` for faster reloads.

### DBLP source (if you need raw SNAP files)

```bash
mkdir -p "$HEDONIC_DBLP_DIR/raw" && cd "$HEDONIC_DBLP_DIR/raw"
# SNAP DBLP co-authorship + communities
# https://snap.stanford.edu/data/com-DBLP.html
wget https://snap.stanford.edu/data/com-dblp.ungraph.txt.gz
wget https://snap.stanford.edu/data/com-dblp.all.cmty.txt.gz
# or the smaller top-5000 communities file if that is all you have
```

Expected scale (full SNAP DBLP): ~317k nodes, ~1.05M edges, ~13k ground-truth communities.

---

## Algorithmic defaults (paper reproduction)

Two rules are **required** for reproduction runs and are the CLI defaults:

| Parameter | Rule | Why |
|-----------|------|-----|
| `n_iterations` | **`-1`** (any negative value) | Iterate local moving until no improving move remains (equilibrium). Positive budgets can stop early and understate gains. |
| `max_memberships` | **Number of GT communities** for the graph under test | Cap memberships at the size of the ground-truth cover, not an arbitrary constant (old scripts often used `4` or `5` iterations). |

How `max_memberships` is resolved when you omit the flag:

| Command | Default `max_memberships` |
|---------|---------------------------|
| `overlapping-subgraph` | Per subgraph: count of GT communities with **≥ 2 nodes** inside the L-hop window (`n_gt_in_subgraph`) |
| `overlapping-full` | `len(gt)` on the full DBLP cover |

Override only when you intentionally want a smaller K:

```bash
hedonic-exp overlapping-subgraph ... --max_memberships 8
```

All detection goes through **`Game.community_hedonic`** (not the removed `OverlappingGame` API).

---

## Methods

### Subgraph experiment (`overlapping-subgraph`)

| Name | Role | Call pattern |
|------|------|----------------|
| **leiden** | Disjoint CPM baseline | `community_hedonic(max_memberships=1, local_move_only=False, n_iterations=-1)` |
| **hedonic_v1** | Overlapping local-moving only | `community_hedonic(max_memberships=K, local_move_only=True)`, warm-started from Leiden membership |
| **hedonic_v2** | Overlapping multi-phase | `community_hedonic(max_memberships=K, local_move_only=False)` |
| **singleton** | Max-granularity control | One community per node |
| **grand_coalition** | Min-granularity control | Single community = all nodes |
| **total_overlap** | Max-redundancy control | `n_gt_in_subgraph` copies of the full vertex set |

Primary metric vs the **seed GT community** (remapped to subgraph ids): **best-match F1** (also precision, recall, Jaccard). Optional Omega is available in `metrics.evaluate_cover` but left off by default (heavier).

Reported deltas:

- `delta_f1` = F1(hedonic_v1) − F1(leiden)
- `delta_f1_v2` = F1(hedonic_v2) − F1(leiden)

### Full-graph experiment (`overlapping-full`)

| Key | Meaning |
|-----|---------|
| `leiden_nonoverlapping` | Full multi-phase disjoint CPM |
| `hedonic_overlapping` | Local-moving overlapping warm-started from Leiden |
| `in_equilibrium` | Nash check on the overlapping cover |
| `quality` | CPM quality (from cover object or pure-Python fallback) |

---

## Quick start (smoke — recommended first)

No DBLP required:

```bash
hedonic-exp smoke
hedonic-exp overlapping-small
```

`overlapping-small` runs Petersen + a tiny two-block SBM (with `max_memberships = n_blocks` on the SBM) and prints F1 / quality / equilibrium checks.

Tiny DBLP subgraph smoke (needs data):

```bash
export HEDONIC_DBLP_DIR=~/Databases/Hedonic/Networks/DBLP
export OUT=/tmp/hedonic-overlapping-cli

mkdir -p "$OUT"

hedonic-exp overlapping-subgraph \
  --levels 1 \
  --n_communities 3 \
  --methods leiden,hedonic_v1 \
  --resolution 0.1 \
  --n_iterations -1 \
  --output "$OUT/subgraph_smoke.json"
```

Defaults already use `--n_iterations -1` and auto `max_memberships`; the flags are shown explicitly for clarity.

### Complexity scale (hardware limits)

Wallclock time to equilibrium as L-hop subnetworks grow. Two lines:
`local_move_only=True` vs `False`. Shared settings: resolution = edge density,
`max_memberships` = #GT communities in the window, `allow_isolation=True`,
`n_iterations=-1`. Each line stops when `--timeout` is exceeded (full DBLP not required).

```bash
# Synthetic smoke (no DBLP)
hedonic-exp overlapping-scale --smoke --output_dir "$OUT/scale_smoke"

# DBLP hop growth from a GT seed (safe timeout)
hedonic-exp overlapping-scale \
  --timeout 30 \
  --max-levels 6 \
  --output_dir "$OUT/scale_dblp"

# Full multi-phase only (local_move_only=False), 10 min per size point
hedonic-exp overlapping-scale \
  --variant full \
  --timeout 600 \
  --max-levels 6 \
  --community_idx 1004 \
  --output_dir "$OUT/scale_dblp_full"
```

Writes `complexity_scale.json` + `complexity_scale.png` under `--output_dir`.

---

## Paper-style subgraph suite

The old `RESULTADOS.md` suite maps 1:1 to CLI flags. Write results under a dedicated directory (example: `$OUT`).

```bash
export HEDONIC_DBLP_DIR=~/Databases/Hedonic/Networks/DBLP
export OUT=~/Databases/Hedonic/Networks/DBLP/results_cli
mkdir -p "$OUT"
```

### Experiment 1 — Random sample, 1-hop, fixed γ=0.1

Communities of size 5–200, seed `np.random.default_rng(42)` (same as the old script).

```bash
# n=200 (fast paper table)
hedonic-exp overlapping-subgraph \
  --levels 1 \
  --n_communities 200 \
  --resolution 0.1 \
  --n_iterations -1 \
  --methods leiden,hedonic_v1,hedonic_v2,singleton,grand_coalition,total_overlap \
  --output "$OUT/subgraph_L1_n200.json"

# n=1000 (stability check; long)
hedonic-exp overlapping-subgraph \
  --levels 1 \
  --n_communities 1000 \
  --resolution 0.1 \
  --n_iterations -1 \
  --output "$OUT/subgraph_L1_n1000.json"
```

Also writes a covers cache next to the JSON:

```text
$OUT/subgraph_L1_n200_covers.pkl
```

### Experiment 2 — Real GT overlap (≥ 2 shared nodes)

```bash
hedonic-exp overlapping-subgraph \
  --levels 1 \
  --n_communities 200 \
  --resolution 0.1 \
  --n_iterations -1 \
  --overlapping_only \
  --min_overlap 2 \
  --output "$OUT/subgraph_overlap_n200.json"
```

### Experiment 3 — Stronger overlap (overlap coefficient ≥ 0.2)

```bash
hedonic-exp overlapping-subgraph \
  --levels 1 \
  --n_communities 200 \
  --resolution 0.1 \
  --n_iterations -1 \
  --overlapping_only \
  --min_overlap 2 \
  --min_overlap_ratio 0.2 \
  --output "$OUT/subgraph_overlap_ratio02_n200.json"
```

### Experiment 4 — Ablation: 2-hop (random sample)

```bash
hedonic-exp overlapping-subgraph \
  --levels 2 \
  --n_communities 100 \
  --resolution 0.1 \
  --n_iterations -1 \
  --output "$OUT/subgraph_L2_n100.json"
```

Expect weaker or negative mean ΔF1: the seed community is a small fraction of the larger 2-hop window (historical ~4% of nodes).

### Experiment 5 / 6 — Density resolution (γ = subgraph density)

Instead of fixed `γ=0.1`, use each subgraph’s edge density:

```bash
hedonic-exp overlapping-subgraph \
  --levels 1 \
  --n_communities 200 \
  --density_resolution \
  --n_iterations -1 \
  --output "$OUT/subgraph_L1_n200_density.json"
```

Historical note: some archived density runs used `n_iterations=5`. **Re-run with the default `-1`** for equilibrium-faithful numbers.

---

## Full-graph experiment (optional / heavy)

Full DBLP with overlapping local moving can be expensive. Prefer subgraphs for the main tables; use full-graph for global quality / equilibrium checks.

```bash
# Single resolution (default γ=1e-4, n_iterations=-1, max_memberships=|GT|)
hedonic-exp overlapping-full \
  --resolution 1e-4 \
  --n_iterations -1 \
  --output "$OUT/dblp_full.json"

# Log-spaced resolution sweep γ ∈ [1e-2, 1] (10 values) — very heavy with n_iterations=-1
hedonic-exp overlapping-full \
  --resolution_sweep \
  --n_iterations -1 \
  --output "$OUT/dblp_resolution_sweep.json"
```

If you need a time-boxed exploratory sweep (not paper equilibrium), you may pass a positive `--n_iterations`, but the CLI will warn. Prefer `-1` for anything you report.

Capping memberships on the full graph (optional, not the paper default):

```bash
hedonic-exp overlapping-full --max_memberships 16 --output "$OUT/dblp_full_k16.json"
```

### Resolution metrics from cached covers

`overlapping-resolution` caches every detected cover under
`<output_dir>/runs/`. Re-score those files without invoking
`community_hedonic`:

```bash
hedonic-exp overlapping-resolution --rescore-only \
  --resolutions 0:1:11 --seeds 0-4 \
  --singleton-mode both \
  --output_dir ~/Databases/Hedonic/Networks/DBLP_CLI/resolution_f1
```

The historical `f1` field remains the symmetric best-match F1 used by earlier
resolution runs. New metrics are stored per singleton mode (`all` and/or
`size_ge_2`) and include:

- one-to-one community precision, recall, and F1 (Hungarian matching by pairwise F1);
- node-membership multilabel micro precision/recall/F1 and macro F1;
- size-weighted community F1;
- community-count, singleton, coverage, size, and memberships-per-vertex diagnostics.

`size_ge_2` filters singleton communities consistently from both predicted and
GT covers. `all` retains them on both sides. `both` reports both evaluations in
one pass. The summary JSON need not embed covers: rescoring reads the per-run
cache files directly.

Sampled Omega is opt-in because it adds work per cached cover. It samples
vertex pairs and never materializes a dense vertex-by-vertex matrix, so it is
safe at full DBLP scale:

```bash
hedonic-exp overlapping-resolution --rescore-only --omega \
  --omega-sample-size 100000 --omega-seed 0 \
  --singleton-mode size_ge_2 \
  --output_dir ~/Databases/Hedonic/Networks/DBLP_CLI/resolution_f1
```

---

## Output layout

### Subgraph JSON

Each element of the results list is one seed community:

```text
community_idx, gt_size, subgraph_nodes, subgraph_edges, levels,
resolution, n_iterations, n_gt_in_subgraph, max_memberships,
leiden / hedonic_v1 / hedonic_v2 / …  →  {f1, precision, recall, jaccard, n_communities, time_s, …},
delta_f1, delta_f1_v2
```

Overlapping-only mode also stores `n_partners` and `shared_nodes`.

Covers pickle (`*_covers.pkl`): list of records with remapped `covers` and `ground_truth` for offline re-scoring (Omega, etc.).

### Full-graph JSON

```text
{
  "params": {resolution, n_iterations, max_memberships, n_gt_communities},
  "leiden_nonoverlapping": {f1, jaccard, quality, time_s, …},
  "hedonic_overlapping": {f1, jaccard, quality, in_equilibrium, time_s, …}
}
```

Or for sweeps: `{"resolution_sweep": [ {...}, ... ]}`.

---

## Interpreting results (historical reference)

The old pipeline (`tmp/hedonic-overlapping/results/RESULTADOS.md`) reported, among other things:

| Setting (1-hop, γ=0.1) | Approx. mean ΔF1 (hedonic − Leiden) |
|------------------------|--------------------------------------|
| Random n=200 / n=1000 | ~+0.11 (older API); later v1/v2 split after C API rewrite |
| Real overlap filter | ~+0.12 |
| Overlap ratio ≥ 0.2 | ~+0.12 |
| 2-hop ablation | ~−0.02 |

**Do not expect bit-identical numbers** after the migration to `lucas-igraph` + `Game.community_hedonic`. What should match is:

- experimental design (sampling, L-hop, filters, metrics),
- method roles (Leiden baseline vs overlapping local-moving / multi-phase),
- defaults for **equilibrium** (`n_iterations=-1`) and **K = |GT|**.

RNG: community sampling uses `default_rng(42)`; Leiden’s internal node order is not fully seed-locked, so F1 can drift by small amounts run-to-run.

---

## Relation to the old `tmp/hedonic-overlapping` pipeline

| Old (`tmp/hedonic-overlapping`) | Current CLI |
|---------------------------------|-------------|
| `OverlappingGame` + `community_leiden_overlapping` | `Game.community_hedonic(max_memberships=K)` |
| `scripts/test_small.py` | `hedonic-exp overlapping-small` |
| `scripts/subgraph_experiment.py` | `hedonic-exp overlapping-subgraph` |
| `scripts/dblp_experiment.py` | `hedonic-exp overlapping-full` |
| Default `n_iterations=5` | Default **`n_iterations=-1`** |
| Default `max_memberships=4` | Default **`max_memberships = n_GT`** (per subgraph / full cover) |
| Hardcoded `./data/dblp` | `HEDONIC_DBLP_DIR` / `--data_dir` |
| Pure-Python v2 intensity model | Multi-phase overlapping via C `community_leiden` (`hedonic_v2`) |

See also the consolidation notes in [`experiments_report.md`](experiments_report.md).

---

## Smoke / CI checks

```bash
# No databases
hedonic-exp smoke
hedonic-exp overlapping-small

# Help surface
hedonic-exp overlapping-subgraph --help
hedonic-exp overlapping-full --help

# Unit tests
.venv/bin/python -m unittest tests.test_overlapping_and_experiments -v
```

---

## Checklist

- [ ] `uv sync --extra experiments`
- [ ] `HEDONIC_DBLP_DIR` points at a tree with `dblp.pkl`, `pkl/`, or SNAP `raw/*.gz`
- [ ] Smoke: `hedonic-exp overlapping-small`
- [ ] Tiny subgraph: `--n_communities 3` writes JSON + `_covers.pkl`
- [ ] Paper runs use **`--n_iterations -1`** (default) and auto **`max_memberships`**
- [ ] Optional: re-run density suite with `-1` (not archived `n_iterations=5` numbers)
- [ ] Optional: full-graph / resolution sweep only if you have the compute budget

---

## See also

- Package layout and agent rules: [`AGENTS.md`](../AGENTS.md)
- Disjoint synthetic reproduction: [`reproduce_disjoint.md`](reproduce_disjoint.md)
- Historical consolidation notes: [`experiments_report.md`](experiments_report.md)
- Old overlapping README / results: `tmp/hedonic-overlapping/README.md`, `tmp/hedonic-overlapping/results/RESULTADOS.md`
- CLI source: `src/hedonic/experiments/CLI.py`
- Subgraph experiment: `src/hedonic/experiments/overlapping/dblp_subgraph.py`
- Full-graph experiment: `src/hedonic/experiments/overlapping/dblp_full.py`
- Metrics: `src/hedonic/experiments/overlapping/metrics.py`
