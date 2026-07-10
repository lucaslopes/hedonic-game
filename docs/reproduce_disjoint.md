# Reproducing the Disjoint Synthetic Experiments (PHYSA V1020)

This guide explains how to **reproduce the disjoint community-detection experiments** on synthetic SBM networks using the current `hedonic` library and the `hedonic-exp` CLI.

The reference archive produced by the old `tmp/hedonic` pipeline lives at:

```text
~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020
```

**Do not write into that folder.** Treat it as read-only. Write all new runs under a separate root, for example:

```text
~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI
```

---

## What “full reproduction” means

A complete run reproduces three artifacts from the paper pipeline:

| Stage | Artifact | CLI command |
|-------|----------|-------------|
| 1. Sweep | JSON results under `resultados/` | `hedonic-exp disjoint` |
| 2. Aggregate | Combined gzipped CSV | `hedonic-exp disjoint-load` |
| 3. Figures | Five paper PDFs | `hedonic-exp plots` |

Or all three in one shot:

```bash
hedonic-exp reproduce-disjoint --preset v1020 \
  --output_root ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI
```

---

## Prerequisites

```bash
# From the hedonic-game repo root
uv sync --extra experiments
```

Optional dependencies used here: `pandas`, `scipy`, `tqdm`, `stopwatch-py`, `matplotlib`, `seaborn`.

Confirm the CLI:

```bash
hedonic-exp list
hedonic-exp disjoint --help
hedonic-exp plots --help
hedonic-exp reproduce-disjoint --help
```

You can also run without installing the console script:

```bash
python -m hedonic.experiments.CLI list
```

### Paths

| Variable | Env override | Default |
|----------|--------------|---------|
| Synthetic root | `HEDONIC_SYNTHETIC_DIR` | `…/Synthetic_Networks/V1020` |

Always pass `--output_root` to a **new** directory for reproduction so the archived V1020 tree is never overwritten. The CLI refuses `--preset v1020` and figure writes that target the archived folder.

---

## Parameter grid (archived V1020)

| Parameter | Values |
|-----------|--------|
| `max_n_nodes` | `1020` |
| `n_communities` | `2, 3, 4, 5, 6` → community sizes `510, 340, 255, 204, 170` |
| `p_in` | `0.01 … 0.10` (step `0.01`) |
| `difficulty` (λ) | `0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75` |
| `noise` | `0.10, 0.25, 0.50, 0.75, 1.00` |
| Network seeds | `0 … 4` (5 graphs) |
| Partition seeds | `0 … 9` (10 noisy initial memberships) |
| Methods | `GroundTruth`, `Mirror`, `OnePass`, `Spectral`, `Leiden`, `Hedonic` |
| Leiden / Hedonic | `n_runs=10` stochastic restarts (unique partitions kept) |
| Resolution | graph edge density (CPM γ) for Leiden / Hedonic |

This grid is encoded as **`--preset v1020`**. A tiny structural clone for smoke tests is **`--preset v1020-smoke`**.

---

## Methods

All methods are run through the current library (`Game` / `community_hedonic` where applicable):

| Name | Role |
|------|------|
| **GroundTruth** | SBM block labels (baseline metrics) |
| **Mirror** | Identity on the noisy initial membership |
| **OnePass** | One-pass neighbour-majority improvement |
| **Spectral** | Leading eigenvector (`clusters = n_communities`) |
| **Leiden** | Full Leiden CPM (`only_local_moving=False`) |
| **Hedonic** | Local-moving phase only (`only_local_moving=True`) |

Hedonic and Leiden both call `Game.community_hedonic(..., max_memberships=1)` with density as resolution.

---

## Quick start (smoke — recommended first)

Structural check: same layout, methods, and figure stems as V1020, on a tiny graph.

```bash
export OUT=~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI

# One-shot: sweep → CSV → figures
hedonic-exp reproduce-disjoint --preset v1020-smoke \
  --output_root "$OUT" \
  --format pdf
```

Expected under `$OUT`:

```text
V1020_CLI/
├── resultados/
│   └── 2C_20N/
│       └── Noise = …/P_in = …/Difficulty = …/Network (000)/
│           ├── partition_000.json
│           └── partition_001.json
├── resultados.csv.gzip
└── figures/
    ├── gt_robustness.pdf
    ├── noise.pdf
    ├── n_communities.pdf
    ├── acc_robustness.pdf
    └── acc_efficiency.pdf
```

Each `partition_*.json` is a **list** of result dicts (one or more rows per method; Leiden/Hedonic may contribute multiple unique runs).

---

## Full V1020 reproduction

### Option A — one command

```bash
export OUT=~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI

hedonic-exp reproduce-disjoint --preset v1020 \
  --output_root "$OUT" \
  --format pdf
```

This is a **large** run (~125 000 noise×partition×network×p_in×difficulty×n_communities cells, times methods and stochastic restarts). Plan disk, time, and RAM accordingly.

### Option B — step by step

```bash
export OUT=~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI

# 1) SBM sweep
hedonic-exp disjoint --preset v1020 --output_root "$OUT"

# 2) Combine JSON → CSV
hedonic-exp disjoint-load \
  --results_folder "$OUT/resultados" \
  --output "$OUT/resultados.csv.gzip" \
  --simple

# 3) Paper figures
hedonic-exp plots \
  --data "$OUT/resultados.csv.gzip" \
  --output_dir "$OUT/figures" \
  --format pdf
```

### Custom grid (no preset)

Any subset of the archived design can be specified explicitly:

```bash
hedonic-exp disjoint \
  --folder_name resultados \
  --layout v1020 \
  --max_n_nodes 1020 \
  --n_communities 2 3 4 5 6 \
  --seeds 0 1 2 3 4 \
  --p_in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 \
  --difficulty 0.10 0.20 0.30 0.40 0.50 0.55 0.60 0.65 0.70 0.75 \
  --noises 0.10 0.25 0.50 0.75 1.00 \
  --partition_seeds 10 \
  --n_runs 10 \
  --methods GroundTruth,Mirror,OnePass,Spectral,Leiden,Hedonic \
  --output_root "$OUT"
```

---

## Output layout (V1020-compatible)

```text
{output_root}/
  resultados/
    {n}C_{size}N/
      Noise = {noise:.2f}/
        P_in = {p_in:.2f}/
          Difficulty = {difficulty:.2f}/
            Network ({seed:03d})/
              partition_{partition_seed:03d}.json
  resultados.csv.gzip          # after disjoint-load
  figures/                     # after plots
  persist/                     # optional figure intermediates
```

### JSON record fields

Each method result includes:

```text
method, number_of_communities, community_size,
p_in, p_out, multiplier, resolution,
duration, accuracy, robustness,
noise, network_seed, partition_seed, partition
```

`accuracy` is the **adjusted Rand index** vs ground truth (same score used in the paper pipeline). The archived `resultados_ari.csv.gzip` also has an explicit `adjusted_rand` column; the plotter accepts either.

### Legacy layout

For debugging, `--layout legacy` writes one `{Method}.json` per method under:

```text
…/{n} Communities of {size} nodes/…/Partition (MMM)/Method.json
```

Presets `v1020` / `v1020-smoke` use `--layout v1020`.

---

## Paper figures

The five stems match archived `V1020/figures/`:

| File | Content |
|------|---------|
| `gt_robustness` | Ground-truth robustness histograms + heatmaps |
| `noise` | Duration / robustness / ARI vs noise (bars + CI) |
| `n_communities` | Same metrics vs #communities (all data vs noise≈1) |
| `acc_robustness` | KDE: ARI vs robustness (all / noisy) |
| `acc_efficiency` | KDE: ARI vs duration (all / noisy) |

### Figures only (from existing CSV)

From a new CLI CSV:

```bash
hedonic-exp plots \
  --data "$OUT/resultados.csv.gzip" \
  --output_dir "$OUT/figures" \
  --format pdf
```

From the **archived** table (read-only), writing figures elsewhere:

```bash
hedonic-exp reproduce-disjoint --plots-only \
  --data ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/resultados_ari.csv.gzip \
  --output_root "$OUT" \
  --format pdf
```

For a quick plot check on the full archive without loading every row:

```bash
hedonic-exp plots \
  --data …/V1020/resultados_ari.csv.gzip \
  --output_dir "$OUT/figures" \
  --max_rows 50000 \
  --no-persist
```

Synthetic figure smoke (no CSV required):

```bash
hedonic-exp plots --smoke --output_dir /tmp/hedonic-figs --format png --no-persist
```

---

## Relation to the old `tmp/hedonic` pipeline

| Old (`tmp/hedonic`) | Current CLI |
|---------------------|-------------|
| Pre-generated `networks/*.pkl` + `memberships/*.csv` | Graphs and noisy memberships generated on the fly |
| `scripts/exp.py` per network pickle | `hedonic-exp disjoint` / `reproduce-disjoint` |
| `community_leiden(only_first_phase=…)` | `community_hedonic(only_local_moving=…)` |
| networkx SBM → igraph | igraph SBM (`Game`) |
| `scripts/plot/paper_plots/plot_figures.py` | `hedonic-exp plots` |
| Hardcoded paths under `~/Databases/…` | `--output_root` + `HEDONIC_SYNTHETIC_DIR` |

Partitions need not be **bit-identical** to the 2020 archive (different SBM RNG path). The **experimental design, methods, metrics schema, directory layout, and figure stems** are what the CLI reproduces.

---

## Safety guards

- `--preset v1020` with `--output_root` equal to the archived V1020 path → **refused**
- `plots` with `--output_dir` equal to archived `V1020/figures` → **refused**
- `reproduce-disjoint` with archived V1020 as `--output_root` → **refused**

Always use `V1020_CLI` (or another new root) for writes.

---

## Smoke / CI checks

```bash
# Isolated library smoke (small graphs + tiny SBM, temp dir)
hedonic-exp smoke

# Disjoint structural smoke only
hedonic-exp disjoint --preset v1020-smoke --output_root /tmp/v1020-smoke

# Unit tests (includes v1020-smoke, plots smoke, reproduce-disjoint smoke)
.venv/bin/python -m unittest tests.test_overlapping_and_experiments -v
```

---

## Checklist

- [ ] `uv sync --extra experiments`
- [ ] Choose a **new** `--output_root` (not archived V1020)
- [ ] Smoke: `reproduce-disjoint --preset v1020-smoke`
- [ ] Full grid (optional): `reproduce-disjoint --preset v1020`
- [ ] Confirm `resultados.csv.gzip` columns match archive schema
- [ ] Confirm `figures/` has the five PDF stems
- [ ] Optional: replot from archived `resultados_ari.csv.gzip` with `--plots-only` for visual comparison

---

## See also

- Package layout and agent rules: [`AGENTS.md`](../AGENTS.md)
- Historical consolidation notes: [`experiments_report.md`](experiments_report.md)
- CLI source: `src/hedonic/experiments/CLI.py`
- Sweep / presets: `src/hedonic/experiments/disjoint/sbm_sweep.py`
- Paper figures: `src/hedonic/experiments/plots/paper_figures.py`
- End-to-end driver: `src/hedonic/experiments/disjoint/reproduce.py`
