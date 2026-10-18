---
name: snap-networks
description: Load, inspect and benchmark the SNAP ground-truth networks (DBLP, Amazon, YouTube, LiveJournal, Friendster, Orkut, Wikipedia) with hedonic 2 — network:cover pairs, ID alignment, the covered-induced subgraph, overlapping detection with Game.community_hedonic, scoring with hedonic.metrics, and the `hedonic run exp`/`show`/`pickle` front door. Use for any task that reads ~/Databases/Hedonic/Networks or scores a cover against a SNAP community file.
---

# SNAP networks (hedonic 2)

The SNAP `com-*` networks ship a graph and two overlapping ground-truth covers
each: `all` (every community) and `top5000` (the 5,000 best, label `top5k`).
Wikipedia has only `all`. Everything here goes through the package's own
loader so IDs, caches and provenance are the ones every experiment uses.

## Ground rules

- **Read-only data.** `~/Databases/Hedonic/Networks` (`HEDONIC_NETWORKS_DIR`,
  user config `network_root`) is the archive. Never write, move or delete
  there. Write our own outputs to a new folder (scratchpad, `artifacts/`, or
  `~/Databases/Hedonic/Overlap/<new-name>`). Copy, never move, other people's data.
- **No downloads unless asked.** `hedonic show covers` lists what is local.
  Nothing is fetched without `--provision` or an explicitly named pair.
- **Heavy loads are guarded and one at a time.** LiveJournal, Orkut, Friendster
  and Wikipedia take GBs. Run them under
  `.agents/skills/hedonic-native-core/scripts/guard.sh LIMIT_GB CMD` and never
  next to a running queue. Prefer a `top5000` pair or a pickle first.
- **Pickles are code.** Load `.pkl` only from this archive or from
  `hedonic pickle`; never one that was downloaded.
- **Use the pinned venv** (`uv run --no-sync ...`), and check
  `runtime_versions()` on a record before trusting it.

## Which pair to use

`hedonic show covers` prints the 13 local pairs and how each loads
(`trusted_archive_pickle` or `streamed_raw_gzip`). `hedonic show networks`
gives sizes, `hedonic show table` the full and covered-subgraph statistics
(Table 1). The paper's eight pairs are DBLP, Amazon, YouTube and LiveJournal ×
`top5000`, then × `all`, in that order. Sizes are in
[references/network-inventory.md](references/network-inventory.md).

## Load a pair

```python
from hedonic.experiments.overlapping import snap
from hedonic.experiments.overlapping.ground_truth_data import covered_induced_dataset

ds = snap.load_snap_dataset("dblp", cover_variant="all")   # the default variant is "top5000"
ds.graph, ds.cover, ds.report        # igraph.Graph on 0..n-1, cover as vertex-index lists, provenance
sub = covered_induced_dataset(ds)    # G[S]: the subgraph induced by the vertices the cover names
```

- `snap.split_pair("dblp:top5k")` → `("dblp", "top5000")`; a bare `dblp` is
  `all`. `snap.pair_files(...)` says which files a pair resolves to.
- **IDs are aligned by the loader.** SNAP IDs are not vertex indices: the DBLP
  pickle keeps them in `vs["label"]`, the Amazon pickle has 548,552 vertices
  against SNAP's 334,863 header, raw files are renumbered. Never index a raw
  community file by its IDs yourself. `ds.report` records the mapping and the
  dropped or invalid IDs; check them when a count looks wrong.
- The loader reads a validated normalized cache in
  `~/Databases/Hedonic/cache/snap` (`HEDONIC_SNAP_CACHE_DIR`), then the archive
  pickles, then streams the `*.txt.gz`. Old pickles of `hedonic.game.HedonicGame`
  load through an alias. Friendster is streamed, never held as a pickle of the
  whole graph.
- Wikipedia is directed in the archive. Benchmarks project it with
  `snap.common_undirected_analysis_dataset` (policy
  `common_undirected_simple_v1`); do the same before computing a density.
- **Pickled pairs.** `hedonic pickle dblp:top5000 ...` (or `--all`, the 11 pairs
  of Table 1) writes `<network>-<cover>.game.pkl` under `pickle_dir`
  (default `~/Databases/Hedonic/Pickles`). `snap_pickle.load_game(path)` returns
  the covered-induced `Game` with `g["cover"]`, `g["network"]`,
  `g["cover_variant"]` and `g["report"]`. This is the fast way to get a big pair.

## Detect

```python
from hedonic import Game

g = Game(sub.graph)
cover = g.community_hedonic(max_memberships=8, resolution=0.1462, seed=0)  # VertexCover
part = g.community_hedonic(seed=0)                                          # max_memberships=1: VertexClustering
```

- Always runs to equilibrium. hedonic 2 has **no** `n_iterations`,
  `local_move_only`, `beta`, `debug_trace` or `phase_policy`; they raise
  `TypeError`. Old notes that pass them are stale.
- The default resolution is the graph's density. A density multiple does not
  transfer between a whole graph and its denser covered subgraph: use an
  absolute `resolution`, ideally the calibrated one (`hoc_tuned.json` of the
  `hedonic-calibration` run, `settings.hedonic["dblp:top5000"]`).
- `max_memberships` is the per-vertex cap (`-1`: no practical cap). A cap taken
  from the reference cover's multiplicity is supervised: label it so.
- `initial_membership` accepts flat labels or per-vertex lists (a ground-truth
  start); `seed` replays the run on a given build.

## Score

```python
from hedonic.metrics import evaluate_cover, partition_to_cover_lists, overlapping_normalized_mutual_information_lfk

pred = [list(c) for c in cover]
scores = evaluate_cover(pred, sub.cover, g.vcount(), compute_omega=True, omega_sample_size=100_000)
f1 = g.evaluate_against(sub.cover, "f1")      # also jaccard, one_to_one_f1, omega, node_micro_f1, size_weighted_f1
```

- Omega is sampled (`omega_sample_size` vertex pairs, `omega_seed`), never a
  dense pair matrix. ONMI is the LFK variant.
- Scoring on the whole graph against a partial cover is a different protocol:
  either detect on `G[S]` (the paper's `--induce covered`) or pass
  `restrict_to_reference=True`. Say which.
- Robustness of a cover is exact, not sampled: `g.stable_intervals(memberships,
  max_memberships=..., allow_isolation=...)` gives each vertex's stable
  resolution intervals and `g.regrets(resolutions, memberships)` the regrets.
  Native termination is not by itself a proof; audit when a certificate is claimed.

## Run it as an experiment

Prefer the durable front door to an ad-hoc script; it reuses the runner,
records, seals and metrics of the paper.

```bash
hedonic run exp --networks dblp:top5000 --methods hedonic,codeseg --induce covered --full --seeds 0-4 --save-covers --dry-run
hedonic run exp --networks dblp:top5000,amazon:top5000 --methods all --induce covered --full --seeds 0-9 \
  --robustness --tuned PATH/hoc_tuned.json,PATH/baseline_tuned.json --name my-bench
hedonic run attach my-bench          # Ctrl-C closes the view only
hedonic pickle run dblp:top5000 --tuned PATH/hoc_tuned.json   # HOC-local per pickle: time, memory, F1, ONMI
```

`hedonic show methods` lists the nine methods (hedonic, codeseg, fox,
clique_percolation, angel, bigclam, ego_splitting, ncgame, neo_kmeans) and which
runtimes are ready; `hedonic-exp codeseg-setup` / `codeseg-doctor` provision
and check them. The paper's whole programme is `hedonic run queue
configs/queue-paper.toml` (`--dry-run`, `--report`).

## Raw inspection

`scripts/inspect_snap_network.py` reads filenames, SNAP headers and a few rows
without loading graphs (bounded, read-only); `--pairs` adds the package's view
of each pair; `--pickles` loads trusted pickles (slow and memory-heavy for the
big networks).

```bash
uv run --no-sync python .agents/skills/snap-networks/scripts/inspect_snap_network.py --dataset DBLP
uv run --no-sync python .agents/skills/snap-networks/scripts/inspect_snap_network.py --pairs --format json
```

## Not ground truth

`email-Eu-core` (departments), `Cora` (7 classes) and `pubmed-diabetes`
(3 classes) carry single-label metadata: partitions, and only when the task
says so. `DBLP_CLI/`, `DBLP/results/` and `DBLP/resolution_spectra/` are old
experiment outputs, never input.
