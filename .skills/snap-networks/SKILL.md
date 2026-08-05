---
name: snap-networks
description: Classify, inspect, load, align, and evaluate SNAP graph datasets by whether their available ground truth is overlapping, disjoint, metadata-only, or absent, then run overlapping community detection in the hedonic-game repository. Use when working with Amazon, DBLP, LiveJournal, YouTube, Wikipedia, email-Eu-core, Cora, PubMed, or similar SNAP-style edge/community files under ~/Databases/Hedonic/Networks; when community IDs need remapping; or when comparing disjoint and overlapping `Game.community_hedonic` results with F1, Jaccard, Omega, quality, and structural diagnostics.
---

# SNAP Networks

Use this skill to move from a saved SNAP archive to a reproducible, ID-safe overlapping community experiment. Keep the large archive read-only unless the user explicitly asks for new caches or results.

## Start with a bounded inspection

Resolve the root from the user, `HEDONIC_NETWORKS_DIR` (or legacy `HEDONIC_DBLP_DIR` for DBLP-focused tools), or the default `~/Databases/Hedonic/Networks`. Do not assume every directory is an overlapping benchmark: first identify whether it contains an edge list, a community cover, labels, attributes, or only prior experiment output.

Run the bundled inspector before loading a large graph:

```bash
python .skills/snap-networks/scripts/inspect_snap_network.py \
  ~/Databases/Hedonic/Networks --dataset DBLP

# Load trusted local pickle caches and inspect graph/cover statistics too.
python .skills/snap-networks/scripts/inspect_snap_network.py \
  ~/Databases/Hedonic/Networks --dataset DBLP --pickles --format json
```

Read [references/network-inventory.md](references/network-inventory.md) for the observed archive layout and known cache/ID caveats. Treat pickle files as executable serialized objects: only inspect them when they are trusted local research artifacts.

## Classify the dataset and ground truth

Use this taxonomy before choosing an evaluation metric:

| Network | Available ground truth | Evaluation interpretation |
|---|---|---|
| Amazon | **Overlapping** product/community cover (`all` or `top5000`) | Evaluate as a node cover; memberships may repeat across communities. |
| DBLP | **Overlapping** co-authorship community cover (`all` or `top5000`) | Primary saved benchmark for the repository’s overlapping experiments. |
| LiveJournal | **Overlapping** social-community cover (`all` or `top5000`) | Evaluate as a cover; expect highly skewed community sizes. |
| YouTube | **Overlapping** channel/community cover (`all` or `top5000`) | Evaluate as a cover; do not turn it into one label per node. |
| Wikipedia | **Overlapping** category cover (`wiki-topcats-categories`) | Categories are multi-membership metadata; preserve the `Category:name; IDs` structure. |
| email-Eu-core | **Disjoint labels**: one department per node | A single-label/partition baseline, not overlapping ground truth. |
| Cora | **Disjoint labels**: one of seven paper classes | A citation graph with class labels and attributes, not an overlapping community cover. |
| PubMed Diabetes | **Disjoint labels**: one of three paper classes | A citation graph with class labels and attributes, not an overlapping community cover. |
| DBLP_CLI | **No ground truth**: prior experiment artifacts | Do not use result JSON, plots, or cached covers as network ground truth. |

“Disjoint” here means each node has one supplied label. “Overlapping” means the supplied community/category file gives a node membership cover and nodes can occur in multiple communities. A graph may be directed or undirected independently of this ground-truth classification.

- **Overlapping SNAP covers:** Amazon, DBLP, LiveJournal, and YouTube. Their `*.cmty.txt.gz` files have one community per line, with space-separated original node IDs. `all` and `top5000` are different covers, not two partitions to merge.
- **Overlapping category cover:** Wikipedia `wiki-topcats-categories.txt.gz` has `Category:name; node IDs` lines; the graph is directed in the saved cache.
- **Disjoint labels:** Email-Eu-core has one department label per node, not an overlapping community cover.
- **Attributed or single-label citation data:** Cora and PubMed provide node features/classes and directed citation edges. Use them for attributed or supervised baselines, not as overlapping ground truth without an explicit cover construction.
- **Prior results:** `DBLP_CLI/` contains JSON, logs, plots, and per-run cover caches. Do not treat these as raw SNAP input.

## Align node IDs before detection

SNAP IDs are labels, while igraph algorithms consume contiguous vertex indices. Always validate the mapping:

1. Read edge and community IDs as integers; skip blank/comment lines.
2. Build or load the graph and determine the set of IDs represented by its vertices.
3. If the graph was built from raw edges, create a deterministic map such as `{old_id: new_id}` from sorted edge IDs and remap every ground-truth community through it.
4. If a cached graph has `g.vs["label"]`, use `{int(label): vertex_index}`. This is required for the saved DBLP pickle: its communities contain original IDs and its vertices are indexed separately.
5. Reject or report communities with missing IDs instead of silently treating an original ID as an index. Deduplicate members inside each community, and normally drop communities with fewer than two in-graph nodes for detection/evaluation.
6. Check `0 <= member < g.vcount()` after remapping, and report graph size, edge count, cover size, community-size summary, and the number of dropped members/communities.

Do not infer identity from a matching-looking maximum ID. The inspected archive contains caches whose vertex counts differ from SNAP header counts, and DBLP’s cached graph has sparse original labels up to 425,956 while containing 317,080 vertices.

## Load efficiently and safely

Prefer an existing validated combined cache, then a validated graph/community pickle pair, then raw compressed SNAP files. For this repository’s DBLP experiment loader, the intended order is `data_dir/dblp.pkl`, `data_dir/pkl/*.pkl`, then `data_dir/raw/*.gz`; see `hedonic.experiments.overlapping.dblp_full.load_dblp` for the current implementation.

For raw files, stream the gzip input and avoid materializing duplicate ID sets unnecessarily. Build an undirected graph for `*.ungraph.txt.gz`; preserve direction only when the dataset is explicitly directed, such as Wikipedia or citation graphs. Store the remap alongside the graph and cover so later subgraph extraction uses the same ID space.

For large networks, inspect headers and a few data lines first. Do not decompress a multi-gigabyte edge list merely to print its first rows. Use the pickle cache for repeated experiments only after checking its vertex attributes and cover ID compatibility.

## Choose a method family deliberately

The source overview groups overlapping detection into these families:

- **Clique-based:** CPM and clique-percolation variants give interpretable cores but are sensitive to `k` and can be expensive on dense graphs.
- **Label propagation:** COPRA and SLPA scale well and expose membership diversity, but randomness and post-processing affect stability.
- **Local expansion/fitness:** OSLOM, LFM, DEMON, and Game grow communities around seeds; they are useful when local structure or statistical significance matters.
- **Link/edge clustering:** partition edges and infer overlapping node communities; this naturally creates overlap but can overproduce communities on dense graphs.
- **Matrix/spectral, ensemble, and deep methods:** NMF/BigCLAM-like soft memberships, link-graph spectral methods, ensembles, or GAT/LPA hybrids are appropriate when low-rank structure, attributes, or multiple base partitions are central.

No family dominates across overlap density, membership diversity, graph type, and graph size. For benchmarks, vary those structural regimes and report both recovery and runtime rather than selecting a method from one scalar score. In this repository, use `Game.community_hedonic` for the hedonic/Leiden comparisons and keep any external baseline isolated in the experiments layer.

## Run hedonic detection

Use the single public detector API:

```python
from hedonic import Game
from hedonic.experiments.overlapping.metrics import (
    evaluate_cover,
    partition_to_cover_lists,
)

game = Game(graph)
resolution = game.density()  # or an explicit CPM gamma

partition = game.community_hedonic(
    resolution=resolution,
    max_memberships=1,
    local_move_only=False,
    n_iterations=-1,
)

k = max(1, len(ground_truth))
cover = game.community_hedonic(
    resolution=resolution,
    max_memberships=k,
    local_move_only=True,
    n_iterations=-1,
    initial_membership=list(partition.membership),
)

metrics = evaluate_cover(
    partition_to_cover_lists(cover), ground_truth, game.vcount(),
    compute_omega=False,
)
```

Follow these defaults unless the experiment intentionally changes them:

- Use `n_iterations=-1` (or another negative value) to run to equilibrium.
- Use `max_memberships=1` for the disjoint baseline and `max_memberships=len(ground_truth)` for an overlapping run. For an L-hop subgraph, use the number of ground-truth communities with at least two nodes in that window.
- Warm-start overlapping detection from the disjoint partition when comparing method roles.
- Use `local_move_only=True` for the hedonic local-moving method and `False` for full Leiden refinement/aggregation.
- Pass `allow_isolation=True` only when empty-community moves are part of the design; record it in the output.

Prefer `hedonic-exp overlapping-benchmark` for the five-network SNAP suite;
use `overlapping-small`, `overlapping-subgraph`, `overlapping-full`,
`overlapping-scale`, and `overlapping-resolution` for focused existing
experiments. Do not resurrect `OverlappingGame` or call raw
`community_leiden` from experiment code.

## Run the saved SNAP benchmarks through the CLI

Use hedonic-exp overlapping-benchmark for the reproducible multi-network
suite. It uses ~/Databases/Hedonic/Networks by default; set
HEDONIC_NETWORKS_DIR or pass --data_root to select another saved archive.
Install the experiments extra first so the CPM and DEMON baselines are
available:

    uv sync --extra experiments
    hedonic-exp overlapping-benchmark --list-networks
    hedonic-exp overlapping-benchmark --list-methods

    # Validate graph/cover paths, ID mappings, and diagnostics without detection.
    hedonic-exp overlapping-benchmark --datasets amazon,dblp --cover top5000 --profile standard --dry-run --output_dir /tmp/hedonic-snap-dry-run

### Choose a profile and dataset/cover scope

    # No archive required: built-in overlapping fixtures for all five datasets.
    hedonic-exp overlapping-benchmark --profile smoke --output_dir /tmp/hedonic-snap-smoke

    # Safe default for saved data: deterministic GT-informed 3,000-node induced subgraphs.
    hedonic-exp overlapping-benchmark --profile standard --datasets amazon,dblp,livejournal,youtube,wikipedia --cover top5000 --output_dir ~/Databases/Hedonic/Networks/SNAP_BENCHMARK_CLI

    # Use every complete supplied cover. Wikipedia categories are included here.
    hedonic-exp overlapping-benchmark --profile standard --cover all --datasets amazon,dblp,livejournal,youtube,wikipedia --output_dir /tmp/hedonic-snap-all

    # Full graph: remove the standard cap deliberately and keep runs resumable.
    hedonic-exp overlapping-benchmark --profile full --datasets dblp --cover top5000 --timeout_per_run 1800 --resume --output_dir ~/Databases/Hedonic/Networks/SNAP_BENCHMARK_CLI/full-dblp

top5000 is a supplied cover for Amazon, DBLP, LiveJournal, and YouTube.
Wikipedia uses wiki-topcats-categories and has only --cover all; requesting
top5000 for Wikipedia writes explicit skipped run records rather than
substituting another ground truth. smoke uses the category-style all cover so
it exercises every one of the five supported datasets.

### Choose methods and detector parameters

The default profile runs all four adapters:

| Method | Family / implementation | Important behavior |
|---|---|---|
| hedonic_local | Game.community_hedonic local-moving | local_move_only=True, n_iterations=-1, max_memberships=K |
| hedonic_multiphase | Game.community_hedonic full Leiden | local_move_only=False, n_iterations=-1, max_memberships=K |
| cpm | NetworkX clique percolation | clique_size=3; clique-rich graphs can be costly |
| demon | external demon.Demon local expansion | epsilon=0.25, min_community_size=2 |

Here K is the number of usable ground-truth communities after any selected
subgraph bound. The two hedonic adapters always use the public Game API and
run to equilibrium. The CLI intentionally exposes stable benchmark method
names rather than ad-hoc per-library flags; exact baseline parameters are
recorded in every run JSON and shown by --list-methods.

    # Compare the two hedonic variants at three explicit CPM resolutions.
    hedonic-exp overlapping-benchmark --datasets amazon --cover top5000 --methods hedonic_local,hedonic_multiphase --resolutions 0:1:3 --seeds 0-4 --output_dir /tmp/amazon-hedonic-resolution

    # Compare independent overlap baselines against local hedonic detection.
    hedonic-exp overlapping-benchmark --datasets dblp --cover top5000 --methods hedonic_local,cpm,demon --seeds 0,1,2 --timeout_per_run 300 --max_nodes 5000 --output_dir /tmp/dblp-method-comparison

    # auto is the graph-density resolution; --max_nodes 0 disables the cap.
    hedonic-exp overlapping-benchmark --datasets youtube --cover all --methods cpm --resolutions auto --max_nodes 0 --timeout_per_run 600 --output_dir /tmp/youtube-cpm-full

--seeds accepts comma lists and inclusive ranges such as 0-4. --resolutions
accepts auto, comma-separated floats, or inclusive start:stop:count grids
such as 0:1:11. --timeout_per_run is enforced in an isolated detector
subprocess. --resume skips terminal completed, unavailable, and intentionally
skipped records but retries timeout/error records. Use a new, clearly named
output directory; never point it at Amazon/, DBLP/, LiveJournal/, Youtube/, or
Wikipedia/ themselves.

### Request and interpret metrics

Every completed run reports recovery, structural overlap behavior, graph
quality, and runtime separately:

- Recovery: symmetric best-match precision/recall/F1/Jaccard, one-to-one
  matching metrics, node-membership micro/macro F1, and size-weighted F1.
- Structural behavior: inclusion, coverage, overlapping, and distribution
  rates; community/covered-node counts; singleton, size, and
  memberships-per-node diagnostics.
- Graph quality: overlapping CPM quality at the run resolution where feasible.
- Scalability: detector runtime plus timeout, unavailable, or error status.

Omega is intentionally opt-in because it is pairwise and can be expensive.
The implementation samples pairs and never allocates a dense vertex-pair
matrix:

    # Recovery plus sampled Omega for a bounded standard DBLP run.
    hedonic-exp overlapping-benchmark --datasets dblp --cover top5000 --methods hedonic_local,demon --omega --omega_sample_size 100000 --output_dir /tmp/dblp-omega

    # Write result tables without plotting, useful for a parameter sweep.
    hedonic-exp overlapping-benchmark --datasets amazon --cover top5000 --resolutions 0:1:11 --seeds 0-4 --no-plots --output_dir /tmp/amazon-resolution-sweep

The output root contains manifest.json, per-run JSON under runs/,
results.jsonl, results.csv.gz, summary.json, summary.csv,
method_availability.json, logs/benchmark.log, and, unless --no-plots is used,
recovery/runtime/overlap/method-comparison plots. Use the manifest and
per-run metadata—not only a single F1 value—to compare methods fairly.

## Evaluate covers, not just partitions

Convert both predictions and ground truth to `list[list[int]]` cover form. Use `evaluate_cover` and its shared helpers for symmetric best-match F1/Jaccard/precision/recall, one-to-one matching, node-membership multilabel F1, size-weighted F1, singleton/coverage/size diagnostics, CPM quality, and optional equilibrium checks. Compute Omega only when requested; at full-network scale use the sampled option and never allocate a dense vertex-pair matrix.

For synthetic or known-cover data, report F1/Jaccard plus membership and overlap diagnostics. For noisy or partial metadata, add CPM/extended modularity-style quality, conductance or density, community-size and memberships-per-node distributions, and stability over seeds/resolution. Do not claim ONMI/ENMI unless an implementation is actually available in the selected environment; the source document lists them as literature metrics, while this repository’s shared evaluator centers on F1/Jaccard/Omega and structural diagnostics.

## Reproducibility and output discipline

Record dataset name, input paths, cover variant (`all`, `top5000`, or categories), ID mapping strategy, graph `n`/`m`/directedness, resolution, `n_iterations`, `max_memberships`, `local_move_only`, seed, runtime, and metric singleton mode. Write generated JSON/CSV/plots outside the archived input tree or under an explicitly named results directory such as `DBLP_CLI/` or `SNAP_BENCHMARK_CLI/`; never overwrite the original SNAP files or archived V1020 data.

When changing an experiment module in this repository, keep the `hedonic-exp` registry and the experiment documentation synchronized, as required by `AGENTS.md`.
