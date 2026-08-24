# Observed SNAP archive

The inspected root was `~/Databases/Hedonic/Networks` (about 2.5 GB at inspection time). Use that home-relative path or an explicit user path in reusable code; do not hard-code a developer’s home directory.

## Ground-truth classification

| Network | Ground-truth type in the saved archive | Files |
|---|---|---|
| Amazon | **Overlapping** community cover | `com-amazon.all.dedup.cmty.txt.gz/.pkl`, `top5000.cmty.txt.gz/.pkl` |
| DBLP | **Overlapping** co-authorship community cover | `com-dblp.all.cmty.txt.gz/.pkl`, `top5000.cmty.txt.gz/.pkl` |
| LiveJournal | **Overlapping** social-community cover | `com-lj.all.cmty.txt.gz/.pkl`, `top5000.cmty.txt.gz/.pkl` |
| YouTube | **Overlapping** channel/community cover | `com-youtube.all.cmty.txt.gz/.pkl`, `top5000.cmty.txt.gz/.pkl` |
| Wikipedia | **Overlapping** category cover | `wiki-topcats-categories.txt.gz/.pkl` |
| email-Eu-core | **Disjoint** department labels, one pair per node | `email-Eu-core-department-labels.txt.gz/.pkl` |
| Cora | **Disjoint** paper class labels (seven classes) | `cora.content` |
| PubMed Diabetes | **Disjoint** paper class labels (three classes) | `Pubmed-Diabetes.NODE.paper.tab` |
| DBLP_CLI | **Absent**; contains prior experiment outputs only | `runs/*.json`, summaries, plots, logs |

The first five datasets can be evaluated directly as node covers. The next three are single-label classification metadata and should be evaluated as partitions only if the task explicitly treats those labels as ground truth. `DBLP_CLI` is never a source of ground truth.

## Overlapping or category-cover datasets

| Directory | Raw graph / cache | Ground-truth or metadata | Observed cache facts |
|---|---|---|---|
| `Amazon/` | `com-amazon.ungraph.txt.gz`; `com-amazon.ungraph.pkl` | `com-amazon.all.dedup.cmty.txt.gz/.pkl`, `top5000.cmty.txt.gz/.pkl` | graph cache `n=548,552`, `m=925,872`, undirected; all cover 75,149 communities / 317,194 unique nodes; top cover 5,000 / 16,716 nodes; community IDs reach 548,551 |
| `DBLP/raw/` + `DBLP/pkl/` | `com-dblp.ungraph.txt.gz`; `pkl/com-dblp.ungraph.pkl` | `com-dblp.all.cmty.txt.gz/.pkl`, `top5000.cmty.txt.gz/.pkl` | graph cache `n=317,080`, `m=1,049,866`, undirected, with vertex attribute `label`; all cover 13,477 communities / 719,820 memberships; top cover 5,000 / 93,432 nodes; labels reach 425,956 |
| `LiveJournal/` | `com-lj.ungraph.txt.gz`; `com-lj.ungraph.pkl` | `com-lj.all.cmty.txt.gz/.pkl`, `top5000.cmty.txt.gz/.pkl` | graph cache `n=4,036,538`, `m=34,681,189`, undirected; all cover 664,414 communities / 1,147,948 nodes; top cover 5,000 / 84,438 nodes; community IDs reach 4,036,471 |
| `Youtube/` | `com-youtube.ungraph.txt.gz`; `com-youtube.ungraph.pkl` | `com-youtube.all.cmty.txt.gz/.pkl`, `top5000.cmty.txt.gz/.pkl` | graph cache `n=1,157,828`, `m=2,987,624`, undirected; all cover 16,386 communities / 52,675 nodes; top cover 5,000 / 39,841 nodes; community IDs reach 663,521 |
| `Wikipedia/` | `wiki-topcats.txt.gz`; `wiki-topcats.pkl` | `wiki-topcats-categories.txt.gz/.pkl`, `wiki-topcats-page-names.txt.gz` | graph cache `n=1,791,489`, `m=28,511,807`, directed; 17,364 category lists; category lines begin `Category:name; node IDs` |

The SNAP headers read from the raw `ungraph` files report approximately 334,863/925,872 for Amazon, 317,080/1,049,866 for DBLP, 3,997,962/34,681,189 for LiveJournal, and 1,134,890/2,987,624 for YouTube. The saved pickle counts do not always equal those headers. Treat this as a provenance/ID-validation signal, not as a reason to overwrite the archive.

Every inspected `*.cmty` pickle was a list of lists with no duplicate member inside an individual community and no singleton communities in the sampled summary. Community sizes are highly skewed, so avoid assuming a fixed community size. The `top5000` file is a selected cover, not the first 5,000 rows of an arbitrary file unless its filename says so.

## Metadata-only or single-label datasets

- `email-Eu-core/`: `email-Eu-core.txt.gz` and `email-Eu-core.pkl` are an undirected graph with `n=1,005`, `m=25,571`; department labels are in `email-Eu-core-department-labels.txt.gz/.pkl`, with one `[node, department]` pair per node. These are disjoint labels unless explicitly converted into a cover.
- `Cora/cora/`: `cora.cites` is a citation edge list and `cora.content` contains paper ID, 1,433 binary word attributes, and one of seven class labels. The bundled README says there are 2,708 papers.
- `pubmed-diabetes/`: tab-delimited citation/feature files; the README says 19,717 publications, 44,338 citation links, 500 word features, and three classes. The `NODE.paper.tab` file starts with schema rows before data rows.

## Cache compatibility notes

Several graph pickles were serialized with a legacy class path (`hedonic.game.HedonicGame`). In the current repository, trusted-cache inspection can require an alias to the current `hedonic.Game` module and `Game` class before `pickle.load`. A failed legacy load is not evidence that the raw file is corrupt.

The DBLP graph pickle has no graph-level attributes but has `g.vs["label"]`; its vertex indices are not the SNAP IDs in the community pickle. Build `id_to_index = {int(label): i for i, label in enumerate(g.vs["label"])}` and remap all communities before calling `Game` or metrics. Check this invariant for every cache rather than copying the DBLP assumption to other datasets.

`DBLP_CLI/` contains prior resolution runs: per-resolution/seed JSON under `runs/`, summary JSON, PNGs, and logs. These are outputs for offline re-scoring and reproducibility checks, not graph input.
