# SNAP archive inventory

Root: `~/Databases/Hedonic/Networks` (read-only). Normalized loader cache:
`~/Databases/Hedonic/cache/snap`. Pair pickles (`hedonic pickle`):
`~/Databases/Hedonic/Pickles`. Regenerate the numbers below with
`hedonic show covers` and `hedonic show table`; they were read on 2026-10-18.

## Pairs

The whole graph is the SNAP header. `G[S]` is the subgraph induced by the
vertices the cover names (`covered_induced_dataset`, `--induce covered`).
Communities are counted as SNAP lists them. The loader merges identical bodies,
so it can return fewer: Amazon `top5000` comes back as 1,517 distinct communities.

| Network | whole graph n / m | cover (communities) | G[S] n (share) / m | loads as |
|---|---|---|---|---|
| DBLP | 317,080 / 1,049,866 | all (13,477) | 260,998 (82.3%) / 950,059 | archive pickle (`DBLP/pkl/`) |
| | | top5000 | 93,432 (29.5%) / 335,520 | |
| Amazon | 334,863 / 925,872 | all (75,149; `all.dedup` file) | 317,194 (94.7%) / 872,935 | archive pickle |
| | | top5000 | 16,716 (5.0%) / 48,739 | |
| YouTube | 1,134,890 / 2,987,624 | all (16,386) | 52,675 (4.6%) / 318,432 | archive pickle (`Youtube/`) |
| | | top5000 | 39,841 (3.5%) / 224,235 | |
| LiveJournal | 3,997,962 / 34,681,189 | all (664,414) | 1,147,948 (28.7%) / 16,899,734 | archive pickle (`com-lj.*`, ~0.4 GB) |
| | | top5000 | 84,438 (2.1%) / 1,521,988 | |
| Orkut | 3,072,441 / 117,185,083 | all (15,301,901) | 2,322,299 (75.6%) / 94,765,983 | streamed `txt.gz` |
| | | top5000 | 731,514 (23.8%) / 21,992,510 | |
| Friendster | 65,608,366 / 1,806,067,135 | all (1,620,991) | 7,944,949 (12.1%) / 447,219,610 | streamed `txt.gz` (~9 GB) |
| | | top5000 | 220,015 (0.3%) / 4,031,793 | |
| Wikipedia | directed `wiki-topcats` | all (categories) only | — | archive pickle; project with `common_undirected_simple_v1` |

The paper uses the eight DBLP/Amazon/YouTube/LiveJournal pairs, plus
Friendster `top5000` in Table 1. Orkut and the Friendster `all` cover appear
only as rows of Table 1.

## ID pitfalls the loader handles

- The DBLP graph pickle keeps the SNAP IDs in `vs["label"]`; vertex indices are
  not SNAP IDs.
- The Amazon graph pickle has 548,552 vertices (the SNAP ID range) against the
  header's 334,863. The cover-induced counts above are the same either way.
- The YouTube and LiveJournal pickles also exceed their headers
  (1,157,828 and 4,036,538 vertices).
- Old pickles name `hedonic.game.HedonicGame`; the loader aliases it to
  `igraph.Graph`.
- Each load's `report` records the mapping, dropped IDs and the source. A
  failed legacy load is not evidence that the raw file is corrupt.

## Not SNAP ground truth

| Folder | What it is |
|---|---|
| `email-Eu-core/` | 1,005 / 25,571 undirected graph, one department label per vertex (disjoint) |
| `Cora/cora/` | citation graph, 2,708 papers, 7 classes (`cora.content`) |
| `pubmed-diabetes/` | 19,717 papers, 44,338 links, 3 classes; `NODE.paper.tab` starts with schema rows |
| `DBLP_CLI/`, `DBLP/results/`, `DBLP/resolution_spectra/` | outputs of old experiments, never input |
| `*.cmty.csv` beside a cover | an older export of the cover, not read by the loader |
