---
title: Disjoint and overlapping communities
summary: Partitions and covers — which one this release detects, and which one this explainer covers.
group: Python API
order: 5
---

## Partitions and covers

In a **partition** every vertex belongs to exactly one community. In a **cover** a vertex may belong to several communities at once, and the hedonic game then needs its own utility for a vertex that splits its participation between them.

| Output | Call | Returns |
| --- | --- | --- |
| Disjoint partition | `max_memberships=1` (the default) | `igraph.VertexClustering` |
| Overlapping cover | — | hedonic 2 |

This release (hedonic 1) detects partitions only: `community_hedonic` refuses any other `max_memberships` with a message that says overlapping covers are hedonic 2. A start state may still give a vertex several labels; each such vertex keeps the one it prefers before the game runs ([community_hedonic](docs:#community-hedonic)).

## What the explainer covers

The explainer, its decision tree, the Familiarity Index and the metagraph all describe the **disjoint** model of the SRC abstract and of arXiv:2509.03834: every agent holds exactly one community label. That is also the model this release's C core plays.

## Where to go next

- Every parameter, including the community-count limits, is on the [community_hedonic](docs:#community-hedonic) page.
- Score a partition against a reference with `Game.evaluate_against` or `hedonic.metrics` ([The Game API](docs:#game-api)); the reference may be a partition or a list of communities.
