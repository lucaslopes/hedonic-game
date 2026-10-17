---
title: Disjoint and overlapping communities
summary: One API, two kinds of output — and where the explainer covers each.
group: Python API
order: 5
---

## One call pattern

| Goal | Call | Returns |
| --- | --- | --- |
| Disjoint partition | `max_memberships=1` (default) | `igraph.VertexClustering` |
| Overlapping cover | `max_memberships=K` with `K > 1` | `igraph.VertexCover` |

In a **partition** every vertex belongs to exactly one community. In a **cover** a vertex may belong to up to `K` communities. `initial_membership` accepts flat labels for either mode; for overlapping runs it also accepts one list of labels per vertex, and flat labels are expanded to singleton rows.

## What the explainer covers

The main explainer, its decision tree, the Familiarity Index and the metagraph describe the **disjoint** model of the SRC abstract and of arXiv:2509.03834: every agent holds exactly one community label. The overlapping mode (`max_memberships > 1`) runs in the same C core with its own utility; the [overlap explainer](story:overlap/) follows it on a six-vertex graph, and [The overlapping game](docs:#overlapping-game) gives its rules.

## Where to go next

- Every parameter of both modes, including the per-vertex cap and the community-count limits, is on the [community_hedonic](docs:#community-hedonic) page.
- Score a partition or a cover against a reference with `Game.evaluate_against` or `hedonic.metrics` ([The Game API](docs:#game-api)).
- Practical guidance for overlapping runs (membership caps, warm starts, recording seeds) is in the repository `README.md`.
