---
title: Disjoint and overlapping communities
summary: One API, two kinds of output — and which one this explainer covers.
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

The explainer, its decision tree, the Familiarity Index and the metagraph all describe the **disjoint** model of the SRC abstract and of arXiv:2509.03834: every agent holds exactly one community label. The overlapping mode lives in lucas-igraph (through `max_memberships`) and is studied separately in this repository's research code; its utility, certificates and experiments are out of scope here.

## Where to go next

- The experiment drivers for both modes are behind the `hedonic-exp` command-line tool ([Reproducibility](docs:#reproducibility)).
- Practical guidance for overlapping runs (membership caps, warm starts, recording seeds) is in the repository `README.md` and `AGENTS.md`.
