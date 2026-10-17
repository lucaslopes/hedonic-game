---
title: The metagraph of partitions
summary: How the explainer generates the 15 metanodes and 52 metaedges of the example, and how it draws them.
group: Theory
order: 8
---

Everything in the explainer's metagraph is computed in the browser from the graph's edge list (`web/src/model/`); nothing is traced from the historical Matplotlib figure. The construction follows the appendix "The Metagraph of Partitions" of arXiv:2509.03834.

## Metanodes: canonical partitions

A partition of $\{0, \dots, n-1\}$ is stored as a **restricted growth string** (RGS): vertex 0 has label 0 and every new label is one larger than the largest seen so far. Relabelling communities therefore never produces a new partition. Enumerating all RGSs yields exactly the Bell number $B_n$ of partitions — $B_4 = 15$ for the example. Each metanode records:

- its **id**, the RGS itself (e.g. `0001` for {0,1,2}{3});
- its **communities**, ordered by smallest member, and their sizes;
- whether it is the **grand coalition** or the **singleton partition**;
- its move distance to both extremes and its **layer**;
- its internal friendships and stranger pairs, from which the quality $\Phi_\gamma$ follows for any $\gamma$.

## Metaedges: single-vertex moves

From every partition, every vertex may join any other existing community, or leave to found a new one (unless it is already alone). Leaving a singleton community removes it; the canonical relabelling closes the gap. Moves are grouped by unordered pair of partitions:

- the example has **52** metaedges;
- **12** of them can be walked by two different vertices: {0,1}{2,3} becomes {0}{1}{2,3} whether 0 or 1 leaves. All movers of an edge always share the same $\Delta d$ and $\Delta\hat d$ — a consequence of the exact potential;
- every edge stores its movers, origin and destination communities, $\Delta d$, $\Delta \hat d$, $F$, its type, and which end offers more friends and which fewer strangers.

The edges split into **37 clear**, **9 frustrated** and **6 indifferent** moves.

## Layers and layout

A breadth-first search gives each partition its move distance $d_G$ from the grand coalition and $d_S$ from the singleton partition. Layers are the ranks of $d_G - d_S$: grand coalition, the four 3+1 partitions, the three 2+2 partitions, the six 2+1+1 partitions and the singleton partition — the same five columns as the paper's figure. Within each column the order is chosen by trying every permutation and keeping the one that minimises the squared vertical stretch of the edges, repeated until nothing improves; the result is deterministic.

The paper's figure omits the 12 moves between three-community partitions (six clear, six indifferent) to stay legible. The explainer draws them as arcs and can hide them.

## Orientation by γ

For a fixed $\gamma$ every edge is oriented towards the side with positive gain $\Delta U$; exact zeros are drawn as ties. Away from the thresholds $1/3$, $1/2$ and $2/3$ the result is a directed acyclic graph, as the potential argument guarantees; `metagraph.test.ts` checks this and the exact-potential identity on every move.

## Accessibility of the drawing

Metanodes are keyboard reachable (a single tab stop; arrow keys move between partitions), every node has a textual description with its quality and sink status, and the Explore section offers the whole metagraph as a table. Colour is always paired with a second cue: arrowheads, dashes, circle/square cursors and labels.
