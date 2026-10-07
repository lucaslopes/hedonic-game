---
title: Better-response dynamics
summary: Why selfish moves always stop, the tie-breaking rules the explainer uses, and the sinks of the example.
group: Theory
order: 9
---

## The dynamic

Start from any partition. Repeatedly pick a vertex that can strictly improve its utility by a unilateral move and move it. Stop when no vertex can: the partition is a **sink** of the oriented metagraph, i.e. a stable (Nash) equilibrium of the hedonic game [@felipe2025hedonic].

## Why it always stops

A move of vertex $i$ from $\mathcal{B}$ to $\mathcal{A}$ changes its own utility by

$$
\Delta\varphi_i = \big(d_i^{\mathcal{A}} - d_i^{\mathcal{B}}\big) - \gamma\,\big(n_{\mathcal{A}} - n_{\mathcal{B}} + 1\big),
$$

and it changes the partition potential by exactly the same amount — the CPM is an **exact potential game** [@monderer1996potential]. Every improving move therefore raises $\Phi_\gamma$, which is bounded, so the walk cannot cycle. For a rational resolution $\gamma = b/c$ each move raises $\Phi_\gamma$ by at least $1/c$, and the potential lies within $[-n^2, n^2]$, so an equilibrium is reached in $O(c\,n^2)$ moves (SRC abstract, Theorem 1; arXiv:2509.03834).

## The explainer's rules

Only moves with $\Delta U > 10^{-9}$ are taken: zero-gain moves (ties, including $\gamma$ exactly at a Familiarity Index) never are. When several improving moves exist, a documented rule picks one, so every walk can be replayed exactly:

| Rule | Choice |
| --- | --- |
| **Best response** (default) | The largest gain. Ties go to the lowest vertex, then to the existing community with the smallest member, then to a new community. |
| **Round-robin** | Vertices take turns $0, 1, \dots, n-1, 0, \dots$ from a pointer; the first vertex that can improve takes its own best move, and the pointer moves past it. A simplified version of the candidate queue in the paper's Algorithm 1. |
| **Random (seeded)** | A uniformly random improving move, drawn with the mulberry32 generator from a visible seed. |

The walk restarts from its start whenever $\gamma$, the rule, the seed or the start changes, so a walk never mixes resolutions.

## Sinks of the example

| Resolution | Sinks |
| --- | --- |
| $0 \le \gamma < 1/3$ | {0,1,2,3} |
| $\gamma = 1/3$ | {0,1,2,3} and {0,1,2}{3} (the edge between them is a tie) |
| $1/3 < \gamma < 1$ | {0,1,2}{3}, including at $\gamma = 1/2$ and $\gamma = 2/3$ |
| $\gamma = 1$ | seven partitions into cliques: {0,1,2}{3}, {0,1}{2}{3}, {0,2}{1}{3}, {0,3}{1,2}, {0}{1,2}{3}, {0,3}{1}{2}, {0}{1}{2}{3} |

From the grand coalition at $\gamma = 1/2$ the walk takes a single step: vertex 3 has one friend and two strangers, $F = 1/3 < \gamma$, so it leaves. From the singleton partition it takes two steps, {0,1}{2}{3} then {0,1,2}{3}.

## Relation to Leiden

The native detector's local-moving phase is a better-response process driven by a queue of candidate vertices, and full Leiden adds refinement and aggregation [@traag2019leiden] [@blondel2008fast]. The explainer's walk is a teaching model on an exhaustively enumerated state space; it is **not** a re-implementation of Leiden. The cross-check on [community_hedonic](docs:#community-hedonic) confirms that the native detector lands on the same sinks for the example graph.
