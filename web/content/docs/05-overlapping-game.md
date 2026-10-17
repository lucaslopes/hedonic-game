---
title: The overlapping game
summary: Intensities, label gains, the prefix rule, the visiting queue and the exact stability audit behind max_memberships > 1.
group: Theory
order: 5.5
slug: overlapping-game
---

With `max_memberships = M > 1`, `Game.community_hedonic` plays a different game from the disjoint one: every vertex holds a set of labels instead of one. The [overlap explainer](story:overlap/#sees) runs a port of the C core's cover engine on a six-vertex graph; this page gives the rules it follows.

## Intensities

Vertex $v$ holds a nonempty set $\sigma_v$ of at most $M$ labels. It gives each of them the intensity

$$
f_v = \frac{1}{\sqrt{|\sigma_v|}},
$$

so $\sum_{c \in \sigma_v} f_v^2 = 1$ whatever the number of labels: a second label adds no attention, it shares the same unit. The **mass** of a label is the total intensity of its holders, $S_c = \sum_{u : c \in \sigma_u} f_u$.

## The gain of a label

For a label $c$, vertex $v$ adds up the intensity of the friends who hold it (the **support**) and of everyone else who holds it (the **crowd**, which includes those friends):

$$
g_c(v) = \sum_{u \sim v,\ c \in \sigma_u} w_{uv} f_u \;-\; \gamma \left(S_c - f_v [c \in \sigma_v]\right).
$$

With unit weights this is $(1-\gamma)$ times the friends' intensity minus $\gamma$ times the strangers' intensity, the disjoint trade-off weighted by intensity. A label that $v$ holds alone scores exactly 0; so does a fresh label nobody holds.

## The utility and the prefix rule

The utility of a set is

$$
U_v(\sigma) = \frac{1}{\sqrt{|\sigma|}} \sum_{c \in \sigma} g_c(v),
$$

where the gains are computed against everyone else's current labels. Among sets of $j$ labels the best is always the $j$ labels of highest gain, so a best response is a **prefix** of the labels sorted by gain (ties by smaller label). The detector compares the prefixes of positive gains up to $M$ and keeps the one of highest value (a longer prefix must be strictly better). When no gain is positive it settles for the single label of highest gain; with `allow_isolation=True` it opens a fresh label instead whenever every reachable label scores below 0. Labels that only strangers hold score $-\gamma S_c$, so only the lightest of them can matter, and the detector looks at it only when every other gain is negative and isolation is off.

A vertex moves when the best set differs from its own and is strictly better.

## The potential

The cover game is an exact potential game. With $A$ the adjacency matrix,

$$
\Phi(\sigma) = \sum_{a < b} |\sigma_a \cap \sigma_b|\, f_a f_b \,(A_{ab} - \gamma)
$$

changes by exactly the mover's change of utility, because the terms of $\Phi$ that involve $v$ add up to $U_v(\sigma_v)$. Every accepted move raises $\Phi$, there are finitely many covers, so the dynamic stops. The quality the package reports is $(\Phi - \gamma n/2)/W$ for unit node weights and $W$ edges, the same potential up to a constant.

## The visiting order

The detector draws a random permutation of the vertices from `random.Random(seed)` and visits them from a queue. A vertex that moves puts its neighbours back in the queue; a vertex that does not move is marked settled at the current move count. When the queue runs empty, every vertex not settled since the last move is queued again in a freshly shuffled order. The run ends when that pass finds nobody: no vertex can improve, an equilibrium. The explainer reproduces this queue and the random draws, so with the same seed it makes the same visits and moves as `community_hedonic(..., seed=seed)`.

## Exact stability of a fixed cover

Freeze every other vertex. Each label's gain is affine in $\gamma$, $g_c(\gamma) = a_c - \gamma\, b_c$, and so is the utility of every set. Vertex $v$'s set is stable where no other set of at most $M$ labels lies above it, which is a finite union of closed intervals of $[0, 1]$. `Game.stable_intervals` computes them exactly, cell by cell between the resolutions where two candidate lines cross; `Game.regrets` reports, on a grid of resolutions, how much each vertex would gain by deviating. The intersection of all the vertices' intervals is the **Nash window** (`hedonic.stability.nash_window`): the resolutions at which the cover is an equilibrium.

On the explainer's graph, two cliques of four sharing the edge 2–3, the cover in which vertices 2 and 3 hold both labels is stable for vertices 2 and 3 on all of $[0, 1]$, and for the other four on $[3 - 2\sqrt{2}, 1] \approx [0.172, 1]$.

```api Game.stable_intervals
```

```api Game.regrets
```

## What the browser port leaves out

The explainer's model (`web/src/model/overlap/`) handles unweighted graphs, resolutions in $[0, 1]$ and no community-count limits. Where the core certifies each comparison with a bound of its rounding error, the port uses a fixed tolerance of $10^{-12}$; on the small graph of the page every genuine difference is far larger. A cross-check script runs the package on the same graph and the tests compare every move, every interval and every regret.
