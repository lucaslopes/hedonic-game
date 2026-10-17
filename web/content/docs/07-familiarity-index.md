---
title: The Familiarity Index
summary: The exact resolution at which a frustrated agent is indifferent — definition, derivation and every edge case.
group: Theory
order: 7
---

## Definition

Consider a vertex $i$ that belongs to community $\mathcal{A}$ and could move to $\mathcal{B}$. Let

$$
\Delta d = d_i^{\mathcal{B}} - d_i^{\mathcal{A}},
\qquad
\Delta \hat d = \hat d_i^{\mathcal{B}} - \hat d_i^{\mathcal{A}},
$$

be the change in friends and in strangers (the vertex itself never counts). The move is worth

$$
\Delta U = \Delta d - \gamma\,(\Delta d + \Delta \hat d),
$$

and it is favourable when $\Delta U > 0$, i.e. $\Delta d > \gamma(\Delta d + \Delta \hat d)$ (Eq. 4 of the SRC abstract). The **Familiarity Index** is the resolution at which the vertex is indifferent (Eq. 5):

$$
F_i(\mathcal{A}, \mathcal{B}) = \frac{\Delta d}{\Delta d + \Delta \hat d}.
$$

$\Delta d + \Delta \hat d$ is the change in the size of the vertex's social circle. Reversing the move negates both deltas, so $F$ does not depend on the direction.

## Reading the index

For a frustrated choice ($0 < F < 1$):

- if $\gamma < F$, the vertex prioritises gaining friends;
- if $\gamma > F$, it prioritises avoiding strangers;
- if $\gamma = F$, it is indifferent: $\Delta U = 0$ exactly.

The explainer draws each frustrated metaedge in two colours split at $F$ (measured from the end with more friends) and places a cursor at $\gamma$: the colour under the cursor wins.

## All the cases

| Deltas | $F$ | Nature | Explainer |
| --- | --- | --- | --- |
| $\Delta d = 0,\ \Delta\hat d = 0$ | undefined ($0/0$) | indifferent for every $\gamma$ | dashed grey edge, never taken |
| same strict sign ($\Delta d\,\Delta\hat d > 0$) | $0 < F < 1$ | **frustrated** | blue/red edge, flips at $F$ |
| $\Delta d = 0,\ \Delta\hat d \ne 0$ | $F = 0$ | clear (fewer strangers wins) | purple arrow; tie only at $\gamma = 0$ |
| $\Delta\hat d = 0,\ \Delta d \ne 0$ | $F = 1$ | clear (more friends wins) | purple arrow; tie only at $\gamma = 1$ |
| opposite strict signs, $\Delta d + \Delta\hat d \ne 0$ | $F < 0$ or $F > 1$ | clear (the ideal community) | purple arrow for every $\gamma \in [0,1]$ |
| opposite strict signs, $\Delta d + \Delta\hat d = 0$ | undefined (zero denominator) | clear: $\Delta U = \Delta d$ for every $\gamma$ | purple arrow for every $\gamma$ |

This matches Table "Analysis of a potential move" of arXiv:2509.03834 (sub-cases a–d) and the decision tree of the SRC abstract: the purple leaves are the clear rows, the red leaf is the frustrated row, and the grey leaf is the indifferent row.

Two boundary subtleties are made explicit in the explainer:

1. **Endpoint ties.** A clear choice with $F = 0$ (same friends, fewer strangers) has $\Delta U = -\gamma\,\Delta\hat d$, which is exactly zero at $\gamma = 0$: strangers cost nothing there, so the scale is level even though the decision tree calls the choice clear. Symmetrically for $F = 1$ at $\gamma = 1$.
2. **Numerical tolerance.** Gains are compared with a tolerance of $10^{-9}$, so a slider value such as $0.1 + 0.2$ still counts as exactly $F = 0.3$. Keyboard stops include every threshold, so $\gamma = 1/3$ can be reached exactly.

## Worked examples on the explainer's graph

| Move | $\Delta d$ | $\Delta\hat d$ | $F$ |
| --- | --- | --- | --- |
| vertex 3 leaves {0,1,2,3} to be alone | $-1$ | $-2$ | $1/3$ |
| vertex 1 or 2 leaves {0,1,2,3} to be alone | $-2$ | $-1$ | $2/3$ |
| vertex 3 leaves {0,1,3} to be alone | $-1$ | $-1$ | $1/2$ |
| vertex 0 leaves {0,1,2,3} to be alone | $-3$ | $0$ | $1$ (clear) |

Nine of the 52 metaedges are frustrated: one at $1/3$, six at $1/2$ and two at $2/3$.
