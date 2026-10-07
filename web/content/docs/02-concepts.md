---
title: Concepts at a glance
summary: The whole story in one page — agents, friends and strangers, the resolution, frustrated choices, the metagraph and equilibria.
group: Guide
order: 2
---

## Agents choose communities

Community detection partitions the vertices of a graph into disjoint groups. The hedonic-game view treats each vertex $i$ as an **agent** that picks the community it belongs to, and cares only about *who else* is in it [@felipe2025src] [@felipe2025hedonic]. Hedonic games are exactly the games where an agent's payoff depends only on the members of its own coalition [@dreze1980hedonic] [@bogomolnaia2002stability].

## Friends and strangers

For a vertex $i$ and a community $\mathcal{C}$:

- the **internal degree** $d_i^{\mathcal{C}} = |\{ j \in \mathcal{C}, j \neq i : A_{ij} = 1\}|$ counts its **friends** (neighbours) in $\mathcal{C}$;
- the **internal non-degree** $\hat d_i^{\mathcal{C}} = |\{ j \in \mathcal{C}, j \neq i : A_{ij} = 0\}|$ counts its **strangers** (non-neighbours) in $\mathcal{C}$.

Two instincts compete: *maximise friends* and *minimise strangers* ([the dilemma](story:#dilemma)).

## One dial resolves the conflict

The Constant Potts Model [@traag2011cpm] gives each pair inside a community the value $A_{ij} - \gamma$: a friend is worth $1-\gamma$ and a stranger costs $\gamma$. The agent's utility in community $\mathcal{C}$ is

$$
\varphi_i^\gamma(\mathcal{C}) = (1-\gamma)\, d_i^{\mathcal{C}} - \gamma\, \hat d_i^{\mathcal{C}}.
$$

See [The resolution parameter](docs:#resolution).

## Clear and frustrated choices

Comparing two communities, most decisions are **clear**: one has at least as many friends and no more strangers. A choice is **frustrated** when one community has more friends and the other fewer strangers — the objectives conflict, as in a frustrated Ising system [@brush1967ising]. The paper's decision tree ([explainer](story:#tree)) separates the cases, and the **Familiarity Index**

$$
F = \frac{\Delta d}{\Delta d + \Delta \hat d}
$$

is the exact resolution at which a frustrated agent is indifferent ([details](docs:#familiarity-index)).

## The metagraph of partitions

Every partition of the graph is a **metanode**; two partitions are joined when a single vertex changing community turns one into the other. Fixing $\gamma$ orients every edge towards the side the mover prefers, turning the metagraph into a directed acyclic graph ([explainer](story:#metagraph), [details](docs:#metagraph)).

## Selfish moves reach an equilibrium

A **better-response dynamic** lets one agent at a time make a strictly improving move. Each move raises the partition quality $\Phi_\gamma$ by exactly the mover's gain — the model is an *exact potential game* [@monderer1996potential] — so the walk ends at a **sink**: a partition where no agent wants to move, i.e. a stable equilibrium ([explainer](story:#walk), [details](docs:#better-response)).

## Robustness

A partition is **fully robust** when every vertex's community simultaneously maximises its friends and minimises its strangers: it is then an equilibrium for every $\gamma \in [0, 1]$ [@lopes2025robustness]. Robustness is explored in the research experiments of this repository; the explainer focuses on the resolution perspective.
