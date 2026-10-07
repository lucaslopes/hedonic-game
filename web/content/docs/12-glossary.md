---
title: Glossary
summary: The vocabulary of the explainer and of the papers, in one place.
group: Project
order: 12
---

| Term | Meaning |
| --- | --- |
| Agent | A vertex of the graph, choosing which community to join. |
| Community | A set of vertices; in the disjoint model each vertex belongs to exactly one. |
| Partition $\pi$ | A split of all vertices into disjoint, non-empty communities. |
| Grand coalition | The partition with a single community containing every vertex. |
| Singleton partition | The partition in which every vertex is alone. |
| Friend | A member of the community linked to the agent. |
| Stranger | A member of the community not linked to the agent. |
| Internal degree $d_i^{\mathcal{C}}$ | Number of friends of $i$ in $\mathcal{C}$ (excluding $i$). |
| Internal non-degree $\hat d_i^{\mathcal{C}}$ | Number of strangers of $i$ in $\mathcal{C}$ (excluding $i$). |
| Resolution $\gamma$ | The price of a stranger; a friend is worth $1-\gamma$. |
| CPM | Constant Potts Model: pair value $A_{ij} - \gamma$ inside communities [@traag2011cpm]. |
| Utility $\varphi_i^\gamma$ | $(1-\gamma)\,d_i - \gamma\,\hat d_i$ in the agent's community. |
| Quality $\Phi_\gamma(\pi)$ | Partition potential $\sum_k [m_k - \gamma\binom{n_k}{2}]$. |
| $\Delta d$, $\Delta\hat d$ | Change in friends and strangers of a move. |
| Gain $\Delta U$ | $\Delta d - \gamma(\Delta d + \Delta\hat d)$; positive gains are improving moves. |
| Familiarity Index $F$ | $\Delta d / (\Delta d + \Delta\hat d)$, the resolution of indifference. |
| Clear choice | Both objectives agree (purple in the figures). |
| Frustrated choice | One option has more friends, the other fewer strangers (red with a blue border). |
| Indifferent | Same friends and same strangers; zero gain for every $\gamma$. |
| Metagraph | Graph whose vertices are partitions and whose edges are single-vertex moves. |
| Metanode, metaedge | A vertex and an edge of the metagraph. |
| Restricted growth string | Canonical label vector of a partition, used as its id. |
| Bell number $B_n$ | Number of partitions of $n$ vertices ($B_4 = 15$). |
| Better response | Any strictly improving move. |
| Best response | The most improving move. |
| Sink | A partition with no improving move: a stable (Nash) equilibrium. |
| Exact potential game | A game where each unilateral gain equals the change of one global potential [@monderer1996potential]. |
| Robust node | A node whose community maximises friends and minimises strangers at once [@lopes2025robustness]. |
| Fully-robust partition | A partition where every node is robust; stable for every $\gamma \in [0, 1]$. |
