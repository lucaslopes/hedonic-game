---
title: The resolution parameter
summary: How γ prices friends and strangers, the quality it induces, and what happens at the extremes.
group: Theory
order: 6
---

## Pair values

Following the Constant Potts Model (CPM) [@traag2011cpm], two vertices $i$ and $j$ in the same community contribute

$$
v_{ij} = (A_{ij} - \gamma)\,\delta(\sigma_i, \sigma_j) =
\begin{cases}
1 - \gamma & \text{friends in the same community,}\\
-\gamma & \text{strangers in the same community,}\\
0 & \text{different communities,}
\end{cases}
$$

where $A$ is the adjacency matrix, $\sigma_i$ is the community of $i$, and $0 \le \gamma \le 1$. In the explainer's scale, a friend is a weight worth $1-\gamma$ and a stranger a balloon lifting by $\gamma$.

## Node, community and partition potentials

Summing pair values gives three nested scores (arXiv:2509.03834, "Hedonic Potential"):

$$
\varphi_i^\gamma(\mathcal{C}_k) = (1-\gamma)\, d_i^k - \gamma\, \hat d_i^k,
\qquad
\phi^\gamma(\mathcal{C}_k) = m_k - \gamma \binom{n_k}{2},
\qquad
\Phi^\gamma(\pi) = \sum_k \Big[ m_k - \gamma \binom{n_k}{2} \Big],
$$

with $m_k$ the edges and $n_k$ the vertices inside community $k$. Equivalently $\Phi^\gamma(\pi) = (1-\gamma)\,E_{\text{in}} - \gamma\,N_{\text{in}}$, where $E_{\text{in}}$ counts friendships and $N_{\text{in}}$ stranger pairs inside communities. The explainer shows $\Phi^\gamma$ as the **quality** of each partition.

The SRC abstract writes the CPM Hamiltonian over ordered pairs, $\mathcal{H}_{\text{CPM}} = -\sum_{i \ne j} (A_{ij}-\gamma)\,\delta(\sigma_i,\sigma_j) = -2\,\Phi^\gamma$, so minimising the Hamiltonian and maximising $\Phi^\gamma$ are the same problem.

## The extremes

- **$\gamma = 0$.** Strangers cost nothing, so adding members never hurts: the grand coalition maximises $\Phi^0 = E_{\text{in}}$. On a connected graph every other partition cuts at least one edge, so the grand coalition is the unique optimum. In the explainer's example it is also the only sink for every $\gamma < 1/3$.
- **$\gamma = 1$.** Friends are worth nothing and every stranger pair costs one point: $\Phi^1 = -N_{\text{in}} \le 0$. Every partition whose communities are cliques reaches the maximum $0$, and the singleton partition is always one of them. In the example seven partitions tie, which is why the explainer says singleton-*like* partitions win.

In between, $\gamma$ acts as a zoom knob: lower values merge denser groups into larger communities, higher values split them. Because the comparison only involves the communities being compared, CPM is *resolution-limit-free* [@traag2011cpm].

## In the package

`community_hedonic(resolution=None)` uses the graph density $2m / (n(n-1))$ as $\gamma$. Any finite value is accepted; the mathematical statements on this site are scoped to $0 \le \gamma \le 1$.
