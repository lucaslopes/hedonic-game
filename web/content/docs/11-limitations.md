---
title: Scope and limitations
summary: What the explainer is — a faithful, tiny teaching model — and what it is not.
group: Project
order: 11
---

- **A four-vertex teaching model.** The explainer enumerates every partition of one small graph. Exhaustive enumeration is only possible for tiny graphs: the number of partitions grows as the Bell numbers ($B_{10} = 115\,975$, $B_{20} \approx 5 \times 10^{13}$). Real networks need a detector that never enumerates partitions, never this page.
- **A simplified dynamic.** The walk moves one vertex at a time over an enumerated state space, with documented tie-breaking rules ([details](docs:#better-response)).
- **Move set.** The metagraph always allows a vertex to leave and found a new community.
- **Unweighted, undirected, simple graphs.** The model code rejects self-loops and duplicate edges; weights are not modelled.
- **Resolution range.** Statements assume $0 \le \gamma \le 1$, the range used by the papers.
- **Numerical ties.** Gains within $10^{-9}$ of zero are treated as exact ties. The resolution slider stops exactly on every threshold via the keyboard, the fraction input or the snap option.
- **Figure differences.** The historical figure omits the 12 moves among three-community partitions; the explainer shows them as optional arcs. Its column order is recomputed and may be a mirror image of the paper's.
- **Visual encodings.** The balance scale's tilt is proportional to the utility difference but clamped, and the sizes of weights and balloons are illustrative. The numbers next to it are exact.
- **Languages.** The story is available in English and Brazilian Portuguese; the technical documentation is in English.
