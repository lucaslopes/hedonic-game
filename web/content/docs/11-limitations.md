---
title: Scope and limitations
summary: What the explainer is — a faithful, tiny teaching model — and what it is not.
group: Project
order: 11
---

- **A four-vertex teaching model.** The explainer enumerates every partition of one small graph. Exhaustive enumeration is only possible for tiny graphs: the number of partitions grows as the Bell numbers ($B_{10} = 115\,975$, $B_{20} \approx 5 \times 10^{13}$). Real networks are handled by the `hedonic` package, never by this page.
- **Not a Leiden implementation.** The walk moves one vertex at a time over an enumerated state space, with documented tie-breaking rules. The native detector (lucas-igraph) uses a queue of candidate vertices and, with `local_move_only=False`, refinement and aggregation. Results on real graphs come from that code.
- **Move set.** The metagraph always allows a vertex to leave and found a new community. In the package this corresponds to `allow_isolation=True`; the default is `False` ([details](docs:#community-hedonic)).
- **Disjoint model only.** Overlapping covers (`max_memberships > 1`) have their own utility and are out of scope here.
- **Unweighted, undirected, simple graphs.** The model code rejects self-loops and duplicate edges; weights are not modelled.
- **Resolution range.** Statements assume $0 \le \gamma \le 1$, the range used by the papers. The package accepts other finite values as an explicit extension.
- **Numerical ties.** Gains within $10^{-9}$ of zero are treated as exact ties. The resolution slider stops exactly on every threshold via the keyboard, the fraction input or the snap option.
- **Figure differences.** The historical figure omits the 12 moves among three-community partitions; the explainer shows them as optional arcs. Its column order is recomputed and may be a mirror image of the paper's.
- **Visual encodings.** The balance scale's tilt is proportional to the utility difference but clamped, and the sizes of weights and balloons are illustrative. The numbers next to it are exact.
- **Languages.** The story is available in English and Brazilian Portuguese; the technical documentation is in English.
