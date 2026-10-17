/**
 * CPM quality of a partition (the potential of the hedonic game).
 *
 * Following the extended version (arXiv:2509.03834, "Partition Potential"):
 *
 *   Φ_γ(π) = Σ_k [ m_k − γ · C(n_k, 2) ]
 *          = (1 − γ) · (internal links) − γ · (internal non-links)
 *
 * where m_k and n_k are the number of edges and vertices inside community k.
 * The CPM Hamiltonian of the SRC abstract (Eq. 3) sums over ordered pairs,
 * so H_CPM = −2 Φ_γ: minimising H and maximising Φ are the same problem.
 * A single move changes Φ_γ by exactly the mover's utility gain
 * ΔU = Δd − γ(Δd + Δd̂) — the exact-potential property.
 */

import type { Adjacency } from './graph';
import { blocksOf, type Rgs } from './partitions';

export interface InternalPairs {
  /** Linked pairs inside communities ("friendships kept"). */
  readonly internalEdges: number;
  /** Unlinked pairs inside communities ("strangers tolerated"). */
  readonly internalNonEdges: number;
}

export function internalPairs(adj: Adjacency, rgs: Rgs): InternalPairs {
  let internalEdges = 0;
  let internalNonEdges = 0;
  for (const block of blocksOf(rgs)) {
    for (let x = 0; x < block.length; x += 1) {
      for (let y = x + 1; y < block.length; y += 1) {
        if (adj[block[x]][block[y]]) internalEdges += 1;
        else internalNonEdges += 1;
      }
    }
  }
  return { internalEdges, internalNonEdges };
}

/** Φ_γ from pre-computed internal pair counts. */
export function potentialFromPairs(pairs: InternalPairs, gamma: number): number {
  return (1 - gamma) * pairs.internalEdges - gamma * pairs.internalNonEdges;
}

/** Φ_γ(π): CPM partition potential at resolution γ. */
export function partitionPotential(adj: Adjacency, rgs: Rgs, gamma: number): number {
  return potentialFromPairs(internalPairs(adj, rgs), gamma);
}
