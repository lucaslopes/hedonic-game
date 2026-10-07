/**
 * Unilateral single-vertex moves between canonical partitions.
 *
 * Two partitions are adjacent in the metagraph when one vertex changing its
 * community turns one into the other. A vertex may join any other existing
 * community, or leave to open a new singleton community (only meaningful
 * when it is not already alone). Leaving a singleton community removes that
 * community; the canonical relabelling takes care of label gaps.
 */

import type { Adjacency } from './graph';
import { blocksOf, canonicalize, partitionId, type Rgs } from './partitions';

export interface Move {
  readonly vertex: number;
  /** Partition ids before and after the move. */
  readonly from: string;
  readonly to: string;
  /** Community of the vertex before the move (including the vertex). */
  readonly originBlock: readonly number[];
  /** Members of the destination community before the move; [] for a new community. */
  readonly targetBlock: readonly number[];
  /** Destination label in the `from` partition, or null for a new community. */
  readonly targetLabel: number | null;
  /** The vertex opens a new singleton community. */
  readonly createsCommunity: boolean;
  /** The vertex was alone, so its old community disappears. */
  readonly removesCommunity: boolean;
  /** d_i and d̂_i in the origin community (the vertex itself excluded). */
  readonly friendsBefore: number;
  readonly strangersBefore: number;
  /** d_i and d̂_i in the destination community. */
  readonly friendsAfter: number;
  readonly strangersAfter: number;
  /** Δd = friendsAfter − friendsBefore; Δd̂ = strangersAfter − strangersBefore. */
  readonly deltaFriends: number;
  readonly deltaStrangers: number;
}

/** Friends and strangers of `vertex` among `members` (the vertex excluded). */
export function countRelations(
  adj: Adjacency,
  vertex: number,
  members: readonly number[],
): { friends: number; strangers: number } {
  let friends = 0;
  let strangers = 0;
  for (const other of members) {
    if (other === vertex) continue;
    if (adj[vertex][other]) friends += 1;
    else strangers += 1;
  }
  return { friends, strangers };
}

/** Apply a move and return the canonical result (`null` target = new community). */
export function applyMove(rgs: Rgs, vertex: number, targetLabel: number | null): number[] {
  const next = [...rgs];
  next[vertex] = targetLabel === null ? Math.max(...rgs) + 1 : targetLabel;
  return canonicalize(next);
}

/**
 * All single-vertex moves out of a partition, in canonical order: by vertex,
 * then by destination label (existing communities first, a new community
 * last). No-op moves (a lone vertex "leaving" to be alone) are excluded.
 */
export function enumerateMoves(adj: Adjacency, rgs: Rgs): Move[] {
  const blocks = blocksOf(rgs);
  const from = partitionId(rgs);
  const moves: Move[] = [];
  for (let vertex = 0; vertex < rgs.length; vertex += 1) {
    const originLabel = rgs[vertex];
    const originBlock = blocks[originLabel];
    const before = countRelations(adj, vertex, originBlock);
    const alone = originBlock.length === 1;
    const targets: (number | null)[] = blocks.map((_, label) => label).filter((label) => label !== originLabel);
    if (!alone) targets.push(null);
    for (const targetLabel of targets) {
      const targetBlock = targetLabel === null ? [] : blocks[targetLabel];
      const after = countRelations(adj, vertex, targetBlock);
      moves.push({
        vertex,
        from,
        to: partitionId(applyMove(rgs, vertex, targetLabel)),
        originBlock,
        targetBlock,
        targetLabel,
        createsCommunity: targetLabel === null,
        removesCommunity: alone,
        friendsBefore: before.friends,
        strangersBefore: before.strangers,
        friendsAfter: after.friends,
        strangersAfter: after.strangers,
        deltaFriends: after.friends - before.friends,
        deltaStrangers: after.strangers - before.strangers,
      });
    }
  }
  return moves;
}
