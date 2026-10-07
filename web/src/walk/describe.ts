import { formatBlocks, type Move } from '../model';
import type { Dictionary } from '../i18n';

/** "{0,1}" or "a new community". */
export function targetText(move: Move, t: Dictionary): string {
  return move.createsCommunity ? t.common.newCommunity : formatBlocks([move.targetBlock]);
}

/** One-sentence description of a unilateral move. */
export function describeMove(move: Move, t: Dictionary): string {
  return move.createsCommunity ? t.walk.leaves(move.vertex) : t.walk.joins(move.vertex, formatBlocks([move.targetBlock]));
}
