/**
 * How a metaedge is drawn in each mode. Pure function of the model, so the
 * encoding can be tested without rendering.
 */

import { edgeState, type MetaEdge } from '../../model';

export type MetagraphMode = 'pair' | 'nodes' | 'edges' | 'types' | 'oriented';

export type EdgeVisual =
  /** Untyped move (before edges are classified). */
  | { readonly style: 'neutral' }
  /** Clear choice: a purple arrow pointing to `head`. */
  | { readonly style: 'clear'; readonly head: string }
  /** Zero gain: dashed grey, no arrowhead. */
  | { readonly style: 'tie'; readonly reason: 'indifferent' | 'threshold' }
  /**
   * Frustrated choice: blue from `friendEnd` up to the Familiarity Index,
   * red from there to `strangerEnd`. `friendHead`/`strangerHead` say which
   * arrowheads are drawn; `cursor` is γ (the position of the γ marker) or
   * null in the γ-free paper view; `tie` marks γ = F exactly.
   */
  | {
      readonly style: 'frustrated';
      readonly friendEnd: string;
      readonly strangerEnd: string;
      readonly familiarity: number;
      readonly friendHead: boolean;
      readonly strangerHead: boolean;
      readonly cursor: number | null;
      readonly tie: boolean;
    };

export function edgeVisual(edge: MetaEdge, mode: MetagraphMode, gamma: number): EdgeVisual {
  const { tradeoff } = edge;
  if (mode === 'nodes' || mode === 'edges') return { style: 'neutral' };

  if (tradeoff.kind === 'indifferent') return { style: 'tie', reason: 'indifferent' };

  if (tradeoff.kind === 'frustrated' && edge.friendEnd && edge.strangerEnd && tradeoff.familiarity !== null) {
    const base = {
      style: 'frustrated' as const,
      friendEnd: edge.friendEnd,
      strangerEnd: edge.strangerEnd,
      familiarity: tradeoff.familiarity,
    };
    if (mode === 'pair' || mode === 'types') {
      return { ...base, friendHead: true, strangerHead: true, cursor: null, tie: false };
    }
    const state = edgeState(edge, gamma);
    if (state.orientation === 'tie') {
      return { ...base, friendHead: false, strangerHead: false, cursor: gamma, tie: true };
    }
    return {
      ...base,
      friendHead: state.to === edge.friendEnd,
      strangerHead: state.to === edge.strangerEnd,
      cursor: gamma,
      tie: false,
    };
  }

  // Clear choice: γ-free direction in the paper view, γ-oriented otherwise.
  if (mode === 'pair' || mode === 'types') {
    return { style: 'clear', head: tradeoff.paretoPreference === 'forward' ? edge.b : edge.a };
  }
  const state = edgeState(edge, gamma);
  if (state.orientation === 'tie' || !state.to) return { style: 'tie', reason: 'threshold' };
  return { style: 'clear', head: state.to };
}
