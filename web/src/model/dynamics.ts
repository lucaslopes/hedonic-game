/**
 * Better-response dynamics on the metagraph.
 *
 * Starting from any partition, one vertex at a time makes a *strictly*
 * improving unilateral move (ΔU > 0). Because CPM is an exact potential game,
 * every such move raises Φ_γ by exactly ΔU, so the walk cannot cycle and
 * must stop at a sink: a partition where no vertex has a strictly improving
 * move (a Nash-stable equilibrium of the hedonic game).
 *
 * Zero-gain moves (ties, including γ exactly at a Familiarity Index) are
 * never taken. Which improving move is taken is decided by a documented,
 * deterministic policy so that every walk is reproducible:
 *
 * - `best`   — best response: the largest gain; ties go to the lowest vertex,
 *              then to the existing community with the smallest member, and
 *              a new community last.
 * - `sweep`  — vertices take turns 0, 1, …, n−1, 0, …; the first vertex (from
 *              the turn pointer) that can improve takes its own best move.
 *              A simplified version of the queue in the paper's Algorithm 1.
 * - `random` — a uniformly random improving move from a seeded PRNG
 *              (mulberry32), reproducible for a given seed.
 */

import type { Move } from './moves';
import { nodePotential, type Metagraph } from './metagraph';
import { EPS, utilityGain } from './tradeoff';

export type WalkPolicy = 'best' | 'sweep' | 'random';
export const WALK_POLICIES: readonly WalkPolicy[] = ['best', 'sweep', 'random'];

export interface Candidate {
  readonly move: Move;
  /** ΔU of the move at the current γ. */
  readonly gain: number;
}

/** Every move out of a partition with its gain at γ, in canonical order. */
export function scoredMoves(metagraph: Metagraph, id: string, gamma: number): Candidate[] {
  return metagraph.movesFrom(id).map((move) => ({
    move,
    gain: utilityGain(move.deltaFriends, move.deltaStrangers, gamma),
  }));
}

/** Moves with a strictly positive gain (beyond the numerical tolerance). */
export function improvingMoves(metagraph: Metagraph, id: string, gamma: number): Candidate[] {
  return scoredMoves(metagraph, id, gamma).filter((candidate) => candidate.gain > EPS);
}

export function isSink(metagraph: Metagraph, id: string, gamma: number): boolean {
  return improvingMoves(metagraph, id, gamma).length === 0;
}

/** All sinks (stable partitions) at γ, in RGS order. */
export function findSinks(metagraph: Metagraph, gamma: number): string[] {
  return metagraph.nodes.filter((node) => isSink(metagraph, node.id, gamma)).map((node) => node.id);
}

/** mulberry32: a tiny, well-known 32-bit PRNG. Returns [value in [0,1), next state]. */
export function mulberry32(state: number): [number, number] {
  const next = (state + 0x6d2b79f5) >>> 0;
  let t = next;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return [((t ^ (t >>> 14)) >>> 0) / 4294967296, next];
}

function bestOf(candidates: readonly Candidate[]): Candidate | undefined {
  let best: Candidate | undefined;
  for (const candidate of candidates) {
    if (!best || candidate.gain > best.gain + EPS) best = candidate;
  }
  return best;
}

export interface WalkStep {
  /** 1-based step number. */
  readonly index: number;
  readonly from: string;
  readonly to: string;
  readonly move: Move;
  readonly gain: number;
  readonly potentialBefore: number;
  readonly potentialAfter: number;
  /** How many strictly improving moves were available at `from`. */
  readonly options: number;
}

export interface WalkState {
  readonly start: string;
  readonly current: string;
  readonly steps: readonly WalkStep[];
  /** `sweep`: the vertex whose turn comes next. */
  readonly pointer: number;
  /** `random`: PRNG state. */
  readonly rngState: number;
  readonly status: 'running' | 'sink' | 'limit';
}

export function startWalk(start: string, seed = 1): WalkState {
  return { start, current: start, steps: [], pointer: 0, rngState: seed >>> 0, status: 'running' };
}

interface Choice {
  readonly candidate: Candidate;
  readonly pointer: number;
  readonly rngState: number;
}

/** Pick one improving move according to the policy (candidates must be non-empty). */
export function chooseMove(
  candidates: readonly Candidate[],
  policy: WalkPolicy,
  n: number,
  pointer: number,
  rngState: number,
): Choice {
  if (candidates.length === 0) throw new RangeError('no improving move to choose from');
  if (policy === 'random') {
    const [value, nextState] = mulberry32(rngState);
    const candidate = candidates[Math.min(candidates.length - 1, Math.floor(value * candidates.length))];
    return { candidate, pointer, rngState: nextState };
  }
  if (policy === 'sweep') {
    for (let offset = 0; offset < n; offset += 1) {
      const vertex = (pointer + offset) % n;
      const own = bestOf(candidates.filter((candidate) => candidate.move.vertex === vertex));
      if (own) return { candidate: own, pointer: (vertex + 1) % n, rngState };
    }
  }
  const best = bestOf(candidates);
  if (!best) throw new RangeError('no improving move to choose from');
  return { candidate: best, pointer, rngState };
}

/** Take one better-response step (or mark the walk as having reached a sink). */
export function advanceWalk(
  metagraph: Metagraph,
  state: WalkState,
  gamma: number,
  policy: WalkPolicy,
  maxSteps = 1000,
): WalkState {
  if (state.status !== 'running') return state;
  const candidates = improvingMoves(metagraph, state.current, gamma);
  if (candidates.length === 0) return { ...state, status: 'sink' };
  if (state.steps.length >= maxSteps) return { ...state, status: 'limit' };
  const { candidate, pointer, rngState } = chooseMove(
    candidates,
    policy,
    metagraph.graph.n,
    state.pointer,
    state.rngState,
  );
  const from = metagraph.node(state.current);
  const to = metagraph.node(candidate.move.to);
  const step: WalkStep = {
    index: state.steps.length + 1,
    from: from.id,
    to: to.id,
    move: candidate.move,
    gain: candidate.gain,
    potentialBefore: nodePotential(from, gamma),
    potentialAfter: nodePotential(to, gamma),
    options: candidates.length,
  };
  const next: WalkState = { ...state, current: to.id, steps: [...state.steps, step], pointer, rngState };
  // Report the sink as soon as it is reached, so the UI can celebrate it.
  return improvingMoves(metagraph, to.id, gamma).length === 0 ? { ...next, status: 'sink' } : next;
}

/** Run a walk to completion (a sink, or the safety limit). */
export function runWalk(
  metagraph: Metagraph,
  start: string,
  gamma: number,
  policy: WalkPolicy,
  options: { seed?: number; maxSteps?: number } = {},
): WalkState {
  const maxSteps = options.maxSteps ?? 1000;
  let state = startWalk(start, options.seed);
  if (isSink(metagraph, start, gamma)) return { ...state, status: 'sink' };
  while (state.status === 'running') state = advanceWalk(metagraph, state, gamma, policy, maxSteps);
  return state;
}

/** Sink reached from every starting partition under a deterministic policy. */
export function basinsOf(
  metagraph: Metagraph,
  gamma: number,
  policy: WalkPolicy,
  seed = 1,
): Map<string, string> {
  const basins = new Map<string, string>();
  for (const node of metagraph.nodes) {
    basins.set(node.id, runWalk(metagraph, node.id, gamma, policy, { seed }).current);
  }
  return basins;
}

export interface VertexCertificate {
  readonly vertex: number;
  /** The vertex's community in the partition. */
  readonly community: readonly number[];
  /** Its most attractive alternative (largest gain), if it has any move. */
  readonly best: Candidate | null;
  /** Alternatives with exactly zero gain (ties). */
  readonly ties: readonly Candidate[];
}

/**
 * Why a partition is (or is not) stable: every vertex's best alternative.
 * At a sink every `best.gain` is ≤ 0.
 */
export function stabilityCertificate(metagraph: Metagraph, id: string, gamma: number): VertexCertificate[] {
  const node = metagraph.node(id);
  const moves = scoredMoves(metagraph, id, gamma);
  return node.rgs.map((label, vertex) => {
    const own = moves.filter((candidate) => candidate.move.vertex === vertex);
    return {
      vertex,
      community: node.blocks[label],
      best: bestOf(own) ?? null,
      ties: own.filter((candidate) => Math.abs(candidate.gain) <= EPS),
    };
  });
}
