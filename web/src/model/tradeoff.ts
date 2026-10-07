/**
 * The friends-versus-strangers trade-off of a single move.
 *
 * Notation follows the SRC extended abstract (§III) and the extended
 * version (arXiv:2509.03834, "The Familiarity Index as a Decision Threshold
 * for Moves"). For a vertex i leaving community A for community B:
 *
 *   Δd  = d_i^B − d_i^A        change in friends   (internal neighbours)
 *   Δd̂  = d̂_i^B − d̂_i^A        change in strangers (internal non-neighbours)
 *
 *   ΔU(γ) = Δd − γ (Δd + Δd̂)   utility gain of the move        (Eq. 4)
 *   F     = Δd / (Δd + Δd̂)     Familiarity Index, when Δd+Δd̂≠0 (Eq. 5)
 *
 * Equivalently, the vertex's utility inside a community with f friends and
 * s strangers is f·(1−γ) − s·γ, and ΔU is the difference of the two values.
 */

/** Numerical tolerance used for every tie and threshold comparison. */
export const EPS = 1e-9;

export function signOf(x: number, eps = EPS): -1 | 0 | 1 {
  if (x > eps) return 1;
  if (x < -eps) return -1;
  return 0;
}

/** Utility of belonging to a community with `friends` and `strangers`. */
export function communityUtility(friends: number, strangers: number, gamma: number): number {
  return friends * (1 - gamma) - strangers * gamma;
}

/** ΔU = Δd − γ(Δd + Δd̂): the gain of the move (positive means "move"). */
export function utilityGain(deltaFriends: number, deltaStrangers: number, gamma: number): number {
  return deltaFriends - gamma * (deltaFriends + deltaStrangers);
}

/**
 * Familiarity Index F = Δd / (Δd + Δd̂), or `null` when the denominator is
 * zero (the move does not change the size of the vertex's social circle, so
 * no resolution value can make it neutral unless Δd is also zero).
 */
export function familiarityIndex(deltaFriends: number, deltaStrangers: number): number | null {
  const denominator = deltaFriends + deltaStrangers;
  if (signOf(denominator) === 0) return null;
  const value = deltaFriends / denominator;
  return value === 0 ? 0 : value; // normalise −0 (Δd = 0 with a negative denominator)
}

export type TradeoffKind = 'indifferent' | 'clear' | 'frustrated';

export interface Tradeoff {
  readonly deltaFriends: number;
  readonly deltaStrangers: number;
  /** Familiarity Index, or null when Δd + Δd̂ = 0. */
  readonly familiarity: number | null;
  /**
   * - `indifferent`: Δd = Δd̂ = 0, the two options look identical.
   * - `clear`: one option has at least as many friends and no more
   *   strangers (F ∉ (0, 1)); every γ agrees on the direction, except for an
   *   exact tie at an endpoint γ = F ∈ {0, 1}.
   * - `frustrated`: Δd and Δd̂ have the same strict sign (0 < F < 1); the
   *   direction depends on γ.
   */
  readonly kind: TradeoffKind;
  /**
   * Direction favoured by both objectives (clear), `none` (indifferent) or
   * `depends` (frustrated). "forward" is the move A → B.
   */
  readonly paretoPreference: 'forward' | 'backward' | 'none' | 'depends';
}

export function analyzeTradeoff(deltaFriends: number, deltaStrangers: number): Tradeoff {
  const f = signOf(deltaFriends);
  const s = signOf(deltaStrangers);
  const familiarity = familiarityIndex(deltaFriends, deltaStrangers);
  let kind: TradeoffKind;
  let paretoPreference: Tradeoff['paretoPreference'];
  if (f === 0 && s === 0) {
    kind = 'indifferent';
    paretoPreference = 'none';
  } else if (f * s > 0) {
    kind = 'frustrated';
    paretoPreference = 'depends';
  } else {
    kind = 'clear';
    // More (or equal) friends and fewer (or equal) strangers, not both equal.
    paretoPreference = f >= 0 && s <= 0 ? 'forward' : 'backward';
  }
  return { deltaFriends, deltaStrangers, familiarity, kind, paretoPreference };
}

export type Orientation = 'forward' | 'backward' | 'tie';

/**
 * Why a move goes the way it does at a given γ:
 * - `friends`: a frustrated choice resolved towards more friends (γ < F);
 * - `strangers`: a frustrated choice resolved towards fewer strangers (γ > F);
 * - `clear`: both objectives agree;
 * - `threshold`: γ equals the Familiarity Index, so the gain is exactly zero;
 * - `indifferent`: nothing changes for the mover at any γ.
 */
export type OrientationReason = 'friends' | 'strangers' | 'clear' | 'threshold' | 'indifferent';

export interface OrientedTradeoff {
  /** ΔU for the forward move A → B. */
  readonly gain: number;
  readonly orientation: Orientation;
  readonly reason: OrientationReason;
}

/** Resolve a trade-off at resolution γ. Exact zero gains are ties. */
export function orientTradeoff(tradeoff: Tradeoff, gamma: number): OrientedTradeoff {
  const gain = utilityGain(tradeoff.deltaFriends, tradeoff.deltaStrangers, gamma);
  const direction = signOf(gain);
  if (tradeoff.kind === 'indifferent') {
    return { gain: 0, orientation: 'tie', reason: 'indifferent' };
  }
  if (direction === 0) {
    return { gain: 0, orientation: 'tie', reason: 'threshold' };
  }
  const orientation: Orientation = direction > 0 ? 'forward' : 'backward';
  if (tradeoff.kind === 'clear') {
    return { gain, orientation, reason: 'clear' };
  }
  // Frustrated: the winning direction either gains friends or sheds strangers.
  const forwardGainsFriends = tradeoff.deltaFriends > 0;
  const winnerGainsFriends = orientation === 'forward' ? forwardGainsFriends : !forwardGainsFriends;
  return { gain, orientation, reason: winnerGainsFriends ? 'friends' : 'strangers' };
}

/** Counts that describe one candidate community from the agent's viewpoint. */
export interface CommunityCounts {
  readonly friends: number;
  readonly strangers: number;
}

/** The trade-off of moving from community A to community B. */
export function compareCommunities(a: CommunityCounts, b: CommunityCounts): Tradeoff {
  return analyzeTradeoff(b.friends - a.friends, b.strangers - a.strangers);
}
