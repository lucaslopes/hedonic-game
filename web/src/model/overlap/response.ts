/**
 * One vertex's best response in the overlapping game: a port of a visit of
 * the C core (native/core/cover/visit.c: hig_cover_visit) without the
 * rounding bounds and the count limits.
 *
 * Vertex v holds the row σ_v of k labels, with intensity f_v = 1/√k on each.
 * For a label c, with v's own contribution taken out,
 *
 *   g_c(v) = Σ_{u ~ v, u holds c} f_u  −  γ · (S_c − f_v [c ∈ σ_v])
 *          = support(c)               −  γ · crowd(c)
 *
 * (cover.c, header comment). The support counts the intensity of v's
 * friends that hold c; the crowd counts the intensity of everyone else that
 * holds c, friends and strangers alike. So each friend is worth (1 − γ) f_u
 * and each stranger costs γ f_u, as in the disjoint game. A set σ is worth
 *
 *   U_v(σ) = Σ_{c ∈ σ} g_c(v) / √|σ|.
 *
 * The best set of at most M labels is a prefix of the labels sorted by g_c
 * (native/README.md, "The best set is a prefix"): the best j positive gains,
 * j ≤ M maximising their sum over √j, or, when no gain is positive, the
 * single best label. A fresh label is worth exactly 0 and competes when
 * isolation is allowed. A label v holds alone is worth exactly 0.
 */

import { adjacencyOf, type Adjacency, type SimpleGraph } from '../graph';
import { intensity, labelMasses, labelMembers, type Cover, type Row } from './cover';

/** A label nobody holds yet (allow_isolation): its gain is exactly 0. */
export const FRESH = -1;

/**
 * Tolerance of every comparison. The core decides with rigorous rounding
 * bounds (native/README.md, "Certified moves"); on these tiny graphs every
 * genuine difference is far above 1e-12, so the decisions agree.
 */
export const TIE = 1e-12;

export interface OverlapGame {
  readonly graph: SimpleGraph;
  readonly adj: Adjacency;
  /** Neighbours of each vertex in edge order (the core's adjacency order). */
  readonly neighbors: readonly (readonly number[])[];
}

export function overlapGame(graph: SimpleGraph): OverlapGame {
  const adj = adjacencyOf(graph);
  const neighbors: number[][] = Array.from({ length: graph.n }, () => []);
  for (const [a, b] of graph.edges) {
    neighbors[a].push(b);
    neighbors[b].push(a);
  }
  return { graph, adj, neighbors };
}

export interface ResponseOptions {
  /** The resolution γ. */
  readonly gamma: number;
  /** max_memberships, the cap M on the labels a vertex may hold. */
  readonly cap: number;
  /** allow_isolation: a vertex may open a fresh label ("open labels"). */
  readonly allowIsolation: boolean;
}

export interface Holder {
  readonly vertex: number;
  /** The holder's intensity f_u = 1/√k_u. */
  readonly f: number;
}

export interface LabelScore {
  readonly label: number;
  /** v holds it. */
  readonly own: boolean;
  /** v holds it and nobody else does: worth exactly 0. */
  readonly alone: boolean;
  /** Friends of v holding the label, with their intensities. */
  readonly friends: readonly Holder[];
  /** Other holders (not v, not linked to v). */
  readonly strangers: readonly Holder[];
  /** Σ f_u over the friends: the support. */
  readonly support: number;
  /** Σ f_u over every other holder: the label's mass without v. */
  readonly crowd: number;
  /** g_c = support − γ · crowd. */
  readonly gain: number;
}

/** The score of any label for v (a label nobody holds scores 0 support and 0 crowd). */
export function scoreLabel(game: OverlapGame, cover: Cover, vertex: number, label: number, gamma: number): LabelScore {
  const row = cover[vertex];
  const own = row.includes(label);
  const friends: Holder[] = [];
  const strangers: Holder[] = [];
  let support = 0;
  let crowd = 0;
  cover.forEach((other, u) => {
    if (u === vertex || !other.includes(label)) return;
    const f = intensity(other.length);
    crowd += f;
    if (game.adj[vertex][u]) {
      friends.push({ vertex: u, f });
      support += f;
    } else strangers.push({ vertex: u, f });
  });
  const alone = own && friends.length === 0 && strangers.length === 0;
  // A label v holds alone: no support, no crowding, exactly 0 (visit.c).
  const gain = alone ? 0 : support - gamma * crowd;
  return { label, own, alone, friends, strangers, support, crowd, gain };
}

/** Sort key of the core: larger gain first, then smaller label (hig_scored_before). */
export function compareScores(x: { gain: number; label: number }, y: { gain: number; label: number }): number {
  if (Math.abs(x.gain - y.gain) <= TIE) return x.label - y.label;
  return y.gain - x.gain;
}

/** U_v(σ) for any non-empty set of labels (FRESH counts as a label worth 0). */
export function payoff(game: OverlapGame, cover: Cover, vertex: number, labels: readonly number[], gamma: number): number {
  if (labels.length === 0) throw new RangeError('a vertex holds at least one label');
  let total = 0;
  for (const label of labels) if (label !== FRESH) total += scoreLabel(game, cover, vertex, label, gamma).gain;
  return total / Math.sqrt(labels.length);
}

export interface BestSet {
  /** The labels, sorted, with FRESH last when a fresh label is opened. */
  readonly labels: readonly number[];
  readonly value: number;
  /** How the core reached it. */
  readonly kind: 'prefix' | 'single' | 'fresh';
}

export interface BestResponse {
  readonly vertex: number;
  readonly row: Row;
  /** U_v(σ_v): Σ of v's own gains over √k. */
  readonly current: number;
  /** Candidates in the core's visiting order: own labels, then the neighbours' labels, then proxies. */
  readonly candidates: readonly LabelScore[];
  /** Candidates sorted by gain (the order the prefix is taken in). */
  readonly ranked: readonly LabelScore[];
  /** Value of every prefix of `ranked` within the cap: prefixValues[j-1] = (g_1+…+g_j)/√j. */
  readonly prefixValues: readonly number[];
  /** Positive gains the prefix may use (at most M). */
  readonly positive: number;
  readonly best: BestSet;
  /** The best set is v's own row. */
  readonly same: boolean;
  readonly improvement: number;
  /** The core moves v: a strictly better set that is not its row. */
  readonly moves: boolean;
}

/**
 * The visit of hig_cover_visit: candidates, gains, the best set, and whether
 * v moves. `options.cap` must be at least |σ_v|.
 */
export function bestResponse(game: OverlapGame, cover: Cover, vertex: number, options: ResponseOptions): BestResponse {
  const { gamma, cap, allowIsolation } = options;
  const row = cover[vertex];
  const k = row.length;
  if (!Number.isInteger(cap) || cap < 1) throw new RangeError(`max_memberships must be a positive integer (got ${cap})`);
  if (k > cap) throw new RangeError(`vertex ${vertex} holds ${k} labels, more than max_memberships = ${cap}`);

  // Candidates: v's labels first, then its neighbours' labels in edge order.
  const order: number[] = [...row];
  const seen = new Set(order);
  for (const u of game.neighbors[vertex]) {
    for (const label of cover[u]) {
      if (!seen.has(label)) {
        seen.add(label);
        order.push(label);
      }
    }
  }
  const candidates: LabelScore[] = order.map((label) => scoreLabel(game, cover, vertex, label, gamma));
  let maxGain = Math.max(...candidates.map((c) => c.gain));
  if (allowIsolation && maxGain < 0) maxGain = 0;

  // Labels v has no support from enter as proxies only when they can win
  // (the core keeps its heap of labels by mass only when isolation is off or γ < 0).
  const useHeap = gamma < 0 || !allowIsolation;
  if (useHeap && (gamma < 0 || maxGain < 0)) {
    const masses = labelMasses(cover);
    const others = [...labelMembers(cover).keys()]
      .filter((label) => !seen.has(label))
      .sort((a, b) => {
        const ma = masses.get(a)!;
        const mb = masses.get(b)!;
        if (ma !== mb) return gamma < 0 ? mb - ma : ma - mb;
        return a - b;
      });
    for (const label of others.slice(0, gamma < 0 ? cap : 1)) candidates.push(scoreLabel(game, cover, vertex, label, gamma));
  }

  let current = 0;
  for (const candidate of candidates) if (candidate.own) current += candidate.gain;
  current /= Math.sqrt(k);

  const ranked = [...candidates].sort(compareScores);
  const prefixValues: number[] = [];
  let sum = 0;
  for (let j = 1; j <= Math.min(cap, ranked.length); j += 1) {
    sum += ranked[j - 1].gain;
    prefixValues.push(sum / Math.sqrt(j));
  }
  const positive = Math.min(cap, ranked.filter((c) => c.gain > TIE).length);

  let best: BestSet;
  let same: boolean;
  if (positive === 0) {
    const single = ranked[0];
    if (allowIsolation && (!single || single.gain < -TIE)) {
      best = { labels: [FRESH], value: 0, kind: 'fresh' };
      same = false;
    } else {
      best = { labels: [single.label], value: single.gain, kind: 'single' };
      same = k === 1 && single.own;
    }
  } else {
    let jBest = 1;
    for (let j = 2; j <= positive; j += 1) if (prefixValues[j - 1] > prefixValues[jBest - 1] + TIE) jBest = j;
    const chosen = ranked.slice(0, jBest);
    best = { labels: chosen.map((c) => c.label).sort((a, b) => a - b), value: prefixValues[jBest - 1], kind: 'prefix' };
    same = jBest === k && chosen.every((c) => c.own);
  }
  const improvement = best.value - current;
  return {
    vertex,
    row,
    current,
    candidates,
    ranked,
    prefixValues,
    positive,
    best,
    same,
    improvement,
    moves: !same && improvement > TIE,
  };
}

/** The best-response certificate of a cover: every vertex's best set and its gain over staying. */
export function coverCertificate(game: OverlapGame, cover: Cover, options: ResponseOptions): BestResponse[] {
  return cover.map((_, vertex) => bestResponse(game, cover, vertex, options));
}

/** True when no vertex has a strictly better set (a Nash equilibrium of the labelled game). */
export function isEquilibrium(game: OverlapGame, cover: Cover, options: ResponseOptions): boolean {
  return cover.every((_, vertex) => !bestResponse(game, cover, vertex, options).moves);
}
