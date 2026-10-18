/**
 * The exact stability audit of a fixed cover: a port of
 * native/core/check/audit.c (hig_audit_prepare, hig_audit_intervals,
 * hig_audit_regrets), which `Game.stable_intervals` and `Game.regrets` call.
 *
 * For a vertex v and a candidate label c, the score is affine in γ:
 *
 *   score_c(γ) = a_c − γ b_c,   a_c = support of c,   b_c = S_c − f_v [c ∈ σ_v]
 *
 * and v's current utility is cur_a − γ cur_b with cur_a = Σ_{own} a_c / √k,
 * cur_b = Σ_{own} b_c / √k. Between two consecutive breakpoints (the γ where
 * two scores cross, or where the current utility crosses ±1, which switches
 * the tolerance) the order of the scores is fixed, so the best response is a
 * fixed prefix and "every prefix value ≤ current + tolerance" is a set of
 * linear inequalities in γ. Solving them cell by cell gives the closed
 * intervals of γ ∈ [0, 1] where v's current set is a best response:
 *
 *   best(γ) − current(γ) ≤ atol + rtol · max(1, |current(γ)|).
 *
 * The arithmetic follows the C code operation by operation (sequential sums,
 * division by √k, the two compensated sums of cur_a and cur_b), so the
 * endpoints match the package's to the last bit on these graphs.
 */

import { compactCover, intensity, normalizeCover, type Cover } from './cover';
import { FRESH, type OverlapGame } from './response';

export interface AuditOptions {
  /** max_memberships of a deviation; defaults to the largest row (and cannot be smaller). */
  readonly cap?: number;
  /** A deviation may open a fresh label scoring 0 ("open labels"). */
  readonly allowIsolation: boolean;
  readonly atol?: number;
  readonly rtol?: number;
}

export interface AuditEntry {
  /** Label in the cover's own numbering, or FRESH. */
  readonly label: number;
  /** a_c: the support. */
  readonly a: number;
  /** b_c: the mass without v. */
  readonly b: number;
  readonly own: boolean;
}

export interface VertexAudit {
  readonly vertex: number;
  readonly entries: readonly AuditEntry[];
  readonly curA: number;
  readonly curB: number;
  /** Sorted breakpoints, 0 and 1 included. */
  readonly cuts: readonly number[];
  /** Closed stable intervals in [0, 1]. */
  readonly intervals: readonly (readonly [number, number])[];
}

interface Prepared {
  readonly rows: number[][];
  /** Cover label of each compacted label. */
  readonly original: number[];
  readonly intensity: number[];
  readonly mass: number[];
  /** Labels by (mass, label), lightest first. */
  readonly lightest: number[];
  readonly maxRow: number;
}

function prepare(cover: Cover): Prepared {
  const normalized = normalizeCover(cover);
  const original = [...new Set(normalized.flat())].sort((a, b) => a - b);
  const rows = compactCover(normalized);
  const f = rows.map((row) => intensity(row.length));
  const mass = Array<number>(original.length).fill(0);
  rows.forEach((row, v) => row.forEach((label) => (mass[label] += f[v])));
  const lightest = mass.map((_, label) => label).sort((x, y) => (mass[x] !== mass[y] ? mass[x] - mass[y] : x - y));
  return { rows, original, intensity: f, mass, lightest, maxRow: Math.max(1, ...rows.map((row) => row.length)) };
}

function capOf(prepared: Prepared, options: AuditOptions): number {
  const cap = options.cap ?? prepared.maxRow;
  if (!Number.isInteger(cap) || cap < prepared.maxRow) {
    throw new RangeError('max_memberships must be at least the largest audit row');
  }
  return cap;
}

/** Candidates of a vertex (audit_vertex, pruned): own, supported, the lightest other, fresh first. */
function auditEntries(game: OverlapGame, prepared: Prepared, v: number, allowIsolation: boolean) {
  const support = new Map<number, number>();
  const own = new Set(prepared.rows[v]);
  for (const label of prepared.rows[v]) support.set(label, 0);
  for (const u of game.neighbors[v]) {
    const term = 1 * prepared.intensity[u];
    for (const label of prepared.rows[u]) support.set(label, (support.get(label) ?? 0) + term);
  }
  const kept = new Set([...support].filter(([label, value]) => own.has(label) || value > 0).map(([label]) => label));
  const extra = prepared.lightest.find((label) => !kept.has(label));
  if (extra !== undefined) {
    kept.add(extra);
    if (!support.has(extra)) support.set(extra, 0);
  }
  const labels = [...kept].sort((x, y) => x - y);
  const entries: (AuditEntry & { compact: number })[] = [];
  if (allowIsolation) entries.push({ label: FRESH, compact: -1, a: 0, b: 0, own: false });
  let curA = 0;
  let curB = 0;
  let correctionA = 0;
  let correctionB = 0;
  for (const label of labels) {
    const isOwn = own.has(label);
    const a = support.get(label)!;
    const b = prepared.mass[label] - (isOwn ? prepared.intensity[v] : 0);
    entries.push({ label: prepared.original[label], compact: label, a, b, own: isOwn });
    if (isOwn) {
      // CPython 3.12's compensated sum(), as audit.c reproduces it.
      let t = curA + a;
      correctionA += Math.abs(curA) >= Math.abs(a) ? curA - t + a : a - t + curA;
      curA = t;
      t = curB + b;
      correctionB += Math.abs(curB) >= Math.abs(b) ? curB - t + b : b - t + curB;
      curB = t;
    }
  }
  const norm = Math.sqrt(prepared.rows[v].length);
  return { entries, curA: (curA + correctionA) / norm, curB: (curB + correctionB) / norm, norm };
}

/** Entry indices by score at γ, best first; ties to the smaller index (audit_order). */
function orderAt(entries: readonly AuditEntry[], gamma: number): { order: number[]; scores: number[] } {
  const scores = entries.map((entry) => entry.a - gamma * entry.b);
  const order = entries.map((_, index) => index).sort((x, y) => (scores[x] !== scores[y] ? scores[y] - scores[x] : x - y));
  return { order, scores };
}

/** Every breakpoint of a vertex in (0, 1), plus 0 and 1, sorted (audit_cuts). */
function cutsOf(entries: readonly AuditEntry[], curA: number, curB: number): number[] {
  const cuts = [0, 1];
  for (let i = 0; i < entries.length; i += 1) {
    for (let j = i + 1; j < entries.length; j += 1) {
      const db = entries[i].b - entries[j].b;
      if (db !== 0) {
        const gamma = (entries[i].a - entries[j].a) / db;
        if (gamma > 0 && gamma < 1) cuts.push(gamma);
      }
    }
  }
  if (curB !== 0) {
    for (const level of [1, -1]) {
      const gamma = (curA - level) / curB;
      if (gamma > 0 && gamma < 1) cuts.push(gamma);
    }
  }
  return cuts.sort((x, y) => x - y);
}

/** The audit of one vertex: its candidates, breakpoints and stable intervals. */
export function auditVertex(game: OverlapGame, cover: Cover, vertex: number, options: AuditOptions): VertexAudit {
  const prepared = prepare(cover);
  return auditPrepared(game, prepared, vertex, capOf(prepared, options), options);
}

function auditPrepared(game: OverlapGame, prepared: Prepared, vertex: number, cap: number, options: AuditOptions): VertexAudit {
  const atol = options.atol ?? 1e-10;
  const rtol = options.rtol ?? 1e-9;
  const { entries, curA, curB } = auditEntries(game, prepared, vertex, options.allowIsolation);
  const cuts = cutsOf(entries, curA, curB);
  const top = Math.min(cap, entries.length);
  const intervals: [number, number][] = [];
  for (let cell = 0; cell + 1 < cuts.length; cell += 1) {
    const p = cuts[cell];
    const q = cuts[cell + 1];
    if (!(q > p)) continue;
    const mid = 0.5 * (p + q);
    let lo = p;
    let hi = q;
    let sumA = 0;
    let sumB = 0;
    let stable = true;
    const uMid = curA - mid * curB;
    const { order } = orderAt(entries, mid);
    let tauA: number;
    let tauB: number;
    if (Math.abs(uMid) <= 1) {
      tauA = atol + rtol;
      tauB = 0;
    } else {
      const sign = uMid > 0 ? 1 : -1;
      tauA = atol + rtol * sign * curA;
      tauB = -rtol * sign * curB;
    }
    for (let k = 1; k <= top; k += 1) {
      const entry = entries[order[k - 1]];
      sumA += entry.a;
      sumB += entry.b;
      const root = Math.sqrt(k);
      const c0 = sumA / root - curA - tauA;
      const c1 = -sumB / root + curB - tauB;
      if (c1 > 0) {
        const x = -c0 / c1;
        if (x < hi) hi = x;
      } else if (c1 < 0) {
        const x = -c0 / c1;
        if (x > lo) lo = x;
      } else if (c0 > 0) {
        stable = false;
        break;
      }
      if (lo > hi) {
        stable = false;
        break;
      }
    }
    if (!stable || lo > hi) continue;
    const last = intervals[intervals.length - 1];
    if (last && lo <= last[1] + 1e-15) {
      if (hi > last[1]) last[1] = hi;
    } else intervals.push([lo, hi]);
  }
  return { vertex, entries: entries.map(({ label, a, b, own }) => ({ label, a, b, own })), curA, curB, cuts, intervals };
}

/** Game.stable_intervals: the stable intervals of every vertex. */
export function stableIntervals(game: OverlapGame, cover: Cover, options: AuditOptions): [number, number][][] {
  const prepared = prepare(cover);
  const cap = capOf(prepared, options);
  return cover.map((_, v) => auditPrepared(game, prepared, v, cap, options).intervals.map(([lo, hi]) => [lo, hi] as [number, number]));
}

export interface Regrets {
  /** regret[v][j]: best utility minus current utility at gammas[j] (0 when ≤ 1e-15). */
  readonly regret: number[][];
  /** tolerance[v][j] = atol + rtol · max(1, |current|). */
  readonly tolerance: number[][];
}

/** Game.regrets: the best-response regret of every vertex on a grid of resolutions. */
export function regrets(game: OverlapGame, cover: Cover, gammas: readonly number[], options: AuditOptions): Regrets {
  const atol = options.atol ?? 1e-10;
  const rtol = options.rtol ?? 1e-9;
  const prepared = prepare(cover);
  const cap = capOf(prepared, options);
  const regret: number[][] = [];
  const tolerance: number[][] = [];
  cover.forEach((_, v) => {
    const { entries, norm } = auditEntries(game, prepared, v, options.allowIsolation);
    const top = Math.min(cap, entries.length);
    const rowRegret: number[] = [];
    const rowTolerance: number[] = [];
    for (const gamma of gammas) {
      const { order, scores } = orderAt(entries, gamma);
      let current = 0;
      entries.forEach((entry, i) => {
        if (entry.own) current += scores[i];
      });
      current /= norm;
      let prefix = 0;
      let best = -Infinity;
      for (let k = 1; k <= top; k += 1) {
        prefix += scores[order[k - 1]];
        const utility = prefix / Math.sqrt(k);
        if (utility > best) best = utility;
      }
      const diff = best - current;
      rowRegret.push(diff > 1e-15 ? diff : 0);
      rowTolerance.push(atol + rtol * (Math.abs(current) > 1 ? Math.abs(current) : 1));
    }
    regret.push(rowRegret);
    tolerance.push(rowTolerance);
  });
  return { regret, tolerance };
}

/** True when γ lies in one of the closed intervals. */
export function inIntervals(intervals: readonly (readonly [number, number])[], gamma: number): boolean {
  return intervals.some(([lo, hi]) => lo <= gamma && gamma <= hi);
}

/** hedonic.stability.nash_window: the intersection of every vertex's stable set. */
export function nashWindow(intervals: readonly (readonly (readonly [number, number])[])[]): [number, number][] {
  let window: [number, number][] = [[0, 1]];
  for (const row of intervals) {
    const next: [number, number][] = [];
    for (const [a, b] of window) {
      for (const [lo, hi] of row) {
        const x = Math.max(a, lo);
        const y = Math.min(b, hi);
        if (x <= y) next.push([x, y]);
      }
    }
    window = next;
    if (window.length === 0) break;
  }
  return window;
}

/** The stable share of [0, 1] (hedonic.stability.stable_measures, linear part, one vertex). */
export function stableLength(intervals: readonly (readonly [number, number])[]): number {
  return intervals.reduce((total, [lo, hi]) => total + (hi - lo), 0);
}
