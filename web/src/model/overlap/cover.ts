/**
 * Covers: every vertex holds a non-empty set of community labels.
 *
 * A cover is stored as one sorted, duplicate-free row of labels per vertex,
 * as the C core stores it (native/core/cover/start.c: hig_cover_start).
 * Vertex v holding k labels puts the *intensity* f_v = 1/√k on each of them
 * (the unit-ℓ2 membership: Σ f_v² = 1), and label c has the *mass*
 * S_c = Σ_{u holds c} f_u (cover.c, header comment).
 */

import type { Adjacency } from '../graph';

export type Row = readonly number[];
export type Cover = readonly Row[];

/** The intensity 1/√k of a vertex that holds k labels. */
export function intensity(k: number): number {
  if (!Number.isInteger(k) || k < 1) throw new RangeError(`a vertex holds at least one label (got ${k})`);
  return 1 / Math.sqrt(k);
}

/** Sort a row and drop repeated labels (hig_cover_start does the same). */
export function normalizeRow(labels: readonly number[]): number[] {
  if (labels.length === 0) throw new RangeError('every vertex needs at least one label');
  for (const label of labels) {
    if (!Number.isInteger(label) || label < 0) throw new RangeError(`labels are non-negative integers (got ${label})`);
  }
  return [...new Set(labels)].sort((a, b) => a - b);
}

export function normalizeCover(rows: readonly (readonly number[])[]): number[][] {
  return rows.map(normalizeRow);
}

/**
 * Labels renumbered 0, 1, … in order of first appearance, rows sorted: the
 * form the package returns (native/core/cover/cover.c: hig_cover_result).
 */
export function relabelCover(cover: Cover): number[][] {
  const relabel = new Map<number, number>();
  return cover.map((row) =>
    row
      .map((label) => {
        let next = relabel.get(label);
        if (next === undefined) {
          next = relabel.size;
          relabel.set(label, next);
        }
        return next;
      })
      .sort((a, b) => a - b),
  );
}

/**
 * Labels renumbered 0, 1, … in numeric order (gaps closed): how the core
 * compacts a start (hig_compact_labels) and how the audit's Python facade
 * renumbers a cover before hig_audit_prepare.
 */
export function compactCover(cover: Cover): number[][] {
  const labels = occupiedLabels(cover);
  const rank = new Map(labels.map((label, index) => [label, index]));
  return cover.map((row) => row.map((label) => rank.get(label)!));
}

/** A stable key of a cover up to relabelling, e.g. "0|0|0,1|0,1|1|1". */
export function coverKey(cover: Cover): string {
  return relabelCover(cover)
    .map((row) => row.join(','))
    .join('|');
}

/** The labels some vertex holds, in increasing order. */
export function occupiedLabels(cover: Cover): number[] {
  return [...new Set(cover.flat())].sort((a, b) => a - b);
}

/** Members of each occupied label, in vertex order. */
export function labelMembers(cover: Cover): Map<number, number[]> {
  const members = new Map<number, number[]>();
  for (const label of occupiedLabels(cover)) members.set(label, []);
  cover.forEach((row, vertex) => row.forEach((label) => members.get(label)!.push(vertex)));
  return members;
}

/** S_c = Σ f_u over the holders of c, summed in vertex order. */
export function labelMasses(cover: Cover): Map<number, number> {
  const masses = new Map<number, number>();
  cover.forEach((row) => {
    const f = intensity(row.length);
    for (const label of row) masses.set(label, (masses.get(label) ?? 0) + f);
  });
  return masses;
}

/** The communities of a cover as vertex sets, ordered by label. */
export function communitiesOf(cover: Cover): { label: number; members: number[] }[] {
  return [...labelMembers(cover)].map(([label, members]) => ({ label, members }));
}

/** Labels shared by two sorted rows. */
export function sharedLabels(a: Row, b: Row): number {
  let i = 0;
  let j = 0;
  let shared = 0;
  while (i < a.length && j < b.length) {
    if (a[i] < b[j]) i += 1;
    else if (a[i] > b[j]) j += 1;
    else {
      shared += 1;
      i += 1;
      j += 1;
    }
  }
  return shared;
}

/**
 * Φ_γ of a cover, the exact potential of the game, written over pairs:
 *
 *   Φ_γ = Σ_{a<b} |σ_a ∩ σ_b| f_a f_b (A_ab − γ)
 *
 * Every pair that shares labels gains 1 − γ per unit of shared intensity
 * when linked and loses γ when not. With one label per vertex it is the
 * disjoint explainer's Φ_γ = (1 − γ)·(internal links) − γ·(internal
 * non-links). It differs by the constant γ n / 2 from the core's form
 * Σ_edges |σ_a ∩ σ_b| f_a f_b − (γ/2) Σ_c S_c² (native/README.md, "The
 * game"); the package reports (that form) / W as the quality (cover.c:
 * hig_cover_quality).
 */
export function coverPotential(adj: Adjacency, cover: Cover, gamma: number): number {
  let total = 0;
  for (let a = 0; a < cover.length; a += 1) {
    for (let b = a + 1; b < cover.length; b += 1) {
      const shared = sharedLabels(cover[a], cover[b]);
      if (shared === 0) continue;
      total += shared * intensity(cover[a].length) * intensity(cover[b].length) * ((adj[a][b] ? 1 : 0) - gamma);
    }
  }
  return total;
}

/** The package's quality: (Φ_γ − γ n / 2) / W, with W the number of edges. */
export function coverQuality(adj: Adjacency, cover: Cover, gamma: number, edgeCount: number): number {
  return (coverPotential(adj, cover, gamma) - (gamma * cover.length) / 2) / edgeCount;
}

/** Letters for labels in the interface: 0 → A, 1 → B, … */
export function labelName(label: number): string {
  if (label < 0) return '∗';
  let name = '';
  let rest = label;
  do {
    name = String.fromCharCode(65 + (rest % 26)) + name;
    rest = Math.floor(rest / 26) - 1;
  } while (rest >= 0);
  return name;
}

/** "{A,B}" for a row. */
export function formatRow(row: Row): string {
  return `{${row.map(labelName).join(',')}}`;
}
