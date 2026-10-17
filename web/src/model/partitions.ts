/**
 * Canonical set partitions.
 *
 * A partition of the vertices {0, …, n-1} is stored as a *restricted growth
 * string* (RGS): `rgs[v]` is the community label of vertex v, labels start at
 * 0 with vertex 0, and each new label is exactly one larger than the largest
 * label seen so far. Every set partition has exactly one RGS, so relabelling
 * communities never produces a "different" partition.
 */

export type Rgs = readonly number[];

/** Relabel communities by order of first appearance (the canonical RGS). */
export function canonicalize(labels: readonly number[]): number[] {
  const relabel = new Map<number, number>();
  return labels.map((label) => {
    let canonical = relabel.get(label);
    if (canonical === undefined) {
      canonical = relabel.size;
      relabel.set(label, canonical);
    }
    return canonical;
  });
}

/** True when `labels` is already a canonical restricted growth string. */
export function isRestrictedGrowth(labels: readonly number[]): boolean {
  let max = -1;
  for (const label of labels) {
    if (!Number.isInteger(label) || label < 0 || label > max + 1) return false;
    max = Math.max(max, label);
  }
  return labels.length > 0;
}

/**
 * Every set partition of n vertices, as restricted growth strings in
 * lexicographic order. The count is the Bell number B_n.
 */
export function enumeratePartitions(n: number): number[][] {
  if (!Number.isInteger(n) || n < 1) {
    throw new RangeError(`n must be a positive integer (got ${n})`);
  }
  if (n > 10) {
    throw new RangeError(`enumerating B_${n} partitions is outside the scope of this explainer`);
  }
  const out: number[][] = [];
  const current = [0];
  const grow = (max: number) => {
    if (current.length === n) {
      out.push([...current]);
      return;
    }
    for (let label = 0; label <= max + 1; label += 1) {
      current.push(label);
      grow(Math.max(max, label));
      current.pop();
    }
  };
  grow(0);
  return out;
}

/** Bell number B_n computed with the Bell triangle. */
export function bellNumber(n: number): number {
  if (!Number.isInteger(n) || n < 0) throw new RangeError(`n must be a non-negative integer`);
  if (n === 0) return 1;
  let row = [1];
  for (let i = 1; i < n; i += 1) {
    const next = [row[row.length - 1]];
    for (const value of row) next.push(next[next.length - 1] + value);
    row = next;
  }
  return row[row.length - 1];
}

/** Communities of a canonical partition, ordered by their smallest vertex. */
export function blocksOf(rgs: Rgs): number[][] {
  const blocks: number[][] = [];
  rgs.forEach((label, vertex) => {
    (blocks[label] ??= []).push(vertex);
  });
  return blocks;
}

/** Stable identifier of a partition: its RGS, e.g. "0001" for {0,1,2}{3}. */
export function partitionId(rgs: Rgs): string {
  return rgs.length <= 10 ? rgs.join('') : rgs.join('.');
}

/** Set notation used in the papers, e.g. "{0,1,2}{3}". */
export function formatBlocks(blocks: readonly (readonly number[])[], separator = ''): string {
  return blocks.map((block) => `{${block.join(',')}}`).join(separator);
}

/** Community sizes in non-increasing order, e.g. [3, 1]. */
export function shapeOf(rgs: Rgs): number[] {
  return blocksOf(rgs)
    .map((block) => block.length)
    .sort((a, b) => b - a);
}

/** Integer-partition key of a shape, e.g. "3+1". */
export function shapeKey(sizes: readonly number[]): string {
  return sizes.join('+');
}
