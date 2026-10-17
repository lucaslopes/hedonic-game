import { describe, expect, it } from 'vitest';
import {
  bellNumber,
  blocksOf,
  canonicalize,
  enumeratePartitions,
  formatBlocks,
  isRestrictedGrowth,
  partitionId,
  shapeKey,
  shapeOf,
} from '../partitions';
import { EXAMPLE_GRAPH, adjacencyOf, neighborsOf, nonNeighborsOf } from '../graph';

describe('partition enumeration', () => {
  it('produces the Bell numbers B1..B7', () => {
    const expected = [1, 2, 5, 15, 52, 203, 877];
    expected.forEach((count, i) => {
      expect(enumeratePartitions(i + 1)).toHaveLength(count);
      expect(bellNumber(i + 1)).toBe(count);
    });
    expect(bellNumber(0)).toBe(1);
  });

  it('lists the 15 partitions of four vertices in lexicographic RGS order', () => {
    const ids = enumeratePartitions(4).map(partitionId);
    expect(ids).toEqual([
      '0000', '0001', '0010', '0011', '0012', '0100', '0101', '0102',
      '0110', '0111', '0112', '0120', '0121', '0122', '0123',
    ]);
  });

  it('only yields canonical, pairwise distinct partitions', () => {
    for (let n = 1; n <= 6; n += 1) {
      const all = enumeratePartitions(n);
      all.forEach((rgs) => expect(isRestrictedGrowth(rgs)).toBe(true));
      const keys = new Set(all.map((rgs) => formatBlocks(blocksOf(rgs))));
      expect(keys.size).toBe(all.length);
    }
  });

  it('rejects out-of-scope sizes', () => {
    expect(() => enumeratePartitions(0)).toThrow(RangeError);
    expect(() => enumeratePartitions(11)).toThrow(RangeError);
  });
});

describe('canonical partition identity', () => {
  it('treats label permutations as the same partition', () => {
    const base = canonicalize([7, 7, 7, 2]);
    expect(base).toEqual([0, 0, 0, 1]);
    expect(canonicalize([1, 1, 1, 0])).toEqual(base);
    expect(canonicalize([5, 5, 5, 9])).toEqual(base);
    expect(partitionId(canonicalize([3, 0, 3, 0]))).toBe(partitionId(canonicalize([0, 1, 0, 1])));
  });

  it('is idempotent and recognises restricted growth strings', () => {
    const labels = [4, 2, 4, 9, 2];
    const once = canonicalize(labels);
    expect(canonicalize(once)).toEqual(once);
    expect(isRestrictedGrowth(once)).toBe(true);
    expect(isRestrictedGrowth([1, 0])).toBe(false);
    expect(isRestrictedGrowth([0, 2])).toBe(false);
    expect(isRestrictedGrowth([])).toBe(false);
  });

  it('exposes blocks, labels and shapes', () => {
    const rgs = [0, 1, 1, 2];
    expect(blocksOf(rgs)).toEqual([[0], [1, 2], [3]]);
    expect(formatBlocks(blocksOf(rgs))).toBe('{0}{1,2}{3}');
    expect(shapeOf(rgs)).toEqual([2, 1, 1]);
    expect(shapeKey(shapeOf(rgs))).toBe('2+1+1');
  });
});

describe('example graph of Figure 1(a)', () => {
  const adj = adjacencyOf(EXAMPLE_GRAPH);

  it('has the edges 0–1, 0–2, 0–3, 1–2', () => {
    expect(neighborsOf(adj, 0)).toEqual([1, 2, 3]);
    expect(neighborsOf(adj, 1)).toEqual([0, 2]);
    expect(neighborsOf(adj, 2)).toEqual([0, 1]);
    expect(neighborsOf(adj, 3)).toEqual([0]);
    expect(nonNeighborsOf(adj, 3)).toEqual([1, 2]);
  });

  it('validates malformed graphs', () => {
    expect(() => adjacencyOf({ n: 2, edges: [[0, 0]] })).toThrow(/self-loop/);
    expect(() => adjacencyOf({ n: 2, edges: [[0, 2]] })).toThrow(/outside/);
    expect(() => adjacencyOf({ n: 2, edges: [[0, 1], [1, 0]] })).toThrow(/duplicate/);
    expect(() => adjacencyOf({ n: 0, edges: [] })).toThrow(RangeError);
  });
});
