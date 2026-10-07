/**
 * Layered layout of the metagraph, generated from the model.
 *
 * Columns follow the move-distance layers (grand coalition first, singleton
 * partition last), as in the historical Matplotlib figure. Column heights are
 * proportional to their size so small columns sit in the middle. Within each
 * column the order is chosen deterministically: every permutation of the
 * column is scored by the squared vertical stretch of its edges to other
 * columns, repeating sweeps until nothing improves. Columns hold at most six
 * partitions for four vertices, so the exhaustive search is instant.
 */

import type { Metagraph } from '../model';

export interface LayoutPoint {
  readonly id: string;
  readonly column: number;
  readonly row: number;
  /** Normalised coordinates in [0, 1]: x along the layers, y within a layer. */
  readonly x: number;
  readonly y: number;
}

export interface MetagraphLayout {
  readonly columns: readonly (readonly string[])[];
  readonly points: ReadonlyMap<string, LayoutPoint>;
}

function permutations<T>(items: readonly T[]): T[][] {
  if (items.length <= 1) return [[...items]];
  const out: T[][] = [];
  items.forEach((item, index) => {
    const rest = [...items.slice(0, index), ...items.slice(index + 1)];
    for (const tail of permutations(rest)) out.push([item, ...tail]);
  });
  return out;
}

export function layoutMetagraph(metagraph: Metagraph, maxSweeps = 8): MetagraphLayout {
  const columnCount = metagraph.layerCount;
  const columns: string[][] = Array.from({ length: columnCount }, () => []);
  metagraph.nodes.forEach((node) => columns[node.layer].push(node.id));
  const tallest = Math.max(...columns.map((column) => column.length));
  const rowY = (row: number, size: number) => (tallest <= 1 ? 0.5 : 0.5 + (row - (size - 1) / 2) / (tallest - 1));

  const y = new Map<string, number>();
  const place = (column: readonly string[]) => column.forEach((id, row) => y.set(id, rowY(row, column.length)));
  columns.forEach(place);

  const neighbours = new Map<string, { id: string; weight: number }[]>();
  for (const node of metagraph.nodes) {
    neighbours.set(
      node.id,
      metagraph
        .incident(node.id)
        .map((edge) => (edge.a === node.id ? edge.b : edge.a))
        .filter((other) => metagraph.node(other).layer !== node.layer)
        .map((other) => ({ id: other, weight: 1 / Math.abs(metagraph.node(other).layer - node.layer) })),
    );
  }

  const stretch = (column: readonly string[]) =>
    column.reduce((total, id, row) => {
      const own = rowY(row, column.length);
      return total + (neighbours.get(id) ?? []).reduce((sum, n) => sum + n.weight * ((y.get(n.id) ?? 0.5) - own) ** 2, 0);
    }, 0);

  for (let sweep = 0; sweep < maxSweeps; sweep += 1) {
    let improved = false;
    const order = sweep % 2 === 0 ? columns.map((_, c) => c) : columns.map((_, c) => columnCount - 1 - c);
    for (const c of order) {
      if (columns[c].length < 2) continue;
      let best = columns[c];
      let bestCost = stretch(best);
      for (const candidate of permutations(columns[c])) {
        const cost = stretch(candidate);
        if (cost < bestCost - 1e-12) {
          best = candidate;
          bestCost = cost;
        }
      }
      if (best !== columns[c]) {
        columns[c] = best;
        place(best);
        improved = true;
      }
    }
    if (!improved) break;
  }

  const points = new Map<string, LayoutPoint>();
  columns.forEach((column, c) =>
    column.forEach((id, row) =>
      points.set(id, { id, column: c, row, x: columnCount <= 1 ? 0.5 : c / (columnCount - 1), y: rowY(row, column.length) }),
    ),
  );
  return { columns, points };
}
