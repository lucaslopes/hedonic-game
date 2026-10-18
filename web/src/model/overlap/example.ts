/**
 * The overlap explainer's network: two cliques of four that share an edge.
 *
 * Vertices 0, 1, 2, 3 form one clique and 2, 3, 4, 5 the other; the edge
 * 2–3 belongs to both. Vertices 2 and 3 are the natural overlap: each is
 * linked to everyone. Vertices 0, 1 and 4, 5 are strangers to each other.
 * The edges are listed in lexicographic order, which is the edge order the
 * package receives from `igraph.Graph(n=6, edges=...)` in the cross-check.
 */

import type { SimpleGraph } from '../graph';

export const OVERLAP_GRAPH: SimpleGraph = {
  n: 6,
  edges: [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 3],
    [2, 4],
    [2, 5],
    [3, 4],
    [3, 5],
    [4, 5],
  ],
};

/** Drawing positions (SVG coordinates, y down): the shared edge stands in the middle. */
export const OVERLAP_POSITIONS: readonly (readonly [number, number])[] = [
  [-1.25, -0.62],
  [-1.25, 0.62],
  [0, -0.95],
  [0, 0.95],
  [1.25, -0.62],
  [1.25, 0.62],
];
