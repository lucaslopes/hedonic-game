/**
 * Simple undirected graphs for the educational simulator.
 *
 * The explainer works on tiny graphs so that every partition can be
 * enumerated. The Python package (`hedonic.Game`) works on arbitrary
 * igraph graphs through the native lucas-igraph detector; nothing in this
 * module is a substitute for it.
 */

export type Edge = readonly [number, number];

export interface SimpleGraph {
  /** Number of vertices, labelled 0..n-1. */
  readonly n: number;
  /** Undirected, loop-free, duplicate-free edge list. */
  readonly edges: readonly Edge[];
}

/** Boolean adjacency matrix: `adj[i][j]` is true when i and j are linked. */
export type Adjacency = readonly (readonly boolean[])[];

/**
 * The four-vertex graph of Figure 1(a) in the SRC extended abstract
 * ("Community Detection as a Hedonic Game"): edges 0–1, 0–2, 0–3 and 1–2.
 * Vertices 0, 1, 2 form a triangle and vertex 3 hangs off vertex 0.
 */
export const EXAMPLE_GRAPH: SimpleGraph = {
  n: 4,
  edges: [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
  ],
};

/**
 * Drawing positions of Figure 1(a) on the unit circle, in SVG coordinates
 * (y grows downwards): 0 at the top, 1 on the right, 2 at the bottom and
 * 3 on the left.
 */
export const EXAMPLE_POSITIONS: readonly (readonly [number, number])[] = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
];

/** Build and validate the adjacency matrix of a simple undirected graph. */
export function adjacencyOf(graph: SimpleGraph): Adjacency {
  const { n, edges } = graph;
  if (!Number.isInteger(n) || n < 1) {
    throw new RangeError(`graph must have at least one vertex (got n=${n})`);
  }
  const adj = Array.from({ length: n }, () => Array<boolean>(n).fill(false));
  for (const [u, v] of edges) {
    if (!Number.isInteger(u) || !Number.isInteger(v) || u < 0 || v < 0 || u >= n || v >= n) {
      throw new RangeError(`edge ${u}–${v} references a vertex outside 0..${n - 1}`);
    }
    if (u === v) {
      throw new RangeError(`self-loop ${u}–${v} is not allowed in a simple graph`);
    }
    if (adj[u][v]) {
      throw new RangeError(`duplicate edge ${u}–${v}`);
    }
    adj[u][v] = true;
    adj[v][u] = true;
  }
  return adj;
}

/** Neighbours ("friends") of `vertex`, in increasing order. */
export function neighborsOf(adj: Adjacency, vertex: number): number[] {
  const out: number[] = [];
  adj[vertex].forEach((linked, other) => {
    if (linked) out.push(other);
  });
  return out;
}

/** Non-neighbours ("strangers") of `vertex`, excluding the vertex itself. */
export function nonNeighborsOf(adj: Adjacency, vertex: number): number[] {
  const out: number[] = [];
  adj[vertex].forEach((linked, other) => {
    if (!linked && other !== vertex) out.push(other);
  });
  return out;
}
