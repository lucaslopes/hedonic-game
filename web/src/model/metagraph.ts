/**
 * The metagraph of partitions (arXiv:2509.03834, appendix "The Metagraph of
 * Partitions"): one metanode per set partition of the original graph, and
 * one metaedge between two partitions whenever a single vertex changing its
 * community turns one into the other.
 *
 * Everything here is generated from the input graph; nothing is hardcoded
 * for the four-vertex example.
 */

import { adjacencyOf, type Adjacency, type SimpleGraph } from './graph';
import { enumerateMoves, type Move } from './moves';
import { blocksOf, enumeratePartitions, formatBlocks, partitionId, shapeKey, shapeOf } from './partitions';
import { internalPairs, potentialFromPairs } from './quality';
import { analyzeTradeoff, orientTradeoff, type OrientationReason, type Tradeoff } from './tradeoff';

export interface MetaNode {
  /** Stable id: the restricted growth string, e.g. "0001". */
  readonly id: string;
  /** Position in lexicographic RGS order. */
  readonly index: number;
  readonly rgs: readonly number[];
  /** Communities ordered by smallest vertex. */
  readonly blocks: readonly (readonly number[])[];
  /** Set notation, e.g. "{0,1,2}{3}". */
  readonly label: string;
  readonly communityCount: number;
  /** Community sizes, largest first. */
  readonly sizes: readonly number[];
  /** Integer-partition key of the sizes, e.g. "3+1". */
  readonly shape: string;
  readonly isGrandCoalition: boolean;
  readonly isSingletons: boolean;
  /** Minimum number of single-vertex moves from the grand coalition. */
  readonly distanceFromGrand: number;
  /** Minimum number of single-vertex moves from the singleton partition. */
  readonly distanceFromSingletons: number;
  /**
   * Column of the layered drawing: the rank of
   * (distanceFromGrand − distanceFromSingletons), so the grand coalition is
   * layer 0 and the singleton partition is the last layer.
   */
  readonly layer: number;
  readonly internalEdges: number;
  readonly internalNonEdges: number;
}

export interface MetaEdge {
  /** `${a}-${b}` with a before b in RGS order. */
  readonly id: string;
  readonly a: string;
  readonly b: string;
  /** Moves a → b. Several vertices can produce the same partition. */
  readonly forward: readonly Move[];
  /** Moves b → a. */
  readonly backward: readonly Move[];
  /** The mover's trade-off for a → b (identical for every forward move). */
  readonly tradeoff: Tradeoff;
  /** Endpoint where the mover has more friends (null when Δd = 0). */
  readonly friendEnd: string | null;
  /** Endpoint where the mover has fewer strangers (null when Δd̂ = 0). */
  readonly strangerEnd: string | null;
  /** Both endpoints sit in the same layer of the drawing. */
  readonly sameLayer: boolean;
}

export interface Metagraph {
  readonly graph: SimpleGraph;
  readonly adjacency: Adjacency;
  readonly nodes: readonly MetaNode[];
  readonly edges: readonly MetaEdge[];
  readonly layerCount: number;
  readonly grandCoalitionId: string;
  readonly singletonsId: string;
  node(id: string): MetaNode;
  incident(id: string): readonly MetaEdge[];
  edgeBetween(x: string, y: string): MetaEdge | undefined;
  /** Every single-vertex move out of a partition, in canonical order. */
  movesFrom(id: string): readonly Move[];
}

function bfsDistances(start: string, neighbours: ReadonlyMap<string, readonly string[]>): Map<string, number> {
  const distance = new Map<string, number>([[start, 0]]);
  const queue = [start];
  for (let head = 0; head < queue.length; head += 1) {
    const current = queue[head];
    for (const next of neighbours.get(current) ?? []) {
      if (!distance.has(next)) {
        distance.set(next, (distance.get(current) ?? 0) + 1);
        queue.push(next);
      }
    }
  }
  return distance;
}

export function buildMetagraph(graph: SimpleGraph): Metagraph {
  const adjacency = adjacencyOf(graph);
  const partitions = enumeratePartitions(graph.n);
  const ids = partitions.map(partitionId);
  const indexOf = new Map(ids.map((id, index) => [id, index]));
  const grandCoalitionId = partitionId(Array(graph.n).fill(0));
  const singletonsId = partitionId(Array.from({ length: graph.n }, (_, v) => v));

  const movesById = new Map<string, Move[]>();
  partitions.forEach((rgs, index) => movesById.set(ids[index], enumerateMoves(adjacency, rgs)));

  // Group moves by unordered partition pair.
  const grouped = new Map<string, { a: string; b: string; forward: Move[]; backward: Move[] }>();
  for (const moves of movesById.values()) {
    for (const move of moves) {
      const forward = (indexOf.get(move.from) ?? 0) < (indexOf.get(move.to) ?? 0);
      const a = forward ? move.from : move.to;
      const b = forward ? move.to : move.from;
      const key = `${a}-${b}`;
      let group = grouped.get(key);
      if (!group) {
        group = { a, b, forward: [], backward: [] };
        grouped.set(key, group);
      }
      (forward ? group.forward : group.backward).push(move);
    }
  }

  const neighbours = new Map<string, string[]>(ids.map((id) => [id, []]));
  for (const { a, b } of grouped.values()) {
    neighbours.get(a)?.push(b);
    neighbours.get(b)?.push(a);
  }
  const fromGrand = bfsDistances(grandCoalitionId, neighbours);
  const fromSingletons = bfsDistances(singletonsId, neighbours);
  const layerValue = (id: string) => (fromGrand.get(id) ?? 0) - (fromSingletons.get(id) ?? 0);
  const layerValues = [...new Set(ids.map(layerValue))].sort((x, y) => x - y);

  const nodes: MetaNode[] = partitions.map((rgs, index) => {
    const id = ids[index];
    const blocks = blocksOf(rgs);
    const sizes = shapeOf(rgs);
    const pairs = internalPairs(adjacency, rgs);
    return {
      id,
      index,
      rgs,
      blocks,
      label: formatBlocks(blocks),
      communityCount: blocks.length,
      sizes,
      shape: shapeKey(sizes),
      isGrandCoalition: id === grandCoalitionId,
      isSingletons: id === singletonsId,
      distanceFromGrand: fromGrand.get(id) ?? Number.POSITIVE_INFINITY,
      distanceFromSingletons: fromSingletons.get(id) ?? Number.POSITIVE_INFINITY,
      layer: layerValues.indexOf(layerValue(id)),
      internalEdges: pairs.internalEdges,
      internalNonEdges: pairs.internalNonEdges,
    };
  });
  const nodeById = new Map(nodes.map((node) => [node.id, node]));

  const edges: MetaEdge[] = [...grouped.entries()]
    .map(([id, { a, b, forward, backward }]) => {
      const reference = forward[0];
      const tradeoff = analyzeTradeoff(reference.deltaFriends, reference.deltaStrangers);
      const friendEnd = tradeoff.deltaFriends > 0 ? b : tradeoff.deltaFriends < 0 ? a : null;
      const strangerEnd = tradeoff.deltaStrangers < 0 ? b : tradeoff.deltaStrangers > 0 ? a : null;
      return {
        id,
        a,
        b,
        forward,
        backward,
        tradeoff,
        friendEnd,
        strangerEnd,
        sameLayer: nodeById.get(a)?.layer === nodeById.get(b)?.layer,
      };
    })
    .sort((x, y) => (indexOf.get(x.a) ?? 0) - (indexOf.get(y.a) ?? 0) || (indexOf.get(x.b) ?? 0) - (indexOf.get(y.b) ?? 0));

  const incident = new Map<string, MetaEdge[]>(ids.map((id) => [id, []]));
  const between = new Map<string, MetaEdge>();
  for (const edge of edges) {
    incident.get(edge.a)?.push(edge);
    incident.get(edge.b)?.push(edge);
    between.set(`${edge.a}|${edge.b}`, edge);
    between.set(`${edge.b}|${edge.a}`, edge);
  }

  return {
    graph,
    adjacency,
    nodes,
    edges,
    layerCount: layerValues.length,
    grandCoalitionId,
    singletonsId,
    node(id) {
      const node = nodeById.get(id);
      if (!node) throw new RangeError(`unknown partition id ${id}`);
      return node;
    },
    incident: (id) => incident.get(id) ?? [],
    edgeBetween: (x, y) => between.get(`${x}|${y}`),
    movesFrom: (id) => movesById.get(id) ?? [],
  };
}

/** Φ_γ of a metanode. */
export function nodePotential(node: MetaNode, gamma: number): number {
  return potentialFromPairs(node, gamma);
}

export interface EdgeState {
  /** ΔU of the move a → b at γ (the move b → a has gain −ΔU). */
  readonly gain: number;
  readonly orientation: 'a-to-b' | 'b-to-a' | 'tie';
  /** Tail and head of the improving direction; null on a tie. */
  readonly from: string | null;
  readonly to: string | null;
  readonly reason: OrientationReason;
}

/** Direction of a metaedge once the resolution γ is fixed. */
export function edgeState(edge: MetaEdge, gamma: number): EdgeState {
  const oriented = orientTradeoff(edge.tradeoff, gamma);
  if (oriented.orientation === 'tie') {
    return { gain: 0, orientation: 'tie', from: null, to: null, reason: oriented.reason };
  }
  const forward = oriented.orientation === 'forward';
  return {
    gain: oriented.gain,
    orientation: forward ? 'a-to-b' : 'b-to-a',
    from: forward ? edge.a : edge.b,
    to: forward ? edge.b : edge.a,
    reason: oriented.reason,
  };
}

/** Distinct Familiarity Index values of the frustrated edges, ascending. */
export function frustrationThresholds(metagraph: Metagraph): number[] {
  const values: number[] = [];
  for (const edge of metagraph.edges) {
    const f = edge.tradeoff.familiarity;
    if (edge.tradeoff.kind === 'frustrated' && f !== null && !values.some((v) => Math.abs(v - f) < 1e-9)) {
      values.push(f);
    }
  }
  return values.sort((x, y) => x - y);
}
