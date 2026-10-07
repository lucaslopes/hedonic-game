import { describe, expect, it } from 'vitest';
import { EXAMPLE_GRAPH } from '../graph';
import { buildMetagraph, edgeState, frustrationThresholds, nodePotential } from '../metagraph';
import { applyMove, enumerateMoves } from '../moves';
import { partitionPotential } from '../quality';
import { utilityGain } from '../tradeoff';

const mg = buildMetagraph(EXAMPLE_GRAPH);
const GRAND = '0000';
const TRIANGLE_PLUS_ONE = '0001'; // {0,1,2}{3}
const SINGLETONS = '0123';

describe('metanodes', () => {
  it('has one node per partition (B4 = 15) with stable ids and labels', () => {
    expect(mg.nodes).toHaveLength(15);
    expect(new Set(mg.nodes.map((node) => node.id)).size).toBe(15);
    expect(mg.node(GRAND).label).toBe('{0,1,2,3}');
    expect(mg.node(TRIANGLE_PLUS_ONE).label).toBe('{0,1,2}{3}');
    expect(mg.node(SINGLETONS).label).toBe('{0}{1}{2}{3}');
  });

  it('marks the grand coalition and the singleton partition', () => {
    expect(mg.grandCoalitionId).toBe(GRAND);
    expect(mg.singletonsId).toBe(SINGLETONS);
    expect(mg.nodes.filter((node) => node.isGrandCoalition).map((node) => node.id)).toEqual([GRAND]);
    expect(mg.nodes.filter((node) => node.isSingletons).map((node) => node.id)).toEqual([SINGLETONS]);
  });

  it('exposes community counts and sizes', () => {
    const shapes = new Map<string, number>();
    for (const node of mg.nodes) shapes.set(node.shape, (shapes.get(node.shape) ?? 0) + 1);
    expect(Object.fromEntries(shapes)).toEqual({ '4': 1, '3+1': 4, '2+2': 3, '2+1+1': 6, '1+1+1+1': 1 });
    expect(mg.node('0112')).toMatchObject({ communityCount: 3, sizes: [2, 1, 1] });
  });

  it('computes move distances to both extremes and five layers', () => {
    const byShape = (shape: string) => mg.nodes.filter((node) => node.shape === shape);
    const expectAll = (shape: string, grand: number, singletons: number, layer: number) =>
      byShape(shape).forEach((node) =>
        expect([node.distanceFromGrand, node.distanceFromSingletons, node.layer]).toEqual([grand, singletons, layer]),
      );
    expectAll('4', 0, 3, 0);
    expectAll('3+1', 1, 2, 1);
    expectAll('2+2', 2, 2, 2);
    expectAll('2+1+1', 2, 1, 3);
    expectAll('1+1+1+1', 3, 0, 4);
    expect(mg.layerCount).toBe(5);
  });

  it('scores each partition with the CPM potential Φγ', () => {
    // Grand coalition: 4 internal links, 2 internal non-links (1–3 and 2–3).
    expect(nodePotential(mg.node(GRAND), 0)).toBe(4);
    expect(nodePotential(mg.node(GRAND), 0.5)).toBeCloseTo(1);
    expect(nodePotential(mg.node(GRAND), 1)).toBeCloseTo(-2);
    expect(nodePotential(mg.node(TRIANGLE_PLUS_ONE), 0.5)).toBeCloseTo(1.5);
    expect(nodePotential(mg.node(SINGLETONS), 0.7)).toBe(0);
  });
});

describe('one-vertex-move adjacency', () => {
  it('has 52 metaedges and no self-loops', () => {
    expect(mg.edges).toHaveLength(52);
    mg.edges.forEach((edge) => expect(edge.a).not.toBe(edge.b));
  });

  it('connects exactly the partitions one move apart (brute force check)', () => {
    const ids = mg.nodes.map((node) => node.id);
    for (const x of ids) {
      for (const y of ids) {
        if (x === y) continue;
        const rx = mg.node(x).rgs;
        const oneMove = rx.some((_, vertex) =>
          [...new Set(rx), null].some((label) => applyMove(rx, vertex, label).join('') === y),
        );
        expect(Boolean(mg.edgeBetween(x, y))).toBe(oneMove);
      }
    }
  });

  it('records the moved vertex, origin and destination', () => {
    const edge = mg.edgeBetween(GRAND, TRIANGLE_PLUS_ONE);
    expect(edge).toBeDefined();
    const [move] = edge?.forward ?? [];
    expect(move).toMatchObject({
      vertex: 3,
      from: GRAND,
      to: TRIANGLE_PLUS_ONE,
      originBlock: [0, 1, 2, 3],
      targetBlock: [],
      createsCommunity: true,
      removesCommunity: false,
      friendsBefore: 1,
      strangersBefore: 2,
      friendsAfter: 0,
      strangersAfter: 0,
      deltaFriends: -1,
      deltaStrangers: -2,
    });
    const [back] = edge?.backward ?? [];
    expect(back).toMatchObject({ vertex: 3, removesCommunity: true, createsCommunity: false, targetBlock: [0, 1, 2] });
  });

  it('keeps several movers when different vertices reach the same partition', () => {
    const shared = mg.edges.filter((edge) => edge.forward.length > 1);
    expect(shared).toHaveLength(12);
    shared.forEach((edge) => expect(edge.backward).toHaveLength(2));
    const split = mg.edgeBetween('0011', '0122'); // {0,1}{2,3} → {0}{1}{2,3}
    expect(split?.forward.map((move) => move.vertex)).toEqual([0, 1]);
  });

  it('gives every mover of an edge the same deltas, reversed on the way back', () => {
    for (const edge of mg.edges) {
      for (const move of edge.forward) {
        expect([move.deltaFriends, move.deltaStrangers]).toEqual([edge.tradeoff.deltaFriends, edge.tradeoff.deltaStrangers]);
      }
      for (const move of edge.backward) {
        expect(move.deltaFriends + edge.tradeoff.deltaFriends).toBe(0);
        expect(move.deltaStrangers + edge.tradeoff.deltaStrangers).toBe(0);
      }
    }
  });

  it('computes friend and stranger deltas for the grand-coalition exits', () => {
    const exit = (vertex: number) => enumerateMoves(mg.adjacency, [0, 0, 0, 0]).find((m) => m.vertex === vertex);
    expect(exit(0)).toMatchObject({ deltaFriends: -3, deltaStrangers: 0 });
    expect(exit(1)).toMatchObject({ deltaFriends: -2, deltaStrangers: -1 });
    expect(exit(2)).toMatchObject({ deltaFriends: -2, deltaStrangers: -1 });
    expect(exit(3)).toMatchObject({ deltaFriends: -1, deltaStrangers: -2 });
  });

  it('draws 12 edges inside the three-community layer (omitted in the paper figure)', () => {
    const inside = mg.edges.filter((edge) => edge.sameLayer);
    expect(inside).toHaveLength(12);
    inside.forEach((edge) => expect(mg.node(edge.a).shape).toBe('2+1+1'));
  });
});

describe('edge types and the Familiarity Index', () => {
  it('splits the 52 moves into 37 clear, 9 frustrated and 6 indifferent', () => {
    const count = (kind: string) => mg.edges.filter((edge) => edge.tradeoff.kind === kind).length;
    expect([count('clear'), count('frustrated'), count('indifferent')]).toEqual([37, 9, 6]);
    // The 40 edges of the historical figure are the ones outside the three-community layer.
    const figureEdges = mg.edges.filter((edge) => !edge.sameLayer);
    expect(figureEdges).toHaveLength(40);
    expect(figureEdges.filter((edge) => edge.tradeoff.kind === 'frustrated')).toHaveLength(9);
    expect(figureEdges.filter((edge) => edge.tradeoff.kind === 'indifferent')).toHaveLength(0);
  });

  it('finds the thresholds 1/3, 1/2 and 2/3', () => {
    const thresholds = frustrationThresholds(mg);
    expect(thresholds).toHaveLength(3);
    [1 / 3, 1 / 2, 2 / 3].forEach((value, i) => expect(thresholds[i]).toBeCloseTo(value, 12));
    const frustrated = mg.edges.filter((edge) => edge.tradeoff.kind === 'frustrated');
    const at = (f: number) => frustrated.filter((edge) => Math.abs((edge.tradeoff.familiarity ?? -1) - f) < 1e-9).length;
    expect([at(1 / 3), at(1 / 2), at(2 / 3)]).toEqual([1, 6, 2]);
  });

  it('identifies which end offers more friends and which fewer strangers', () => {
    const edge = mg.edgeBetween(GRAND, TRIANGLE_PLUS_ONE);
    expect(edge?.friendEnd).toBe(GRAND);
    expect(edge?.strangerEnd).toBe(TRIANGLE_PLUS_ONE);
    for (const e of mg.edges.filter((x) => x.tradeoff.kind === 'frustrated')) {
      expect(e.friendEnd).not.toBeNull();
      expect(e.friendEnd).not.toBe(e.strangerEnd);
    }
  });

  it('orients edges by the resolution, with ties exactly at γ = F', () => {
    const edge = mg.edgeBetween(GRAND, TRIANGLE_PLUS_ONE);
    if (!edge) throw new Error('missing edge');
    expect(edgeState(edge, 0.2)).toMatchObject({ from: TRIANGLE_PLUS_ONE, to: GRAND, reason: 'friends' });
    expect(edgeState(edge, 0.5)).toMatchObject({ from: GRAND, to: TRIANGLE_PLUS_ONE, reason: 'strangers' });
    expect(edgeState(edge, 1 / 3)).toMatchObject({ orientation: 'tie', from: null, to: null });
    const ties = mg.edges.filter((e) => edgeState(e, 0.5).orientation === 'tie');
    expect(ties).toHaveLength(6 + 6); // six indifferent edges + six frustrated edges with F = 1/2
  });

  it('turns the metagraph into a DAG once γ is fixed (away from thresholds)', () => {
    for (const gamma of [0.1, 0.4, 0.6, 0.9]) {
      const out = new Map<string, string[]>(mg.nodes.map((node) => [node.id, []]));
      mg.edges.forEach((edge) => {
        const state = edgeState(edge, gamma);
        if (state.from && state.to) out.get(state.from)?.push(state.to);
      });
      const visiting = new Set<string>();
      const done = new Set<string>();
      const hasCycle = (id: string): boolean => {
        if (visiting.has(id)) return true;
        if (done.has(id)) return false;
        visiting.add(id);
        const cyclic = (out.get(id) ?? []).some(hasCycle);
        visiting.delete(id);
        done.add(id);
        return cyclic;
      };
      expect(mg.nodes.some((node) => hasCycle(node.id))).toBe(false);
    }
  });
});

describe('exact potential', () => {
  it('changes Φγ by exactly the mover’s utility gain on every move', () => {
    for (const gamma of [0, 0.15, 1 / 3, 0.5, 0.8, 1]) {
      for (const node of mg.nodes) {
        for (const move of mg.movesFrom(node.id)) {
          const delta =
            partitionPotential(mg.adjacency, mg.node(move.to).rgs, gamma) -
            partitionPotential(mg.adjacency, node.rgs, gamma);
          expect(delta).toBeCloseTo(utilityGain(move.deltaFriends, move.deltaStrangers, gamma), 12);
        }
      }
    }
  });
});

describe('generality', () => {
  it('builds the metagraph of other small graphs (triangle, path of five)', () => {
    const triangle = buildMetagraph({ n: 3, edges: [[0, 1], [1, 2], [0, 2]] });
    expect(triangle.nodes).toHaveLength(5);
    // 3 exits from the grand coalition, 3 among the two-community partitions, 3 into the singletons.
    expect(triangle.edges).toHaveLength(9);
    const path5 = buildMetagraph({ n: 5, edges: [[0, 1], [1, 2], [2, 3], [3, 4]] });
    expect(path5.nodes).toHaveLength(52);
    path5.nodes.forEach((node) => expect(Number.isFinite(node.distanceFromGrand)).toBe(true));
  });
});
