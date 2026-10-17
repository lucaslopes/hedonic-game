import { describe, expect, it } from 'vitest';
import {
  advanceWalk,
  basinsOf,
  findSinks,
  improvingMoves,
  isSink,
  mulberry32,
  runWalk,
  stabilityCertificate,
  startWalk,
  WALK_POLICIES,
} from '../dynamics';
import { EXAMPLE_GRAPH } from '../graph';
import { buildMetagraph } from '../metagraph';

const mg = buildMetagraph(EXAMPLE_GRAPH);

describe('sink detection', () => {
  it('finds the grand coalition as the only sink for γ < 1/3', () => {
    for (const gamma of [0, 0.1, 0.25, 0.33]) expect(findSinks(mg, gamma)).toEqual(['0000']);
  });

  it('keeps both endpoints of the tied edge as sinks at γ = 1/3', () => {
    expect(findSinks(mg, 1 / 3)).toEqual(['0000', '0001']);
  });

  it('finds {0,1,2}{3} as the only sink for 1/3 < γ < 1, including the other thresholds', () => {
    for (const gamma of [0.34, 0.5, 0.6, 2 / 3, 0.75, 0.99]) expect(findSinks(mg, gamma)).toEqual(['0001']);
  });

  it('finds seven sinks at γ = 1: every partition into cliques', () => {
    const sinks = findSinks(mg, 1);
    expect(sinks).toEqual(['0001', '0012', '0102', '0110', '0112', '0120', '0123']);
    sinks.forEach((id) => expect(mg.node(id).internalNonEdges).toBe(0));
  });

  it('never counts a zero-gain move as improving', () => {
    // At γ = 1/2 six frustrated edges tie; none of their moves is "improving".
    for (const node of mg.nodes) {
      improvingMoves(mg, node.id, 0.5).forEach(({ gain }) => expect(gain).toBeGreaterThan(0));
    }
    expect(isSink(mg, '0000', 1 / 3)).toBe(true);
  });
});

describe('deterministic better-response walks', () => {
  it('leaves the grand coalition in one step at γ = 0.5 (vertex 3 avoids strangers)', () => {
    const walk = runWalk(mg, '0000', 0.5, 'best');
    expect(walk.status).toBe('sink');
    expect(walk.current).toBe('0001');
    expect(walk.steps).toHaveLength(1);
    expect(walk.steps[0]).toMatchObject({ from: '0000', to: '0001', gain: 0.5 });
    expect(walk.steps[0].move.vertex).toBe(3);
  });

  it('stays put when the start is already a sink', () => {
    const walk = runWalk(mg, '0000', 0.2, 'best');
    expect(walk).toMatchObject({ status: 'sink', current: '0000', steps: [] });
  });

  it('assembles the triangle from the singletons at γ = 0.5 with documented tie-breaks', () => {
    const walk = runWalk(mg, '0123', 0.5, 'best');
    expect(walk.steps.map((step) => step.to)).toEqual(['0012', '0001']);
    // Four moves tie at +0.5 in the first step; the lowest vertex (0) joins
    // the community with the smallest member ({1}).
    expect(walk.steps[0].move).toMatchObject({ vertex: 0, targetBlock: [1] });
    expect(walk.steps[0].options).toBe(8);
    expect(walk.steps[1].move).toMatchObject({ vertex: 2, targetBlock: [0, 1] });
  });

  it('merges everything at low resolution', () => {
    const walk = runWalk(mg, '0123', 0.2, 'best');
    expect(walk.steps.map((step) => step.to)).toEqual(['0012', '0001', '0000']);
  });

  it('follows the round-robin sweep policy reproducibly', () => {
    const walk = runWalk(mg, '0123', 0.5, 'sweep');
    expect(walk.steps.map((step) => [step.move.vertex, step.to])).toEqual([
      [0, '0012'],
      [2, '0001'],
    ]);
    expect(runWalk(mg, '0123', 0.5, 'sweep')).toEqual(walk);
  });

  it('reproduces seeded random walks exactly', () => {
    for (const seed of [1, 7, 2024]) {
      const first = runWalk(mg, '0123', 0.5, 'random', { seed });
      const second = runWalk(mg, '0123', 0.5, 'random', { seed });
      expect(second).toEqual(first);
      expect(first.current).toBe('0001');
    }
    const [value, next] = mulberry32(1);
    expect(value).toBeGreaterThanOrEqual(0);
    expect(value).toBeLessThan(1);
    expect(mulberry32(1)).toEqual([value, next]);
  });

  it('strictly increases the potential at every step and always ends at a sink', () => {
    for (let k = 0; k <= 20; k += 1) {
      const gamma = k / 20;
      for (const policy of WALK_POLICIES) {
        for (const node of mg.nodes) {
          const walk = runWalk(mg, node.id, gamma, policy, { seed: 3 });
          expect(walk.status).toBe('sink');
          expect(isSink(mg, walk.current, gamma)).toBe(true);
          for (const step of walk.steps) {
            expect(step.gain).toBeGreaterThan(0);
            expect(step.potentialAfter - step.potentialBefore).toBeCloseTo(step.gain, 12);
          }
        }
      }
    }
  });

  it('advances one step at a time and reports the sink as soon as it is reached', () => {
    let state = startWalk('0123');
    state = advanceWalk(mg, state, 0.5, 'best');
    expect(state).toMatchObject({ current: '0012', status: 'running' });
    state = advanceWalk(mg, state, 0.5, 'best');
    expect(state).toMatchObject({ current: '0001', status: 'sink' });
    expect(advanceWalk(mg, state, 0.5, 'best')).toBe(state);
  });
});

describe('basins and stability certificates', () => {
  it('sends every start to the unique sink when there is one', () => {
    const basins = basinsOf(mg, 0.5, 'best');
    expect(new Set(basins.values())).toEqual(new Set(['0001']));
  });

  it('splits the space into several basins at γ = 1', () => {
    const basins = basinsOf(mg, 1, 'best');
    expect(new Set(basins.values()).size).toBeGreaterThan(1);
    for (const [start, sink] of basins) {
      if (isSink(mg, start, 1)) expect(sink).toBe(start);
    }
  });

  it('shows that no vertex can strictly improve at a sink', () => {
    const certificate = stabilityCertificate(mg, '0001', 0.5);
    expect(certificate).toHaveLength(4);
    certificate.forEach(({ best }) => expect(best?.gain ?? 0).toBeLessThanOrEqual(0));
    const vertex3 = certificate[3];
    expect(vertex3.community).toEqual([3]);
    expect(vertex3.best?.move.targetBlock).toEqual([0, 1, 2]);
    expect(vertex3.best?.gain).toBeCloseTo(-0.5);
  });

  it('lists exact ties separately', () => {
    const certificate = stabilityCertificate(mg, '0001', 1 / 3);
    expect(certificate[3].ties.map((tie) => tie.move.to)).toEqual(['0000']);
  });
});
