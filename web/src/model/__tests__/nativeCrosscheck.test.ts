import { describe, expect, it } from 'vitest';
import { findSinks } from '../dynamics';
import { EXAMPLE_GRAPH } from '../graph';
import { buildMetagraph } from '../metagraph';
import fixture from './fixtures/native-crosscheck.json';

/**
 * Cross-check against the native detector. The fixture was recorded with
 * web/scripts/crosscheck_native.py (hedonic + lucas-igraph); see its
 * `versions` field. The native Leiden local-moving phase may visit vertices
 * in any order, so we only require its result to be a sink of the browser
 * model, never a particular sink.
 */
describe('native community_hedonic cross-check', () => {
  const mg = buildMetagraph(EXAMPLE_GRAPH);

  it('uses the same graph as the explainer', () => {
    expect(fixture.graph.n).toBe(EXAMPLE_GRAPH.n);
    expect(fixture.graph.edges).toEqual(EXAMPLE_GRAPH.edges.map(([u, v]) => [u, v]));
  });

  it('only returns sinks of the browser model when isolation moves are allowed', () => {
    const runs = fixture.runs.filter((run) => run.allowIsolation);
    expect(runs.length).toBeGreaterThan(0);
    for (const run of runs) {
      expect(findSinks(mg, run.gamma)).toContain(run.partition);
    }
  });

  it('matches the unique sink whenever there is one', () => {
    for (const run of fixture.runs.filter((r) => r.allowIsolation)) {
      const sinks = findSinks(mg, run.gamma);
      if (sinks.length === 1) expect(run.partition).toBe(sinks[0]);
    }
  });

  it('keeps the grand coalition when isolation moves are disabled (the package default)', () => {
    // Without "leave to found a new community", no vertex can leave the grand coalition.
    for (const run of fixture.runs.filter((r) => !r.allowIsolation && r.start === 'grand')) {
      expect(run.partition).toBe(mg.grandCoalitionId);
    }
  });
});
