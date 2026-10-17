import { describe, expect, it } from 'vitest';
import {
  coverQuality,
  movesOf,
  OVERLAP_GRAPH,
  overlapGame,
  regrets,
  relabelCover,
  runCover,
  stableIntervals,
  startCoverRun,
} from '../overlap';
import fixture from './fixtures/overlap-native.json';

/**
 * Cross-check against the package. The fixture was recorded with
 * web/scripts/crosscheck_overlap_native.py (hedonic 2.0 and its C core; see
 * its `versions` and `calls` fields). The browser port draws the same
 * visiting order from the same seed, so it must reproduce every accepted
 * move, the returned cover and its quality; the audit port follows the C
 * arithmetic operation by operation, so the intervals and regrets agree to
 * the last bit.
 */
const game = overlapGame(OVERLAP_GRAPH);
const STARTS: Record<string, number[][] | null> = { singletons: null, together: Array.from({ length: 6 }, () => [0]) };


describe('overlap model against the native core', () => {
  it('uses the same graph', () => {
    expect(fixture.graph.n).toBe(OVERLAP_GRAPH.n);
    expect(fixture.graph.edges).toEqual(OVERLAP_GRAPH.edges.map(([u, v]) => [u, v]));
  });

  it('covers every resolution, cap, isolation policy, seed and start of the script', () => {
    expect(fixture.runs.length).toBe(2 * 8 * 3 * 2 * 3);
  });

  for (const run of fixture.runs) {
    const name = `${run.start}, γ = ${run.gamma}, M = ${run.cap}, isolation ${run.allowIsolation}, seed ${run.seed}`;
    it(`reproduces the moves and the result: ${name}`, () => {
      const options = { gamma: run.gamma, cap: run.cap, allowIsolation: run.allowIsolation, seed: run.seed };
      const done = runCover(startCoverRun(game, STARTS[run.start], options));
      expect(movesOf(done).map(({ vertex, row }) => [vertex, [...row]])).toEqual(run.moves);
      expect(relabelCover(done.rows)).toEqual(run.cover);
      expect(coverQuality(game.adj, done.rows, run.gamma, OVERLAP_GRAPH.edges.length)).toBeCloseTo(run.quality, 12);
      expect(stableIntervals(game, run.cover, { cap: run.cap, allowIsolation: run.allowIsolation })).toEqual(run.intervals);
    });
  }

  for (const audit of fixture.audits) {
    it(`audits the ${audit.cover} cover like Game.stable_intervals and Game.regrets (M = ${audit.cap}, isolation ${audit.allowIsolation})`, () => {
      const options = { cap: audit.cap, allowIsolation: audit.allowIsolation };
      expect(stableIntervals(game, audit.rows, options)).toEqual(audit.intervals);
      const grid = regrets(game, audit.rows, fixture.grid, options);
      expect(grid.regret).toEqual(audit.regret);
      expect(grid.tolerance).toEqual(audit.tolerance);
    });
  }
});
