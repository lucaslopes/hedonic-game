import { describe, expect, it } from 'vitest';
import {
  auditVertex,
  bestResponse,
  compactCover,
  coverKey,
  coverPotential,
  FRESH,
  inIntervals,
  intensity,
  isEquilibrium,
  labelMasses,
  movesOf,
  nashWindow,
  nextWord,
  occupiedLabels,
  OVERLAP_GRAPH,
  overlapGame,
  payoff,
  pythonRandom,
  randomBelow,
  regrets,
  relabelCover,
  runCover,
  stableIntervals,
  startCoverRun,
  stepCoverRun,
  type Cover,
} from '../overlap';
import { mulberry32 } from '../dynamics';

const game = overlapGame(OVERLAP_GRAPH);
const NATURAL: Cover = [[0], [0], [0, 1], [0, 1], [1], [1]];
const SQRT2 = Math.SQRT2;

/** Every non-empty set of at most `cap` labels from `labels` (FRESH at most once). */
function subsets(labels: readonly number[], cap: number): number[][] {
  const out: number[][] = [];
  const grow = (start: number, current: number[]) => {
    if (current.length > 0) out.push([...current]);
    if (current.length === cap) return;
    for (let i = start; i < labels.length; i += 1) {
      current.push(labels[i]);
      grow(i + 1, current);
      current.pop();
    }
  };
  grow(0, []);
  return out;
}

/** A seeded random cover with rows of 1..3 labels drawn from 0..4. */
function randomCover(seed: number): number[][] {
  let state = seed;
  const draw = (k: number) => {
    const [value, next] = mulberry32(state);
    state = next;
    return Math.floor(value * k);
  };
  return Array.from({ length: OVERLAP_GRAPH.n }, () => {
    const size = 1 + draw(3);
    return [...new Set(Array.from({ length: size }, () => draw(5)))].sort((a, b) => a - b);
  });
}

describe('Python’s random.Random, as the core draws from it', () => {
  it('yields getrandbits(32) of random.Random(0) and random.Random(7)', () => {
    const zero = pythonRandom(0);
    expect([nextWord(zero), nextWord(zero), nextWord(zero)]).toEqual([3626764237, 1654615998, 3255389356]);
    const seven = pythonRandom(7);
    expect([nextWord(seven), nextWord(seven), nextWord(seven)]).toEqual([1390851128, 4071050724, 647892279]);
  });

  it('draws bounded integers in range', () => {
    const mt = pythonRandom(3);
    for (let i = 0; i < 200; i += 1) {
      const value = randomBelow(mt, 1 + (i % 7));
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1 + (i % 7));
    }
  });
});

describe('intensities and masses', () => {
  it('spreads a unit of membership as 1/√k on each of k labels', () => {
    for (let k = 1; k <= 6; k += 1) {
      expect(intensity(k)).toBeCloseTo(1 / Math.sqrt(k), 15);
      expect(k * intensity(k) ** 2).toBeCloseTo(1, 14);
    }
    expect(() => intensity(0)).toThrow(RangeError);
  });

  it('sums the holders’ intensities into a label’s mass', () => {
    const masses = labelMasses(NATURAL);
    expect(masses.get(0)).toBeCloseTo(2 + SQRT2, 14);
    expect(masses.get(1)).toBeCloseTo(2 + SQRT2, 14);
  });

  it('relabels in order of first appearance like the package, and compacts in numeric order like the core', () => {
    expect(relabelCover([[5], [2, 5], [2]])).toEqual([[0], [0, 1], [1]]);
    expect(compactCover([[5], [2, 5], [2]])).toEqual([[1], [0, 1], [0]]);
    expect(occupiedLabels([[5], [2, 5], [2]])).toEqual([2, 5]);
  });
});

describe('a vertex’s best response', () => {
  it('scores labels by support − γ · crowd (vertex 0 in the natural cover)', () => {
    const response = bestResponse(game, NATURAL, 0, { gamma: 0.5, cap: 2, allowIsolation: false });
    const [a, b] = [0, 1].map((label) => response.candidates.find((c) => c.label === label)!);
    // A: friends 1 (f = 1), 2 and 3 (f = 1/√2 each); nobody else.
    expect(a.support).toBeCloseTo(1 + SQRT2, 14);
    expect(a.crowd).toBeCloseTo(1 + SQRT2, 14);
    expect(a.gain).toBeCloseTo(0.5 * (1 + SQRT2), 14);
    // B: friends 2 and 3 (1/√2 each), strangers 4 and 5 (1 each).
    expect(b.friends.map((h) => h.vertex)).toEqual([2, 3]);
    expect(b.strangers.map((h) => h.vertex)).toEqual([4, 5]);
    expect(b.gain).toBeCloseTo(SQRT2 - 0.5 * (2 + SQRT2), 14);
    expect(response.best.labels).toEqual([0]);
    expect(response.same).toBe(true);
    expect(response.moves).toBe(false);
  });

  it('adds the second label below γ = 3 − 2√2 (where g_B = (√2 − 1) g_A)', () => {
    const threshold = 3 - 2 * SQRT2;
    const below = bestResponse(game, NATURAL, 0, { gamma: threshold - 0.01, cap: 2, allowIsolation: false });
    expect(below.best.labels).toEqual([0, 1]);
    expect(below.moves).toBe(true);
    const above = bestResponse(game, NATURAL, 0, { gamma: threshold + 0.01, cap: 2, allowIsolation: false });
    expect(above.best.labels).toEqual([0]);
    // With a cap of one label the second label is never an option.
    expect(bestResponse(game, NATURAL, 0, { gamma: 0, cap: 1, allowIsolation: false }).best.labels).toEqual([0]);
  });

  it('keeps both labels for the shared vertices: every holder is a friend', () => {
    for (const gamma of [0, 0.3, 0.7, 0.99]) {
      const response = bestResponse(game, NATURAL, 2, { gamma, cap: 2, allowIsolation: true });
      expect(response.best.labels).toEqual([0, 1]);
      // A: friends 0, 1 (f = 1) and 3 (f = 1/√2); the same for B.
      expect(response.current).toBeCloseTo(((2 + 1 / SQRT2) * (1 - gamma) * 2) / SQRT2, 12);
    }
  });

  it('opens a fresh label only when every label loses and isolation is allowed', () => {
    const together: Cover = Array.from({ length: 6 }, () => [0]);
    // Vertex 0 in the grand coalition at γ = 0.9: 3 friends, 2 strangers: 3(0.1) − 2(0.9) < 0.
    const open = bestResponse(game, together, 0, { gamma: 0.9, cap: 2, allowIsolation: true });
    expect(open.best).toEqual({ labels: [FRESH], value: 0, kind: 'fresh' });
    expect(open.moves).toBe(true);
    const fixed = bestResponse(game, together, 0, { gamma: 0.9, cap: 2, allowIsolation: false });
    expect(fixed.best.labels).toEqual([0]);
    expect(fixed.moves).toBe(false);
  });

  it('treats a label held alone as worth exactly 0', () => {
    const cover: Cover = [[0], [1], [1], [1], [1], [1]];
    const response = bestResponse(game, cover, 0, { gamma: 0.9, cap: 2, allowIsolation: false });
    const own = response.candidates.find((c) => c.label === 0)!;
    expect(own.alone).toBe(true);
    expect(own.gain).toBe(0);
    expect(response.current).toBe(0);
  });

  it('finds the best of every feasible set, and that set is a prefix of the labels by gain', () => {
    let checked = 0;
    for (let seed = 1; seed <= 60; seed += 1) {
      const cover = randomCover(seed);
      for (const cap of [3, 4]) {
        for (const allowIsolation of [false, true]) {
          for (const gamma of [0, 0.2, 0.45, 0.8, 1]) {
            for (let v = 0; v < cover.length; v += 1) {
              const response = bestResponse(game, cover, v, { gamma, cap, allowIsolation });
              const labels = [...occupiedLabels(cover), ...(allowIsolation ? [FRESH] : [])];
              let brute = -Infinity;
              for (const set of subsets(labels, cap)) brute = Math.max(brute, payoff(game, cover, v, set, gamma));
              expect(response.best.value).toBeCloseTo(brute, 10);
              expect(payoff(game, cover, v, response.best.labels, gamma)).toBeCloseTo(response.best.value, 10);
              expect(response.current).toBeCloseTo(payoff(game, cover, v, cover[v], gamma), 10);
              if (response.best.kind === 'prefix') {
                const prefix = response.ranked.slice(0, response.best.labels.length).map((c) => c.label);
                expect([...prefix].sort((x, y) => x - y)).toEqual(response.best.labels);
              }
              checked += 1;
            }
          }
        }
      }
    }
    expect(checked).toBe(60 * 2 * 2 * 5 * 6);
  });

  it('only extends the prefix with positive gains, never beyond the cap', () => {
    for (let seed = 1; seed <= 40; seed += 1) {
      const cover = randomCover(seed);
      for (const cap of [3, 4]) {
        for (const gamma of [0.1, 0.5]) {
          for (let v = 0; v < cover.length; v += 1) {
            const response = bestResponse(game, cover, v, { gamma, cap, allowIsolation: true });
            expect(response.best.labels.length).toBeLessThanOrEqual(cap);
            if (response.best.kind === 'prefix') {
              for (const label of response.best.labels) {
                expect(response.candidates.find((c) => c.label === label)!.gain).toBeGreaterThan(0);
              }
            }
          }
        }
      }
    }
  });
});

describe('the potential', () => {
  it('reduces to the disjoint (1 − γ)·links − γ·non-links with one label each', () => {
    const halves: Cover = [[0], [0], [0], [1], [1], [1]];
    // {0,1,2}: 3 links; {3,4,5}: 3 links.
    expect(coverPotential(game.adj, halves, 0.25)).toBeCloseTo(6 * 0.75, 14);
    const together: Cover = Array.from({ length: 6 }, () => [0]);
    expect(coverPotential(game.adj, together, 0.5)).toBeCloseTo(11 * 0.5 - 4 * 0.5, 14);
  });

  it('changes by exactly the mover’s improvement on every accepted move', () => {
    let moves = 0;
    for (const gamma of [0, 0.15, 0.4, 0.5, 0.7, 1]) {
      for (const cap of [1, 2, 3]) {
        for (const allowIsolation of [false, true]) {
          for (const seed of [0, 1, 2, 3]) {
            const done = runCover(startCoverRun(game, null, { gamma, cap, allowIsolation, seed }));
            for (const visit of done.visits) {
              if (!visit.moved) {
                expect(visit.potentialAfter).toBe(visit.potentialBefore);
                continue;
              }
              moves += 1;
              expect(visit.response.improvement).toBeGreaterThan(0);
              expect(visit.potentialAfter - visit.potentialBefore).toBeCloseTo(visit.response.improvement, 12);
            }
          }
        }
      }
    }
    expect(moves).toBeGreaterThan(100);
  });
});

describe('local moving on the example', () => {
  it('converges from singletons to the two cliques sharing the edge at γ = 1/2, M = 2', () => {
    for (let seed = 0; seed < 8; seed += 1) {
      for (const allowIsolation of [false, true]) {
        const done = runCover(startCoverRun(game, null, { gamma: 0.5, cap: 2, allowIsolation, seed }));
        expect(done.status).toBe('equilibrium');
        expect(coverKey(done.rows)).toBe('0|0|0,1|0,1|1|1');
        expect(isEquilibrium(game, done.rows, { gamma: 0.5, cap: 2, allowIsolation })).toBe(true);
      }
    }
  });

  it('ends only after every vertex was visited without moving after the last move', () => {
    const done = runCover(startCoverRun(game, null, { gamma: 0.4, cap: 3, allowIsolation: true, seed: 5 }));
    const lastMove = done.visits.map((visit) => visit.moved).lastIndexOf(true);
    const settled = new Set(done.visits.slice(lastMove + 1).map((visit) => visit.vertex));
    expect(settled.size).toBe(OVERLAP_GRAPH.n);
    expect(stepCoverRun(done)).toBe(done);
  });

  it('plays the disjoint game when M = 1', () => {
    for (const gamma of [0.2, 0.5, 0.8]) {
      const done = runCover(startCoverRun(game, null, { gamma, cap: 1, allowIsolation: true, seed: 0 }));
      done.rows.forEach((row) => expect(row.length).toBe(1));
    }
  });

  it('records every move with its vertex and new row', () => {
    const done = runCover(startCoverRun(game, null, { gamma: 0.5, cap: 2, allowIsolation: false, seed: 0 }));
    expect(movesOf(done).length).toBe(done.moves);
    expect(done.visits.filter((visit) => visit.rescan).length).toBeGreaterThanOrEqual(1);
  });

  it('refuses a start with more labels than the cap', () => {
    expect(() => startCoverRun(game, NATURAL, { gamma: 0.5, cap: 1, allowIsolation: false, seed: 0 })).toThrow(RangeError);
  });
});

describe('stable intervals', () => {
  it('finds vertex 0 of the natural cover stable on [3 − 2√2, 1]', () => {
    const audit = auditVertex(game, NATURAL, 0, { cap: 2, allowIsolation: false });
    expect(audit.intervals.length).toBe(1);
    expect(audit.intervals[0][0]).toBeCloseTo(3 - 2 * SQRT2, 8);
    expect(audit.intervals[0][1]).toBe(1);
    expect(audit.cuts).toContain(0);
    expect(audit.cuts).toContain(1);
  });

  it('finds the shared vertices stable everywhere and the whole cover stable on [3 − 2√2, 1]', () => {
    const intervals = stableIntervals(game, NATURAL, { cap: 2, allowIsolation: true });
    expect(intervals[2]).toEqual([[0, 1]]);
    expect(intervals[3]).toEqual([[0, 1]]);
    const window = nashWindow(intervals);
    expect(window.length).toBe(1);
    expect(window[0][0]).toBeCloseTo(3 - 2 * SQRT2, 8);
    expect(window[0][1]).toBe(1);
  });

  it('finds singletons stable only at γ = 1 (up to the tolerance)', () => {
    const singletons = Array.from({ length: 6 }, (_, v) => [v]);
    for (const row of stableIntervals(game, singletons, { allowIsolation: false })) {
      expect(row.length).toBe(1);
      expect(row[0][0]).toBeGreaterThan(0.999);
      expect(row[0][1]).toBe(1);
    }
  });

  it('agrees with the regrets: zero regret inside the intervals, positive outside', () => {
    const grid = Array.from({ length: 41 }, (_, i) => i / 40);
    for (const cover of [NATURAL, [[0], [0], [0], [1], [1], [1]], Array.from({ length: 6 }, () => [0])] as Cover[]) {
      for (const allowIsolation of [false, true]) {
        const intervals = stableIntervals(game, cover, { cap: 2, allowIsolation });
        const { regret, tolerance } = regrets(game, cover, grid, { cap: 2, allowIsolation });
        intervals.forEach((row, v) =>
          grid.forEach((gamma, j) => {
            const near = row.some(([lo, hi]) => Math.abs(gamma - lo) < 1e-6 || Math.abs(gamma - hi) < 1e-6);
            if (!near) expect(regret[v][j] <= tolerance[v][j]).toBe(inIntervals(row, gamma));
          }),
        );
      }
    }
  });

  it('matches the best response away from the endpoints', () => {
    const intervals = stableIntervals(game, NATURAL, { cap: 2, allowIsolation: false });
    for (const gamma of [0.05, 0.3, 0.6, 0.95]) {
      intervals.forEach((row, v) => {
        const response = bestResponse(game, NATURAL, v, { gamma, cap: 2, allowIsolation: false });
        expect(response.improvement <= 1e-9).toBe(inIntervals(row, gamma));
      });
    }
  });

  it('refuses a cap below the largest row', () => {
    expect(() => stableIntervals(game, NATURAL, { cap: 1, allowIsolation: false })).toThrow(RangeError);
  });
});
