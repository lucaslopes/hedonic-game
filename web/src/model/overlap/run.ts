/**
 * Local moving on a cover, visit by visit, as the C core runs it
 * (native/core/cover/visit.c: hig_cover_run; native/core/shared/queue.c).
 *
 * Every vertex is queued once in a random order (a Fisher–Yates shuffle).
 * A visit takes the vertex at the head of the FIFO queue and computes its
 * best response; when the best set is strictly better, the vertex adopts it
 * and its neighbours are queued (each vertex is in the queue at most once).
 * A visit that does not move stamps the vertex with the number of moves so
 * far. When the queue runs dry, every vertex not stamped since the last move
 * is queued again, in a fresh random order; when there is none, every vertex
 * was visited without moving in the final state and the run stops. Every
 * move raises the exact potential by the mover's improvement, so the run
 * always stops.
 *
 * Label ids follow the core: a start is compacted to 0, 1, … in numeric
 * order, a vertex that opens a fresh label takes the most recently freed id
 * (a stack) or the next unused one, and ties between equal gains go to the
 * smaller id. With the seed of `community_hedonic(seed=…)` the visits, the
 * moves and the result are the package's (see the cross-check fixture).
 */

import { compactCover, coverPotential, normalizeCover, type Cover } from './cover';
import { bestResponse, FRESH, type BestResponse, type OverlapGame, type ResponseOptions } from './response';
import { cloneTwister, pythonRandom, randomBelow, type MersenneTwister } from './random';

export interface RunOptions extends ResponseOptions {
  /** The seed of `random.Random(seed)`, as `community_hedonic(seed=…)` uses it. */
  readonly seed: number;
}

export interface CoverVisit {
  /** 1-based visit number. */
  readonly index: number;
  readonly vertex: number;
  /** The visit began a new pass over the vertices that were not settled. */
  readonly rescan: boolean;
  readonly before: readonly number[];
  readonly after: readonly number[];
  readonly moved: boolean;
  readonly response: BestResponse;
  /** The id the vertex took for a fresh label, if it opened one. */
  readonly freshLabel: number | null;
  readonly potentialBefore: number;
  readonly potentialAfter: number;
}

interface QueueState {
  readonly ring: number[];
  head: number;
  count: number;
  readonly queued: boolean[];
  /** The first shuffle; rescans walk it to collect the unsettled vertices. */
  readonly order: readonly number[];
  /** Number of moves when the vertex was last visited without moving; −1 after it moved. */
  readonly settledAt: number[];
}

export interface CoverRun {
  readonly game: OverlapGame;
  readonly options: RunOptions;
  /** The start, compacted (internal label ids). */
  readonly start: readonly (readonly number[])[];
  readonly rows: readonly (readonly number[])[];
  readonly members: readonly number[];
  readonly freeIds: readonly number[];
  readonly labelCount: number;
  readonly queue: Readonly<QueueState>;
  readonly rng: MersenneTwister;
  readonly visits: readonly CoverVisit[];
  readonly moves: number;
  readonly status: 'running' | 'equilibrium';
}

function push(queue: QueueState, u: number): void {
  if (queue.queued[u]) return;
  const n = queue.ring.length;
  let tail = queue.head + queue.count;
  if (tail >= n) tail -= n;
  queue.ring[tail] = u;
  queue.queued[u] = true;
  queue.count += 1;
}

/** The singleton start: vertex v holds label v. */
export function singletonCover(n: number): number[][] {
  return Array.from({ length: n }, (_, v) => [v]);
}

/** Prepare a run from a start cover (null: singletons), drawing the first visiting order. */
export function startCoverRun(game: OverlapGame, start: Cover | null, options: RunOptions): CoverRun {
  const n = game.graph.n;
  const rows = compactCover(normalizeCover(start ?? singletonCover(n)));
  if (rows.length !== n) throw new RangeError(`the start has ${rows.length} rows for ${n} vertices`);
  rows.forEach((row, v) => {
    if (row.length > options.cap) throw new RangeError(`vertex ${v} starts with more than max_memberships = ${options.cap} labels`);
  });
  const labelCount = Math.max(...rows.flat()) + 1;
  const members = Array<number>(labelCount).fill(0);
  rows.forEach((row) => row.forEach((label) => (members[label] += 1)));

  const rng = pythonRandom(options.seed);
  const order = Array.from({ length: n }, (_, v) => v);
  for (let i = n - 1; i > 0; i -= 1) {
    const j = randomBelow(rng, i + 1);
    [order[i], order[j]] = [order[j], order[i]];
  }
  const queue: QueueState = {
    ring: Array<number>(n).fill(0),
    head: 0,
    count: 0,
    queued: Array<boolean>(n).fill(false),
    order,
    settledAt: Array<number>(n).fill(-1),
  };
  for (const v of order) push(queue, v);
  return { game, options, start: rows, rows, members, freeIds: [], labelCount, queue, rng, visits: [], moves: 0, status: 'running' };
}

/** One visit (or, when every vertex is settled, the end of the run). */
export function stepCoverRun(run: CoverRun): CoverRun {
  if (run.status !== 'running') return run;
  const { game, options } = run;
  const n = game.graph.n;
  const queue: QueueState = {
    ring: [...run.queue.ring],
    head: run.queue.head,
    count: run.queue.count,
    queued: [...run.queue.queued],
    order: run.queue.order,
    settledAt: [...run.queue.settledAt],
  };
  const rng = cloneTwister(run.rng);
  let rescan = false;

  if (queue.count === 0) {
    queue.head = 0;
    for (const u of queue.order) if (queue.settledAt[u] !== run.moves) push(queue, u);
    if (queue.count === 0) return { ...run, status: 'equilibrium' };
    for (let i = queue.count - 1; i > 0; i -= 1) {
      const j = randomBelow(rng, i + 1);
      [queue.ring[i], queue.ring[j]] = [queue.ring[j], queue.ring[i]];
    }
    rescan = true;
  }
  const vertex = queue.ring[queue.head];
  queue.head = queue.head + 1 === n ? 0 : queue.head + 1;
  queue.count -= 1;
  queue.queued[vertex] = false;

  const response = bestResponse(game, run.rows, vertex, options);
  const potentialBefore = coverPotential(game.adj, run.rows, options.gamma);
  const before = run.rows[vertex];
  if (!response.moves) {
    queue.settledAt[vertex] = run.moves;
    const visit: CoverVisit = {
      index: run.visits.length + 1,
      vertex,
      rescan,
      before,
      after: before,
      moved: false,
      response,
      freshLabel: null,
      potentialBefore,
      potentialAfter: potentialBefore,
    };
    return { ...run, queue, rng, visits: [...run.visits, visit] };
  }

  // The new row: the chosen labels sorted, a fresh label (if any) appended.
  const members = [...run.members];
  const freeIds = [...run.freeIds];
  let labelCount = run.labelCount;
  const after = response.best.labels.filter((label) => label !== FRESH);
  let freshLabel: number | null = null;
  if (response.best.labels.includes(FRESH)) {
    freshLabel = freeIds.length > 0 ? freeIds.pop()! : labelCount++;
    while (members.length < labelCount) members.push(0);
    after.push(freshLabel);
  }
  // Leave, then join, in increasing label order (the core merges the two sorted rows).
  for (const label of before) {
    if (after.includes(label)) continue;
    members[label] -= 1;
    if (members[label] === 0) freeIds.push(label);
  }
  for (const label of after) if (!before.includes(label)) members[label] += 1;

  const rows = run.rows.map((row, v) => (v === vertex ? after : row));
  const moves = run.moves + 1;
  queue.settledAt[vertex] = -1;
  for (const u of game.neighbors[vertex]) push(queue, u);
  const visit: CoverVisit = {
    index: run.visits.length + 1,
    vertex,
    rescan,
    before,
    after,
    moved: true,
    response,
    freshLabel,
    potentialBefore,
    potentialAfter: coverPotential(game.adj, rows, options.gamma),
  };
  return { ...run, rows, members, freeIds, labelCount, queue, rng, visits: [...run.visits, visit], moves };
}

/** Visit until every vertex is settled (or `maxVisits`, a safety net). */
export function runCover(run: CoverRun, maxVisits = 100000): CoverRun {
  let current = run;
  while (current.status === 'running') {
    if (current.visits.length >= maxVisits) throw new RangeError(`no equilibrium within ${maxVisits} visits`);
    current = stepCoverRun(current);
  }
  return current;
}

/** The accepted moves of a run, as (vertex, new row). */
export function movesOf(run: CoverRun): { vertex: number; row: readonly number[] }[] {
  return run.visits.filter((visit) => visit.moved).map((visit) => ({ vertex: visit.vertex, row: visit.after }));
}

/** The vertices still waiting in the queue, head first. */
export function queuedVertices(run: CoverRun): number[] {
  const n = run.game.graph.n;
  return Array.from({ length: run.queue.count }, (_, i) => run.queue.ring[(run.queue.head + i) % n]);
}
