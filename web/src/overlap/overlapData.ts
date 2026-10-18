import { formatNumber } from '../model';
import { OVERLAP_GRAPH, labelName, overlapGame, type Cover, type Row } from '../model/overlap';

/** The overlap explainer's game: two cliques of four that share the edge 2–3. */
export const GAME = overlapGame(OVERLAP_GRAPH);

export type PresetName = 'natural' | 'halves' | 'together' | 'singletons';

export const PRESETS: Record<PresetName, Cover> = {
  natural: [[0], [0], [0, 1], [0, 1], [1], [1]],
  halves: [[0], [0], [0], [1], [1], [1]],
  together: [[0], [0], [0], [0], [0], [0]],
  singletons: [[0], [1], [2], [3], [4], [5]],
};

export const PRESET_ORDER: readonly PresetName[] = ['natural', 'halves', 'together', 'singletons'];

export type StartName = 'singletons' | 'together' | 'halves';

export const START_ORDER: readonly StartName[] = ['singletons', 'together', 'halves'];

/** Label colours: the four block colours first, then the remaining accents. */
const PALETTE = [
  'var(--block-0)',
  'var(--block-1)',
  'var(--block-2)',
  'var(--block-3)',
  'var(--clear)',
  'var(--mid)',
  'var(--metal)',
  'var(--tie)',
];

export function labelColor(label: number): string {
  if (label < 0) return 'var(--ink-3)';
  return PALETTE[label % PALETTE.length];
}

/** "{A, B}" for a row of labels. */
export function rowText(row: Row): string {
  return `{${row.map(labelName).join(', ')}}`;
}

/** "[0.17, 1.00] ∪ …", or null when the set is empty. */
export function intervalsText(intervals: readonly (readonly [number, number])[]): string | null {
  if (intervals.length === 0) return null;
  return intervals
    .map(([lo, hi]) => (Math.abs(hi - lo) < 1e-12 ? `{${formatNumber(lo, 3)}}` : `[${formatNumber(lo, 3)}, ${formatNumber(hi, 3)}]`))
    .join(' ∪ ');
}

/** Largest number of labels any vertex holds. */
export function widestRow(cover: Cover): number {
  return cover.reduce((max, row) => Math.max(max, row.length), 1);
}

export const CAPS = [1, 2, 3, 4] as const;
