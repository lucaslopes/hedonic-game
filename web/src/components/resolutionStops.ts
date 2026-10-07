/**
 * Keyboard and pointer behaviour of the resolution slider.
 *
 * Keyboard stops are every multiple of 0.01 plus the exact threshold values,
 * so arrow keys can land precisely on γ = 1/3 even though 1/3 is not a
 * multiple of 0.01. While dragging, values are rounded to 0.01 and, when
 * snapping is on, pulled onto a nearby threshold.
 */

const GRID = 100;
const SAME = 1e-9;

export const SNAP_RADIUS = 0.012;

export function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function keyboardStops(markers: readonly number[]): number[] {
  const stops = Array.from({ length: GRID + 1 }, (_, i) => i / GRID);
  for (const marker of markers) {
    if (marker >= 0 && marker <= 1 && !stops.some((stop) => Math.abs(stop - marker) < SAME)) stops.push(marker);
  }
  return stops.sort((a, b) => a - b);
}

export function nextStop(gamma: number, markers: readonly number[], direction: 1 | -1): number {
  const stops = keyboardStops(markers);
  if (direction > 0) return stops.find((stop) => stop > gamma + SAME) ?? 1;
  for (let i = stops.length - 1; i >= 0; i -= 1) if (stops[i] < gamma - SAME) return stops[i];
  return 0;
}

export function pageStep(gamma: number, direction: 1 | -1): number {
  return clamp01(Math.round((gamma + direction * 0.1) * GRID) / GRID);
}

/** Round a dragged value to the grid and optionally snap it onto a threshold. */
export function pointerValue(raw: number, markers: readonly number[], snap: boolean): number {
  const value = clamp01(raw);
  if (snap) {
    let best: number | null = null;
    for (const marker of markers) {
      if (Math.abs(marker - value) <= SNAP_RADIUS && (best === null || Math.abs(marker - value) < Math.abs(best - value))) {
        best = marker;
      }
    }
    if (best !== null) return best;
  }
  return Math.round(value * GRID) / GRID;
}
