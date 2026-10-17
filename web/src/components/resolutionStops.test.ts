import { describe, expect, it } from 'vitest';
import { keyboardStops, nextStop, pageStep, pointerValue, SNAP_RADIUS } from './resolutionStops';

describe('resolution slider stops', () => {
  const thresholds = [1 / 3, 1 / 2, 2 / 3];

  it('includes every hundredth plus the exact thresholds', () => {
    const stops = keyboardStops(thresholds);
    expect(stops).toContain(1 / 3);
    expect(stops).toContain(2 / 3);
    expect(stops.filter((s) => Math.abs(s - 0.5) < 1e-12)).toHaveLength(1);
    expect(stops).toHaveLength(101 + 2);
  });

  it('lands exactly on 1/3 between 0.33 and 0.34', () => {
    expect(nextStop(0.33, thresholds, 1)).toBe(1 / 3);
    expect(nextStop(1 / 3, thresholds, 1)).toBe(0.34);
    expect(nextStop(0.34, thresholds, -1)).toBe(1 / 3);
    expect(nextStop(1 / 3, thresholds, -1)).toBe(0.33);
  });

  it('clamps at the ends', () => {
    expect(nextStop(1, thresholds, 1)).toBe(1);
    expect(nextStop(0, thresholds, -1)).toBe(0);
    expect(pageStep(0.95, 1)).toBe(1);
    expect(pageStep(0.05, -1)).toBe(0);
    expect(pageStep(0.5, 1)).toBe(0.6);
  });

  it('rounds drags to 0.01 and snaps only within the radius', () => {
    expect(pointerValue(0.4567, thresholds, false)).toBe(0.46);
    expect(pointerValue(0.5 + SNAP_RADIUS / 2, thresholds, true)).toBe(0.5);
    expect(pointerValue(0.3301, thresholds, true)).toBe(1 / 3);
    expect(pointerValue(0.3301, thresholds, false)).toBe(0.33);
    expect(pointerValue(0.4, thresholds, true)).toBe(0.4);
    expect(pointerValue(-3, thresholds, true)).toBe(0);
  });
});
