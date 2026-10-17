import { describe, expect, it } from 'vitest';
import {
  EPS,
  analyzeTradeoff,
  communityUtility,
  compareCommunities,
  familiarityIndex,
  orientTradeoff,
  utilityGain,
} from '../tradeoff';

describe('utility gain', () => {
  it('matches ΔU = Δd − γ(Δd + Δd̂)', () => {
    expect(utilityGain(-1, -2, 0.5)).toBeCloseTo(0.5);
    expect(utilityGain(-2, -1, 0.5)).toBeCloseTo(-0.5);
    expect(utilityGain(1, 0, 0)).toBe(1);
  });

  it('equals the difference of weighted scores friends·(1−γ) − strangers·γ', () => {
    for (const gamma of [0, 0.2, 1 / 3, 0.5, 0.9, 1]) {
      for (let fa = 0; fa <= 4; fa += 1) {
        for (let sa = 0; sa <= 4; sa += 1) {
          const a = { friends: fa, strangers: sa };
          const b = { friends: 4 - sa, strangers: fa };
          const tradeoff = compareCommunities(a, b);
          const expected = communityUtility(b.friends, b.strangers, gamma) - communityUtility(a.friends, a.strangers, gamma);
          expect(utilityGain(tradeoff.deltaFriends, tradeoff.deltaStrangers, gamma)).toBeCloseTo(expected, 12);
        }
      }
    }
  });
});

describe('Familiarity Index', () => {
  it('is Δd / (Δd + Δd̂)', () => {
    expect(familiarityIndex(-1, -2)).toBeCloseTo(1 / 3);
    expect(familiarityIndex(-2, -1)).toBeCloseTo(2 / 3);
    expect(familiarityIndex(1, 1)).toBeCloseTo(1 / 2);
    expect(familiarityIndex(2, -1)).toBeCloseTo(2);
  });

  it('is undefined (null) for a zero denominator', () => {
    expect(familiarityIndex(1, -1)).toBeNull();
    expect(familiarityIndex(0, 0)).toBeNull();
  });

  it('is symmetric under reversing the move', () => {
    for (let f = -4; f <= 4; f += 1) {
      for (let s = -4; s <= 4; s += 1) {
        expect(familiarityIndex(-f, -s)).toEqual(familiarityIndex(f, s));
      }
    }
  });
});

describe('trade-off classification', () => {
  it('distinguishes indifferent, clear and frustrated choices', () => {
    expect(analyzeTradeoff(0, 0)).toMatchObject({ kind: 'indifferent', paretoPreference: 'none', familiarity: null });
    expect(analyzeTradeoff(1, 1)).toMatchObject({ kind: 'frustrated', paretoPreference: 'depends' });
    expect(analyzeTradeoff(-1, -2)).toMatchObject({ kind: 'frustrated' });
    expect(analyzeTradeoff(1, -1)).toMatchObject({ kind: 'clear', paretoPreference: 'forward', familiarity: null });
    expect(analyzeTradeoff(0, -1)).toMatchObject({ kind: 'clear', paretoPreference: 'forward', familiarity: 0 });
    expect(analyzeTradeoff(2, 0)).toMatchObject({ kind: 'clear', paretoPreference: 'forward', familiarity: 1 });
    expect(analyzeTradeoff(-1, 1)).toMatchObject({ kind: 'clear', paretoPreference: 'backward' });
    expect(analyzeTradeoff(0, 3)).toMatchObject({ kind: 'clear', paretoPreference: 'backward' });
  });

  it('is frustrated exactly when 0 < F < 1', () => {
    for (let f = -5; f <= 5; f += 1) {
      for (let s = -5; s <= 5; s += 1) {
        const { kind, familiarity } = analyzeTradeoff(f, s);
        if (f === 0 && s === 0) expect(kind).toBe('indifferent');
        else if (familiarity !== null && familiarity > 0 && familiarity < 1) expect(kind).toBe('frustrated');
        else expect(kind).toBe('clear');
      }
    }
  });

  it('never lets γ ∈ [0, 1] reverse a clear choice', () => {
    for (let f = -4; f <= 4; f += 1) {
      for (let s = -4; s <= 4; s += 1) {
        const tradeoff = analyzeTradeoff(f, s);
        if (tradeoff.kind !== 'clear') continue;
        for (let step = 0; step <= 20; step += 1) {
          const { orientation } = orientTradeoff(tradeoff, step / 20);
          expect([tradeoff.paretoPreference, 'tie']).toContain(orientation);
        }
      }
    }
  });
});

describe('orientation at a resolution', () => {
  const leaveGrand = analyzeTradeoff(-1, -2); // vertex 3 leaving the grand coalition, F = 1/3

  it('favours friends below F and fewer strangers above F', () => {
    expect(orientTradeoff(leaveGrand, 0.2)).toMatchObject({ orientation: 'backward', reason: 'friends' });
    expect(orientTradeoff(leaveGrand, 0.5)).toMatchObject({ orientation: 'forward', reason: 'strangers' });
  });

  it('reports an exact tie at γ = F instead of picking a side', () => {
    expect(orientTradeoff(leaveGrand, 1 / 3)).toMatchObject({ orientation: 'tie', reason: 'threshold', gain: 0 });
  });

  it('treats floating-point noise at the threshold as a tie', () => {
    const tradeoff = analyzeTradeoff(3, 7); // F = 0.3
    const gamma = 0.1 + 0.2; // 0.30000000000000004
    expect(Math.abs(utilityGain(3, 7, gamma))).toBeLessThan(EPS);
    expect(orientTradeoff(tradeoff, gamma).orientation).toBe('tie');
  });

  it('ties clear choices only at the matching endpoint', () => {
    const fewerStrangers = analyzeTradeoff(0, -1); // F = 0
    expect(orientTradeoff(fewerStrangers, 0).orientation).toBe('tie');
    expect(orientTradeoff(fewerStrangers, 0.01)).toMatchObject({ orientation: 'forward', reason: 'clear' });
    const moreFriends = analyzeTradeoff(2, 0); // F = 1
    expect(orientTradeoff(moreFriends, 1).orientation).toBe('tie');
    expect(orientTradeoff(moreFriends, 0.99)).toMatchObject({ orientation: 'forward', reason: 'clear' });
  });

  it('keeps indifferent moves tied for every γ', () => {
    const none = analyzeTradeoff(0, 0);
    for (const gamma of [0, 0.5, 1]) {
      expect(orientTradeoff(none, gamma)).toMatchObject({ orientation: 'tie', reason: 'indifferent' });
    }
  });

  it('flips frustrated choices exactly at F', () => {
    for (let f = -4; f <= 4; f += 1) {
      for (let s = -4; s <= 4; s += 1) {
        const tradeoff = analyzeTradeoff(f, s);
        if (tradeoff.kind !== 'frustrated' || tradeoff.familiarity === null) continue;
        const F = tradeoff.familiarity;
        const below = orientTradeoff(tradeoff, Math.max(0, F - 1e-3));
        const above = orientTradeoff(tradeoff, Math.min(1, F + 1e-3));
        expect(below.orientation).not.toBe('tie');
        expect(above.orientation).not.toBe(below.orientation);
        expect(below.reason).toBe('friends');
        expect(above.reason).toBe('strangers');
      }
    }
  });
});
