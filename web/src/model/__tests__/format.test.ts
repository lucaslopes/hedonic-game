import { describe, expect, it } from 'vitest';
import { asFraction, formatDelta, formatFraction, formatGamma, formatNumber, formatSigned, parseGamma } from '../format';

describe('number formatting', () => {
  it('uses a typographic minus and hides negative zero', () => {
    expect(formatNumber(-0.5)).toBe('−0.50');
    expect(formatNumber(-0)).toBe('0.00');
    expect(formatNumber(-1e-12)).toBe('0.00');
    expect(formatSigned(0.5)).toBe('+0.50');
    expect(formatSigned(-1)).toBe('−1.00');
    expect(formatSigned(0)).toBe('0.00');
    expect(formatDelta(-2)).toBe('−2');
    expect(formatDelta(1)).toBe('+1');
    expect(formatDelta(0)).toBe('0');
  });

  it('recognises simple fractions', () => {
    expect(asFraction(1 / 3)).toEqual({ numerator: 1, denominator: 3 });
    expect(asFraction(0.1 + 0.2)).toEqual({ numerator: 3, denominator: 10 });
    expect(asFraction(Math.PI)).toBeNull();
    expect(formatFraction(0.5)).toBe('1/2');
    expect(formatFraction(2 / 3)).toBe('2/3');
    expect(formatFraction(1)).toBe('1');
    expect(formatFraction(-2)).toBe('−2');
  });

  it('formats resolution values', () => {
    expect(formatGamma(0.5)).toBe('0.50');
    expect(formatGamma(1 / 3)).toBe('1/3 ≈ 0.333');
    expect(formatGamma(0.25)).toBe('0.25');
    expect(formatGamma(0)).toBe('0.00');
  });

  it('parses typed resolution values', () => {
    expect(parseGamma('0.5')).toBe(0.5);
    expect(parseGamma('.25')).toBe(0.25);
    expect(parseGamma('0,75')).toBe(0.75);
    expect(parseGamma('1/3')).toBeCloseTo(1 / 3, 15);
    expect(parseGamma(' 2 / 3 ')).toBeCloseTo(2 / 3, 15);
    expect(parseGamma('1')).toBe(1);
    expect(parseGamma('1.5')).toBeNull();
    expect(parseGamma('-0.1')).toBeNull();
    expect(parseGamma('1/0')).toBeNull();
    expect(parseGamma('abc')).toBeNull();
    expect(parseGamma('')).toBeNull();
  });
});
