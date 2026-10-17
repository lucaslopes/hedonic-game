/** Display helpers shared by the model tests and the interface. */

const MINUS = '−';

/** Replace the ASCII hyphen of negative numbers with a typographic minus. */
function typographic(text: string): string {
  return text.startsWith('-') ? MINUS + text.slice(1) : text;
}

/** Clean −0 and tiny floating-point noise before printing. */
function tidy(x: number, digits: number): number {
  const rounded = Number(x.toFixed(digits));
  return Object.is(rounded, -0) ? 0 : rounded;
}

export function formatNumber(x: number, digits = 2): string {
  return typographic(tidy(x, digits).toFixed(digits));
}

/** Signed value: "+0.50", "−1.00", or "0.00". */
export function formatSigned(x: number, digits = 2): string {
  const value = tidy(x, digits);
  if (value > 0) return `+${value.toFixed(digits)}`;
  return typographic(value.toFixed(digits));
}

/** Integer delta with an explicit sign: "+1", "−2", "0". */
export function formatDelta(x: number): string {
  if (x > 0) return `+${x}`;
  return typographic(String(x));
}

/** The simplest fraction p/q (q ≤ maxDenominator) equal to x, if any. */
export function asFraction(
  x: number,
  maxDenominator = 12,
  eps = 1e-9,
): { numerator: number; denominator: number } | null {
  if (!Number.isFinite(x)) return null;
  for (let denominator = 1; denominator <= maxDenominator; denominator += 1) {
    const numerator = Math.round(x * denominator);
    if (Math.abs(numerator / denominator - x) <= eps) return { numerator, denominator };
  }
  return null;
}

/** "1/3", "−2/3", "1/2", "1" — or a decimal when x is not a simple fraction. */
export function formatFraction(x: number, digits = 3): string {
  const fraction = asFraction(x);
  if (!fraction) return formatNumber(x, digits);
  if (fraction.denominator === 1) return typographic(String(fraction.numerator));
  return typographic(`${fraction.numerator}/${fraction.denominator}`);
}

/**
 * Format a resolution value: two decimals, or "1/3 ≈ 0.333" when it sits
 * exactly on a fraction that two decimals cannot show.
 */
export function formatGamma(gamma: number): string {
  const fraction = asFraction(gamma);
  if (fraction && fraction.denominator > 1 && Math.abs(Number(gamma.toFixed(2)) - gamma) > 1e-9) {
    return `${fraction.numerator}/${fraction.denominator} ≈ ${gamma.toFixed(3)}`;
  }
  return formatNumber(gamma, 2);
}

/**
 * Parse a user-typed resolution: "0.5", ".5", "0,5", "1/3". Returns null for
 * anything that is not a finite number in [0, 1].
 */
export function parseGamma(text: string): number | null {
  const cleaned = text.trim().replace(',', '.').replace(MINUS, '-');
  if (cleaned === '') return null;
  let value: number;
  const fraction = /^(-?\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)$/.exec(cleaned);
  if (fraction) {
    const denominator = Number(fraction[2]);
    if (denominator === 0) return null;
    value = Number(fraction[1]) / denominator;
  } else if (/^-?(\d+\.?\d*|\.\d+)$/.test(cleaned)) {
    value = Number(cleaned);
  } else {
    return null;
  }
  if (!Number.isFinite(value) || value < 0 || value > 1) return null;
  return value;
}
