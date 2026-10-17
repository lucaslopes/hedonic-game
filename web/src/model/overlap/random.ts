/**
 * The C core's random visiting order, reproduced exactly.
 *
 * `Game.community_hedonic(seed=s)` hands the core `random.Random(s)`; the
 * binding (native/binding/random.c) draws its 32-bit words with CPython's
 * Mersenne Twister, i.e. `getrandbits(32)`. The core turns words into
 * bounded integers with Lemire's unbiased multiply-and-reject rule
 * (native/core/shared/random.c: hig_random_below). With the same seed this
 * module yields the same words and the same bounded draws, so the explainer
 * visits vertices in the same order as the package.
 */

const N = 624;
const M = 397;

export interface MersenneTwister {
  readonly state: Uint32Array;
  index: number;
}

function initGenrand(seed: number): MersenneTwister {
  const state = new Uint32Array(N);
  state[0] = seed >>> 0;
  for (let i = 1; i < N; i += 1) {
    const prev = state[i - 1] ^ (state[i - 1] >>> 30);
    state[i] = (Math.imul(1812433253, prev) + i) >>> 0;
  }
  return { state, index: N };
}

/** CPython's `init_by_array`, as `random.Random(seed)` calls it. */
function initByArray(key: readonly number[]): MersenneTwister {
  const mt = initGenrand(19650218);
  const s = mt.state;
  let i = 1;
  let j = 0;
  for (let k = Math.max(N, key.length); k > 0; k -= 1) {
    const prev = s[i - 1] ^ (s[i - 1] >>> 30);
    s[i] = ((s[i] ^ Math.imul(prev, 1664525)) + key[j] + j) >>> 0;
    i += 1;
    j += 1;
    if (i >= N) {
      s[0] = s[N - 1];
      i = 1;
    }
    if (j >= key.length) j = 0;
  }
  for (let k = N - 1; k > 0; k -= 1) {
    const prev = s[i - 1] ^ (s[i - 1] >>> 30);
    s[i] = ((s[i] ^ Math.imul(prev, 1566083941)) - i) >>> 0;
    i += 1;
    if (i >= N) {
      s[0] = s[N - 1];
      i = 1;
    }
  }
  s[0] = 0x80000000;
  return mt;
}

/**
 * The generator of `random.Random(seed)` for an integer seed. Python seeds
 * with the 32-bit words of |seed|, least significant first ([0] for 0).
 */
export function pythonRandom(seed: number): MersenneTwister {
  if (!Number.isSafeInteger(seed)) throw new RangeError(`seed must be a safe integer (got ${seed})`);
  let rest = Math.abs(seed);
  const key: number[] = [];
  do {
    key.push(rest % 0x100000000);
    rest = Math.floor(rest / 0x100000000);
  } while (rest > 0);
  return initByArray(key);
}

export function cloneTwister(mt: MersenneTwister): MersenneTwister {
  return { state: new Uint32Array(mt.state), index: mt.index };
}

/** The next 32-bit word: CPython's `genrand_uint32`, i.e. `getrandbits(32)`. */
export function nextWord(mt: MersenneTwister): number {
  const s = mt.state;
  if (mt.index >= N) {
    let k = 0;
    for (; k < N - M; k += 1) {
      const y = (s[k] & 0x80000000) | (s[k + 1] & 0x7fffffff);
      s[k] = s[k + M] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
    }
    for (; k < N - 1; k += 1) {
      const y = (s[k] & 0x80000000) | (s[k + 1] & 0x7fffffff);
      s[k] = s[k + (M - N)] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
    }
    const y = (s[N - 1] & 0x80000000) | (s[0] & 0x7fffffff);
    s[N - 1] = s[M - 1] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
    mt.index = 0;
  }
  let y = s[mt.index];
  mt.index += 1;
  y ^= y >>> 11;
  y ^= (y << 7) & 0x9d2c5680;
  y ^= (y << 15) & 0xefc60000;
  y ^= y >>> 18;
  return y >>> 0;
}

const TWO_32 = 0x100000000;

/**
 * A uniform integer in [0, range): hig_random_below. The 64-bit product
 * word × range is exact in a double while range < 2^21, far above the
 * explainer's graphs.
 */
export function randomBelow(mt: MersenneTwister, range: number): number {
  if (!Number.isInteger(range) || range < 1 || range >= 2 ** 21) {
    throw new RangeError(`range must be an integer in [1, 2^21) (got ${range})`);
  }
  const threshold = (TWO_32 - range) % range;
  for (;;) {
    const product = nextWord(mt) * range;
    const high = Math.floor(product / TWO_32);
    const low = product - high * TWO_32;
    if (low >= threshold) return high;
  }
}
