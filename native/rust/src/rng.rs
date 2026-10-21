//! Random numbers (native/core/shared/random.c, native/binding/random.c): the
//! source of 32-bit words the visiting order is drawn from, unbiased bounded
//! draws, PCG32 for Rust callers and tests, and CPython's Mersenne Twister.

use crate::error::{Error, Kind, Result};

/// A source of random 32-bit words. An error aborts the call.
pub trait Rng {
    fn next32(&mut self) -> Result<u32>;
}

/// A uniform integer in [0, range), range >= 1 (Lemire's multiply-and-reject
/// on 32-bit words).
pub fn random_below<R: Rng + ?Sized>(rng: &mut R, range: u32) -> Result<u32> {
    let threshold = 0u32.wrapping_sub(range) % range;
    loop {
        let word = rng.next32()?;
        let product = word as u64 * range as u64;
        if product as u32 >= threshold {
            return Ok((product >> 32) as u32);
        }
    }
}

/// PCG32 (XSH-RR, 64-bit state), O'Neill 2014: `hig_pcg32_t`.
#[derive(Clone, Debug)]
pub struct Pcg32 {
    state: u64,
    increment: u64,
}

impl Pcg32 {
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut pcg = Pcg32 { state: 0, increment: (stream << 1) | 1 };
        pcg.step();
        pcg.state = pcg.state.wrapping_add(seed);
        pcg.step();
        pcg
    }

    fn step(&mut self) -> u32 {
        let old = self.state;
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rotation = (old >> 59) as u32;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.increment);
        xorshifted.rotate_right(rotation)
    }
}

impl Rng for Pcg32 {
    fn next32(&mut self) -> Result<u32> {
        Ok(self.step())
    }
}

pub const MT_N: usize = 624;
const MT_M: usize = 397;

/// CPython's Mersenne Twister state (Modules/_randommodule.c, MT19937 of
/// Matsumoto and Nishimura): the words of `random.Random.getstate()[1]`.
#[derive(Clone)]
pub struct Mt19937 {
    pub state: [u32; MT_N],
    pub index: usize,
}

impl Mt19937 {
    /// `init_genrand` of mt19937ar.c (tests only: Python seeds by key).
    pub fn from_seed32(seed: u32) -> Self {
        let mut state = [0u32; MT_N];
        state[0] = seed;
        for i in 1..MT_N {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { state, index: MT_N }
    }

    /// `init_by_array` of mt19937ar.c: what `random.Random(seed)` runs for an
    /// int seed (its absolute value's 32-bit words, least significant first).
    pub fn from_key(key: &[u32]) -> Self {
        let mut mt = Mt19937::from_seed32(19650218);
        let s = &mut mt.state;
        let key: &[u32] = if key.is_empty() { &[0] } else { key };
        let (mut i, mut j) = (1usize, 0usize);
        let mut k = MT_N.max(key.len());
        while k > 0 {
            s[i] = (s[i] ^ (s[i - 1] ^ (s[i - 1] >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= MT_N {
                s[0] = s[MT_N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = MT_N - 1;
        while k > 0 {
            s[i] = (s[i] ^ (s[i - 1] ^ (s[i - 1] >> 30)).wrapping_mul(1566083941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= MT_N {
                s[0] = s[MT_N - 1];
                i = 1;
            }
            k -= 1;
        }
        s[0] = 0x8000_0000;
        mt
    }

    /// `random.Random(seed)` for a non-negative int seed.
    pub fn python_seed(seed: u64) -> Self {
        let key: Vec<u32> = if seed >> 32 != 0 {
            vec![seed as u32, (seed >> 32) as u32]
        } else {
            vec![seed as u32]
        };
        Mt19937::from_key(&key)
    }

    #[inline]
    pub fn next(&mut self) -> u32 {
        const MAG01: [u32; 2] = [0, 0x9908_b0df];
        let s = &mut self.state;
        if self.index >= MT_N {
            let mut k = 0;
            while k < MT_N - MT_M {
                let y = (s[k] & 0x8000_0000) | (s[k + 1] & 0x7fff_ffff);
                s[k] = s[k + MT_M] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
                k += 1;
            }
            while k < MT_N - 1 {
                let y = (s[k] & 0x8000_0000) | (s[k + 1] & 0x7fff_ffff);
                s[k] = s[k + MT_M - MT_N] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
                k += 1;
            }
            let y = (s[MT_N - 1] & 0x8000_0000) | (s[0] & 0x7fff_ffff);
            s[MT_N - 1] = s[MT_M - 1] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
            self.index = 0;
        }
        let mut y = s[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }
}

impl Rng for Mt19937 {
    fn next32(&mut self) -> Result<u32> {
        Ok(self.next())
    }
}

/// A source that fails (tests of the error path).
pub struct Failing;

impl Rng for Failing {
    fn next32(&mut self) -> Result<u32> {
        Err(Error::new(Kind::Callback, "The random number generator failed."))
    }
}
