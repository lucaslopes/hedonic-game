//! The fused multiply-add: `x y + z` rounded once (native/core/internal.h,
//! `hig_fma`, and native/core/softfma.h).
//!
//! Every fused multiply-add of the core is [`fma`], never a library call: the
//! instruction where the target has it (aarch64 always; x86_64 when the CPU
//! reports FMA, checked once), otherwise [`soft_fma`], a correctly rounded fma
//! in integer arithmetic. Both give the IEEE 754 result bit for bit, so the
//! choice only changes the speed. rustc never contracts `a * b + c` on its
//! own, so every other expression rounds as written.

/// `x y + z` rounded once (to nearest, ties to even).
#[inline(always)]
pub fn fma(x: f64, y: f64, z: f64) -> f64 {
    #[cfg(all(target_arch = "aarch64", not(feature = "force-soft-fma")))]
    {
        // fmadd: aarch64 always has it, and mul_add lowers to it.
        x.mul_add(y, z)
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "force-soft-fma")))]
    {
        if x86::has_fma() {
            // SAFETY: the CPU reports FMA (and the OS saves its state).
            unsafe { x86::fma_instruction(x, y, z) }
        } else {
            soft_fma(x, y, z)
        }
    }
    #[cfg(any(
        feature = "force-soft-fma",
        not(any(target_arch = "aarch64", target_arch = "x86_64"))
    ))]
    {
        soft_fma(x, y, z)
    }
}

/// Which fma this build runs: "instruction" or "soft" (x86_64: as detected).
pub fn fma_path() -> &'static str {
    #[cfg(all(target_arch = "aarch64", not(feature = "force-soft-fma")))]
    {
        "instruction"
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "force-soft-fma")))]
    {
        if x86::has_fma() {
            "dispatch: instruction"
        } else {
            "dispatch: soft"
        }
    }
    #[cfg(any(
        feature = "force-soft-fma",
        not(any(target_arch = "aarch64", target_arch = "x86_64"))
    ))]
    {
        "soft"
    }
}

#[cfg(all(target_arch = "x86_64", not(feature = "force-soft-fma")))]
mod x86 {
    use std::sync::atomic::{AtomicU8, Ordering};

    static STATE: AtomicU8 = AtomicU8::new(0); // 0 unknown, 1 soft, 2 instruction

    #[inline(always)]
    pub fn has_fma() -> bool {
        match STATE.load(Ordering::Relaxed) {
            2 => true,
            1 => false,
            _ => detect(),
        }
    }

    #[cold]
    fn detect() -> bool {
        let yes = std::arch::is_x86_feature_detected!("fma");
        STATE.store(if yes { 2 } else { 1 }, Ordering::Relaxed);
        yes
    }

    #[target_feature(enable = "fma")]
    pub unsafe fn fma_instruction(x: f64, y: f64, z: f64) -> f64 {
        use std::arch::x86_64::*;
        _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(x), _mm_set_sd(y), _mm_set_sd(z)))
    }
}

// ---- The soft fma (adapted from musl libc 1.2.5, src/math/fma.c; MIT) -------
//
// Copyright © 2005-2020 Rich Felker, et al.; the modifications of
// native/core/softfma.h, Copyright (C) 2026 Lucas Lopes Felipe, MIT licence.
// This is a line-by-line port of native/core/softfma.h: the exact 106-bit
// product of the significands, the addend aligned with a sticky bit, an exact
// 128-bit sum, and one rounding to nearest-even on the integer bits.

const ZEROINFNAN: i32 = 0x7ff - 0x3ff - 52 - 1;

struct Num {
    m: u64,
    e: i32,
    sign: u64,
}

#[inline]
fn normalize(x: f64) -> Num {
    let mut ix = x.to_bits();
    let mut e = (ix >> 52) as i32;
    let sign = (e & 0x800) as u64;
    e &= 0x7ff;
    ix &= (1u64 << 52) - 1;
    if e == 0 {
        if ix != 0 {
            // subnormal: x = ix 2^-1074
            let s = ix.leading_zeros() as i32 - 11;
            ix <<= s;
            e = 1 - s;
        } else {
            e = 0x800; // zero
        }
    }
    ix &= (1u64 << 52) - 1;
    ix |= 1u64 << 52;
    Num { m: ix << 1, e: e - (0x3ff + 52 + 1), sign }
}

#[inline]
fn mul(x: u64, y: u64) -> (u64, u64) {
    let (xlo, xhi) = (x & 0xffff_ffff, x >> 32);
    let (ylo, yhi) = (y & 0xffff_ffff, y >> 32);
    let t1 = xlo * ylo;
    let t2 = xlo * yhi + xhi * ylo;
    let t3 = xhi * yhi;
    let lo = t1.wrapping_add(t2 << 32);
    let hi = t3 + (t2 >> 32) + (t1 > lo) as u64;
    (hi, lo)
}

/// `x y + z` rounded once, in integer arithmetic (no library call, nothing
/// of the floating-point environment): bit for bit the IEEE result.
pub fn soft_fma(x: f64, y: f64, z: f64) -> f64 {
    let nx = normalize(x);
    let ny = normalize(y);
    let nz = normalize(z);

    if nx.e >= ZEROINFNAN || ny.e >= ZEROINFNAN {
        return x * y + z; // x y is exact, or infinite, or NaN
    }
    if nz.e >= ZEROINFNAN {
        if nz.e > ZEROINFNAN {
            return x * y; // z = +-0, x y != 0: round(x y)
        }
        return z; // z infinite or NaN, x y finite
    }

    let (mut rhi, mut rlo) = mul(nx.m, ny.m);
    let (zhi, zlo);
    let mut e = nx.e + ny.e;
    let mut d = nz.e - e;
    if d > 0 {
        if d < 64 {
            zlo = nz.m << d;
            zhi = nz.m >> (64 - d);
        } else {
            zlo = 0;
            zhi = nz.m;
            e = nz.e - 64;
            d -= 64;
            if d == 0 {
            } else if d < 64 {
                rlo = (rhi << (64 - d)) | (rlo >> d) | ((rlo << (64 - d)) != 0) as u64;
                rhi >>= d;
            } else {
                rlo = 1;
                rhi = 0;
            }
        }
    } else {
        zhi = 0;
        d = -d;
        if d == 0 {
            zlo = nz.m;
        } else if d < 64 {
            zlo = (nz.m >> d) | ((nz.m << (64 - d)) != 0) as u64;
        } else {
            zlo = 1;
        }
    }

    let mut sign = nx.sign ^ ny.sign;
    let samesign = (sign ^ nz.sign) == 0;
    let mut nonzero = true;
    if samesign {
        rlo = rlo.wrapping_add(zlo);
        rhi = rhi.wrapping_add(zhi).wrapping_add((rlo < zlo) as u64);
    } else {
        let t = rlo;
        rlo = rlo.wrapping_sub(zlo);
        rhi = rhi.wrapping_sub(zhi).wrapping_sub((t < rlo) as u64);
        if rhi >> 63 != 0 {
            rlo = 0u64.wrapping_sub(rlo);
            rhi = 0u64.wrapping_sub(rhi).wrapping_sub((rlo != 0) as u64);
            sign = (sign == 0) as u64;
        }
        nonzero = rhi != 0;
    }

    if nonzero {
        e += 64;
        d = rhi.leading_zeros() as i32 - 1;
        // d > 0 (the top bits of the product are 0)
        rhi = (rhi << d) | (rlo >> (64 - d)) | ((rlo << d) != 0) as u64;
    } else if rlo != 0 {
        d = rlo.leading_zeros() as i32 - 1;
        if d < 0 {
            rhi = (rlo >> 1) | (rlo & 1);
        } else {
            rhi = rlo << d;
        }
    } else {
        // exact zero: x y = -z, so fl(x y) = -z and the sum is +0
        return x * y + z;
    }
    e -= d;

    let biased = e + 1085;
    let k = if biased >= 1 { 10 } else { 11 - biased };
    let mut q;
    if k >= 64 {
        q = 0; // below 2^-1075: rounds to zero
    } else {
        q = rhi >> k;
        let rem = rhi & ((1u64 << k) - 1);
        let half = 1u64 << (k - 1);
        if rem > half || (rem == half && (q & 1) != 0) {
            q += 1;
        }
    }
    let mut bits;
    if biased >= 1 {
        if biased >= 0x7ff {
            bits = 0x7ffu64 << 52; // overflow: infinity
        } else {
            bits = ((biased as u64 - 1) << 52) + q;
            if bits >= 0x7ffu64 << 52 {
                bits = 0x7ffu64 << 52;
            }
        }
    } else {
        bits = q; // subnormal, or 2^52: the smallest normal after rounding
    }
    if sign != 0 {
        bits |= 1u64 << 63;
    }
    f64::from_bits(bits)
}
