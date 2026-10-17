/*
   hedonic -- a correctly rounded fused multiply-add in integer arithmetic.

   Adapted from musl libc 1.2.5, src/math/fma.c, which carries this notice:

   Copyright © 2005-2020 Rich Felker, et al.

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

   The modifications, Copyright (C) 2026 Lucas Lopes Felipe, are released
   under the same MIT licence. This file alone is MIT-licensed; the rest of
   the core is GPL-3.0-or-later, with which the MIT licence is compatible.
*/

/*
 * hig_soft_fma(x, y, z) is x y + z rounded once, to nearest with ties to
 * even: the IEEE 754 binary64 fusedMultiplyAdd in the default rounding mode,
 * bit for bit (a NaN result is some quiet NaN; its payload is unspecified).
 * The core uses it wherever the target has no fused multiply-add
 * instruction it may rely on at compile time (internal.h, hig_fma), so its
 * results never depend on the C library's fma().
 *
 * musl's algorithm: the exact product of the two 53-bit significands as a
 * 128-bit integer (four 32x32 -> 64-bit products), the addend aligned to it
 * with a sticky bit for every bit shifted out, an exact 128-bit sum or
 * difference, and its top 63 bits (the lowest one sticky). Changes from musl:
 *
 *   - the final rounding is done here in integer arithmetic (53 bits, or
 *     fewer for a subnormal result, ties to even, overflow to infinity) and
 *     the double is assembled from its bits, instead of converting to double
 *     and scaling with scalbn(): no libm call, and nothing depends on the
 *     floating-point environment (musl raises the underflow and inexact
 *     flags; nothing here does);
 *   - subnormal operands are normalized with a count of leading zeros
 *     instead of a multiplication by 2^63;
 *   - fma(x, y, +-0) with x y finite and nonzero returns fl(x y), whose zero,
 *     if x y underflows, has the sign of x y as IEEE 754 requires (musl
 *     returns fl(x y) + z, which is +0 for an underflowing negative product
 *     and z = +0);
 *   - C portability: memcpy instead of a union, no unary minus on unsigned
 *     operands, explicit conversions.
 *
 * The only floating-point operations left are on the special cases, where a
 * product is exact or the result is a zero, infinity or NaN that the
 * correctly rounded x * y and + give. Assumed: binary64 doubles and the
 * default environment (round to nearest, subnormals not flushed), as for the
 * rest of the core. tests/test_core.c compares it with the hardware
 * instruction where there is one; native/README.md reports the exhaustive
 * comparison.
 */

#ifndef HIG_SOFTFMA_H
#define HIG_SOFTFMA_H

#include <stdint.h>
#include <string.h>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

static inline uint64_t hig_sf_bits(double x) {
    uint64_t i;
    memcpy(&i, &x, sizeof i);
    return i;
}

static inline double hig_sf_double(uint64_t i) {
    double x;
    memcpy(&x, &i, sizeof x);
    return x;
}

/* The number of leading zero bits of x != 0. */
static inline int hig_sf_clz64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clzll((unsigned long long) x);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    unsigned long index;
    _BitScanReverse64(&index, x);
    return 63 - (int) index;
#else
    int n = 0;
    if (!(x >> 32)) { n += 32; x <<= 32; }
    if (!(x >> 48)) { n += 16; x <<= 16; }
    if (!(x >> 56)) { n += 8; x <<= 8; }
    if (!(x >> 60)) { n += 4; x <<= 4; }
    if (!(x >> 62)) { n += 2; x <<= 2; }
    if (!(x >> 63)) { n += 1; }
    return n;
#endif
}

/* x = sign m 2^e with m in [2^53, 2^54) and even ("top 10 bits and last bit
 * 0"); e >= HIG_SF_ZEROINFNAN marks an infinity or NaN (== ) or zero (>). */
typedef struct {
    uint64_t m;
    int e;
    int sign;
} hig_sf_num;

#define HIG_SF_ZEROINFNAN (0x7ff - 0x3ff - 52 - 1)

static inline hig_sf_num hig_sf_normalize(double x) {
    uint64_t ix = hig_sf_bits(x);
    int e = (int) (ix >> 52);
    hig_sf_num r;

    r.sign = e & 0x800;
    e &= 0x7ff;
    ix &= ((uint64_t) 1 << 52) - 1;
    if (!e) {
        if (ix) {                       /* subnormal: x = ix 2^-1074 */
            const int s = hig_sf_clz64(ix) - 11;
            ix <<= s;                   /* bit 52 set */
            e = 1 - s;
        } else {
            e = 0x800;                  /* zero */
        }
    }
    ix &= ((uint64_t) 1 << 52) - 1;
    ix |= (uint64_t) 1 << 52;
    r.m = ix << 1;
    r.e = e - (0x3ff + 52 + 1);
    return r;
}

/* *hi 2^64 + *lo = x y exactly. */
static inline void hig_sf_mul(uint64_t *hi, uint64_t *lo, uint64_t x, uint64_t y) {
    const uint64_t xlo = (uint32_t) x, xhi = x >> 32;
    const uint64_t ylo = (uint32_t) y, yhi = y >> 32;
    const uint64_t t1 = xlo * ylo;
    const uint64_t t2 = xlo * yhi + xhi * ylo;
    const uint64_t t3 = xhi * yhi;

    *lo = t1 + (t2 << 32);
    *hi = t3 + (t2 >> 32) + (t1 > *lo);
}

static inline double hig_soft_fma(double x, double y, double z) {
    const hig_sf_num nx = hig_sf_normalize(x);
    const hig_sf_num ny = hig_sf_normalize(y);
    const hig_sf_num nz = hig_sf_normalize(z);
    uint64_t rhi, rlo, zhi, zlo, q, rem, half, bits;
    int e, d, sign, samesign, nonzero, biased, k;

    if (nx.e >= HIG_SF_ZEROINFNAN || ny.e >= HIG_SF_ZEROINFNAN) {
        return x * y + z;               /* x y is exact, or infinite, or NaN */
    }
    if (nz.e >= HIG_SF_ZEROINFNAN) {
        if (nz.e > HIG_SF_ZEROINFNAN) {
            return x * y;               /* z = +-0, x y != 0: round(x y) */
        }
        return z;                       /* z infinite or NaN, x y finite */
    }

    /* r = x y; either the top 20 or 21 bits of rhi and the last 2 bits of
     * rlo are 0 */
    hig_sf_mul(&rhi, &rlo, nx.m, ny.m);

    /* align exponents: z <<= kz, r >>= kr, kz + kr == d, e = e + kr (== ez - kz) */
    e = nx.e + ny.e;
    d = nz.e - e;
    if (d > 0) {
        if (d < 64) {
            zlo = nz.m << d;
            zhi = nz.m >> (64 - d);
        } else {
            zlo = 0;
            zhi = nz.m;
            e = nz.e - 64;
            d -= 64;
            if (d == 0) {
            } else if (d < 64) {
                rlo = (rhi << (64 - d)) | (rlo >> d) | (uint64_t) ((rlo << (64 - d)) != 0);
                rhi = rhi >> d;
            } else {
                rlo = 1;
                rhi = 0;
            }
        }
    } else {
        zhi = 0;
        d = -d;
        if (d == 0) {
            zlo = nz.m;
        } else if (d < 64) {
            zlo = (nz.m >> d) | (uint64_t) ((nz.m << (64 - d)) != 0);
        } else {
            zlo = 1;
        }
    }

    /* add */
    sign = nx.sign ^ ny.sign;
    samesign = !(sign ^ nz.sign);
    nonzero = 1;
    if (samesign) {
        /* r += z */
        rlo += zlo;
        rhi += zhi + (rlo < zlo);
    } else {
        /* r -= z */
        const uint64_t t = rlo;
        rlo -= zlo;
        rhi = rhi - zhi - (t < rlo);
        if (rhi >> 63) {
            rlo = 0 - rlo;
            rhi = 0 - rhi - (uint64_t) (rlo != 0);
            sign = !sign;
        }
        nonzero = rhi != 0;
    }

    /* set rhi to the top 63 bits of the result (the last bit is sticky) */
    if (nonzero) {
        e += 64;
        d = hig_sf_clz64(rhi) - 1;
        /* note: d > 0 */
        rhi = (rhi << d) | (rlo >> (64 - d)) | (uint64_t) ((rlo << d) != 0);
    } else if (rlo) {
        d = hig_sf_clz64(rlo) - 1;
        if (d < 0) {
            rhi = (rlo >> 1) | (rlo & 1);
        } else {
            rhi = rlo << d;
        }
    } else {
        /* exact zero: x y = -z, so fl(x y) = -z and the sum is +0 */
        return x * y + z;
    }
    e -= d;

    /* The result is rhi 2^e, rhi in [2^62, 2^63) with a sticky last bit.
     * Keep 53 bits (k = 10 dropped) when it is normal, with biased exponent
     * e + 1085 (a normal double is M 2^(E - 1075), M in [2^52, 2^53));
     * otherwise keep the bits from 2^-1074 up (k = -1074 - e > 10). Round to
     * nearest, ties to even; the sticky bit is below the rounding position
     * (k >= 10), so it decides the ties correctly. */
    biased = e + 1085;
    k = biased >= 1 ? 10 : 11 - biased;
    if (k >= 64) {
        q = 0;                          /* below 2^-1075: rounds to zero */
    } else {
        q = rhi >> k;
        rem = rhi & (((uint64_t) 1 << k) - 1);
        half = (uint64_t) 1 << (k - 1);
        if (rem > half || (rem == half && (q & 1))) {
            q++;
        }
    }
    if (biased >= 1) {
        /* q in [2^52, 2^53]: the implicit bit carries into the exponent, as
         * does a rounding up to 2^53 */
        if (biased >= 0x7ff) {
            bits = (uint64_t) 0x7ff << 52;          /* overflow: infinity */
        } else {
            bits = ((uint64_t) (biased - 1) << 52) + q;
            if (bits >= (uint64_t) 0x7ff << 52) {
                bits = (uint64_t) 0x7ff << 52;
            }
        }
    } else {
        bits = q;   /* subnormal, or 2^52: the smallest normal after rounding */
    }
    if (sign) {
        bits |= (uint64_t) 1 << 63;
    }
    return hig_sf_double(bits);
}

#endif
