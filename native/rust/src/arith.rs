//! Certified arithmetic (native/core/internal.h): upward-rounded bounds of
//! rounding errors under the standard model of IEEE binary64 with rounding to
//! nearest. A bound is a non-negative double no smaller than the exact
//! quantity it bounds; every helper returns exactly 0 when all its arguments
//! are 0, so exact computations get exact bounds.

use crate::fma::fma;

/// u = 2^-53.
pub const U: f64 = f64::EPSILON / 2.0;
/// The smallest subnormal.
pub const ETA: f64 = 4.9406564584124654e-324;
/// DBL_MIN, the smallest normal double.
pub const DBL_MIN: f64 = f64::MIN_POSITIVE;
/// 2^-968: the error of a product at least this large is a double, and
/// fma(a, b, -fl(a b)) computes it exactly.
pub const EXACT_PRODUCT: f64 = DBL_MIN * 18014398509481984.0;
/// A mass is rebuilt when its accumulated error exceeds this fraction of it.
pub const REBUILD_RATIO: f64 = 16.0 * U;

/// x >= 0 rounded up: >= x (1 + u)^2, and >= x.
#[inline(always)]
pub fn up(x: f64) -> f64 {
    fma(x, 4.0 * U, x)
}

/// An upper bound of a + b for bounds a, b >= 0.
#[inline(always)]
pub fn add_up(a: f64, b: f64) -> f64 {
    up(a + b)
}

/// An upper bound of a b for bounds a, b >= 0.
#[inline(always)]
pub fn mul_up(a: f64, b: f64) -> f64 {
    let p = a * b;
    if p >= DBL_MIN {
        up(p)
    } else if a > 0.0 && b > 0.0 {
        p + ETA
    } else {
        0.0
    }
}

/// An upper bound of gamma_k = (1 + u)^k - 1, with room, for k below 2^40.
#[inline(always)]
pub fn gamma_k(k: f64) -> f64 {
    k * (1.125 * U)
}

/// a + b = sum + error exactly (Knuth's TwoSum).
#[inline(always)]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let error = (a - (s - bb)) + (b - bb);
    (s, error)
}

/// The improvement a - b of computed payoffs whose errors are bounded
/// together by `bound`: (d, e) with |exact improvement - d| <= e.
#[inline(always)]
pub fn difference(a: f64, b: f64, bound: f64) -> (f64, f64) {
    let (d, de) = two_sum(a, -b);
    (d, add_up(bound, de.abs()))
}

/// An upper bound of an exact gain known as d +- e (0 when d + e <= 0).
#[inline(always)]
pub fn gain_up(d: f64, e: f64) -> f64 {
    let t = d + e;
    if t > 0.0 {
        up(t)
    } else {
        0.0
    }
}

/// |exact a b - p| for p = fl(a b).
#[inline]
pub fn product_error(a: f64, b: f64, p: f64) -> f64 {
    if p.abs() >= EXACT_PRODUCT || a == 0.0 || b == 0.0 {
        return fma(a, b, -p).abs();
    }
    add_up(mul_up(U, p.abs()), ETA)
}

/// An upper bound of err / hi (hi > 0), at least eta.
#[inline(always)]
pub fn ratio_up(err: f64, hi: f64) -> f64 {
    up(err / hi) + ETA
}

/// hi + lo += x, both two-sums exact; the one rounded step adds its exact
/// error to err, a bound of |exact sum - (hi + lo)|.
#[inline(always)]
pub fn mass_add(hi: &mut f64, lo: &mut f64, err: &mut f64, x: f64) {
    let (s, e) = two_sum(*hi, x);
    let (e2, r) = two_sum(e, *lo);
    let (h, l) = two_sum(s, e2);
    *hi = h;
    *lo = l;
    if r != 0.0 {
        *err = add_up(*err, r.abs());
    }
}

/// The raw slack divided by |W| (rounded up): NaN when W = 0.
pub fn normalized_slack(slack: f64, total: f64) -> f64 {
    if total == 0.0 {
        return f64::NAN;
    }
    if slack > 0.0 {
        up(slack / total.abs()) + ETA
    } else {
        0.0
    }
}
