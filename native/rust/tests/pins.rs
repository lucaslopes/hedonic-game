//! The bit pins of native/tests/test_core.c (Zachary, random.Random(seed)'s
//! order), fixed points, the verifier on equilibria, and the soft fma
//! against the instruction. Fast: `cargo test --release`.

use hedonic_core_rs::cover::{cover, Cover};
use hedonic_core_rs::fma::{fma, soft_fma};
use hedonic_core_rs::graph::{Edges, Graph, Options};
use hedonic_core_rs::partition::partition;
use hedonic_core_rs::queue::Callbacks;
use hedonic_core_rs::rng::{Failing, Mt19937, Pcg32};
use hedonic_core_rs::trace::Trace;
use hedonic_core_rs::verify::{verify_cover, verify_partition};

const ZACHARY: [i64; 156] = [
    0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 10, 0, 11, 0, 12, 0, 13, 0, 17, 0, 19, 0, 21, 0,
    31, 1, 2, 1, 3, 1, 7, 1, 13, 1, 17, 1, 19, 1, 21, 1, 30, 2, 3, 2, 7, 2, 27, 2, 28, 2, 32, 2, 9, 2,
    8, 2, 13, 3, 7, 3, 12, 3, 13, 4, 6, 4, 10, 5, 6, 5, 10, 5, 16, 6, 16, 8, 30, 8, 32, 8, 33, 9, 33,
    13, 33, 14, 32, 14, 33, 15, 32, 15, 33, 18, 32, 18, 33, 19, 33, 20, 32, 20, 33, 22, 32, 22, 33,
    23, 25, 23, 27, 23, 32, 23, 33, 23, 29, 24, 25, 24, 27, 24, 31, 25, 31, 26, 29, 26, 33, 27, 33,
    28, 31, 28, 33, 29, 32, 29, 33, 30, 32, 30, 33, 31, 32, 31, 33, 32, 33,
];
const DENSITY: f64 = 78.0 / 561.0;

struct Pin {
    name: &'static str,
    seed: u32,
    resolution: f64,
    cap: i64,
    isolation: bool,
    weighted: bool,
    node_weighted: bool,
    max_total: i64,
    exact: i64,
    k: i64,
    quality: &'static str,
    slack: &'static str,
    membership: &'static [i64],
    masks: &'static [u64],
}

/// C's %a notation, e.g. 0x1.b8f407b8f407dp-2.
fn hex(s: &str) -> f64 {
    let (neg, s) = s.strip_prefix('-').map_or((false, s), |r| (true, r));
    let s = s.strip_prefix("0x").unwrap();
    let (mant, exp) = s.split_once('p').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let mut bits: u64 = u64::from_str_radix(int, 16).unwrap();
    let mut shift = 0;
    for c in frac.chars() {
        bits = bits * 16 + c.to_digit(16).unwrap() as u64;
        shift += 4;
    }
    let v = bits as f64 * 2f64.powi(exp - shift);
    if neg {
        -v
    } else {
        v
    }
}

const PINS: &[Pin] = &[
    Pin { name: "zachary partition seed 0", seed: 0, resolution: f64::NAN, cap: 1, isolation: false, weighted: false, node_weighted: false, max_total: -1, exact: -1, k: 5, quality: "0x1.b8f407b8f407dp-2", slack: "0x0p+0",
        membership: &[0, 0, 0, 0, 1, 2, 2, 0, 3, 0, 1, 1, 0, 0, 3, 3, 2, 0, 3, 0, 3, 0, 3, 3, 4, 4, 3, 4, 4, 3, 3, 4, 3, 3], masks: &[] },
    Pin { name: "zachary cover M=3 gamma 0.1 seed 0", seed: 0, resolution: 0.1, cap: 3, isolation: false, weighted: false, node_weighted: false, max_total: -1, exact: -1, k: 9, quality: "0x1.0fc298892fe39p-1", slack: "0x1.f1d4abc914b81p-44",
        membership: &[], masks: &[0x10060, 0x10470, 0x10470, 0x2a388f, 0x10000204, 0x402a318f, 0x8b800000, 0x3f4d4c100, 0x3f4d4c100] },
    Pin { name: "weighted partition gamma 0.2 isolation seed 1", seed: 1, resolution: 0.2, cap: 1, isolation: true, weighted: true, node_weighted: false, max_total: -1, exact: -1, k: 10, quality: "0x1.96f439aec498ep-2", slack: "0x1.0c0000000000ap-50",
        membership: &[0, 0, 0, 0, 1, 1, 1, 0, 2, 3, 1, 4, 0, 0, 5, 5, 1, 0, 6, 0, 6, 7, 5, 8, 9, 9, 8, 6, 6, 8, 2, 6, 5, 6], masks: &[] },
    Pin { name: "negative resolution cover M=2 gamma -0.02 seed 2", seed: 2, resolution: -0.02, cap: 2, isolation: false, weighted: false, node_weighted: false, max_total: -1, exact: -1, k: 5, quality: "0x1.c89a7b611699ep-1", slack: "0x1.1078050fee7cap-43",
        membership: &[], masks: &[0x10060, 0x2a388f, 0xdb2b3fff, 0x364d4c100, 0x3bfd4c200] },
    Pin { name: "exactly 3 communities gamma 0.1 seed 3", seed: 3, resolution: 0.1, cap: 1, isolation: false, weighted: false, node_weighted: false, max_total: -1, exact: 3, k: 3, quality: "0x1.01f81f81f81f8p-1", slack: "0x0p+0",
        membership: &[0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 0, 0, 2, 2, 1, 0, 2, 0, 2, 0, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2], masks: &[] },
    Pin { name: "weighted cover M=4 gamma 0.05 isolation at most 6 seed 5", seed: 5, resolution: 0.05, cap: 4, isolation: true, weighted: true, node_weighted: true, max_total: 6, exact: -1, k: 6, quality: "0x1.523b9503a6befp-1", slack: "0x1.7c5aa71afd091p-43",
        membership: &[], masks: &[0x800, 0x10470, 0x2a309f, 0x930b34ff, 0x3fcd4c304, 0x3fcd4c304] },
    Pin { name: "exact arithmetic partition gamma 0.5 isolation seed 7", seed: 7, resolution: 0.5, cap: 1, isolation: true, weighted: false, node_weighted: false, max_total: -1, exact: -1, k: 19, quality: "0x1.20d20d20d20d2p-4", slack: "0x0p+0",
        membership: &[0, 0, 0, 0, 1, 2, 2, 0, 3, 4, 1, 5, 6, 0, 7, 8, 2, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17, 15, 18, 17, 3, 16, 3, 3], masks: &[] },
];

fn weights() -> (Vec<f64>, Vec<f64>) {
    let ew = (0..78).map(|e| 0.5 + 0.25 * (e % 7) as f64).collect();
    let nw = (0..34).map(|v| 0.5 * (1 + v % 4) as f64).collect();
    (ew, nw)
}

fn options(pin: &Pin) -> Options {
    Options {
        resolution: if pin.resolution.is_nan() { DENSITY } else { pin.resolution },
        max_memberships: pin.cap,
        max_total_communities: pin.max_total,
        n_communities: pin.exact,
        allow_isolation: pin.isolation,
    }
}

#[test]
fn bit_pins() {
    let (ew, nw) = weights();
    for pin in PINS {
        let g = Graph::new(
            34,
            Edges::Flat(&ZACHARY),
            pin.weighted.then_some(&ew[..]),
            pin.node_weighted.then_some(&nw[..]),
        )
        .unwrap();
        let o = options(pin);
        let mut mt = Mt19937::python_seed(pin.seed as u64);
        let mut cb = Callbacks::new(&mut mt);
        let mut trace = Trace::default();
        let (k, quality, slack) = if pin.cap == 1 {
            let r = partition(&g, &o, None, &mut cb, Some(&mut trace)).unwrap();
            let mut names = vec![-1i64; 34];
            let mut next = 0;
            let relabelled: Vec<i64> = r
                .membership
                .iter()
                .map(|&c| {
                    if names[c as usize] < 0 {
                        names[c as usize] = next;
                        next += 1;
                    }
                    names[c as usize]
                })
                .collect();
            assert_eq!(relabelled, pin.membership, "{}", pin.name);
            let report = verify_partition(&g, &o, &r.membership).unwrap();
            assert_eq!(report.improvable, 0, "{}: {:?}", pin.name, report);
            (r.count, r.quality, r.raw_slack)
        } else {
            let r = cover(&g, &o, None, &mut cb, Some(&mut trace)).unwrap();
            let mut masks = vec![0u64; r.count as usize];
            for v in 0..34 {
                for &l in r.cover.row(v) {
                    masks[l as usize] |= 1 << v;
                }
            }
            masks.sort();
            assert_eq!(masks, pin.masks, "{}", pin.name);
            let report = verify_cover(&g, &o, &r.cover).unwrap();
            assert_eq!(report.improvable, 0, "{}: {:?}", pin.name, report);
            (r.count, r.quality, r.raw_slack)
        };
        assert_eq!(k, pin.k, "{}", pin.name);
        assert_eq!(quality.to_bits(), hex(pin.quality).to_bits(), "{}: quality {:e}", pin.name, quality);
        assert_eq!(slack.to_bits(), hex(pin.slack).to_bits(), "{}: slack {:e}", pin.name, slack);
    }
}

#[test]
fn equilibria_are_fixed_points() {
    let g = Graph::new(34, Edges::Flat(&ZACHARY), None, None).unwrap();
    for isolation in [false, true] {
        let o = Options { resolution: DENSITY, allow_isolation: isolation, ..Options::default() };
        let mut r = Pcg32::new(0, 54);
        let first = partition(&g, &o, None, &mut Callbacks::new(&mut r), None).unwrap();
        assert!(first.count >= 2 && first.count <= 8 && first.quality > 0.3);
        let mut r = Pcg32::new(99, 54);
        let again = partition(&g, &o, Some(&first.membership), &mut Callbacks::new(&mut r), None).unwrap();
        assert_eq!(again.membership, first.membership);
        assert_eq!(again.quality.to_bits(), first.quality.to_bits());
        for cap in 2..=4 {
            let o = Options { max_memberships: cap, ..o };
            let mut r = Pcg32::new(0, 54);
            let first = cover(&g, &o, None, &mut Callbacks::new(&mut r), None).unwrap();
            assert!(first.quality > 0.3);
            let mut r = Pcg32::new(5, 54);
            let again = cover(&g, &o, Some(&first.cover), &mut Callbacks::new(&mut r), None).unwrap();
            assert_eq!(again.cover, first.cover);
            assert_eq!(again.quality.to_bits(), first.quality.to_bits());
        }
    }
}

#[test]
fn errors_are_values() {
    let g = Graph::new(34, Edges::Flat(&ZACHARY), None, None).unwrap();
    let o = Options { resolution: DENSITY, ..Options::default() };
    let mut f = Failing;
    assert!(partition(&g, &o, None, &mut Callbacks::new(&mut f), None).is_err());
    let bad = [0i64, 1];
    assert!(Graph::new(34, Edges::Flat(&bad[..1]), None, None).is_err());
    let loops = [0i64, 0];
    let g = Graph::new(2, Edges::Flat(&loops), None, None).unwrap();
    let mut r = Pcg32::new(0, 54);
    // Covers refuse loops (partitions take them, as the C core does).
    let err = cover(&g, &Options { max_memberships: 2, ..o }, None, &mut Callbacks::new(&mut r), None).unwrap_err();
    assert_eq!(err.message, "The overlapping game requires a loopless graph.");
    let o = Options { resolution: f64::NAN, ..o };
    let g = Graph::new(34, Edges::Flat(&ZACHARY), None, None).unwrap();
    assert!(cover(&g, &Options { max_memberships: 2, ..o }, Some(&Cover::default()), &mut Callbacks::new(&mut r), None).is_err());
}

/// The soft fma against the instruction on adversarial cases.
#[test]
fn soft_fma_matches_the_instruction() {
    let mut s = 0x243f6a8885a308d3u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut mismatches = 0;
    for i in 0..400_000 {
        let (x, y, z) = match i % 4 {
            0 => (f64::from_bits(next()), f64::from_bits(next()), f64::from_bits(next())),
            1 => {
                // Cancellation: z close to -x y.
                let x = f64::from_bits((next() & 0x800f_ffff_ffff_ffff) | (((next() % 200) + 923) << 52));
                let y = f64::from_bits((next() & 0x800f_ffff_ffff_ffff) | (((next() % 200) + 923) << 52));
                let p = -(x * y);
                (x, y, f64::from_bits(p.to_bits() ^ (next() & 0xff)))
            }
            2 => {
                // Subnormal results.
                let x = f64::from_bits((next() & 0x800f_ffff_ffff_ffff) | ((next() % 60 + 480) << 52));
                let y = f64::from_bits((next() & 0x800f_ffff_ffff_ffff) | ((next() % 60 + 480) << 52));
                (x, y, f64::from_bits(next() & 0x801f_ffff_ffff_ffff))
            }
            _ => {
                // Ties: z half an ulp of the product.
                let x = (next() >> 11) as f64;
                let y = (next() >> 11) as f64;
                let p = x * y;
                let ulp = f64::from_bits(p.to_bits() + 1) - p;
                (x, y, ulp * 0.5 * if next() & 1 == 1 { 1.0 } else { -1.0 })
            }
        };
        let (soft, hard) = (soft_fma(x, y, z), x.mul_add(y, z));
        if !(soft.is_nan() && hard.is_nan()) && soft.to_bits() != hard.to_bits() {
            if mismatches < 10 {
                eprintln!("soft fma({:e}, {:e}, {:e}) = {:e}, the instruction {:e}", x, y, z, soft, hard);
            }
            mismatches += 1;
        }
        let _ = fma(x, y, z);
    }
    assert_eq!(mismatches, 0);
}
