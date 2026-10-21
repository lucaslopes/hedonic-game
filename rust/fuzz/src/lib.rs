//! What the fuzz targets check, as plain functions of the input bytes (the
//! targets in fuzz_targets/ only hand libFuzzer's bytes to them, and the
//! tests below feed them pseudo-random bytes on stable Rust).
//!
//! - `inputs`: arbitrary graphs (any vertex count, edge ends, weights, both
//!   edge layouts), options, partition starts, cover offsets and labels,
//!   verifier inputs and audit queries. Every call returns Ok or Err; none
//!   panics, overflows or reads out of bounds (the release profile keeps
//!   debug assertions and overflow checks on).
//! - `runs`: small valid games. Every partition and cover ends at a certified
//!   equilibrium: each traced move's computed improvement beats its rounding
//!   bound, the independent verifier certifies no gain above the reported
//!   slack (tests/test_equilibrium.py's check), the result is well formed,
//!   and the same seed gives the same bits.

use hedonic_core_rs::audit::{AuditOptions, AuditState, Selection};
use hedonic_core_rs::cover::{cover, Cover};
use hedonic_core_rs::graph::{Edges, Graph, Options};
use hedonic_core_rs::partition::partition;
use hedonic_core_rs::queue::Callbacks;
use hedonic_core_rs::rng::Pcg32;
use hedonic_core_rs::trace::Trace;
use hedonic_core_rs::verify::{verify_cover, verify_partition};

/// How far inputs got (the tests check that the checks are not vacuous).
#[cfg(test)]
static REACHED: [std::sync::atomic::AtomicUsize; 5] = [const { std::sync::atomic::AtomicUsize::new(0) }; 5];
const ENGINES: usize = 0;
const AUDIT: usize = 1;
const QUERIES: usize = 2;
const PARTITIONS: usize = 3;
const COVERS: usize = 4;

#[inline]
fn reached(_what: usize) {
    #[cfg(test)]
    REACHED[_what].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// The input bytes, read front to back; zeros once they run out. A `tame`
/// reader draws only in-range integers and positive finite reals, so that
/// inputs also get past validation, deep into the engines and the audit.
struct Bytes<'a>(&'a [u8], bool);

impl Bytes<'_> {
    fn new(data: &[u8]) -> Bytes<'_> {
        let mut b = Bytes(data, false);
        b.1 = b.u8() % 4 != 0;
        b
    }

    fn u8(&mut self) -> u8 {
        match self.0.split_first() {
            Some((&b, rest)) => {
                self.0 = rest;
                b
            }
            None => 0,
        }
    }

    fn u64(&mut self) -> u64 {
        (0..8).fold(0u64, |x, _| x << 8 | self.u8() as u64)
    }

    fn bool(&mut self) -> bool {
        self.u8() & 1 == 1
    }

    /// Mostly small, sometimes extreme.
    fn int(&mut self, small: i64) -> i64 {
        if self.1 {
            return (self.u8() as i64) % small.max(1);
        }
        match self.u8() % 16 {
            0 => -1,
            1 => i64::MAX,
            2 => i64::MIN,
            3 => self.u64() as i64,
            4 => 1 << 31,
            5 => -(self.u8() as i64),
            _ => (self.u8() as i64) % small.max(1),
        }
    }

    fn real(&mut self) -> f64 {
        if self.1 {
            return (1 + self.u8() as u32) as f64 / 64.0;
        }
        match self.u8() % 16 {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => -1.0,
            3 => 0.0,
            4 => f64::MIN_POSITIVE / 4.0,
            5 => 1e300,
            6 => f64::from_bits(self.u64()),
            7 => -0.0,
            8 => 1.0,
            _ => self.u8() as f64 / 64.0,
        }
    }

    /// `ints`/`reals` of a length drawn first (`len(around)`).
    fn ints_near(&mut self, around: usize, small: i64) -> Vec<i64> {
        let len = self.len(around);
        self.ints(len, small)
    }

    fn reals_near(&mut self, around: usize) -> Vec<f64> {
        let len = self.len(around);
        self.reals(len)
    }

    fn ints(&mut self, len: usize, small: i64) -> Vec<i64> {
        (0..len).map(|_| self.int(small)).collect()
    }

    fn reals(&mut self, len: usize) -> Vec<f64> {
        (0..len).map(|_| self.real()).collect()
    }

    /// A length near `around` (off by one either way now and then).
    fn len(&mut self, around: usize) -> usize {
        if self.1 {
            return around;
        }
        match self.u8() % 8 {
            0 => around + 1,
            1 => around.saturating_sub(1),
            2 => self.u8() as usize % 40,
            _ => around,
        }
    }
}

/// The largest graph `inputs` runs an engine on (a check on arbitrary data
/// must stay fast).
const RUN_LIMIT: i64 = 64;

pub fn inputs(data: &[u8]) {
    let mut b = Bytes::new(data);
    let n = match b.u8() % 8 {
        0 => b.int(1 << 40),
        _ => b.u8() as i64 % (RUN_LIMIT + 4) - 2,
    };
    let small = n.clamp(1, RUN_LIMIT) + 2;
    // Ends: in range when tame, else up to two past the last vertex.
    let ends = if b.1 { n.clamp(1, RUN_LIMIT) } else { small };
    let m = b.u8() as usize % 48;
    let odd = (!b.1 && b.u8() % 4 == 0) as usize;
    let flat = b.ints(2 * m + odd, ends);
    let (first, second) = (b.ints(m, ends), b.ints_near(m, ends));
    let edges = if b.bool() { Edges::Flat(&flat) } else { Edges::Split(&first, &second) };
    let ew = b.reals_near(m);
    let nw = b.reals_near(n.clamp(0, RUN_LIMIT + 2) as usize);
    let ew = b.bool().then_some(&ew[..]);
    let nw = b.bool().then_some(&nw[..]);
    let Ok(g) = Graph::new(n, edges, ew, nw) else { return };
    if !(0..=RUN_LIMIT).contains(&g.vertex_count) {
        return;
    }
    reached(ENGINES);
    let size = g.vertex_count as usize;
    let o = Options {
        resolution: b.real(),
        max_memberships: b.int(small),
        max_total_communities: b.int(small),
        n_communities: b.int(small),
        allow_isolation: b.bool(),
    };
    let start = b.ints_near(size, small);
    let mut rng = Pcg32::new(b.u64(), 54);
    let _ = partition(&g, &o, b.bool().then_some(&start[..]), &mut Callbacks::new(&mut rng), None);
    let _ = verify_partition(&g, &o, &start);

    // Cover rows: arbitrary offsets and labels, or a well-formed cover with
    // at most one entry spoiled.
    let (offset, labels) = if !b.1 && b.bool() {
        let mut at = 0i64;
        let mut offset: Vec<i64> = vec![if b.u8() % 16 == 0 { b.int(4) } else { 0 }];
        let rows = b.len(size);
        for _ in 0..rows {
            at = at.wrapping_add(match b.u8() % 16 {
                0 => -1,
                1 => b.int(small),
                k => (k % 4) as i64,
            });
            offset.push(at);
        }
        let labels = b.ints_near(offset.last().map_or(0, |&x| x.clamp(0, 4 * RUN_LIMIT)) as usize, small);
        (offset, labels)
    } else {
        let k = 1 + b.u8() as i64 % (size as i64 + 1);
        let (mut offset, mut labels) = (vec![0i64], Vec::new());
        for v in 0..size as i64 {
            let first = (v + b.u8() as i64) % k;
            labels.push(first);
            for l in 0..k {
                if l != first && b.u8() % 4 == 0 {
                    labels.push(l);
                }
            }
            let row = offset.len() - 1;
            labels[offset[row] as usize..].sort_unstable();
            offset.push(labels.len() as i64);
        }
        if b.u8() % 4 == 0 {
            let (at, value) = (b.u8() as usize, b.int(small));
            if b.bool() && !labels.is_empty() {
                let at = at % labels.len();
                labels[at] = value;
            } else {
                let at = at % offset.len();
                offset[at] = value;
            }
        }
        (offset, labels)
    };
    let rows = Cover { offset: offset.clone(), labels: labels.clone() };
    let mut rng = Pcg32::new(b.u64(), 54);
    let _ = cover(&g, &o, b.bool().then_some(&rows), &mut Callbacks::new(&mut rng), None);
    let _ = verify_cover(&g, &o, &rows);

    let mut polls = b.u8() as i64 * 8;
    let mut stop = move || {
        polls -= 1;
        polls == 0
    };
    let interrupt = b.bool();
    let Ok(state) = AuditState::prepare(&g, &offset, &labels, if interrupt { Some(&mut stop) } else { None }) else {
        return;
    };
    reached(AUDIT);
    let ao = AuditOptions {
        max_memberships: b.int(small),
        allow_isolation: b.bool(),
        prune: b.bool(),
        atol: b.real(),
        rtol: b.real(),
    };
    let count = b.u8() as usize % 12;
    let list = b.ints(count, small);
    let sel = if b.bool() {
        Selection::List(&list)
    } else {
        let begin = b.int(small);
        Selection::Range(begin, b.int(small))
    };
    if state.intervals(&ao, sel, None).is_ok() {
        reached(QUERIES);
    }
    let _ = state.intervals_exhaustive(&ao, sel, None);
    let count = b.u8() as usize % 6;
    let gammas = b.reals(count);
    // The outputs' length is the caller's to get right (the binding checks
    // it); a wrong one must still be an error.
    let wanted = match sel {
        Selection::List(list) => list.len(),
        Selection::Range(begin, end) => end.saturating_sub(begin).clamp(0, 4 * RUN_LIMIT) as usize,
    } * gammas.len();
    let out = b.len(wanted);
    let (mut regret, mut tolerance) = (vec![0.0; out], vec![0.0; out]);
    let _ = state.regrets(&ao, sel, &gammas, &mut regret, &mut tolerance, None);
}

/// Labels 0 .. k-1, all used, in 0 .. k.
fn partition_well_formed(membership: &[i64], k: i64) -> bool {
    let mut seen = vec![false; k.max(0) as usize];
    for &c in membership {
        if c < 0 || c >= k {
            return false;
        }
        seen[c as usize] = true;
    }
    seen.iter().all(|&s| s)
}

/// Non-empty sorted rows of at most `cap` distinct labels, 0 .. k-1 all used.
fn cover_well_formed(c: &Cover, n: i64, k: i64, cap: i64) -> bool {
    if c.vertex_count() != n || c.validate().is_err() {
        return false;
    }
    let mut seen = vec![false; k.max(0) as usize];
    for v in 0..n as usize {
        let row = c.row(v);
        if row.is_empty() || row.len() as i64 > cap {
            return false;
        }
        for (i, &l) in row.iter().enumerate() {
            if l < 0 || l >= k || (i > 0 && row[i - 1] >= l) {
                return false;
            }
            seen[l as usize] = true;
        }
    }
    seen.iter().all(|&s| s)
}

fn moves_certified(trace: &Trace) {
    for (i, m) in trace.moves.iter().enumerate() {
        assert!(m.bound >= 0.0 && m.improvement > m.bound, "move {i}: improvement {:e}, bound {:e}", m.improvement, m.bound);
    }
}

pub fn runs(data: &[u8]) {
    let mut b = Bytes(data, true);
    let n = 1 + b.u8() as i64 % 24;
    let m = b.u8() as usize % 64;
    let mut flat = Vec::with_capacity(2 * m);
    for _ in 0..m {
        let u = b.u8() as i64 % n;
        let v = b.u8() as i64 % n;
        if u != v {
            flat.extend([u, v]);
        }
    }
    let m = flat.len() / 2;
    let weight = |b: &mut Bytes| match b.u8() % 8 {
        0 => 1.0,
        1 => 0.5,
        2 => 3.0,
        _ => (1 + b.u8() as u32) as f64 / 37.0,
    };
    let ew: Vec<f64> = (0..m).map(|_| weight(&mut b)).collect();
    let nw: Vec<f64> = (0..n).map(|_| weight(&mut b)).collect();
    let ew = b.bool().then_some(&ew[..]);
    let nw = (b.u8() % 4 == 0).then_some(&nw[..]);
    let g = Graph::new(n, Edges::Flat(&flat), ew, nw).expect("a valid graph");
    let resolution = match b.u8() % 8 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        _ => b.u8() as f64 / 128.0,
    };
    let cap = match b.u8() % 6 {
        0 => n,
        k => k.min(4) as i64,
    };
    let o = Options {
        resolution,
        max_memberships: cap,
        max_total_communities: if b.u8() % 8 == 0 { 1 + b.u8() as i64 % n } else { -1 },
        n_communities: if b.u8() % 16 == 0 { 1 + b.u8() as i64 % n } else { -1 },
        allow_isolation: b.bool(),
    };
    let seed = b.u64();
    let start: Option<Vec<i64>> = b.bool().then(|| (0..n).map(|_| b.u8() as i64 % n).collect());

    if cap == 1 {
        let run = |trace: Option<&mut Trace>| {
            partition(&g, &o, start.as_deref(), &mut Callbacks::new(&mut Pcg32::new(seed, 54)), trace)
        };
        let mut trace = Trace::default();
        // An infeasible count (n_communities, max_total) is an error, never a panic.
        let Ok(r) = run(Some(&mut trace)) else { return };
        reached(PARTITIONS);
        moves_certified(&trace);
        assert!(partition_well_formed(&r.membership, r.count), "{:?} ({} communities)", r.membership, r.count);
        let report = verify_partition(&g, &o, &r.membership).expect("the verifier takes the result");
        assert!(report.max_certified <= r.raw_slack, "certified gain {:e} above the slack {:e}", report.max_certified, r.raw_slack);
        let again = run(None).expect("the same call again");
        assert_eq!((again.membership, again.quality.to_bits(), again.slack.to_bits()), (r.membership, r.quality.to_bits(), r.slack.to_bits()));
    } else {
        let start = start.map(|labels| Cover { offset: (0..=n).collect(), labels });
        let run = |trace: Option<&mut Trace>| cover(&g, &o, start.as_ref(), &mut Callbacks::new(&mut Pcg32::new(seed, 54)), trace);
        let mut trace = Trace::default();
        let Ok(r) = run(Some(&mut trace)) else { return };
        reached(COVERS);
        moves_certified(&trace);
        assert!(cover_well_formed(&r.cover, n, r.count, cap), "{:?} ({} labels, cap {cap})", r.cover, r.count);
        let report = verify_cover(&g, &o, &r.cover).expect("the verifier takes the result");
        assert!(report.max_certified <= r.raw_slack, "certified gain {:e} above the slack {:e}", report.max_certified, r.raw_slack);
        let again = run(None).expect("the same call again");
        assert_eq!((again.cover, again.quality.to_bits(), again.slack.to_bits()), (r.cover, r.quality.to_bits(), r.slack.to_bits()));
    }
}

#[cfg(test)]
mod tests {
    /// Pseudo-random inputs of random lengths (HIG_FUZZ_CASES of each,
    /// default 20 000): the targets' checks without libFuzzer.
    fn smoke(target: fn(&[u8])) {
        let cases: u64 = std::env::var("HIG_FUZZ_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000);
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut next = move || {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        for _ in 0..cases {
            let len = (next() % 600) as usize;
            let data: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            target(&data);
        }
    }

    fn count(what: usize) -> usize {
        super::REACHED[what].load(std::sync::atomic::Ordering::Relaxed)
    }

    #[test]
    fn inputs_never_panic() {
        smoke(super::inputs);
        let (engines, audit, queries) = (count(super::ENGINES), count(super::AUDIT), count(super::QUERIES));
        println!("inputs: {engines} graphs reached the engines, {audit} audits prepared, {queries} answered");
        assert!(engines > 0 && audit > 0 && queries > 0);
    }

    #[test]
    fn runs_end_at_certified_equilibria() {
        smoke(super::runs);
        let (partitions, covers) = (count(super::PARTITIONS), count(super::COVERS));
        println!("runs: {partitions} partitions and {covers} covers checked");
        assert!(partitions > 0 && covers > 0);
    }
}
