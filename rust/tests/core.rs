//! The checks of native/tests/test_core.c that pins.rs does not port: every
//! partition and cover of small random games against an independent
//! brute-force best response computed from the definition of the payoffs,
//! with the count limits, the structure of the result, the returned quality
//! and the verifier; then Zachary, the two edge layouts, a former tie cycle,
//! count limits, a planted partition, callbacks, error paths, and the soft
//! fma on test_core.c's adversarial classes with its pinned hash.
//!
//! The games come from the C test's own generators (its SplitMix64 test
//! stream, PCG32 stream 54, the same seeds and sizes), drawn in the same
//! order, so these are the cases test_core.c checks. Where test_core.c draws
//! two numbers in one argument list (an order C leaves unspecified), this
//! file draws them left to right, as clang does.

#![allow(clippy::neg_cmp_op_on_partial_ord)] // test_core.c's comparisons

use std::cell::Cell;

use hedonic_core_rs::cover::{cover, Cover};
use hedonic_core_rs::error::{Error, Kind};
use hedonic_core_rs::fma::{fma, fma_path, soft_fma};
use hedonic_core_rs::graph::{Edges, Graph, Options};
use hedonic_core_rs::partition::partition;
use hedonic_core_rs::queue::Callbacks;
use hedonic_core_rs::rng::{Failing, Pcg32};
use hedonic_core_rs::trace::Trace;
use hedonic_core_rs::verify::{verify_cover, verify_partition};

/// Zachary's karate club, 34 vertices, 78 edges.
const ZACHARY: [i64; 156] = [
    0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 10, 0, 11, 0, 12, 0, 13, 0, 17, 0, 19, 0,
    21, 0, 31, 1, 2, 1, 3, 1, 7, 1, 13, 1, 17, 1, 19, 1, 21, 1, 30, 2, 3, 2, 7, 2, 27, 2, 28, 2,
    32, 2, 9, 2, 8, 2, 13, 3, 7, 3, 12, 3, 13, 4, 6, 4, 10, 5, 6, 5, 10, 5, 16, 6, 16, 8, 30, 8,
    32, 8, 33, 9, 33, 13, 33, 14, 32, 14, 33, 15, 32, 15, 33, 18, 32, 18, 33, 19, 33, 20, 32, 20,
    33, 22, 32, 22, 33, 23, 25, 23, 27, 23, 32, 23, 33, 23, 29, 24, 25, 24, 27, 24, 31, 25, 31, 26,
    29, 26, 33, 27, 33, 28, 31, 28, 33, 29, 32, 29, 33, 30, 32, 30, 33, 31, 32, 31, 33, 32, 33,
];

/// The number of random games of each kind (test_core.c's 1500 at scale 1;
/// HIG_RANDOM_CASES sets another, as the Miri job does).
fn random_cases() -> usize {
    std::env::var("HIG_RANDOM_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(1500)
}

// ---- Random numbers for the tests -------------------------------------------

/// test_core.c's test stream (SplitMix64 from 0x9e3779b97f4a7c15). The C test
/// draws the planted graph, then the random partitions, then the random
/// covers from one stream; each test here replays the draws before its own.
struct Draw(u64);

impl Draw {
    fn new() -> Self {
        Draw(0x9e3779b97f4a7c15)
    }

    fn next64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next64() >> 11) as f64 / 9007199254740992.0 // 2^53
    }

    fn below(&mut self, n: i64) -> i64 {
        (self.next64() % n as u64) as i64
    }
}

/// The core's random source of the tests: PCG32 on stream 54.
fn seeded(seed: u64) -> Pcg32 {
    Pcg32::new(seed, 54)
}

fn options(resolution: f64, max_memberships: i64, allow_isolation: bool) -> Options {
    Options {
        resolution,
        max_memberships,
        max_total_communities: -1,
        n_communities: -1,
        allow_isolation,
    }
}

// ---- Brute-force best responses ---------------------------------------------

fn ends(g: &Graph, e: i64) -> (i64, i64) {
    (g.end(2 * e), g.end(2 * e + 1))
}

/// A slack far above the core's tolerance and far below any real gain.
fn slack(g: &Graph, v: i64, gamma: f64) -> f64 {
    let (mut a, mut total) = (0.0f64, 0.0f64);
    for e in 0..g.edge_count {
        let (x, y) = ends(g, e);
        if x == v || y == v {
            a += g.edge_weight(e).abs();
        }
    }
    for u in 0..g.vertex_count {
        total += g.node_weight(u);
    }
    1e-9 * (1.0 + a + gamma.abs() * g.node_weight(v) * total)
}

/// The largest gain of a feasible unilateral move in a partition, from the
/// definition u_v(C) = w(v, C \ v) - gamma n_v N(C \ v), in units of the
/// vertex's slack: at most 1 at an equilibrium.
fn partition_regret(g: &Graph, membership: &[i64], k: i64, o: &Options) -> f64 {
    let n = g.vertex_count;
    let k = k as usize;
    let (mut to, mut mass, mut size) = (vec![0.0f64; k], vec![0.0f64; k], vec![0i64; k]);
    let exact = o.n_communities > 0;
    let limited = o.max_total_communities > 0;
    let mut worst = 0.0f64;

    for v in 0..n {
        mass[membership[v as usize] as usize] += g.node_weight(v);
        size[membership[v as usize] as usize] += 1;
    }
    for v in 0..n {
        let nv = g.node_weight(v);
        let own = membership[v as usize] as usize;
        let alone = size[own] == 1;
        if exact && alone {
            continue; // leaving would empty a community
        }
        to.iter_mut().for_each(|t| *t = 0.0);
        for e in 0..g.edge_count {
            let (a, b) = ends(g, e);
            if a == v && b != v {
                to[membership[b as usize] as usize] += g.edge_weight(e);
            }
            if b == v && a != v {
                to[membership[a as usize] as usize] += g.edge_weight(e);
            }
        }
        let current = to[own] - o.resolution * nv * (mass[own] - nv);
        let mut best = current;
        for c in 0..k {
            if c != own {
                let u = to[c] - o.resolution * nv * mass[c];
                if u > best {
                    best = u;
                }
            }
        }
        if o.allow_isolation
            && !exact
            && !alone
            && (!limited || (k as i64) < o.max_total_communities)
            && 0.0 > best
        {
            best = 0.0;
        }
        let regret = (best - current) / slack(g, v, o.resolution);
        if regret > worst {
            worst = regret;
        }
    }
    worst
}

fn holds(c: &Cover, v: i64, label: i64) -> bool {
    c.row(v as usize).contains(&label)
}

fn row_size(c: &Cover, v: i64) -> i64 {
    c.offset[v as usize + 1] - c.offset[v as usize]
}

const MAX_ENUMERATED_LABELS: i64 = 16;

/// The same for a cover: every set of at most max_memberships labels among the
/// k occupied ones, alone or with a fresh label (payoff 0), under the count
/// limits, with g_c(v) = sum_{u ~ v holding c} w_uv f_u - gamma n_v sum_{u != v
/// holding c} n_u f_u and U_v(sigma) = sum_{c in sigma} g_c / sqrt(|sigma|).
/// Returns -1 when k is too large to enumerate.
fn cover_regret(g: &Graph, c: &Cover, k: i64, o: &Options) -> f64 {
    let n = g.vertex_count;
    let cap = o.max_memberships;
    let exact = o.n_communities > 0;
    let limited = o.max_total_communities > 0;
    let mut gain = [0.0f64; MAX_ENUMERATED_LABELS as usize];
    let mut worst = 0.0f64;

    if k > MAX_ENUMERATED_LABELS {
        return -1.0;
    }
    for v in 0..n {
        let nv = g.node_weight(v);
        let kv = row_size(c, v);
        let mut current = 0.0f64;
        let (mut mandatory, mut sole) = (0u32, 0u32);
        for l in 0..k {
            let (mut support, mut crowd) = (0.0f64, 0.0f64);
            let mut holders = 0;
            for u in 0..n {
                if holds(c, u, l) {
                    holders += 1;
                    if u != v {
                        crowd += g.node_weight(u) / (row_size(c, u) as f64).sqrt();
                    }
                }
            }
            for e in 0..g.edge_count {
                let (a, b) = ends(g, e);
                let u = if a == v {
                    b
                } else if b == v {
                    a
                } else {
                    -1
                };
                if u >= 0 && u != v && holds(c, u, l) {
                    support += g.edge_weight(e) / (row_size(c, u) as f64).sqrt();
                }
            }
            gain[l as usize] = support - o.resolution * nv * crowd;
            if holds(c, v, l) {
                current += gain[l as usize];
                if holders == 1 {
                    sole |= 1 << l;
                    if exact {
                        mandatory |= 1 << l;
                    }
                }
            }
        }
        current /= (kv as f64).sqrt();
        let mut best = current;
        for set in 0u32..(1 << k) {
            let size = set.count_ones() as i64;
            if set & mandatory != mandatory {
                continue;
            }
            for fresh in 0..=1i64 {
                let count_after = k - (sole & !set).count_ones() as i64 + fresh;
                if size + fresh < 1 || size + fresh > cap {
                    continue;
                }
                if fresh == 1 && (!o.allow_isolation || exact) {
                    continue;
                }
                if limited && count_after > o.max_total_communities {
                    continue;
                }
                let mut sum = 0.0f64;
                for l in 0..k {
                    if set & (1 << l) != 0 {
                        sum += gain[l as usize];
                    }
                }
                sum /= ((size + fresh) as f64).sqrt();
                if sum > best {
                    best = sum;
                }
            }
        }
        let regret = (best - current) / slack(g, v, o.resolution);
        if regret > worst {
            worst = regret;
        }
    }
    worst
}

/// Potential / W from the definitions.
fn partition_potential(g: &Graph, membership: &[i64], k: i64, gamma: f64) -> f64 {
    let (mut q, mut total) = (0.0f64, 0.0f64);
    let mut mass = vec![0.0f64; k as usize];
    for e in 0..g.edge_count {
        let (a, b) = ends(g, e);
        total += g.edge_weight(e);
        if membership[a as usize] == membership[b as usize] {
            q += 2.0 * g.edge_weight(e);
        }
    }
    for v in 0..g.vertex_count {
        mass[membership[v as usize] as usize] += g.node_weight(v);
    }
    for m in mass {
        q -= gamma * m * m;
    }
    q / (2.0 * total)
}

fn cover_potential(g: &Graph, c: &Cover, k: i64, gamma: f64) -> f64 {
    let (mut q, mut total) = (0.0f64, 0.0f64);
    let mut mass = vec![0.0f64; k as usize];
    for e in 0..g.edge_count {
        let (a, b) = ends(g, e);
        let (ka, kb) = (row_size(c, a) as f64, row_size(c, b) as f64);
        let shared = c
            .row(a as usize)
            .iter()
            .filter(|&&l| holds(c, b, l))
            .count();
        total += g.edge_weight(e);
        q += 2.0 * g.edge_weight(e) * shared as f64 / (ka * kb).sqrt();
    }
    for v in 0..g.vertex_count {
        let kv = row_size(c, v) as f64;
        for &l in c.row(v as usize) {
            mass[l as usize] += g.node_weight(v) / kv.sqrt();
        }
    }
    for m in mass {
        q -= gamma * m * m;
    }
    q / (2.0 * total)
}

fn close_to(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-10 * (1.0 + a.abs() + b.abs())
}

/// Labels 0 .. k-1 in order of first appearance, all used.
fn partition_well_formed(membership: &[i64], n: i64, k: i64) -> bool {
    let mut next = 0;
    if membership.len() as i64 != n {
        return false;
    }
    for &l in membership {
        if l < 0 || l > next {
            return false;
        }
        if l == next {
            next += 1;
        }
    }
    next == k
}

/// Non-empty sorted rows of at most cap labels, labels 0 .. k-1 all used.
fn cover_well_formed(c: &Cover, n: i64, k: i64, cap: i64) -> bool {
    let mut used = 0;
    let mut seen = vec![false; k as usize + 1];
    if c.vertex_count() != n || c.offset[0] != 0 || c.validate().is_err() {
        return false;
    }
    for v in 0..n as usize {
        let row = c.row(v);
        if row.is_empty() || row.len() as i64 > cap {
            return false;
        }
        for (i, &l) in row.iter().enumerate() {
            if l < 0 || l >= k || (i > 0 && row[i - 1] >= l) {
                return false;
            }
            if !seen[l as usize] {
                seen[l as usize] = true;
                used += 1;
            }
        }
    }
    used == k
}

// ---- Random small games ------------------------------------------------------

const SMALL_N: i64 = 10;
const SMALL_M: usize = 64;

struct Small {
    n: i64,
    edges: Vec<i64>,
    edge_weights: Vec<f64>,
    node_weights: Vec<f64>,
    weighted: bool,
    node_weighted: bool,
}

impl Small {
    fn graph(&self) -> Graph<'_> {
        Graph::new(
            self.n,
            Edges::Flat(&self.edges),
            self.weighted.then_some(&self.edge_weights[..]),
            self.node_weighted.then_some(&self.node_weights[..]),
        )
        .unwrap()
    }
}

/// A random graph of 2 .. largest vertices.
fn random_graph(d: &mut Draw, cover: bool, largest: i64) -> Small {
    let p = 0.15 + 0.6 * d.uniform();
    let signed_weights = !cover && d.below(4) == 0;
    let weighted = signed_weights || d.below(3) == 0;
    let node_weighted = d.below(3) == 0;
    let n = 2 + d.below(largest - 1);
    let (mut edges, mut edge_weights) = (Vec::new(), Vec::new());
    loop {
        edges.clear();
        edge_weights.clear();
        for a in 0..n {
            // Partitions get the odd self-loop.
            let from = a + (cover || d.below(8) != 0) as i64;
            for b in from..n {
                if d.uniform() < p && edge_weights.len() < SMALL_M - 1 {
                    edges.extend_from_slice(&[a, b]);
                    edge_weights.push(if signed_weights {
                        d.uniform() * 3.0 - 1.0
                    } else if d.below(5) == 0 {
                        (1 + d.below(3)) as f64
                    } else {
                        0.1 + 2.0 * d.uniform()
                    });
                }
            }
        }
        if !edge_weights.is_empty() && d.below(5) == 0 {
            // A multi-edge.
            edges.extend_from_slice(&[edges[0], edges[1]]);
            edge_weights.push(edge_weights[0]);
        }
        if !edge_weights.is_empty() {
            break;
        }
    }
    let node_weights = (0..n)
        .map(|_| {
            if d.below(6) == 0 {
                0.0
            } else {
                0.2 + 2.0 * d.uniform()
            }
        })
        .collect();
    Small {
        n,
        edges,
        edge_weights,
        node_weights,
        weighted,
        node_weighted,
    }
}

fn total_edge_weight(g: &Graph) -> f64 {
    let mut w = 0.0;
    for e in 0..g.edge_count {
        w += g.edge_weight(e);
    }
    w
}

const RESOLUTIONS: [f64; 8] = [-0.7, -0.05, 0.0, 0.02, 0.1, 0.35, 0.8, 2.0];

fn random_limits(d: &mut Draw, o: &mut Options, n: i64, cap: i64) {
    match d.below(4) {
        0 => o.max_total_communities = 1 + d.below(n),
        1 => o.n_communities = 1 + d.below(n * cap),
        _ => {}
    }
}

fn start_labels(o: &Options, n: i64) -> i64 {
    if o.n_communities > 0 {
        o.n_communities
    } else if o.max_total_communities > 0 {
        o.max_total_communities
    } else {
        n
    }
}

/// Unit weights at a resolution whose products are exact: the slack is 0.
fn exact_arithmetic(s: &Small, gamma: f64) -> bool {
    !s.weighted && !s.node_weighted && (gamma == 0.0 || gamma == 2.0)
}

/// What the verifier saw on the random games.
#[derive(Default)]
struct Verified {
    vertices: i64,
    improvable: i64,
}

/// One random partition game: the graph, the options and the start, or None
/// when its total edge weight is not positive (skipped, as test_core.c does).
fn partition_case(d: &mut Draw) -> Option<(Small, Options, Option<Vec<i64>>)> {
    let start = d.below(3) == 0;
    let s = random_graph(d, false, SMALL_N);
    let resolution = RESOLUTIONS[d.below(8) as usize];
    let isolation = d.below(2) != 0;
    let mut o = options(resolution, 1, isolation);
    random_limits(d, &mut o, s.n, 1);
    if !(total_edge_weight(&s.graph()) > 0.0) {
        return None;
    }
    let membership = start.then(|| {
        let labels = start_labels(&o, s.n);
        (0..s.n)
            .map(|v| 3 * if v < labels { v } else { d.below(labels) })
            .collect()
    });
    Some((s, o, membership))
}

/// The traced run's record: a start row per vertex, and every accepted move
/// certain (its computed improvement beats the bound of its rounding error).
fn check_trace(trace: &Trace, n: i64, case: usize) {
    assert_eq!(trace.start.len() as i64, n, "case {case}: trace start");
    for m in &trace.moves {
        assert!(
            m.improvement > m.bound,
            "case {case}: a traced move of vertex {} gains {:e}, bound {:e}",
            m.v,
            m.improvement,
            m.bound
        );
    }
}

/// The planted graph of `planted` (1500 vertices, 15 blocks), drawn first
/// from the test stream.
fn planted_graph(d: &mut Draw) -> (Vec<i64>, Vec<f64>) {
    const N: i64 = 1500;
    const BLOCKS: i64 = 15;
    let (mut edges, mut weights) = (Vec::new(), Vec::new());
    for a in 0..N {
        for b in a + 1..N {
            let p = if a % BLOCKS == b % BLOCKS { 0.12 } else { 0.004 };
            if d.uniform() < p {
                edges.extend_from_slice(&[a, b]);
                weights.push(0.5 + d.uniform());
            }
        }
    }
    (edges, weights)
}

/// Random partitions: brute-force regret, count limits, labels, quality,
/// the slack (0 in exact arithmetic) and the verifier.
#[test]
fn random_partitions() {
    let mut d = Draw::new();
    planted_graph(&mut d);
    let (mut run, mut not_equilibria, mut exact) = (0, 0, 0);
    let mut seen = Verified::default();

    let cases = random_cases();
    for i in 0..cases {
        let Some((s, o, start)) = partition_case(&mut d) else {
            continue;
        };
        let g = s.graph();
        let mut rng = seeded(i as u64);
        let mut trace = Trace::default();
        let r = partition(
            &g,
            &o,
            start.as_deref(),
            &mut Callbacks::new(&mut rng),
            Some(&mut trace),
        )
        .unwrap_or_else(|e| panic!("case {i}: {}", e.message));
        run += 1;
        let k = r.count;
        assert!(partition_well_formed(&r.membership, s.n, k), "case {i}");
        assert!(o.n_communities <= 0 || k == o.n_communities, "case {i}: k={k}");
        assert!(
            o.max_total_communities <= 0 || k <= o.max_total_communities,
            "case {i}"
        );
        let potential = partition_potential(&g, &r.membership, k, o.resolution);
        assert!(
            close_to(r.quality, potential),
            "case {i}: quality {:e} vs {:e}",
            r.quality,
            potential
        );
        let regret = partition_regret(&g, &r.membership, k, &o);
        not_equilibria += (regret > 1.0) as i32;
        assert!(
            regret <= 1.0,
            "case {i}: partition regret {regret} slacks (gamma {}, isolation {})",
            o.resolution,
            o.allow_isolation
        );
        assert!(r.raw_slack >= 0.0, "case {i}: slack {:e}", r.raw_slack);
        if exact_arithmetic(&s, o.resolution) {
            exact += 1;
            assert_eq!(
                r.raw_slack.to_bits(),
                0.0f64.to_bits(),
                "case {i}: slack {:e} in exact arithmetic",
                r.raw_slack
            );
        }
        check_trace(&trace, s.n, i);
        // The independent verifier: no deviation it can certify gains more
        // than the raw slack the engine reported.
        let report = verify_partition(&g, &o, &r.membership)
            .unwrap_or_else(|e| panic!("case {i}: verifier: {}", e.message));
        assert!(
            report.max_certified <= r.raw_slack,
            "case {i}: the verifier certifies a gain {:e} above the slack {:e}",
            report.max_certified,
            r.raw_slack
        );
        assert_eq!(report.improvable, 0, "case {i}: {report:?}");
        seen.vertices += report.vertices;
        seen.improvable += report.improvable;
    }
    println!(
        "random partitions: {run} games checked by brute force, {not_equilibria} not at \
         equilibrium, {exact} in exact arithmetic (slack 0); verifier: {} vertices, {} with a \
         certainly improving deviation",
        seen.vertices, seen.improvable
    );
    assert!(run > cases / 2);
}

/// Random covers: every result enumerated by brute force (n cap <= 16),
/// count limits, rows, quality and the verifier.
#[test]
fn random_covers() {
    const CAPS: [i64; 4] = [1, 2, 3, -1];
    let mut d = Draw::new();
    planted_graph(&mut d);
    let cases = random_cases();
    for _ in 0..cases {
        partition_case(&mut d);
    }
    let (mut run, mut enumerated, mut not_equilibria) = (0, 0, 0);
    let mut seen = Verified::default();

    for i in 0..cases {
        let use_start = d.below(3) == 0;
        // n cap <= MAX_ENUMERATED_LABELS: every result is enumerated.
        let mut cap = CAPS[d.below(4) as usize];
        let s = random_graph(
            &mut d,
            true,
            match cap {
                2 => 8,
                3 => 5,
                _ => 4,
            },
        );
        let mut rng = seeded(i as u64 + 1000003);
        if cap < 0 || cap > s.n {
            cap = s.n;
        }
        let resolution = RESOLUTIONS[d.below(8) as usize];
        let isolation = d.below(2) != 0;
        let mut o = options(resolution, cap, isolation);
        random_limits(&mut d, &mut o, s.n, cap);
        let start = use_start.then(|| {
            // Row v holds v, v + n, ... below K (every label used, at most
            // cap of them as K <= n cap), then random extras up to its size.
            let target = start_labels(&o, s.n);
            let (mut offset, mut labels) = (vec![0i64], Vec::new());
            for v in 0..s.n {
                let size = 1 + d.below(cap);
                let begin = labels.len() as i64;
                labels.push(7 * (v % target));
                let mut l = v + s.n;
                while l < target {
                    labels.push(7 * l);
                    l += s.n;
                }
                while (labels.len() as i64) - begin < size {
                    labels.push(7 * d.below(target));
                }
                offset.push(labels.len() as i64);
            }
            Cover { offset, labels }
        });
        let g = s.graph();
        let mut trace = Trace::default();
        let r = cover(
            &g,
            &o,
            start.as_ref(),
            &mut Callbacks::new(&mut rng),
            Some(&mut trace),
        )
        .unwrap_or_else(|e| panic!("case {i}: {}", e.message));
        run += 1;
        let k = r.count;
        assert!(cover_well_formed(&r.cover, s.n, k, cap), "case {i}");
        assert!(o.n_communities <= 0 || k == o.n_communities, "case {i}: k={k}");
        assert!(
            o.max_total_communities <= 0 || k <= o.max_total_communities,
            "case {i}"
        );
        let potential = cover_potential(&g, &r.cover, k, o.resolution);
        assert!(
            close_to(r.quality, potential),
            "case {i}: quality {:e} vs {:e}",
            r.quality,
            potential
        );
        let regret = cover_regret(&g, &r.cover, k, &o);
        if regret >= 0.0 {
            enumerated += 1;
            not_equilibria += (regret > 1.0) as i32;
            assert!(
                regret <= 1.0,
                "case {i}: cover regret {regret} slacks (gamma {}, cap {cap}, iso {})",
                o.resolution,
                o.allow_isolation
            );
        }
        assert!(r.raw_slack >= 0.0, "case {i}: slack {:e}", r.raw_slack);
        check_trace(&trace, s.n, i);
        let report = verify_cover(&g, &o, &r.cover)
            .unwrap_or_else(|e| panic!("case {i}: verifier: {}", e.message));
        assert!(
            report.max_certified <= r.raw_slack,
            "case {i}: the verifier certifies a gain {:e} above the slack {:e}",
            report.max_certified,
            r.raw_slack
        );
        assert_eq!(report.improvable, 0, "case {i}: {report:?}");
        seen.vertices += report.vertices;
        seen.improvable += report.improvable;
    }
    println!(
        "random covers: {run} games, {enumerated} checked by brute force, {not_equilibria} not \
         at equilibrium; verifier: {} vertices, {} with a certainly improving deviation",
        seen.vertices, seen.improvable
    );
    assert_eq!(enumerated, run, "only {enumerated} of {run} covers enumerated");
}

// ---- Fixed games --------------------------------------------------------------

fn zachary() -> Graph<'static> {
    Graph::new(34, Edges::Flat(&ZACHARY), None, None).unwrap()
}

/// Zachary at its density: brute-force regret, the community count, the
/// quality, cover rows, and the same random words giving the same result.
/// (That an equilibrium is a fixed point is pins.rs's
/// `equilibria_are_fixed_points`.)
#[test]
fn zachary_determinism_and_regret() {
    let g = zachary();
    let density = 78.0 / (34.0 * 33.0 / 2.0);
    for isolation in [false, true] {
        let o = options(density, 1, isolation);
        let first = partition(&g, &o, None, &mut Callbacks::new(&mut seeded(0)), None).unwrap();
        let k = first.count;
        assert!(
            partition_regret(&g, &first.membership, k, &o) <= 1.0,
            "isolation {isolation}"
        );
        assert!((2..=8).contains(&k), "k = {k}");
        assert!(first.quality > 0.3, "quality {}", first.quality);
        let again = partition(&g, &o, None, &mut Callbacks::new(&mut seeded(0)), None).unwrap();
        assert_eq!(again.count, k, "not deterministic");
        assert_eq!(again.quality.to_bits(), first.quality.to_bits());
        assert_eq!(again.membership, first.membership, "not deterministic");
    }
    for cap in 2..=4 {
        for isolation in [false, true] {
            let o = options(density, cap, isolation);
            let c = cover(&g, &o, None, &mut Callbacks::new(&mut seeded(0)), None).unwrap();
            assert!(cover_well_formed(&c.cover, 34, c.count, cap), "cap {cap}");
            assert!(
                close_to(c.quality, cover_potential(&g, &c.cover, c.count, density)),
                "quality"
            );
            assert!(c.quality > 0.3, "quality {}", c.quality);
            let c2 = cover(&g, &o, None, &mut Callbacks::new(&mut seeded(0)), None).unwrap();
            assert_eq!(c2.cover, c.cover, "cover not deterministic");
            assert_eq!(c2.quality.to_bits(), c.quality.to_bits());
        }
    }
}

/// The two edge layouts give the same game.
#[test]
fn edge_layouts() {
    let g = zachary();
    let first: Vec<i64> = ZACHARY.iter().step_by(2).copied().collect();
    let second: Vec<i64> = ZACHARY.iter().skip(1).step_by(2).copied().collect();
    let split = Graph::new(34, Edges::Split(&first, &second), None, None).unwrap();
    let o = options(0.1, 1, false);
    let a = partition(&g, &o, None, &mut Callbacks::new(&mut seeded(3)), None).unwrap();
    let b = partition(&split, &o, None, &mut Callbacks::new(&mut seeded(3)), None).unwrap();
    assert_eq!(a.count, b.count, "partition layouts differ");
    assert_eq!(a.quality.to_bits(), b.quality.to_bits());
    assert_eq!(a.membership, b.membership, "partition layouts differ");
    let o = options(0.1, 3, true);
    let ca = cover(&g, &o, None, &mut Callbacks::new(&mut seeded(3)), None).unwrap();
    let cb = cover(&split, &o, None, &mut Callbacks::new(&mut seeded(3)), None).unwrap();
    assert_eq!(ca.count, cb.count, "cover layouts differ");
    assert_eq!(ca.quality.to_bits(), cb.quality.to_bits());
    assert_eq!(ca.cover, cb.cover, "cover layouts differ");
}

/// A game on which the earlier core cycled between equal-payoff moves: it
/// must stop, at an equilibrium, from every order.
#[test]
fn former_tie_cycle() {
    const EDGES: [i64; 30] = [
        9, 20, 20, 19, 9, 3, 8, 11, 4, 10, 10, 17, 16, 7, 0, 17, 20, 5, 3, 15, 2, 8, 15, 13, 17,
        13, 21, 20, 8, 4,
    ];
    const NW: [f64; 22] = [
        0.5, 1.3, 1.7, 0.7, 1.9, 1.3, 1.3, 0.7, 0.9, 1.7, 2.3, 1.7, 0.9, 0.7, 1.1, 1.9, 2.1, 0.7,
        1.5, 1.9, 0.9, 1.7,
    ];
    let g = Graph::new(22, Edges::Flat(&EDGES), None, Some(&NW)).unwrap();
    for seed in 0..50 {
        for isolation in [false, true] {
            let o = options(0.342, 1, isolation);
            let r = partition(&g, &o, None, &mut Callbacks::new(&mut seeded(seed)), None)
                .unwrap_or_else(|e| panic!("{}", e.message));
            assert!(
                partition_regret(&g, &r.membership, r.count, &o) <= 1.0,
                "seed {seed}"
            );
        }
    }
}

/// Count limits on a ring, from the default starts (also with more labels
/// than vertices for covers). One random source per target serves its five
/// calls, as in test_core.c.
#[test]
fn counts() {
    let edges: Vec<i64> = (0..12).flat_map(|v| [v, (v + 1) % 12]).collect();
    let g = Graph::new(12, Edges::Flat(&edges), None, None).unwrap();
    for target in 1..=12i64 {
        let mut rng = seeded(target as u64);
        let mut o = options(0.3, 1, true);
        o.n_communities = target;
        let r = partition(&g, &o, None, &mut Callbacks::new(&mut rng), None)
            .unwrap_or_else(|e| panic!("{}", e.message));
        assert_eq!(r.count, target, "target {target} gave {}", r.count);
        assert!(
            partition_regret(&g, &r.membership, r.count, &o) <= 1.0,
            "target {target}"
        );
        for cap in 2..=3 {
            let mut o = options(0.3, cap, true);
            o.n_communities = target * cap;
            let c = cover(&g, &o, None, &mut Callbacks::new(&mut rng), None)
                .unwrap_or_else(|e| panic!("{}", e.message));
            assert!(
                c.count == target * cap && cover_well_formed(&c.cover, 12, c.count, cap),
                "cover target {} gave {}",
                target * cap,
                c.count
            );
            o.n_communities = -1;
            o.max_total_communities = target;
            let c = cover(&g, &o, None, &mut Callbacks::new(&mut rng), None)
                .unwrap_or_else(|e| panic!("{}", e.message));
            assert!(
                c.count <= target && cover_well_formed(&c.cover, 12, c.count, cap),
                "cover max {target} gave {}",
                c.count
            );
        }
    }
}

/// A larger planted partition (1500 vertices, 15 blocks), unweighted and
/// weighted, with and without isolation; covers at caps 2, 4, 8.
#[test]
#[cfg_attr(miri, ignore = "1500 vertices: hours under Miri")]
fn planted() {
    const N: i64 = 1500;
    const BLOCKS: i64 = 15;
    let (edges, weights) = planted_graph(&mut Draw::new());
    for weighted in [false, true] {
        let g = Graph::new(N, Edges::Flat(&edges), weighted.then_some(&weights[..]), None).unwrap();
        for isolation in [false, true] {
            let o = options(0.02, 1, isolation);
            let r = partition(&g, &o, None, &mut Callbacks::new(&mut seeded(11)), None)
                .unwrap_or_else(|e| panic!("{}", e.message));
            let k = r.count;
            assert!(partition_well_formed(&r.membership, N, k), "planted");
            assert!(
                close_to(r.quality, partition_potential(&g, &r.membership, k, 0.02)),
                "quality"
            );
            assert!(
                partition_regret(&g, &r.membership, k, &o) <= 1.0,
                "planted regret"
            );
            assert!((BLOCKS / 2..=4 * BLOCKS).contains(&k), "planted k = {k}");
            for cap in [2, 4, 8] {
                for negative in [false, true] {
                    let o = options(if negative { -0.001 } else { 0.02 }, cap, isolation);
                    let c = cover(&g, &o, None, &mut Callbacks::new(&mut seeded(12)), None)
                        .unwrap_or_else(|e| panic!("{}", e.message));
                    assert!(
                        cover_well_formed(&c.cover, N, c.count, cap),
                        "planted cover"
                    );
                    let potential = cover_potential(&g, &c.cover, c.count, o.resolution);
                    assert!(
                        close_to(c.quality, potential),
                        "cover quality {:e} vs {:e}",
                        c.quality,
                        potential
                    );
                }
            }
        }
    }
}

// ---- Callbacks and errors ----------------------------------------------------

/// A failing random source and an interruption stop both engines; a 40000-
/// vertex ring makes the run reach its first interruption check. (A failed
/// call returns no cover: `Result` makes that so.)
#[test]
fn callbacks() {
    const N: i64 = 40000;
    let edges: Vec<i64> = (0..N).flat_map(|v| [v, (v + 1) % N]).collect();
    let g = Graph::new(N, Edges::Flat(&edges), None, None).unwrap();
    let mut o = options(0.1, 1, false);
    let err = partition(&g, &o, None, &mut Callbacks::new(&mut Failing), None).unwrap_err();
    assert_eq!(err.kind, Kind::Callback, "rng failure: {}", err.message);
    o.max_memberships = 2;
    let err = cover(&g, &o, None, &mut Callbacks::new(&mut Failing), None).unwrap_err();
    assert_eq!(err.kind, Kind::Callback, "rng failure (cover): {}", err.message);

    let calls = Cell::new(0);
    let mut interrupt_now = || {
        calls.set(calls.get() + 1);
        true
    };
    let mut rng = seeded(1);
    let mut o = options(0.1, 1, false);
    let err = partition(
        &g,
        &o,
        None,
        &mut Callbacks {
            rng: &mut rng,
            interrupt: Some(&mut interrupt_now),
        },
        None,
    )
    .unwrap_err();
    assert_eq!(err.kind, Kind::Interrupted, "interrupt: {}", err.message);
    o.max_memberships = 2;
    let err = cover(
        &g,
        &o,
        None,
        &mut Callbacks {
            rng: &mut rng,
            interrupt: Some(&mut interrupt_now),
        },
        None,
    )
    .unwrap_err();
    assert_eq!(
        err.kind,
        Kind::Interrupted,
        "interrupt (cover): {}",
        err.message
    );
    assert_eq!(calls.get(), 2, "interrupt called {} times", calls.get());
}

/// Every error path of test_core.c, with its kind and (where the C test reads
/// it) its message. Not ported, as Rust's types rule them out: a missing
/// random source (`o.rng = NULL`: `Callbacks` holds a `&mut dyn Rng`), and a
/// failed call handing back a cover (`Result` returns none).
#[test]
fn errors() {
    let g = zachary();
    let mut rng = seeded(0);
    let mut cb = Callbacks::new(&mut rng);
    fn kind_p(g: &Graph, o: &Options, start: Option<&[i64]>, cb: &mut Callbacks) -> Result<(), Error> {
        partition(g, o, start, cb, None).map(|_| ())
    }
    fn kind_c(g: &Graph, o: &Options, start: Option<&Cover>, cb: &mut Callbacks) -> Result<(), Error> {
        cover(g, o, start, cb, None).map(|_| ())
    }

    let err = kind_p(&g, &options(f64::NAN, 1, false), None, &mut cb).unwrap_err();
    assert_eq!(err.kind, Kind::Inval, "NaN resolution");
    assert!(
        err.message.contains("resolution must be finite"),
        "{}",
        err.message
    );
    let err = kind_p(&g, &options(0.1, 2, false), None, &mut cb).unwrap_err();
    assert_eq!(err.kind, Kind::Inval, "partition cap 2");
    let err = kind_c(&g, &options(0.1, 35, false), None, &mut cb).unwrap_err();
    assert_eq!(err.kind, Kind::Inval, "cap above n");
    let mut o = options(0.1, 1, false);
    o.n_communities = 35;
    let err = kind_p(&g, &o, None, &mut cb).unwrap_err();
    assert_eq!(err.kind, Kind::Inval, "K above n");
    o.n_communities = 5;
    o.max_total_communities = 4;
    let err = kind_p(&g, &o, None, &mut cb).unwrap_err();
    assert_eq!(err.kind, Kind::Inval, "K above max");
    let err = kind_c(&g, &options(1e305, 2, false), None, &mut cb).unwrap_err();
    assert_eq!(err.kind, Kind::Overflow, "overflowing resolution");

    let looped = [0i64, 1, 1, 1];
    let bad = Graph::new(2, Edges::Flat(&looped), None, None).unwrap();
    let err = kind_c(&bad, &options(0.1, 2, false), None, &mut cb).unwrap_err();
    assert_eq!(err.kind, Kind::Inval, "loop in a cover");
    assert!(err.message.contains("loopless"), "{}", err.message);
    kind_p(&bad, &options(0.1, 1, false), None, &mut cb)
        .unwrap_or_else(|e| panic!("a loop is fine in a partition: {}", e.message));
    let outside = [0i64, 5];
    let bad = Graph::new(2, Edges::Flat(&outside), None, None).unwrap();
    let err = kind_p(&bad, &options(0.1, 1, false), None, &mut cb).unwrap_err();
    assert_eq!(err.kind, Kind::Inval, "end outside");
    let double_edge = [0i64, 1, 0, 1];
    let negative = [1.0f64, -1.0];
    let bad = Graph::new(2, Edges::Flat(&double_edge), Some(&negative), None).unwrap();
    let err = kind_c(&bad, &options(0.1, 2, false), None, &mut cb).unwrap_err();
    assert_eq!(err.kind, Kind::Inval, "negative cover weight");
    kind_p(&bad, &options(0.1, 1, false), None, &mut cb)
        .unwrap_or_else(|e| panic!("signed partition weights: {}", e.message));
    let nan_weight = [f64::NAN, 1.0];
    let bad = Graph::new(2, Edges::Flat(&double_edge), Some(&nan_weight), None).unwrap();
    let err = kind_p(&bad, &options(0.1, 1, false), None, &mut cb).unwrap_err();
    assert_eq!(err.kind, Kind::Inval, "NaN weight");
    let bad = Graph::new(2, Edges::Flat(&double_edge), None, Some(&negative)).unwrap();
    let err = kind_p(&bad, &options(0.1, 1, false), None, &mut cb).unwrap_err();
    assert_eq!(err.kind, Kind::Inval, "negative node weight");

    // Starts.
    let o = options(0.1, 1, false);
    let mut membership: Vec<i64> = (0..34).map(|v| v % 3).collect();
    membership[5] = -1;
    let err = kind_p(&g, &o, Some(&membership), &mut cb).unwrap_err();
    assert_eq!(err.kind, Kind::Inval, "negative label");
    let membership: Vec<i64> = (0..34).map(|v| v % 3).collect();
    let mut o = options(0.1, 1, false);
    o.max_total_communities = 2;
    let err = kind_p(&g, &o, Some(&membership), &mut cb).unwrap_err();
    assert_eq!(err.kind, Kind::Inval, "start above max");
    o.max_total_communities = -1;
    o.n_communities = 4;
    let err = kind_p(&g, &o, Some(&membership), &mut cb).unwrap_err();
    assert_eq!(err.kind, Kind::Inval, "start count");

    let o = options(0.1, 2, true);
    let mut offset = vec![0i64; 35];
    for v in 0..34 {
        offset[v + 1] = offset[v] + 1 + 2 * (v == 7) as i64;
    }
    let mut start = Cover {
        labels: (0..offset[34]).collect(),
        offset,
    };
    let err = kind_c(&g, &o, Some(&start), &mut cb).unwrap_err();
    assert_eq!(err.kind, Kind::Inval, "row above the cap");
    assert!(
        err.message.contains("exceeds max_memberships"),
        "{}",
        err.message
    );
    // A repeated label counts once.
    start.labels[8] = start.labels[7];
    start.labels[9] = start.labels[7];
    kind_c(&g, &o, Some(&start), &mut cb).unwrap_or_else(|e| panic!("{}", e.message));
    start.offset[8] = start.offset[7];
    let err = kind_c(&g, &o, Some(&start), &mut cb).unwrap_err();
    assert_eq!(err.kind, Kind::Inval, "empty row");
    assert!(err.message.contains("non-empty"), "{}", err.message);
}

// ---- The soft fma -------------------------------------------------------------

/// test_core.c's fma stream (SplitMix64 from 0x243f6a8885a308d3) and its
/// adversarial cases.
struct FmaDraw(u64);

impl FmaDraw {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo + 1) as u64) as i32
    }

    /// A random double with unbiased exponent in [lo, hi] (clamped to the
    /// subnormals and the largest finite exponent), random sign and
    /// significand.
    fn double(&mut self, lo: i32, hi: i32) -> f64 {
        let e = (self.range(lo, hi) + 1023).clamp(0, 2046);
        let r = self.next();
        f64::from_bits((r & 0x800f_ffff_ffff_ffff) | ((e as u64) << 52))
    }

    fn sign(&mut self, x: f64) -> f64 {
        if self.next() & 1 != 0 {
            -x
        } else {
            x
        }
    }

    /// 0, -0, +-inf, a quiet NaN, +-DBL_MIN, +-the least subnormal,
    /// +-DBL_MAX, +-1 and the largest subnormal, or a random double.
    fn special(&mut self) -> f64 {
        const VALUES: [u64; 14] = [
            0x0000000000000000,
            0x8000000000000000,
            0x7ff0000000000000,
            0xfff0000000000000,
            0x7ff8000000000000,
            0x0010000000000000,
            0x8010000000000000,
            0x0000000000000001,
            0x8000000000000001,
            0x7fefffffffffffff,
            0xffefffffffffffff,
            0x3ff0000000000000,
            0xbff0000000000000,
            0x000fffffffffffff,
        ];
        let i = (self.next() % 16) as usize;
        if i < 14 {
            f64::from_bits(VALUES[i])
        } else {
            self.double(-1080, 1030)
        }
    }

    fn zero(&mut self) -> f64 {
        if self.next() & 1 != 0 {
            0.0
        } else {
            -0.0
        }
    }

    /// One adversarial case of class `cls`, drawn in test_core.c's order.
    fn case(&mut self, cls: usize) -> (f64, f64, f64) {
        match cls {
            0 => {
                let x = f64::from_bits(self.next());
                let y = f64::from_bits(self.next());
                (x, y, f64::from_bits(self.next()))
            }
            1 => {
                let x = self.double(-1080, 1030);
                let y = self.double(-1080, 1030);
                (x, y, self.double(-1080, 1030))
            }
            2 => {
                let (ex, ey) = (self.range(-40, 40), self.range(-40, 40));
                let x = self.double(ex, ex);
                let y = self.double(ey, ey);
                (x, y, self.double(ex + ey - 60, ex + ey + 60))
            }
            3 => {
                // z = -fl(x y) moved by a few ulps: the result is the
                // product's error.
                let (ex, ey) = (self.range(-300, 300), self.range(-300, 300));
                let x = self.double(ex, ex);
                let y = self.double(ey, ey);
                let shift = self.range(-3, 3) as i64 as u64;
                (x, y, f64::from_bits((-(x * y)).to_bits().wrapping_add(shift)))
            }
            4 => {
                // Short significands: x y exact, z = -x y.
                let a = (self.next() >> 38) as f64;
                let b = (self.next() >> 38) as f64;
                let s = self.range(-500, 500);
                let x = self.sign(ldexp(a, s));
                let t = self.range(-500, 500) - s;
                let y = self.sign(ldexp(b, t));
                (x, y, -(x * y))
            }
            5 => {
                // Zeros, and products that underflow to a signed zero.
                let r = self.next() % 4;
                let mut x = self.double(-700, -400);
                let mut y = self.double(-700, -400);
                let mut z = self.double(-1080, 100);
                if r == 0 {
                    x = self.zero();
                }
                if r == 1 {
                    y = self.zero();
                }
                if r >= 2 {
                    z = self.zero();
                }
                (x, y, z)
            }
            6 => {
                // Subnormal products and results.
                let (s, ex) = (self.range(-1140, -990), self.range(-600, -400));
                let x = self.double(ex, ex);
                let y = self.double(s - ex, s - ex);
                let t = self.double(-1080, -1015);
                let mut z = self.sign(t);
                if self.next() & 1 != 0 {
                    let t = self.double(-1080, -1070);
                    z = -(x * y) + self.sign(t);
                }
                (x, y, z)
            }
            7 => {
                // Products near and above the overflow threshold.
                let (s, ex) = (self.range(990, 1060), self.range(500, 520));
                let x = self.double(ex, ex);
                let y = self.double(s - ex, s - ex);
                let t = self.double(1000, 1023);
                (x, y, self.sign(t))
            }
            8 => {
                let x = self.special();
                let y = self.special();
                (x, y, self.special())
            }
            9 => {
                // a b + c in integers, scaled: up to 55 bits, often exactly
                // halfway.
                let a = ((self.next() >> 37) | (1 << 26)) as f64;
                let b = ((self.next() >> 37) | (1 << 26)) as f64;
                let wide = self.next() % 3 == 2;
                let c = ((self.next() >> if wide { 13 } else { 41 }) as i64
                    - if wide { 1i64 << 50 } else { 1i64 << 22 }) as f64;
                let band = self.next() % 4; // anywhere, subnormal results, overflow
                let s = match band {
                    2 => self.range(-1135, -1060),
                    3 => self.range(940, 990),
                    _ => self.range(-1180, 990),
                };
                let s1 = s / 2;
                let x = self.sign(ldexp(a, s1));
                let y = self.sign(ldexp(b, s - s1));
                (x, y, ldexp(c, s))
            }
            10 => {
                // Exponent gaps from 54 to 1100 between x y and z.
                let (ex, ey) = (self.range(-30, 30), self.range(-30, 30));
                let magnitude = self.range(54, 1100);
                let gap = magnitude * if self.next() & 1 != 0 { 1 } else { -1 };
                let x = self.double(ex, ex);
                let y = self.double(ey, ey);
                let t = self.double(0, 0);
                (x, y, self.sign(ldexp(t, ex + ey + gap)))
            }
            _ => {
                // z a small multiple of a quarter ulp of fl(x y).
                let (ex, ey) = (self.range(-200, 200), self.range(-200, 200));
                let x = self.double(ex, ex);
                let y = self.double(ey, ey);
                let e = frexp_exponent(x * y);
                let multiple = self.range(1, 8) as f64;
                let drop = self.range(0, 3);
                (x, y, self.sign(ldexp(multiple, e - 53 - drop)))
            }
        }
    }
}

/// ldexp without libm: musl's scalbn, which rounds once (in the last
/// product), so subnormal results are correctly rounded.
fn ldexp(x: f64, mut n: i32) -> f64 {
    let p1023 = f64::from_bits(0x7fe0_0000_0000_0000); // 0x1p1023
    let m969 = f64::from_bits(54 << 52); // 0x1p-1022 * 0x1p53
    let mut y = x;
    if n > 1023 {
        y *= p1023;
        n -= 1023;
        if n > 1023 {
            y *= p1023;
            n -= 1023;
            n = n.min(1023);
        }
    } else if n < -1022 {
        // Keep the final n below -53 to avoid double rounding in the
        // subnormal range.
        y *= m969;
        n += 1022 - 53;
        if n < -1022 {
            y *= m969;
            n += 1022 - 53;
            n = n.max(-1022);
        }
    }
    y * f64::from_bits(((0x3ff + n) as u64) << 52)
}

/// The exponent frexp gives a finite x: x = m 2^e, 0.5 <= |m| < 1 (0 for 0).
fn frexp_exponent(x: f64) -> i32 {
    let e = ((x.to_bits() >> 52) & 0x7ff) as i32;
    if e == 0 {
        if x == 0.0 {
            0
        } else {
            frexp_exponent(x * f64::from_bits((1023 + 64) << 52)) - 64
        }
    } else {
        e - 1022
    }
}

const FMA_CLASS_NAMES: [&str; 12] = [
    "random bits",
    "random doubles",
    "moderate exponents",
    "cancellation",
    "exact zero",
    "signed zeros",
    "subnormal",
    "overflow",
    "specials",
    "halfway (integers)",
    "exponent gaps",
    "near half an ulp",
];

/// test_core.c's FMA_CASES and the hash of the soft fma's results on them,
/// pinned on the instruction.
const FMA_CASES: usize = 1_000_000;
const FMA_HASH: u64 = 0x6dd6a440fccd0f72;

/// The soft fma on test_core.c's 10^6 adversarial cases: the hash of its
/// results against the one pinned on the instruction, and the results
/// against the instruction itself where this build runs it (`fma` is the
/// instruction then; never libm's fma).
#[test]
#[cfg_attr(miri, ignore = "10^6 soft fma cases; no unsafe code")]
fn soft_fma_classes_and_hash() {
    let instruction = fma_path().contains("instruction");
    let mut d = FmaDraw(0x243f6a8885a308d3);
    let mut counts = [0usize; 12];
    let mut mismatches = 0;
    let mut hash: u64 = 0xcbf29ce484222325;
    for _ in 0..FMA_CASES {
        let cls = (d.next() % 12) as usize;
        let (x, y, z) = d.case(cls);
        let soft = soft_fma(x, y, z);
        let bits = if soft.is_nan() {
            0x7ff8_0000_0000_0000
        } else {
            soft.to_bits()
        };
        hash = (hash ^ bits).wrapping_mul(0x100000001b3);
        counts[cls] += 1;
        if instruction {
            let hard = fma(x, y, z);
            if !(soft.is_nan() && hard.is_nan()) && soft.to_bits() != hard.to_bits() {
                if mismatches < 10 {
                    eprintln!(
                        "soft fma({:e}, {:e}, {:e}) = {:e}, the instruction {:e} ({})",
                        x, y, z, soft, hard, FMA_CLASS_NAMES[cls]
                    );
                }
                mismatches += 1;
            }
        }
    }
    let classes: Vec<String> = FMA_CLASS_NAMES
        .iter()
        .zip(counts)
        .map(|(name, count)| format!("{name} {count}"))
        .collect();
    println!(
        "soft fma: {FMA_CASES} cases ({}), {}",
        classes.join(", "),
        if instruction {
            format!("{mismatches} differ from the instruction")
        } else {
            "no instruction to compare with".to_string()
        }
    );
    assert_eq!(
        mismatches, 0,
        "{mismatches} soft fma results differ from the instruction"
    );
    assert_eq!(
        hash, FMA_HASH,
        "soft fma results: hash {hash:#x}, pinned {FMA_HASH:#x}"
    );
}

/// HIG_EXPECT_FMA=soft|instruction: the fma this build runs is that one (CI
/// runs x86_64 tests on a CPU with FMA and, under QEMU, on one without, and
/// checks that the dispatch picked the instruction or the soft fma).
#[test]
fn fma_dispatch_as_expected() {
    let path = fma_path();
    match std::env::var("HIG_EXPECT_FMA").as_deref() {
        Ok("soft") => assert!(path.ends_with("soft"), "fma: {path}"),
        Ok("instruction") => assert!(path.ends_with("instruction"), "fma: {path}"),
        Ok(other) => panic!("HIG_EXPECT_FMA must be soft or instruction, not {other}"),
        Err(_) => println!("fma: {path}"),
    }
}

/// Caps above SHORT_SELECTION (32): the best set is chosen by sorting, not by
/// insertion. A dense game at a low resolution, from a start of many labels
/// per vertex, so visits rank dozens of positive gains; checked by the
/// verifier, the rows' shape and determinism.
#[test]
fn large_caps_sort_the_gains() {
    let mut d = Draw::new();
    let n = if cfg!(miri) { 40 } else { 90 };
    let mut flat = Vec::new();
    for u in 0..n {
        for v in u + 1..n {
            if d.uniform() < 0.5 {
                flat.extend([u, v]);
            }
        }
    }
    let g = Graph::new(n, Edges::Flat(&flat), None, None).unwrap();
    let mut offset = vec![0i64];
    let mut labels = Vec::new();
    for v in 0..n {
        labels.extend((0..48).filter(|l| (v + l) % 3 != 0));
        offset.push(labels.len() as i64);
    }
    let start = Cover { offset, labels };
    for cap in [33, 40, n] {
        let o = options(0.002, cap, false);
        let r = cover(&g, &o, Some(&start), &mut Callbacks::new(&mut seeded(1)), None).unwrap();
        assert!(cover_well_formed(&r.cover, n, r.count, cap), "cap {cap}");
        assert!((0..n as usize).any(|v| r.cover.row(v).len() > 32), "cap {cap}: no row past the short selection");
        let report = verify_cover(&g, &o, &r.cover).unwrap();
        assert!(report.max_certified <= r.raw_slack, "cap {cap}: {:e} > {:e}", report.max_certified, r.raw_slack);
        let again = cover(&g, &o, Some(&start), &mut Callbacks::new(&mut seeded(1)), None).unwrap();
        assert_eq!(again.cover, r.cover);
        assert_eq!(again.quality.to_bits(), r.quality.to_bits());
    }
}

/// Prefetching (on above a working-set size, forced on here) changes no
/// result, and its hints stay inside their slices (debug builds check the
/// indices; Miri runs it with the hints compiled out).
#[test]
fn prefetching_changes_nothing() {
    use hedonic_core_rs::prefetch::set_override;
    let g = zachary();
    let mut runs = Vec::new();
    for on in [false, true] {
        set_override(Some(on));
        let p = partition(&g, &options(0.1, 1, false), None, &mut Callbacks::new(&mut seeded(5)), None).unwrap();
        let c = cover(&g, &options(0.1, 3, true), None, &mut Callbacks::new(&mut seeded(5)), None).unwrap();
        runs.push((p.membership, p.quality.to_bits(), c.cover, c.quality.to_bits()));
    }
    set_override(None);
    assert!(runs[0] == runs[1], "prefetching changed a result");
}
