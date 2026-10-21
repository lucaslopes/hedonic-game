//! The engines on fixed synthetic graphs, timed, for A/B builds (bench/ab_unsafe.py).
//!
//! cargo run --release --example bench_core -- [ROUNDS] [FILTER]
//!
//! Prints one line per case: the fastest of ROUNDS runs (ms) and a hash of
//! every result (memberships, quality and slack bits, audit intervals), so two
//! builds can be checked to compute the same thing while their times are
//! compared. The graphs are drawn from a fixed SplitMix64 stream: the same
//! graphs in every build.
use hedonic_core_rs::audit::{AuditOptions, AuditState, Selection};
use hedonic_core_rs::cover::cover;
use hedonic_core_rs::graph::{Adjacency, Edges, Graph, Options};
use hedonic_core_rs::partition::partition;
use hedonic_core_rs::queue::Callbacks;
use hedonic_core_rs::rng::Mt19937;
use std::hint::black_box;
use std::time::Instant;

struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / 9007199254740992.0
    }
}

/// Preferential attachment: each new vertex links to `k` ends of earlier edges.
fn barabasi(n: i64, k: usize, seed: u64) -> Vec<i64> {
    let mut r = Mix(seed);
    let mut flat: Vec<i64> = Vec::new();
    for v in 1..n {
        for _ in 0..k.min(v as usize) {
            let u = if flat.is_empty() || r.below(4) == 0 { r.below(v as u64) as i64 } else { flat[r.below(flat.len() as u64) as usize] };
            if u != v {
                flat.push(v);
                flat.push(u);
            }
        }
    }
    flat
}

/// A planted partition: `m` edges, a fraction `p_in` inside blocks of `size`.
fn planted(n: i64, size: i64, m: usize, p_in: f64, seed: u64) -> Vec<i64> {
    let mut r = Mix(seed);
    let mut flat = Vec::with_capacity(2 * m);
    while flat.len() < 2 * m {
        let u = r.below(n as u64) as i64;
        let v = if r.uniform() < p_in { u / size * size + r.below(size as u64) as i64 } else { r.below(n as u64) as i64 };
        if u != v && v < n {
            flat.push(u);
            flat.push(v);
        }
    }
    flat
}

fn hash(h: &mut u64, x: u64) {
    *h = (*h ^ x).wrapping_mul(0x100000001b3);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rounds: usize = args.get(1).map_or(5, |x| x.parse().unwrap());
    let filter = args.get(2).cloned().unwrap_or_default();
    let ba = barabasi(50_000, 3, 1);
    let sbm = planted(20_000, 500, 200_000, 0.8, 2);
    let er = planted(100_000, 100_000, 500_000, 0.0, 3);
    let mut r = Mix(4);
    let sbm_w: Vec<f64> = (0..sbm.len() / 2).map(|_| 0.1 + 1.9 * r.uniform()).collect();
    let density = |flat: &Vec<i64>, n: i64| flat.len() as f64 / (n as f64 * (n - 1) as f64);
    type Case<'a> = (&'a str, i64, &'a Vec<i64>, Option<&'a [f64]>, i64, f64, bool);
    let cases: Vec<Case> = vec![
        ("BA 50k partition", 50_000, &ba, None, 1, density(&ba, 50_000), false),
        ("BA 50k cover M=3", 50_000, &ba, None, 3, 3.0 * density(&ba, 50_000), true),
        ("SBM 20k weighted partition", 20_000, &sbm, Some(&sbm_w), 1, density(&sbm, 20_000), false),
        ("SBM 20k weighted cover M=3", 20_000, &sbm, Some(&sbm_w), 3, 3.0 * density(&sbm, 20_000), true),
        ("ER 100k partition", 100_000, &er, None, 1, density(&er, 100_000), false),
    ];
    for (name, n, flat, weights, cap, gamma, audit) in cases {
        if !name.contains(filter.as_str()) {
            continue;
        }
        let o = Options { resolution: gamma, max_memberships: cap, allow_isolation: false, ..Options::default() };
        let (mut build, mut run, mut audited) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut h = 0xcbf29ce484222325u64;
        for round in 0..rounds {
            let t = Instant::now();
            let g = Graph::new(n, Edges::Flat(flat), weights, None).unwrap();
            let a = Adjacency::build(&g).unwrap();
            black_box(&a);
            build = build.min(t.elapsed().as_secs_f64());
            if round == 0 {
                a.adj.iter().for_each(|&u| hash(&mut h, u as u64));
            }
            drop(a);
            let mut mt = Mt19937::python_seed(round as u64 % 2);
            let mut cb = Callbacks::new(&mut mt);
            let t = Instant::now();
            let (q, slack, rows) = if cap == 1 {
                let p = partition(&g, &o, None, &mut cb, None).unwrap();
                let rows = hedonic_core_rs::cover::Cover {
                    offset: (0..=n).collect(),
                    labels: p.membership.clone(),
                };
                (p.quality, p.slack, rows)
            } else {
                let c = cover(&g, &o, None, &mut cb, None).unwrap();
                (c.quality, c.slack, c.cover)
            };
            run = run.min(t.elapsed().as_secs_f64());
            if round < 2 {
                hash(&mut h, q.to_bits());
                hash(&mut h, slack.to_bits());
                rows.labels.iter().for_each(|&x| hash(&mut h, x as u64));
            }
            if audit {
                let t = Instant::now();
                let state = AuditState::prepare(&g, &rows.offset, &rows.labels, None).unwrap();
                let ao = AuditOptions { max_memberships: cap, allow_isolation: false, prune: true, atol: 1e-10, rtol: 1e-9 };
                let found = state.intervals(&ao, Selection::Range(0, n.min(5000)), None).unwrap();
                audited = audited.min(t.elapsed().as_secs_f64());
                if round < 2 {
                    found.iter().flatten().for_each(|&(a, b)| {
                        hash(&mut h, a.to_bits());
                        hash(&mut h, b.to_bits());
                    });
                }
            }
        }
        let audit_ms = if audit { format!(" audit {:8.2} ms", audited * 1e3) } else { String::new() };
        println!("{name:<28} build {:7.2} ms  run {:8.2} ms{audit_ms}  hash {h:016x}", build * 1e3, run * 1e3);
    }
}
