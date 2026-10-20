//! The independent verifier of a partition or a cover
//! (native/core/check/check.c, tests only): no queue, no settlement record,
//! no heap and no accumulator shared with the engines. The masses are summed
//! afresh, each candidate's error is bounded on its own, and the communities
//! a vertex has no edge to are read off one sorted array. For each deviation
//! the report keeps the certified lower bound of its exact gain (when it is
//! certainly positive) and an upper bound.

use crate::arith::*;
use crate::cover::Cover;
use crate::error::{filled, Error, Result};
use crate::fma::fma;
use crate::graph::{check_input, compact_labels, limit_set, Adjacency, Graph, Id, Options};

const KAPPA_CHECK: f64 = 5.0 * U;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VerifyReport {
    /// Vertices checked (a vertex alone under an exact count has no deviation).
    pub vertices: i64,
    /// Vertices with a certainly improving deviation.
    pub improvable: i64,
    /// The largest certified lower bound of a deviation's gain (raw).
    pub max_certified: f64,
    /// The largest upper bound of a deviation's gain (raw).
    pub max_bound: f64,
}

/// A lower bound of d - e > 0 (0 otherwise).
fn certified(d: f64, e: f64) -> f64 {
    let t = d - e;
    if t > 0.0 {
        fma(t, -4.0 * U, t)
    } else {
        0.0
    }
}

fn note(report: &mut VerifyReport, d: f64, e: f64, improvable: &mut bool) {
    let (lower, upper) = (certified(d, e), gain_up(d, e));
    if lower > report.max_certified {
        report.max_certified = lower;
    }
    if upper > report.max_bound {
        report.max_bound = upper;
    }
    if d > e {
        *improvable = true;
    }
}

/// |exact w - gn m - r| for r = fl(w - gn m): one rounding, inputs' errors.
fn check_bound(r: f64, gn: f64, m: f64, ew: f64, em: f64, eg: f64) -> f64 {
    add_up(
        add_up(add_up(mul_up(U, r.abs()), ETA), ew),
        add_up(mul_up(gn.abs(), em), mul_up(eg, add_up(m.abs(), em))),
    )
}

/// (mass, id) ascending, or descending by mass when gamma < 0; ties by id.
fn rank(hi: &[f64], descending: bool) -> Vec<Id> {
    let mut ranked: Vec<Id> = (0..hi.len() as Id).collect();
    ranked.sort_unstable_by(|&x, &y| {
        let (mx, my) = (hi[x as usize], hi[y as usize]);
        let by_mass = if descending { my.total_cmp(&mx) } else { mx.total_cmp(&my) };
        if mx != my {
            by_mass
        } else {
            x.cmp(&y)
        }
    });
    ranked
}

fn theta_of(hi: &[f64], lo: &[f64], err: &[f64]) -> f64 {
    let mut theta = 0.0f64;
    for c in 0..hi.len() {
        let e = add_up(err[c], lo[c].abs());
        if e != 0.0 {
            let ratio = if hi[c] > 0.0 { ratio_up(e, hi[c]) } else { f64::INFINITY };
            if ratio > theta {
                theta = ratio;
            }
        }
    }
    theta
}

/// Checks a partition (one label per vertex) under `options`.
pub fn verify_partition(graph: &Graph, options: &Options, membership: &[i64]) -> Result<VerifyReport> {
    let mut options = *options;
    let mut report = VerifyReport::default();
    check_input(graph, &mut options, false)?;
    let n = graph.vertex_count as usize;
    let exact = limit_set(options.n_communities);
    let adj = Adjacency::build(graph)?;
    if membership.len() != n {
        return Err(Error::inval("The membership array has the wrong length."));
    }
    if membership.iter().any(|&x| x < 0) {
        return Err(Error::inval("Membership labels must be non-negative."));
    }
    let mut label = filled(n, 0 as Id)?;
    let k = compact_labels(membership, &mut label)? as usize;
    let mut members = filled(k, 0 as Id)?;
    let (mut hi, mut lo, mut err) = (filled(k, 0.0)?, filled(k, 0.0)?, filled(k, 0.0)?);
    let mut to = filled(k, 0.0f64)?;
    let mut seen = filled(k, 0u8)?;
    let mut touched: Vec<Id> = Vec::new();
    for v in 0..n {
        let c = label[v] as usize;
        mass_add(&mut hi[c], &mut lo[c], &mut err[c], adj.node_weight(v as Id));
        members[c] += 1;
    }
    let theta = theta_of(&hi, &lo, &err);
    let ranked = rank(&hi, options.resolution < 0.0);

    for v in 0..n {
        let c0 = label[v] as usize;
        let alone = members[c0] == 1;
        let nv = adj.node_weight(v as Id);
        let gn = options.resolution * nv;
        let eg = if adj.node_weighted { product_error(options.resolution, nv, gn) } else { 0.0 };
        let mut abs_weight = 0.0f64;
        let (mut stay_mass, mut stay_error) = (0.0f64, 0.0f64);
        let mut improvable = false;
        if alone && exact {
            continue;
        }
        report.vertices += 1;
        touched.clear();
        for i in adj.range(v as Id) {
            let c = label[adj.adj[i] as usize] as usize;
            let w = if adj.weighted { adj.weight[i] } else { 1.0 };
            abs_weight += w.abs();
            if seen[c] == 0 {
                seen[c] = 1;
                to[c] = 0.0;
                touched.push(c as Id);
            }
            to[c] += w;
        }
        let degree = adj.range(v as Id).len();
        let ew = if adj.weighted { mul_up(gamma_k(degree as f64 + 2.0), abs_weight) } else { 0.0 };
        if !alone {
            let (m1, t1) = two_sum(hi[c0], -nv);
            let (m, t2) = two_sum(m1, lo[c0]);
            stay_mass = m;
            stay_error = add_up(add_up(t1.abs(), t2.abs()), err[c0]);
        }
        let stay = fma(-gn, stay_mass, if seen[c0] != 0 { to[c0] } else { 0.0 });
        let stay_bound = check_bound(stay, gn, stay_mass, ew, stay_error, eg);

        // Every community of a neighbour.
        for &c in &touched {
            let c = c as usize;
            if c != c0 {
                let r = fma(-gn, hi[c], to[c]);
                let bound = check_bound(r, gn, hi[c], ew, add_up(err[c], lo[c].abs()), eg);
                let (d, e) = difference(r, stay, add_up(bound, stay_bound));
                note(&mut report, d, e, &mut improvable);
            }
        }
        // An empty community.
        if options.allow_isolation
            && !exact
            && !alone
            && (!limit_set(options.max_total_communities) || (k as i64) < options.max_total_communities)
        {
            let (d, e) = difference(0.0, stay, stay_bound);
            note(&mut report, d, e, &mut improvable);
        }
        // The best community v has no edge to stands for all of them.
        for &c in &ranked {
            let c = c as usize;
            if c != c0 && seen[c] == 0 {
                let r = -gn * hi[c];
                let bound = check_bound(r, gn, hi[c], 0.0, mul_up(theta, hi[c]), eg);
                let (d, e) = difference(r, stay, add_up(bound, stay_bound));
                note(&mut report, d, e, &mut improvable);
                break;
            }
        }
        for &c in &touched {
            seen[c as usize] = 0;
        }
        report.improvable += improvable as i64;
    }
    Ok(report)
}

#[derive(Clone, Copy)]
struct Scored {
    gain: f64,
    bound: f64,
    label: Id,
}

/// Checks a cover (sorted rows of distinct labels) under `options`.
pub fn verify_cover(graph: &Graph, options: &Options, cover: &Cover) -> Result<VerifyReport> {
    let mut options = *options;
    let mut report = VerifyReport::default();
    check_input(graph, &mut options, true)?;
    let n = graph.vertex_count as usize;
    let cap = options.max_memberships as Id;
    let exact = limit_set(options.n_communities);
    if cover.offset.len() != n + 1 {
        return Err(Error::inval("The cover does not match the graph."));
    }
    cover.validate()?;
    for v in 0..n {
        let row = cover.row(v);
        if row.is_empty() || row.len() > cap as usize {
            return Err(Error::inval(format!("Row {} has {} labels.", v, row.len())));
        }
        for i in 0..row.len() {
            if row[i] < 0 || (i > 0 && row[i] <= row[i - 1]) {
                return Err(Error::inval(format!("Row {} is not sorted and distinct.", v)));
            }
        }
    }
    let adj = Adjacency::build(graph)?;
    let mut label = filled(cover.labels.len(), 0 as Id)?;
    let k = compact_labels(&cover.labels, &mut label)? as usize;
    let mut members = filled(k, 0 as Id)?;
    let (mut hi, mut lo, mut err) = (filled(k, 0.0)?, filled(k, 0.0)?, filled(k, 0.0)?);
    let mut support = filled(k, 0.0f64)?;
    let mut seen = filled(k, 0u8)?;
    let mut touched: Vec<Id> = Vec::new();
    let mut inv_sqrt = filled(cap as usize + 1, 0.0f64)?;
    for j in 1..=cap as usize {
        inv_sqrt[j] = 1.0 / (j as f64).sqrt();
    }
    let sqrt_cap_up = up((cap as f64).sqrt());
    let mut tiny_mass = 0.0;
    let mut tiny_support = false;
    {
        let smallest = 4.0 * DBL_MIN * sqrt_cap_up;
        if adj.node_weighted {
            for &w in &adj.node_weight {
                if w > 0.0 && w < smallest {
                    tiny_mass = n as f64 * ETA;
                }
            }
        }
        if adj.weighted {
            for &w in &adj.weight {
                if w > 0.0 && w < smallest {
                    tiny_support = true;
                }
            }
        }
    }
    for v in 0..n {
        let size = cover.row(v).len();
        let term = adj.node_weight(v as Id) * inv_sqrt[size];
        for i in cover.offset[v] as usize..cover.offset[v + 1] as usize {
            let c = label[i] as usize;
            mass_add(&mut hi[c], &mut lo[c], &mut err[c], term);
            members[c] += 1;
        }
    }
    let theta = theta_of(&hi, &lo, &err);
    let theta_c = add_up(theta, mul_up(KAPPA_CHECK, add_up(1.0, theta)));
    let ranked = rank(&hi, options.resolution < 0.0);
    let mut scored: Vec<Scored> = Vec::new();

    for v in 0..n {
        let (row_begin, row_end) = (cover.offset[v] as usize, cover.offset[v + 1] as usize);
        let size = row_end - row_begin;
        let nv = adj.node_weight(v as Id);
        let gn = options.resolution * nv;
        let eg = if adj.node_weighted { product_error(options.resolution, nv, gn) } else { 0.0 };
        let old_term = nv * inv_sqrt[size];
        let degree = adj.range(v as Id).len();
        let es_coef = gamma_k(degree as f64 + 8.0);
        let es_floor = if tiny_support { degree as f64 * ETA } else { 0.0 };
        let fresh = options.allow_isolation
            && !exact
            && (!limit_set(options.max_total_communities) || (k as i64) < options.max_total_communities);
        let mut mandatory: Id = 0;
        let (mut current, mut abs_current, mut own_bounds, mut bmax) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut e_pre, mut e_best) = (0.0f64, 0.0f64);
        let best;
        let mut improvable = false;

        // Supports: v's labels first, then its neighbours'.
        touched.clear();
        for i in row_begin..row_end {
            let c = label[i] as usize;
            seen[c] = 2;
            support[c] = 0.0;
            touched.push(c as Id);
        }
        for i in adj.range(v as Id) {
            let u = adj.adj[i] as usize;
            let ku = cover.row(u).len();
            let s = (if adj.weighted { adj.weight[i] } else { 1.0 }) * inv_sqrt[ku];
            for j in cover.offset[u] as usize..cover.offset[u + 1] as usize {
                let c = label[j] as usize;
                if seen[c] == 0 {
                    seen[c] = 1;
                    support[c] = 0.0;
                    touched.push(c as Id);
                }
                support[c] += s;
            }
        }
        scored.clear();

        // Gains with their bounds.
        for &c in &touched {
            let c = c as usize;
            let es = add_up(mul_up(es_coef, support[c]), es_floor);
            let (gain, bound);
            if seen[c] == 2 {
                if members[c] == 1 {
                    gain = 0.0;
                    bound = 0.0;
                    mandatory += exact as Id;
                } else {
                    let (m1, t1) = two_sum(hi[c], -old_term);
                    let (mass, t2) = two_sum(m1, lo[c]);
                    gain = fma(-gn, mass, support[c]);
                    let mut em = add_up(add_up(t1.abs(), t2.abs()), err[c]);
                    em = add_up(add_up(em, mul_up(KAPPA_CHECK, add_up(mass.abs(), em))), tiny_mass);
                    bound = check_bound(gain, gn, mass.abs(), es, em, eg);
                }
                current += gain;
                abs_current += gain.abs();
                own_bounds = add_up(own_bounds, bound);
                if exact && members[c] == 1 {
                    continue; // a mandatory label is in every set: not scored
                }
            } else {
                let em = add_up(
                    add_up(
                        add_up(err[c], lo[c].abs()),
                        mul_up(KAPPA_CHECK, add_up(hi[c], add_up(err[c], lo[c].abs()))),
                    ),
                    tiny_mass,
                );
                gain = fma(-gn, hi[c], support[c]);
                bound = check_bound(gain, gn, hi[c], es, em, eg);
            }
            scored.push(Scored { gain, bound, label: c as Id });
        }
        // The labels v has no support from: the cap heaviest (gamma n_v < 0)
        // or the lightest stand for all of them.
        {
            let want = if gn < 0.0 { cap } else { 1 };
            let mut taken: Id = 0;
            for &c in &ranked {
                if taken >= want {
                    break;
                }
                let c = c as usize;
                if seen[c] == 0 {
                    let em = add_up(mul_up(theta_c, hi[c]), tiny_mass);
                    let gain = -gn * hi[c];
                    scored.push(Scored { gain, bound: check_bound(gain, gn, hi[c], 0.0, em, eg), label: c as Id });
                    taken += 1;
                }
            }
        }
        for &c in &touched {
            seen[c as usize] = 0;
        }
        for s in &scored {
            if s.bound > bmax {
                bmax = s.bound;
            }
        }
        current *= inv_sqrt[size];
        let e_cur = add_up(
            mul_up(
                fma(inv_sqrt[size], 8.0 * U, inv_sqrt[size]),
                add_up(own_bounds, mul_up(gamma_k(size as f64), abs_current)),
            ),
            add_up(mul_up(6.0 * U, current.abs()), 2.0 * ETA),
        );

        // The best set: a prefix of the positive gains (with the mandatory
        // labels), or the single best label, or a fresh one.
        scored.sort_unstable_by(|x, y| {
            if x.gain != y.gain {
                if x.gain > y.gain {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            } else {
                x.label.cmp(&y.label)
            }
        });
        let used = scored.len();
        let mut positive = 0usize;
        while positive < used
            && (positive as i64) < (cap - mandatory) as i64
            && scored[positive].gain > 0.0
        {
            positive += 1;
        }
        if mandatory == 0 && positive == 0 {
            let mut b = if used > 0 { scored[0].gain } else { f64::NEG_INFINITY };
            e_best = if used > 0 { scored[0].bound } else { 0.0 };
            if fresh && !(b >= 0.0) {
                b = 0.0;
                e_best = 0.0;
            }
            best = b;
        } else {
            let (mut prefix, mut bounds) = (0.0f64, 0.0f64);
            let mut j_best = 0usize;
            let mut b = 0.0f64;
            for j in 1..=positive {
                prefix += scored[j - 1].gain;
                let score = prefix * inv_sqrt[mandatory as usize + j];
                bounds = add_up(bounds, scored[j - 1].bound);
                if (mandatory == 0 && j == 1) || score > b {
                    b = score;
                    j_best = j;
                    let f = inv_sqrt[mandatory as usize + j];
                    e_best = mul_up(fma(f, 8.0 * U, f), bounds);
                }
            }
            if b > 0.0 {
                e_pre = add_up(mul_up(gamma_k(cap as f64 + 6.0), b), 2.0 * ETA);
            }
            if j_best > 0 {
                e_best = add_up(e_best, e_pre);
            }
            best = b;
        }
        report.vertices += 1;
        // The computed best set's certified gain, and an upper bound of every
        // set's gain.
        let (d, e) = difference(best, current, add_up(e_best, e_cur));
        if d > e {
            improvable = true;
            if certified(d, e) > report.max_certified {
                report.max_certified = certified(d, e);
            }
        }
        let (d, e) = difference(best, current, add_up(add_up(e_pre, mul_up(sqrt_cap_up, bmax)), e_cur));
        if gain_up(d, e) > report.max_bound {
            report.max_bound = gain_up(d, e);
        }
        report.improvable += improvable as i64;
    }
    Ok(report)
}
