//! The disjoint game (native/core/partition/): best-response local moving of
//! a partition.
//!
//! A visit of v scores, with v taken out of its community c0, the payoff
//! g(c) = w(v, c) - gamma n_v N(c) of every action: staying, each community
//! holding a neighbour, an empty community when isolation is allowed and the
//! count limits permit one more, and the best community v has no edge to
//! (read off the heap of communities by mass, needed only when gamma n_v < 0
//! or every other action loses). v moves when the move certainly raises its
//! exact payoff: the computed improvement exceeds a rigorous bound of its
//! rounding error. A visit that does not move v bounds, from above, the
//! exact gain of each of v's deviations: the largest such bound after the
//! last move is the equilibrium slack. Every expression is the C core's, in
//! the same order, so the results are the same bits.

use crate::arith::*;
use crate::error::{filled, Error, Result};
use crate::fma::fma;
use crate::graph::{check_input, compact_labels, limit_set, Adjacency, Graph, Id, Options};
use crate::heap::{Group, Heap, Tail};
use crate::prefetch::{prefetch, wanted};
use crate::queue::{Callbacks, Queue};
use crate::trace::Trace;

/// A partition's result: communities 0 .. count - 1 in order of first
/// appearance, the quality (potential / W; NaN when W = 0) and the raw slack.
#[derive(Clone, Debug)]
pub struct PartitionResult {
    pub membership: Vec<i64>,
    pub count: i64,
    pub quality: f64,
    pub raw_slack: f64,
    /// The slack divided by |W| as the quality is (NaN when W = 0).
    pub slack: f64,
}

struct State<'t> {
    graph: Adjacency,
    gamma: f64,
    isolation: bool,
    exact: bool,
    max_total: i64,
    membership: Vec<Id>,
    group: Vec<Group>,
    tail: Vec<Tail>,
    theta: f64,
    crowd: f64,
    rebuild: bool,
    total_up: f64,
    free_ids: Vec<Id>,
    occupied: Id,
    use_heap: bool,
    heap_ready: bool,
    heap: Heap,
    candidate: Vec<Id>,
    candidate_weight: Vec<f64>,
    queue: Queue,
    slack: f64,
    trace: Option<&'t mut Trace>,
    prefetch: bool,
}

const PREFETCH_FAR: usize = 8;
const PREFETCH_MID: usize = 4;
const PREFETCH_NEAR: usize = 2;
const PREFETCH_DEGREE: usize = 16;

/// What a visit found, for the rare paths that bound its payoffs.
struct Recall {
    c0: Id,
    walked: Id,
    count: usize,
    alone: bool,
    isolation: bool,
    walk: bool,
    begin: usize,
    end: usize,
    nv: f64,
    gn: f64,
    eg: f64,
    ew: f64,
    stay: f64,
    own_weight: f64,
    walked_gain: f64,
    stay_bound: f64,
}

/// A bound of |exact payoff - r|, r = fl(w - gn m): w the computed edge
/// weight (error <= ew), m the computed mass (error <= em), gn = fl(gamma
/// n_v) (error <= eg). When gn m is not tiny, the exact value of w - gn m is
/// known as a sum of doubles, so an exactly computed payoff gets bound 0.
#[inline]
fn payoff_bound(r: f64, w: f64, gn: f64, m: f64, ew: f64, em: f64, eg: f64) -> f64 {
    let mut rounding = add_up(mul_up(U, r.abs()), ETA);
    let p = gn * m;
    if p.abs() >= EXACT_PRODUCT || gn == 0.0 || m == 0.0 {
        let pe = fma(gn, m, -p); // gn m = p + pe exactly
        let (sum, te) = two_sum(w, -p); // w - p = sum + te exactly
        let refined = add_up(add_up(up((sum - r).abs()), te.abs()), pe.abs());
        if refined < rounding {
            rounding = refined;
        }
    }
    add_up(add_up(rounding, ew), add_up(mul_up(gn.abs(), em), mul_up(eg, add_up(m.abs(), em))))
}

impl<'t> State<'t> {
    fn set_theta(&mut self, theta: f64) {
        self.theta = theta;
        self.crowd = mul_up(4.0 * self.total_up, add_up(theta, 5.0 * U));
    }

    /// Keeps theta above c's error ratio, or asks for a rebuild.
    #[inline(always)]
    fn track(&mut self, c: Id) {
        let t = self.tail[c as usize];
        if t.err != 0.0 || t.lo != 0.0 {
            let error = add_up(t.err, t.lo.abs());
            let hi = self.group[c as usize].mass;
            if !(error <= REBUILD_RATIO * hi) {
                self.rebuild = true;
            } else {
                let ratio = ratio_up(error, hi);
                if ratio > self.theta {
                    self.set_theta(ratio);
                }
            }
        }
    }

    /// Every mass summed afresh from the members (and the heap reordered).
    fn rebuild_masses(&mut self) {
        let n = self.graph.n as usize;
        for c in 0..n {
            self.group[c].mass = 0.0;
            self.tail[c] = Tail::default();
        }
        for v in 0..n {
            let c = self.membership[v] as usize;
            let w = self.graph.node_weight(v as Id);
            let (g, t) = (&mut self.group[c], &mut self.tail[c]);
            mass_add(&mut g.mass, &mut t.lo, &mut t.err, w);
        }
        self.set_theta(0.0);
        self.rebuild = false;
        for c in 0..n {
            if self.group[c].members > 0 {
                self.track(c as Id);
            }
        }
        self.rebuild = false; // a fresh sum is as good as it gets
        if self.heap_ready {
            self.heap.clear();
            for c in 0..n {
                if self.group[c].members > 0 {
                    self.heap.insert(&self.group, c as Id);
                }
            }
        }
    }

    #[inline(always)]
    fn leave(&mut self, c: Id, weight: f64) {
        let cu = c as usize;
        self.group[cu].members -= 1;
        if self.group[cu].members == 0 {
            self.group[cu].mass = 0.0;
            self.tail[cu] = Tail::default();
            self.free_ids.push(c);
            self.occupied -= 1;
            if self.heap_ready {
                self.heap.remove(&self.group, c);
            }
        } else {
            let (g, t) = (&mut self.group[cu], &mut self.tail[cu]);
            mass_add(&mut g.mass, &mut t.lo, &mut t.err, -weight);
            self.track(c);
            if self.heap_ready {
                self.heap.update(&self.group, c);
            }
        }
    }

    #[inline(always)]
    fn join(&mut self, c: Id, weight: f64) {
        let cu = c as usize;
        let was = self.group[cu].members;
        self.group[cu].members += 1;
        if was == 0 {
            self.group[cu].mass = weight;
            self.tail[cu] = Tail::default();
            self.occupied += 1;
            if self.heap_ready {
                self.heap.insert(&self.group, c);
            }
        } else {
            let (g, t) = (&mut self.group[cu], &mut self.tail[cu]);
            mass_add(&mut g.mass, &mut t.lo, &mut t.err, weight);
            self.track(c);
            if self.heap_ready {
                self.heap.update(&self.group, c);
            }
        }
    }

    /// Builds the heap from the current masses the first time a visit needs
    /// a community without edge support: the same order as one maintained
    /// since the start.
    fn prepare_heap(&mut self) -> Result<()> {
        self.heap.reserve(self.graph.n as usize)?;
        for c in 0..self.graph.n {
            if self.group[c as usize].members > 0 {
                self.heap.insert(&self.group, c);
            }
        }
        self.heap_ready = true;
        Ok(())
    }

    /// Recomputes what the visit of v computed, with the same expressions in
    /// the same order (so the same values).
    fn recall(&self, v: Id, count: usize, walked: Id, walk: bool) -> Recall {
        let group = &self.group;
        let c0 = self.membership[v as usize];
        let alone = group[c0 as usize].members == 1;
        let begin = self.graph.start[v as usize] as usize;
        let end = self.graph.start[v as usize + 1] as usize;
        let nv = self.graph.node_weight(v);
        let gn = self.gamma * nv;
        let mut own_weight = 0.0;
        for k in 0..count {
            if self.candidate[k] == c0 {
                own_weight = self.candidate_weight[k];
            }
        }
        let ew = if self.graph.weighted {
            let mut abs_weight = 0.0f64;
            for i in begin..end {
                abs_weight += self.graph.weight[i].abs();
            }
            mul_up(gamma_k((end - begin) as f64 + 2.0), abs_weight)
        } else {
            0.0
        };
        let eg = if self.graph.node_weighted { product_error(self.gamma, nv, gn) } else { 0.0 };
        let stay = fma(
            -gn,
            if alone { 0.0 } else { (group[c0 as usize].mass - nv) + self.tail[c0 as usize].lo },
            own_weight,
        );
        let isolation = self.isolation
            && !self.exact
            && !alone
            && (self.max_total < 0 || (self.occupied as i64) < self.max_total);
        let walked_gain = if walked >= 0 { -gn * group[walked as usize].mass } else { 0.0 };
        Recall {
            c0,
            walked,
            count,
            alone,
            isolation,
            walk,
            begin,
            end,
            nv,
            gn,
            eg,
            ew,
            stay,
            own_weight,
            walked_gain,
            stay_bound: -1.0,
        }
    }

    /// c0's mass without v is (hi - n_v) + lo rounded twice: its error is
    /// the two exact residuals and c0's own error.
    fn stay_bound(&self, x: &mut Recall) -> f64 {
        if x.stay_bound < 0.0 {
            let c0 = x.c0 as usize;
            let hi = if x.alone { 0.0 } else { self.group[c0].mass };
            let lo = if x.alone { 0.0 } else { self.tail[c0].lo };
            let err = if x.alone { 0.0 } else { self.tail[c0].err };
            let (mut mass, mut error) = (0.0, 0.0);
            if hi != 0.0 || lo != 0.0 || err != 0.0 {
                let (m1, t1) = two_sum(hi, -x.nv);
                let (m, t2) = two_sum(m1, lo);
                mass = m;
                error = add_up(add_up(t1.abs(), t2.abs()), err);
            }
            x.stay_bound = payoff_bound(x.stay, x.own_weight, x.gn, mass, x.ew, error, x.eg);
        }
        x.stay_bound
    }

    /// The bound of a community's payoff: mass error from the community's
    /// own record, or, for the walk's community (which stands for every
    /// community v has no edge to), theta M.
    fn community_bound(&self, x: &Recall, c: Id, gain: f64, w: f64, proxy: bool) -> f64 {
        let hi = self.group[c as usize].mass;
        let t = self.tail[c as usize];
        let em = if proxy { mul_up(self.theta, hi) } else { add_up(t.err, t.lo.abs()) };
        payoff_bound(gain, w, x.gn, hi, if proxy { 0.0 } else { x.ew }, em, x.eg)
    }

    /// Moves v from c0 to best (-1: an empty community) and requeues its
    /// neighbours outside the community it joined. Returns that community.
    #[inline(always)]
    fn make_move(&mut self, v: Id, c0: Id, mut best: Id, nv: f64, begin: usize, end: usize) -> Id {
        if best < 0 {
            best = self.free_ids.pop().expect("c0 keeps v's company: never empty");
        }
        self.leave(c0, nv);
        self.join(best, nv);
        self.membership[v as usize] = best;
        self.queue.moved(v);
        self.slack = 0.0;
        if self.rebuild {
            self.rebuild_masses();
        }
        debug_assert!(begin <= end && end <= self.graph.adj.len());
        // SAFETY: begin..end is v's block of the well-formed adjacency, and
        // a neighbour is a vertex id, below n = membership.len().
        let adj = unsafe { self.graph.adj.get_unchecked(begin..end) };
        for &u in adj {
            debug_assert!((u as usize) < self.membership.len());
            if unsafe { *self.membership.get_unchecked(u as usize) } != best {
                // native/README.md, "Requeueing"
                self.queue.push(u);
            }
        }
        best
    }

    /// The visit's best action beats staying: moves v when the improvement
    /// is certain (`certain`: the coarse bound already shows it; otherwise,
    /// and for a trace, the improvement gets its own bound).
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn try_move(
        &mut self,
        v: Id,
        count: usize,
        walked: Id,
        walk: bool,
        best: Id,
        alt: usize,
        certain: bool,
    ) -> bool {
        let mut x = self.recall(v, count, walked, walk);
        let (mut d, mut e) = (0.0, 0.0);
        if !certain || self.trace.is_some() {
            let (best_gain, bound);
            if best < 0 {
                best_gain = 0.0;
                bound = 0.0;
            } else if best == walked {
                best_gain = x.walked_gain;
                bound = self.community_bound(&x, best, best_gain, 0.0, true);
            } else {
                let w = self.candidate_weight[alt];
                best_gain = fma(-x.gn, self.group[best as usize].mass, w);
                bound = self.community_bound(&x, best, best_gain, w, false);
            }
            let sb = self.stay_bound(&mut x);
            (d, e) = difference(best_gain, x.stay, add_up(bound, sb));
            if !certain && !(d > e) {
                return false;
            }
        }
        let joined = self.make_move(v, x.c0, best, x.nv, x.begin, x.end);
        if let Some(trace) = self.trace.as_deref_mut() {
            trace.record_move(v as i64, d, e, &[joined]);
        }
        true
    }

    /// v stays: raises the slack to the largest upper bound of the exact
    /// gain of a deviation that the coarse bound did not discard.
    #[inline(never)]
    fn raise_slack(&mut self, v: Id, count: usize, walked: Id, walk: bool, cutoff: f64, near: bool) {
        let mut x = self.recall(v, count, walked, walk);
        let mut slack = 0.0f64;
        if near {
            for k in 0..x.count {
                let c = self.candidate[k];
                if c != x.c0 {
                    let w = self.candidate_weight[k];
                    let gain = fma(-x.gn, self.group[c as usize].mass, w);
                    if gain > cutoff {
                        let bound = self.community_bound(&x, c, gain, w, false);
                        let sb = self.stay_bound(&mut x);
                        let (d, e) = difference(gain, x.stay, add_up(bound, sb));
                        let g = gain_up(d, e);
                        if g > slack {
                            slack = g;
                        }
                    }
                }
            }
        }
        if x.walked >= 0 && x.walked_gain > cutoff {
            let bound = self.community_bound(&x, x.walked, x.walked_gain, 0.0, true);
            let sb = self.stay_bound(&mut x);
            let (d, e) = difference(x.walked_gain, x.stay, add_up(bound, sb));
            let g = gain_up(d, e);
            if g > slack {
                slack = g;
            }
        }
        if (x.isolation || !x.walk) && 0.0 > cutoff {
            let sb = self.stay_bound(&mut x);
            let (d, e) = difference(0.0, x.stay, sb);
            let g = gain_up(d, e);
            if g > slack {
                slack = g;
            }
        }
        if slack > self.slack {
            self.slack = slack;
        }
    }

    /// A visit of v. `W` is a constant at each call: a unit-weight and a
    /// weighted version of the scan.
    #[inline(always)]
    fn visit<const W: bool>(&mut self, v: Id) -> Result<()> {
        let vu = v as usize;
        debug_assert!(vu < self.membership.len());
        // SAFETY: the queue holds vertex ids below n = membership.len(); a
        // community id is below n = group.len().
        let (c0, alone) = unsafe {
            let c0 = *self.membership.get_unchecked(vu);
            debug_assert!((c0 as usize) < self.group.len());
            (c0, self.group.get_unchecked(c0 as usize).members == 1)
        };
        let (begin, end) = (self.graph.range(v).start, self.graph.range(v).end);
        let nv = self.graph.node_weight(v);
        let gn = self.gamma * nv;
        let mut count = 0usize;
        let mut best = c0;
        let mut alt = usize::MAX;
        let mut walked: Id = -1;
        let mut abs_weight = 0.0f64;
        let mut own_weight = 0.0f64;
        let mut other_gain = f64::NEG_INFINITY;
        let mut walked_gain = 0.0f64;

        if alone && self.exact {
            // Leaving would empty c0 and break the exact count: no deviation.
            self.queue.settle(v);
            return Ok(());
        }
        {
            debug_assert!(begin <= end && end <= self.graph.adj.len());
            debug_assert!(!W || self.graph.weight.len() == self.graph.adj.len());
            // SAFETY: begin <= end <= adj.len() = weight.len() when weighted.
            let adj = unsafe { self.graph.adj.get_unchecked(begin..end) };
            let weights: &[f64] = if W { unsafe { self.graph.weight.get_unchecked(begin..end) } } else { &[] };
            let membership = &self.membership[..];
            let group = &mut self.group[..];
            let candidate = &mut self.candidate[..];
            let candidate_weight = &mut self.candidate_weight[..];
            debug_assert!(adj.len() < candidate.len());
            for (i, &u) in adj.iter().enumerate() {
                debug_assert!((u as usize) < membership.len());
                // SAFETY: a neighbour is a vertex id below n = membership.len().
                let c = unsafe { *membership.get_unchecked(u as usize) };
                // Own support needs no sparse lookup: the same unit additions
                // in CSR order (0 + 1 is exactly the first assignment of 1).
                if !W && c == c0 {
                    own_weight += 1.0;
                    continue;
                }
                debug_assert!(!W || weights.len() == adj.len());
                // SAFETY: a weighted adjacency has one weight per neighbour.
                let w = if W { unsafe { *weights.get_unchecked(i) } } else { 1.0 };
                if W {
                    abs_weight += w.abs();
                }
                debug_assert!((c as usize) < group.len() && count < candidate.len());
                // SAFETY: a community id is below n = group.len(); count is at
                // most the degree, below candidate.len() = max degree + 1; a
                // slot is a position already written, below count.
                unsafe {
                    let g = group.get_unchecked_mut(c as usize);
                    let slot = g.slot;
                    if slot < 0 {
                        g.slot = count as Id;
                        *candidate.get_unchecked_mut(count) = c;
                        *candidate_weight.get_unchecked_mut(count) = w;
                        count += 1;
                    } else {
                        *candidate_weight.get_unchecked_mut(slot as usize) += w;
                    }
                }
            }
        }
        if !W {
            abs_weight = (end - begin) as f64;
            if own_weight > 0.0 {
                // The rare bound paths recall it from the candidate array;
                // scoring and the heap walk skip c0, so every alternative
                // keeps its encounter order and ties.
                self.candidate[count] = c0;
                self.candidate_weight[count] = own_weight;
                count += 1;
            }
        }
        debug_assert!((c0 as usize) < self.group.len() && self.group.len() == self.tail.len());
        debug_assert!(count <= self.candidate.len() && self.candidate.len() == self.candidate_weight.len());
        // SAFETY: c0 is a community id, below n = group.len() = tail.len();
        // a slot is a position written by the scan, below count; count is at
        // most the degree + 1 <= candidate.len() = candidate_weight.len().
        let (g0, lo0) = unsafe { (*self.group.get_unchecked(c0 as usize), self.tail.get_unchecked(c0 as usize).lo) };
        if W && g0.slot >= 0 {
            debug_assert!((g0.slot as usize) < count);
            own_weight = unsafe { *self.candidate_weight.get_unchecked(g0.slot as usize) };
        }
        let stay = fma(-gn, if alone { 0.0 } else { (g0.mass - nv) + lo0 }, own_weight);
        let mut best_gain = stay;
        let group = &self.group[..];
        // SAFETY: as above, count <= candidate.len() = candidate_weight.len().
        let (candidate, candidate_weight) =
            unsafe { (self.candidate.get_unchecked(..count), self.candidate_weight.get_unchecked(..count)) };
        // Selects, not branches: the order of encounter decides ties, and
        // the loop stays short (most visits score a few candidates).
        let mut k = 0;
        while k < count {
            // SAFETY: k < count, the length of both slices.
            let (c, weight) = unsafe { (*candidate.get_unchecked(k), *candidate_weight.get_unchecked(k)) };
            if c != c0 {
                debug_assert!((c as usize) < group.len());
                // SAFETY: a candidate is a community id, below n = group.len().
                let mass = unsafe { group.get_unchecked(c as usize).mass };
                let gain = fma(-gn, mass, weight);
                other_gain = if gain > other_gain { gain } else { other_gain };
                let better = gain > best_gain;
                best = if better { c } else { best };
                alt = if better { k } else { alt };
                best_gain = if better { gain } else { best_gain };
            }
            k += 1;
        }
        let isolation = self.isolation
            && !self.exact
            && !alone
            && (self.max_total < 0 || (self.occupied as i64) < self.max_total);
        if isolation && best_gain < 0.0 {
            best = -1; // an empty community: payoff 0
            best_gain = 0.0;
        }
        let walk = self.use_heap && (gn < 0.0 || best_gain < 0.0);
        let mut status = Ok(());
        if walk {
            if !self.heap_ready {
                status = self.prepare_heap();
            }
            if status.is_ok() {
                self.heap.walk_begin();
                loop {
                    match self.heap.walk_next(&self.group) {
                        Err(err) => {
                            status = Err(err);
                            break;
                        }
                        Ok(None) => break,
                        Ok(Some(c)) => {
                            if c != c0 && self.group[c as usize].slot < 0 {
                                walked = c;
                                walked_gain = -gn * self.group[c as usize].mass;
                                if walked_gain > best_gain {
                                    best = c;
                                    best_gain = walked_gain;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
        let group = &mut self.group[..];
        for &c in &self.candidate[..count] {
            debug_assert!((c as usize) < group.len());
            // SAFETY: a candidate is a community id, below n = group.len().
            unsafe { group.get_unchecked_mut(c as usize).slot = -1 };
        }
        status?;

        // The errors of gamma n_v and of each w(v, c).
        let eg = if self.graph.node_weighted { product_error(self.gamma, nv, gn) } else { 0.0 };
        let ew = if W { mul_up(gamma_k((end - begin) as f64 + 2.0), abs_weight) } else { 0.0 };
        // coarse is at least four times a bound of every payoff's error here,
        // so payoffs whose computed values differ by more than 3 coarse
        // certainly compare the same way exactly.
        let coarse =
            4.0 * (U * abs_weight + ETA + ew + eg * self.total_up) + gn.abs() * self.crowd;
        if best != c0 {
            let certain = best_gain > stay + 3.0 * coarse;
            if certain && self.trace.is_none() {
                self.make_move(v, c0, best, nv, begin, end);
                return Ok(());
            } else if self.try_move(v, count, walked, walk, best, alt, certain) {
                return Ok(());
            }
        }

        // v stays. The deviations the coarse bound does not discard get their
        // own bounds.
        let cutoff = stay - 3.0 * coarse;
        if other_gain > cutoff
            || (walked >= 0 && walked_gain > cutoff)
            || ((isolation || !walk) && 0.0 > cutoff && !(alone && ew == 0.0))
        {
            self.raise_slack(v, count, walked, walk, cutoff, other_gain > cutoff);
        }
        self.queue.settle(v);
        Ok(())
    }

    /// Prefetches, in stages, what the visits of the next queued vertices
    /// will read: their adjacency offsets 8 visits ahead, adjacency blocks 4
    /// ahead, the neighbours' communities 2 ahead and those communities'
    /// records 1 ahead. Only cache hints: no value depends on them.
    #[inline(always)]
    fn prefetch_ahead(&self) {
        let (g, q) = (&self.graph, &self.queue);
        if let Some(u) = q.peek(PREFETCH_FAR) {
            prefetch(&g.start, u as usize);
            prefetch(&self.membership, u as usize);
        }
        if let Some(u) = q.peek(PREFETCH_MID) {
            let b = g.start[u as usize] as usize;
            prefetch(&g.adj, b);
            prefetch(&self.group, self.membership[u as usize] as usize);
        }
        if let Some(u) = q.peek(PREFETCH_NEAR) {
            let (b, e) = (g.start[u as usize] as usize, g.start[u as usize + 1] as usize);
            for &w in &g.adj[b..e.min(b + PREFETCH_DEGREE)] {
                prefetch(&self.membership, w as usize);
            }
        }
        if let Some(u) = q.peek(0) {
            let (b, e) = (g.start[u as usize] as usize, g.start[u as usize + 1] as usize);
            for &w in &g.adj[b..e.min(b + PREFETCH_DEGREE)] {
                prefetch(&self.group, self.membership[w as usize] as usize);
            }
        }
    }

    fn run(&mut self, cb: &mut Callbacks) -> Result<()> {
        let weighted = self.graph.weighted;
        let ahead = self.prefetch;
        while let Some(v) = self.queue.next(cb)? {
            if ahead {
                self.prefetch_ahead();
            }
            if weighted {
                self.visit::<true>(v)?;
            } else {
                self.visit::<false>(v)?;
            }
        }
        Ok(())
    }
}

/// The start: the caller's labels or the default, compacted, checked
/// against the count limits. Returns the compacted ids and their count.
fn start_labels(options: &Options, n: Id, start: Option<&[i64]>) -> Result<(Vec<Id>, i64)> {
    let default: Vec<i64>;
    let labels: &[i64] = match start {
        Some(labels) => {
            if labels.len() != n as usize {
                return Err(Error::inval("The membership array has the wrong length."));
            }
            labels
        }
        None => {
            let target: i64 = if limit_set(options.n_communities) {
                options.n_communities
            } else if limit_set(options.max_total_communities)
                && options.max_total_communities < n as i64
            {
                options.max_total_communities
            } else {
                n as i64
            };
            default = (0..n as i64).map(|v| v % target).collect();
            &default
        }
    };
    if labels.iter().any(|&x| x < 0) {
        return Err(Error::inval("Initial membership labels must be non-negative."));
    }
    let mut membership = filled(n as usize, 0 as Id)?;
    let distinct = compact_labels(labels, &mut membership)?;
    if limit_set(options.max_total_communities) && distinct > options.max_total_communities {
        return Err(Error::inval("Initial membership exceeds max_total_communities."));
    }
    if limit_set(options.n_communities) && distinct != options.n_communities {
        return Err(Error::inval(
            "Initial membership does not contain exactly n_communities communities.",
        ));
    }
    Ok((membership, distinct))
}

/// (2 sum_internal w - gamma sum_C N(C)^2) / 2W, in the caller's edge list.
fn quality(graph: &Graph, gamma: f64, membership: &[i64], count: i64) -> Result<f64> {
    let mut mass = filled(count as usize, 0.0f64)?;
    let (mut q, mut total) = (0.0f64, 0.0f64);
    let weights = graph.edge_weights;
    graph.each_edge(|e, a, b| {
        let w = weights.map_or(1.0, |w| w[e]);
        total += w;
        if membership[a as usize] == membership[b as usize] {
            q += 2.0 * w;
        }
    });
    for v in 0..graph.vertex_count {
        mass[membership[v as usize] as usize] += graph.node_weight(v);
    }
    for &m in &mass {
        q = fma(-(gamma * m), m, q);
    }
    Ok(if total != 0.0 { q / (2.0 * total) } else { f64::NAN })
}

/// Local moving of a partition (`hig_partition` / `hig_partition_traced`).
/// `start`: one non-negative label per vertex, or None for singletons (under
/// a count limit, vertex v in community v mod K).
pub fn partition(
    graph: &Graph,
    options: &Options,
    start: Option<&[i64]>,
    cb: &mut Callbacks,
    trace: Option<&mut Trace>,
) -> Result<PartitionResult> {
    let mut call = PartitionCall::prepare(graph, options, start, trace)?;
    call.run(cb)?;
    call.finish(graph)
}

/// `partition` in three steps, for a caller that must not hold the graph
/// while the visits run (the CPython binding runs them without the GIL):
/// `prepare` reads the graph, the options and the start; `run` reads only
/// what the call owns (and the callbacks); `finish` reads the graph again,
/// for the quality, and refuses one of another size. The same draws, visits
/// and bits as `partition`, which is these three calls.
pub struct PartitionCall<'t> {
    state: State<'t>,
    options: Options,
    edge_count: i64,
}

impl<'t> PartitionCall<'t> {
    pub fn prepare(
        graph: &Graph,
        options: &Options,
        start: Option<&[i64]>,
        mut trace: Option<&'t mut Trace>,
    ) -> Result<PartitionCall<'t>> {
        let mut options = *options;
        check_input(graph, &mut options, false)?;
        let n = graph.vertex_count as Id;
        let (membership, occupied) = start_labels(&options, n, start)?;

        let gamma = options.resolution;
        let isolation = options.allow_isolation;
        let exact = limit_set(options.n_communities);
        let max_total = options.max_total_communities;
        let adjacency = Adjacency::build(graph)?;
        // The visits read it unchecked: build makes it so (n + 1 offsets that
        // are prefix sums of the degrees, every neighbour an edge end it checked
        // below n), and the state, private to this module, never changes it.
        debug_assert!(adjacency.well_formed());
        // N is a sum of n non-negative terms: N(1 + gamma_n) bounds the exact one,
        // and every community's mass stays below twice that.
        let total = adjacency.total_node_weight;
        let total_up = up(2.0 * add_up(total, mul_up(gamma_k(n as f64), total)));
        let mut group = filled(n as usize, Group::EMPTY)?;
        for &c in &membership {
            group[c as usize].members += 1;
        }
        let mut free_ids = filled(0, 0 as Id)?;
        free_ids.try_reserve_exact(n as usize).map_err(|_| Error::nomem())?;
        let mut c = n - 1;
        while c >= occupied as Id {
            free_ids.push(c);
            c -= 1;
        }
        let max_degree = adjacency.max_degree as usize;
        // membership, group, tail, start, the queue's ring, flags, order and
        // record per vertex; the adjacency (and weights) per half-edge.
        let working_set = n as usize * 69 + adjacency.adj.len() * if adjacency.weighted { 12 } else { 4 };
        let mut s = State {
            gamma,
            isolation,
            exact,
            max_total,
            tail: filled(n as usize, Tail::default())?,
            group,
            membership,
            theta: 0.0,
            crowd: 0.0,
            rebuild: false,
            total_up,
            free_ids,
            occupied: occupied as Id,
            use_heap: gamma < 0.0 || !isolation || limit_set(max_total) || exact,
            heap_ready: false,
            heap: Heap::new(gamma < 0.0),
            candidate: filled(max_degree + 1, 0 as Id)?,
            candidate_weight: filled(max_degree + 1, 0.0)?,
            graph: adjacency,
            queue: Queue::empty(), // the visiting order is drawn by `run`
            slack: 0.0,
            trace: None,
            prefetch: wanted(working_set),
        };
        s.rebuild_masses(); // the masses, theta and the heap
        if let Some(t) = trace.as_deref_mut() {
            for v in 0..n as usize {
                t.record_start(&[s.membership[v]]);
            }
        }
        s.trace = trace;
        Ok(PartitionCall { state: s, options, edge_count: graph.edge_count })
    }

    /// The visiting order, then the visits until every vertex is settled.
    pub fn run(&mut self, cb: &mut Callbacks) -> Result<()> {
        let s = &mut self.state;
        s.queue = Queue::new(s.graph.n, cb)?;
        s.run(cb)
    }

    /// Communities 0, 1, ... in order of first appearance, the quality in
    /// `graph`'s edge order and the slack.
    pub fn finish(self, graph: &Graph) -> Result<PartitionResult> {
        let mut s = self.state;
        let n = s.graph.n;
        if graph.vertex_count != n as i64 || graph.edge_count != self.edge_count || !graph.ends_in_range() {
            return Err(Error::inval("The graph changed during the call."));
        }
        // The run state goes before the result is built: only the labels stay.
        let (membership, raw_slack) = (std::mem::take(&mut s.membership), s.slack);
        drop(s);

        // Communities 0, 1, ... in order of first appearance.
        let mut relabel = filled(n as usize, -1 as Id)?;
        let mut next: Id = 0;
        let mut out = filled(n as usize, 0i64)?;
        for v in 0..n as usize {
            let c = membership[v] as usize;
            if relabel[c] < 0 {
                relabel[c] = next;
                next += 1;
            }
            out[v] = relabel[c] as i64;
        }
        drop(membership);
        drop(relabel);
        let next = next as i64;
        let q = quality(graph, self.options.resolution, &out, next)?;
        Ok(PartitionResult {
            membership: out,
            count: next,
            quality: q,
            raw_slack,
            slack: normalized_slack(raw_slack, graph.total_edge_weight()),
        })
    }
}
