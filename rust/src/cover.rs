//! The overlapping game (native/core/cover/): best-response local moving of
//! a cover, the unit-l2 game.
//!
//! Vertex v holds a sorted row of k_v labels and puts intensity
//! f_v = 1 / sqrt(k_v) on each; label c has mass S_c = sum of n_u f_u over its
//! holders. With v's own contribution taken out, label c is worth
//!
//! ```text
//! g_c(v) = sum_{u ~ v holding c} w_uv f_u - gamma n_v (S_c - n_v f_v [c in sigma_v])
//! ```
//!
//! to v, and a set sigma is worth sum_{c in sigma} g_c / sqrt(|sigma|). The best
//! set of at most M labels is a prefix of the labels sorted by g_c
//! (native/README.md, "The best set is a prefix"). v adopts it when that
//! certainly raises its exact payoff. A visit that does not move v bounds,
//! from above, the exact gain of every set v could hold instead: the largest
//! such bound after the last move is the equilibrium slack. Every expression
//! is the C core's, in the same order, so the results are the same bits.

use crate::arith::*;
use crate::error::{filled, push, Error, Result};
use crate::fma::fma;
use crate::prefetch::{prefetch, wanted};
use crate::graph::{check_input, compact_labels, limit_set, Adjacency, Graph, Id, Options, ID_MAX};
use crate::heap::{Group, Heap, Tail};
use crate::queue::{Callbacks, Queue};
use crate::trace::Trace;

/// KAPPA bounds the relative error of a mass term n_u f_u as stored.
const KAPPA: f64 = 5.0 * U;
const SHORT_SELECTION: Id = 32;
/// Prefetch stages, in visits ahead of the next one (see `prefetch_ahead`).
const PREFETCH_FAR: usize = 8;
const PREFETCH_MID: usize = 4;
const PREFETCH_NEAR: usize = 2;
const PREFETCH_ONE: usize = 1;
const PREFETCH_DEGREE: usize = 16;

/// A cover as CSR: row v is `labels[offset[v]..offset[v + 1]]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cover {
    pub offset: Vec<i64>,
    pub labels: Vec<i64>,
}

impl Cover {
    pub fn vertex_count(&self) -> i64 {
        self.offset.len().saturating_sub(1) as i64
    }

    pub fn row(&self, v: usize) -> &[i64] {
        &self.labels[self.offset[v] as usize..self.offset[v + 1] as usize]
    }

    /// From one row per vertex.
    pub fn from_rows<R: AsRef<[i64]>>(rows: &[R]) -> Cover {
        let mut offset = vec![0i64];
        let mut labels = Vec::new();
        for row in rows {
            labels.extend_from_slice(row.as_ref());
            offset.push(labels.len() as i64);
        }
        Cover { offset, labels }
    }

    /// Offsets start at 0, never decrease and end at the number of labels.
    pub fn validate(&self) -> Result<()> {
        let ok = !self.offset.is_empty()
            && self.offset[0] == 0
            && self.offset.windows(2).all(|w| w[0] <= w[1])
            && *self.offset.last().unwrap() == self.labels.len() as i64;
        if ok {
            Ok(())
        } else {
            Err(Error::inval("The cover's offsets do not describe its labels."))
        }
    }
}

/// A cover's result: sorted rows with labels 0 .. count - 1 in order of
/// first appearance, the quality and the slack.
#[derive(Clone, Debug)]
pub struct CoverResult {
    pub cover: Cover,
    pub count: i64,
    pub quality: f64,
    pub raw_slack: f64,
    pub slack: f64,
}

#[derive(Clone, Copy, Default)]
struct RowRef {
    start: usize,
    size: Id,
    capacity: Id,
}

/// A positive gain of a visit. Aligned as a u128: the long selection sorts
/// integer keys in the same buffer (`top_positive`).
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
struct Scored {
    gain: f64,
    label: Id,
    index: Id,
}

#[inline(always)]
fn scored_before(gain_a: f64, label_a: Id, gain_b: f64, label_b: Id) -> bool {
    gain_a > gain_b || (gain_a == gain_b && label_a < label_b)
}

/// A bound of |exact gain - r|, r = fl(s - gn m): s the computed support
/// (error <= es), m the mass without v (error <= em), gn = fl(gamma n_v)
/// (error <= eg).
#[inline]
fn gain_bound(r: f64, es: f64, gn: f64, m: f64, em: f64, eg: f64) -> f64 {
    add_up(
        add_up(add_up(mul_up(U, r.abs()), ETA), es),
        add_up(mul_up(gn.abs(), em), mul_up(eg, add_up(m, em))),
    )
}

/// >= 1 / sqrt(k) given f = fl(1 / fl(sqrt(k))).
#[inline(always)]
fn inv_sqrt_up(f: f64) -> f64 {
    fma(f, 8.0 * U, f)
}

/// What a visit computed, for its decision and for the bounds.
struct VisitRecord {
    k: Id,
    mandatory: Id,
    single: i64,
    j_best: Id,
    count: usize,
    uses_fresh: bool,
    gn: f64,
    old_term: f64,
    eg: f64,
    es_coef: f64,
    es_floor: f64,
    current: f64,
    abs_current: f64,
    e_pre: f64,
    max_support: f64,
    max_mass: f64,
    min_other: f64,
    max_other: f64,
}

/// The overlapping game's state. The visits read it unchecked where these
/// hold (made so by `cover` and kept by every change; `invariants` checks
/// them in debug builds): the adjacency is as `Adjacency::build` made it;
/// row has n entries, each row's capacity lies in the pool and holds at most
/// cap labels; every label in a row is below label_count <= group.len() =
/// tail.len(); inv_sqrt and own_support have cap + 1 entries; candidate and
/// value have the same length; a group's slot is -1 outside a visit.
struct State<'t> {
    graph: Adjacency,
    gamma: f64,
    cap: Id,
    isolation: bool,
    exact: bool,
    max_total: i64,
    sqrt_cap_up: f64,

    row: Vec<RowRef>,
    pool: Vec<Id>,
    pool_live: usize,

    group: Vec<Group>,
    tail: Vec<Tail>,
    theta: f64,
    theta_below: f64,
    rebuild: bool,
    tiny_mass: f64,
    tiny_support: bool,
    label_count: Id,
    free_ids: Vec<Id>,
    occupied: Id,

    inv_sqrt: Vec<f64>,
    use_heap: bool,
    heap_ready: bool,
    heap: Heap,

    candidate: Vec<Id>,
    value: Vec<f64>,
    top: Vec<Scored>,
    chosen: Vec<Id>,
    queue: Queue,
    own_support: Vec<f64>,
    slack: f64,
    trace: Option<&'t mut Trace>,
    /// Whether the visits prefetch ahead (a working set past the caches).
    prefetch: bool,
}

fn row_capacity(cap: Id, size: Id) -> Id {
    let mut capacity: Id = 2;
    while capacity < size {
        capacity *= 2;
    }
    capacity.min(cap)
}

impl<'t> State<'t> {
    /// The invariants the visits rely on (debug builds only).
    #[cfg(debug_assertions)]
    fn invariants(&self) -> bool {
        let n = self.graph.n as usize;
        let cap = self.cap as usize;
        self.graph.well_formed()
            && self.row.len() == n
            && self.row.iter().all(|r| {
                0 <= r.size
                    && r.size <= r.capacity
                    && r.capacity as usize <= cap
                    && r.start + r.capacity as usize <= self.pool.len()
                    && self.pool[r.start..r.start + r.size as usize]
                        .iter()
                        .all(|&c| 0 <= c && c < self.label_count)
            })
            && self.label_count as usize <= self.group.len()
            && self.group.len() == self.tail.len()
            && self.inv_sqrt.len() == cap + 1
            && self.own_support.len() == cap + 1
            && self.candidate.len() == self.value.len()
            && self.group.iter().all(|g| g.slot == -1)
    }

    // ---- Labels ------------------------------------------------------------

    fn reserve_labels(&mut self, ids: usize) -> Result<()> {
        let capacity = self.group.len();
        if ids <= capacity {
            return Ok(());
        }
        let grown = if capacity < 8 { 16 } else { 2 * capacity }.max(ids);
        self.group.try_reserve_exact(grown - capacity).map_err(|_| Error::nomem())?;
        self.tail.try_reserve_exact(grown - capacity).map_err(|_| Error::nomem())?;
        self.group.resize(grown, Group::EMPTY);
        self.tail.resize(grown, Tail::default());
        if self.heap_ready {
            self.heap.reserve(grown)?;
        }
        Ok(())
    }

    fn fresh_label(&mut self) -> Result<Id> {
        if let Some(label) = self.free_ids.pop() {
            return Ok(label);
        }
        if self.label_count >= ID_MAX - 1 {
            return Err(Error::overflow("The overlapping game ran out of label ids."));
        }
        self.reserve_labels(self.label_count as usize + 1)?;
        let label = self.label_count;
        self.label_count += 1;
        Ok(label)
    }

    /// Keeps theta above c's error ratio, or asks for a rebuild. The division
    /// is skipped when error <= fl(theta_below hi), a normal double: then
    /// neither theta nor the rebuild flag would change.
    #[inline(always)]
    fn track(&mut self, c: Id) {
        let t = self.tail[c as usize];
        let error = add_up(t.err, t.lo.abs());
        if error != 0.0 {
            let hi = self.group[c as usize].mass;
            let below = self.theta_below * hi;
            if below >= DBL_MIN && error <= below {
                return;
            }
            if !(error <= REBUILD_RATIO * hi) {
                self.rebuild = true;
            } else {
                let ratio = ratio_up(error, hi);
                if ratio > self.theta {
                    self.theta = ratio;
                    self.theta_below = if ratio >= DBL_MIN { ratio * (1.0 - 16.0 * U) } else { 0.0 };
                }
            }
        }
    }

    /// Every mass summed afresh from the rows (and the heap reordered).
    fn rebuild_masses(&mut self) {
        let labels = self.label_count as usize;
        for c in 0..labels {
            self.group[c].mass = 0.0;
            self.tail[c] = Tail::default();
        }
        for v in 0..self.graph.n as usize {
            let r = self.row[v];
            let term = self.graph.node_weight(v as Id) * self.inv_sqrt[r.size as usize];
            for i in 0..r.size as usize {
                let c = self.pool[r.start + i] as usize;
                let (g, t) = (&mut self.group[c], &mut self.tail[c]);
                mass_add(&mut g.mass, &mut t.lo, &mut t.err, term);
            }
        }
        self.theta = 0.0;
        self.theta_below = 0.0;
        for c in 0..labels {
            if self.group[c].members > 0 {
                self.track(c as Id);
            }
        }
        self.rebuild = false; // a fresh sum is as good as it gets
        if self.heap_ready {
            self.heap.clear();
            for c in 0..labels {
                if self.group[c].members > 0 {
                    self.heap.insert(&self.group, c as Id);
                }
            }
        }
    }

    fn leave(&mut self, c: Id, term: f64) -> Result<()> {
        let cu = c as usize;
        self.group[cu].members -= 1;
        if self.group[cu].members == 0 {
            self.group[cu].mass = 0.0;
            self.tail[cu] = Tail::default();
            push(&mut self.free_ids, c)?;
            self.occupied -= 1;
            if self.heap_ready {
                self.heap.remove(&self.group, c);
            }
        } else {
            let (g, t) = (&mut self.group[cu], &mut self.tail[cu]);
            mass_add(&mut g.mass, &mut t.lo, &mut t.err, -term);
            self.track(c);
            if self.heap_ready {
                self.heap.update(&self.group, c);
            }
        }
        Ok(())
    }

    fn join(&mut self, c: Id, term: f64) {
        let cu = c as usize;
        let was = self.group[cu].members;
        self.group[cu].members += 1;
        if was == 0 {
            self.group[cu].mass = term;
            self.tail[cu] = Tail::default();
            self.occupied += 1;
            if self.heap_ready {
                self.heap.insert(&self.group, c);
            }
        } else {
            let (g, t) = (&mut self.group[cu], &mut self.tail[cu]);
            mass_add(&mut g.mass, &mut t.lo, &mut t.err, term);
            self.track(c);
            if self.heap_ready {
                self.heap.update(&self.group, c);
            }
        }
    }

    fn rescale(&mut self, c: Id, old_term: f64, new_term: f64) {
        let cu = c as usize;
        {
            let (g, t) = (&mut self.group[cu], &mut self.tail[cu]);
            mass_add(&mut g.mass, &mut t.lo, &mut t.err, -old_term);
            mass_add(&mut g.mass, &mut t.lo, &mut t.err, new_term);
        }
        self.track(c);
        if self.heap_ready {
            self.heap.update(&self.group, c);
        }
    }

    /// Built only when a visit first needs labels without edge support: the
    /// same mass/id order as one maintained since the start.
    fn prepare_heap(&mut self) -> Result<()> {
        self.heap.reserve(self.group.len())?;
        for c in 0..self.label_count {
            if self.group[c as usize].members > 0 {
                self.heap.insert(&self.group, c);
            }
        }
        self.heap_ready = true;
        Ok(())
    }

    // ---- Rows --------------------------------------------------------------

    /// Rewrites the pool with every row at the front of its own capacity.
    fn compact_pool(&mut self) -> Result<()> {
        let mut pool: Vec<Id> = Vec::new();
        pool.try_reserve_exact(self.pool_live + 1024).map_err(|_| Error::nomem())?;
        for r in self.row.iter_mut() {
            let used = pool.len();
            pool.extend_from_slice(&self.pool[r.start..r.start + r.size as usize]);
            pool.resize(used + r.capacity as usize, 0);
            r.start = used;
        }
        self.pool = pool;
        Ok(())
    }

    fn write_row(&mut self, v: Id, size: Id) -> Result<()> {
        let vu = v as usize;
        if size > self.row[vu].capacity {
            let capacity = row_capacity(self.cap, size);
            let start = self.pool.len();
            self.pool.try_reserve(capacity as usize).map_err(|_| Error::nomem())?;
            self.pool.resize(start + capacity as usize, 0);
            self.pool_live += (capacity - self.row[vu].capacity) as usize;
            self.row[vu].start = start;
            self.row[vu].capacity = capacity;
        }
        let start = self.row[vu].start;
        self.pool[start..start + size as usize].copy_from_slice(&self.chosen[..size as usize]);
        self.row[vu].size = size;
        if self.pool.len() > 2 * self.pool_live + 4096 {
            self.compact_pool()?;
        }
        Ok(())
    }

    // ---- A visit -----------------------------------------------------------

    #[inline(always)]
    fn reserve_candidates(&mut self, needed: usize) -> Result<()> {
        if needed <= self.candidate.len() {
            return Ok(());
        }
        self.grow_candidates(needed)
    }

    #[cold]
    #[inline(never)]
    fn grow_candidates(&mut self, needed: usize) -> Result<()> {
        let capacity = self.candidate.len();
        let grown = if capacity < 8 { 16 } else { 2 * capacity }.max(needed);
        self.candidate.try_reserve_exact(grown - capacity).map_err(|_| Error::nomem())?;
        self.value.try_reserve_exact(grown - capacity).map_err(|_| Error::nomem())?;
        self.candidate.resize(grown, 0);
        self.value.resize(grown, 0.0);
        Ok(())
    }

    /// The `limit` best positive gains, best first, in the first entries of
    /// `top`; returns how many.
    fn top_positive(&mut self, count: usize, limit: Id) -> Result<usize> {
        if limit <= 0 {
            return Ok(0);
        }
        let (values, labels) = (&self.value[..count], &self.candidate[..count]);
        if limit <= SHORT_SELECTION {
            let limit = limit as usize;
            if self.top.len() < limit {
                self.top.try_reserve(limit - self.top.len()).map_err(|_| Error::nomem())?;
                self.top.resize(limit, Scored { gain: 0.0, label: 0, index: 0 });
            }
            // Insertion into a sorted buffer of at most `limit` entries (the
            // first `kept` of self.top; what lies past them is stale).
            let top = &mut self.top[..limit];
            let mut kept = 0usize;
            for (i, (&gain, &label)) in values.iter().zip(labels).enumerate() {
                if !(gain > 0.0) {
                    continue;
                }
                // SAFETY (this loop): 1 <= kept <= limit = top.len() wherever
                // top[kept - 1] is read, and 0 <= slot - 1 < slot < kept.
                let mut slot;
                if kept < limit {
                    slot = kept;
                    kept += 1;
                } else {
                    let last = unsafe { *top.get_unchecked(kept - 1) };
                    if scored_before(gain, label, last.gain, last.label) {
                        slot = kept - 1;
                    } else {
                        continue;
                    }
                }
                while slot > 0 {
                    let previous = unsafe { *top.get_unchecked(slot - 1) };
                    if !scored_before(gain, label, previous.gain, previous.label) {
                        break;
                    }
                    unsafe { *top.get_unchecked_mut(slot) = previous };
                    slot -= 1;
                }
                debug_assert!(slot < top.len());
                unsafe { *top.get_unchecked_mut(slot) = Scored { gain, label, index: i as Id } };
            }
            Ok(kept)
        } else {
            // Sorted as integer keys in `scored_before`'s order, in top's
            // own buffer: the gain's bits inverted (for a gain > 0 they
            // ascend with its value), then the label, then the index. Labels
            // are distinct, so (gain desc, label asc) is a total order and the
            // index never decides; an integer sort is the comparator's sort.
            self.top.clear();
            if self.top.capacity() < count {
                self.top.try_reserve(count).map_err(|_| Error::nomem())?;
            }
            let base = self.top.as_mut_ptr() as *mut u128;
            let mut positive = 0usize;
            for (i, (&gain, &label)) in values.iter().zip(labels).enumerate() {
                if gain > 0.0 {
                    let key = (!gain.to_bits() as u128) << 64 | (label as u32 as u128) << 32 | i as u32 as u128;
                    // SAFETY: positive < count <= capacity; Scored is 16 bytes
                    // aligned to 16 (repr(C, align(16))), as a u128 is.
                    unsafe { base.add(positive).write(key) };
                    positive += 1;
                }
            }
            // SAFETY: the first `positive` slots hold initialized u128s; no
            // other reference to top's buffer lives while `keys` does.
            let keys = unsafe { std::slice::from_raw_parts_mut(base, positive) };
            keys.sort_unstable();
            let kept = positive.min(limit as usize);
            for j in 0..kept {
                // SAFETY: j < positive; each slot is read as a key before the
                // Scored it decodes to is written over it. Every bit pattern
                // is a valid Scored (an f64 and two i32s).
                unsafe {
                    let key = base.add(j).read();
                    let gain = f64::from_bits(!((key >> 64) as u64));
                    let (label, index) = ((key >> 32) as u32 as Id, key as u32 as Id);
                    (base.add(j) as *mut Scored).write(Scored { gain, label, index });
                }
            }
            // SAFETY: kept <= positive <= capacity, and the first kept slots
            // are Scored values now.
            unsafe { self.top.set_len(kept) };
            Ok(kept)
        }
    }

    /// The rigorous bounds of a visit's comparison: `move` for the
    /// improvement of the best set over v's row, `gain` for the gain of any
    /// set v could hold. v's own labels are candidates 0..k-1.
    #[inline(never)]
    fn bounds(&self, x: &VisitRecord) -> (f64, f64) {
        let k = x.k as usize;
        let (mut own_bounds, mut own_max) = (0.0f64, 0.0f64);
        for i in 0..k {
            let c = self.candidate[i] as usize;
            let g = self.group[c];
            if g.members != 1 {
                let (m1, t1) = two_sum(g.mass, -x.old_term);
                let (mass, t2) = two_sum(m1, self.tail[c].lo);
                let mut em = add_up(add_up(t1.abs(), t2.abs()), self.tail[c].err);
                em = add_up(add_up(em, mul_up(KAPPA, add_up(mass.abs(), em))), self.tiny_mass);
                let bound = gain_bound(
                    self.value[i],
                    add_up(mul_up(x.es_coef, self.own_support[i]), x.es_floor),
                    x.gn,
                    mass.abs(),
                    em,
                    x.eg,
                );
                own_bounds = add_up(own_bounds, bound);
                if bound > own_max {
                    own_max = bound;
                }
            }
        }
        let theta_c = add_up(self.theta, mul_up(KAPPA, add_up(1.0, self.theta)));
        let em = add_up(mul_up(theta_c, x.max_mass), self.tiny_mass);
        let abs_gain = if x.max_other > -x.min_other { x.max_other } else { -x.min_other };
        let mut bmax = if x.count > k {
            gain_bound(
                abs_gain,
                add_up(mul_up(x.es_coef, x.max_support), x.es_floor),
                x.gn,
                x.max_mass,
                em,
                x.eg,
            )
        } else {
            0.0
        };
        if own_max > bmax {
            bmax = own_max;
        }
        let e_cur = add_up(
            mul_up(
                inv_sqrt_up(self.inv_sqrt[k]),
                add_up(own_bounds, mul_up(gamma_k(x.k as f64), x.abs_current)),
            ),
            add_up(mul_up(6.0 * U, x.current.abs()), 2.0 * ETA),
        );
        let e_best = if x.uses_fresh || (x.single < 0 && x.j_best == 0) {
            0.0 // a fresh label, or the mandatory labels alone: exactly 0
        } else if x.single >= 0 {
            bmax
        } else {
            add_up(
                mul_up(
                    inv_sqrt_up(self.inv_sqrt[(x.mandatory + x.j_best) as usize]),
                    mul_up(x.j_best as f64, bmax),
                ),
                x.e_pre,
            )
        };
        let mv = add_up(e_best, e_cur);
        let gain = add_up(add_up(x.e_pre, mul_up(self.sqrt_cap_up, bmax)), e_cur);
        (mv, gain)
    }

    /// Candidates of v after its own k labels: its neighbours' labels with
    /// their supports, in the order the C core finds them; returns the count.
    /// One reservation covers every neighbour (a candidate is a distinct
    /// label held by v or a neighbour, at most cap of them per neighbour).
    #[inline(always)]
    fn collect<const W: bool>(&mut self, ku: usize, begin: usize, end: usize) -> Result<usize> {
        let most = ku + (end - begin) * self.cap as usize;
        self.reserve_candidates(most.min(self.label_count as usize + 1))?;
        let State { graph, row, pool, inv_sqrt, group, candidate, value, .. } = self;
        debug_assert!(begin <= end && end <= graph.adj.len() && (!W || graph.weight.len() == graph.adj.len()));
        // SAFETY (this block): begin..end is a block of the well-formed
        // adjacency (weighted: one weight per half-edge); a neighbour is a
        // vertex, below n = row.len(); a row's labels lie in the pool and
        // its size is at most cap, below inv_sqrt.len() (State's invariants).
        let adj = unsafe { graph.adj.get_unchecked(begin..end) };
        let weight: &[f64] = if W { unsafe { graph.weight.get_unchecked(begin..end) } } else { &[] };
        let mut count = ku;
        let (row, pool, inv_sqrt) = (&row[..], &pool[..], &inv_sqrt[..]);
        let (group, candidate, value) = (&mut group[..], &mut candidate[..], &mut value[..]);
        for (i, &u) in adj.iter().enumerate() {
            debug_assert!((u as usize) < row.len());
            let ru = unsafe { *row.get_unchecked(u as usize) };
            let w = if W { unsafe { *weight.get_unchecked(i) } } else { 1.0 };
            debug_assert!((ru.size as usize) < inv_sqrt.len() && ru.start + ru.size as usize <= pool.len());
            let support = w * unsafe { *inv_sqrt.get_unchecked(ru.size as usize) };
            for &c in unsafe { pool.get_unchecked(ru.start..ru.start + ru.size as usize) } {
                debug_assert!((c as usize) < group.len() && count < candidate.len());
                // SAFETY: a row holds label ids below label_count <= group.len();
                // count stays below the reservation above (distinct labels);
                // a slot is a position already written, below count.
                unsafe {
                    let g = group.get_unchecked_mut(c as usize);
                    if g.slot < 0 {
                        g.slot = count as Id;
                        *candidate.get_unchecked_mut(count) = c;
                        *value.get_unchecked_mut(count) = support;
                        count += 1;
                    } else {
                        *value.get_unchecked_mut(g.slot as usize) += support;
                    }
                }
            }
        }
        Ok(count)
    }

    /// A visit of v. With `probe`, v does not move: the bound of the exact
    /// gain of v's best deviation is returned, and the state is unchanged.
    #[inline(always)]
    fn visit<const W: bool>(&mut self, v: Id, probe: bool) -> Result<f64> {
        let vu = v as usize;
        debug_assert!(vu < self.row.len());
        // SAFETY: v is a vertex (from the queue, or the slack's pass over
        // the settled ones), below n = row.len(); the adjacency is well formed; a row's size is at most cap, below
        // inv_sqrt.len() (State's invariants).
        let r = unsafe { *self.row.get_unchecked(vu) };
        let k = r.size;
        let ku = k as usize;
        let (begin, end) = unsafe { self.graph.range_unchecked(v) };
        let nv = unsafe { self.graph.node_weight_unchecked(v) };
        let gn = self.gamma * nv;
        debug_assert!(ku < self.inv_sqrt.len());
        let old_term = nv * unsafe { *self.inv_sqrt.get_unchecked(ku) };
        let fresh_allowed = self.isolation
            && !self.exact
            && (self.max_total < 0 || (self.occupied as i64) < self.max_total);
        let mut count: usize;
        let slotted: usize;
        let mut mandatory: Id = 0;
        let mut single: i64 = -1;
        let mut j_best: Id = 0;
        let mut current = 0.0f64;
        let mut max_gain = f64::NEG_INFINITY;
        let best_score: f64;
        let mut min_other = f64::INFINITY;
        let mut max_other = f64::NEG_INFINITY;
        let mut max_support = 0.0f64;
        let mut max_mass = 0.0f64;
        let mut abs_current = 0.0f64;
        let mut own_hi = 0.0f64;
        let mut own_support_max = 0.0f64;
        let mut e_pre = 0.0f64;
        let mut uses_fresh = false;
        let same: bool;

        // Candidates: v's labels first, then its neighbours' labels.
        self.reserve_candidates(ku + 1)?;
        {
            debug_assert!(r.start + ku <= self.pool.len() && ku < self.candidate.len());
            // SAFETY: v's row lies in the pool, its labels below group.len();
            // ku + 1 candidates are reserved (State's invariants).
            let own = unsafe { self.pool.get_unchecked(r.start..r.start + ku) };
            let (group, candidate, value) = (&mut self.group[..], &mut self.candidate[..], &mut self.value[..]);
            for (i, &c) in own.iter().enumerate() {
                debug_assert!((c as usize) < group.len());
                unsafe {
                    group.get_unchecked_mut(c as usize).slot = i as Id;
                    *candidate.get_unchecked_mut(i) = c;
                    *value.get_unchecked_mut(i) = 0.0;
                }
            }
        }
        let mut status: Result<()> = Ok(());
        match self.collect::<W>(ku, begin, end) {
            Ok(c) => count = c,
            Err(e) => {
                count = ku;
                status = Err(e);
            }
        }
        slotted = count;

        let eg = if self.graph.node_weighted { product_error(self.gamma, nv, gn) } else { 0.0 };
        let es_coef = gamma_k((end - begin) as f64 + 8.0);
        let es_floor = if self.tiny_support { (end - begin) as f64 * ETA } else { 0.0 };

        if status.is_ok() {
            // Gains, and the payoff of v's row.
            let exact = self.exact as Id;
            let (group, tail) = (&self.group[..], &self.tail[..]);
            debug_assert!(ku <= count && ku < self.own_support.len());
            // SAFETY: the first ku candidates are v's labels (below
            // group.len() = tail.len()); ku <= cap < own_support.len().
            let own_support = unsafe { self.own_support.get_unchecked_mut(..ku) };
            let own = unsafe { self.candidate.get_unchecked(..ku) };
            let values = unsafe { self.value.get_unchecked_mut(..ku) };
            for ((value, support), &c) in values.iter_mut().zip(own_support).zip(own) {
                let c = c as usize;
                debug_assert!(c < group.len());
                let g = unsafe { *group.get_unchecked(c) };
                let gain;
                *support = *value;
                if g.members == 1 {
                    gain = 0.0; // v alone holds it: no support, no crowding (exactly)
                    mandatory += exact;
                } else {
                    let lo = unsafe { tail.get_unchecked(c).lo };
                    gain = fma(-gn, (g.mass - old_term) + lo, *value);
                    own_hi = if g.mass > own_hi { g.mass } else { own_hi };
                    own_support_max = if *value > own_support_max { *value } else { own_support_max };
                }
                current += gain;
                abs_current += gain.abs();
                *value = gain;
                if gain > max_gain {
                    max_gain = gain;
                }
            }
            // The other labels: their bounds are taken together.
            let group = &self.group[..];
            debug_assert!(count <= self.candidate.len());
            // SAFETY: ku <= count <= candidate.len() = value.len(); a
            // candidate is a label, below group.len().
            let values = unsafe { self.value.get_unchecked_mut(ku..count) };
            for (value, &c) in values.iter_mut().zip(unsafe { self.candidate.get_unchecked(ku..count) }) {
                debug_assert!((c as usize) < group.len());
                let mass = unsafe { group.get_unchecked(c as usize).mass };
                let support = *value;
                let gain = fma(-gn, mass, support);
                if support > max_support {
                    max_support = support;
                }
                if mass > max_mass {
                    max_mass = mass;
                }
                if gain < min_other {
                    min_other = gain;
                }
                if gain > max_other {
                    max_other = gain;
                }
                *value = gain;
            }
            if max_other > max_gain {
                max_gain = max_other;
            }
            current *= self.inv_sqrt[ku];
            if fresh_allowed && max_gain < 0.0 {
                max_gain = 0.0;
            }

            // Labels v has no support from: the walk's labels stand for all.
            if self.use_heap && (gn < 0.0 || max_gain < 0.0) {
                let want = if gn < 0.0 { self.cap } else { 1 };
                let mut taken: Id = 0;
                if !self.heap_ready {
                    status = self.prepare_heap();
                }
                if status.is_ok() {
                    self.heap.walk_begin();
                    while taken < want {
                        let c = match self.heap.walk_next(&self.group) {
                            Err(err) => {
                                status = Err(err);
                                break;
                            }
                            Ok(None) => break,
                            Ok(Some(c)) => c,
                        };
                        if self.group[c as usize].slot >= 0 {
                            continue;
                        }
                        if let Err(err) = self.reserve_candidates(count + 1) {
                            status = Err(err);
                            break;
                        }
                        let mass = self.group[c as usize].mass;
                        let gain = -gn * mass;
                        self.candidate[count] = c;
                        self.value[count] = gain;
                        if mass > max_mass {
                            max_mass = mass;
                        }
                        if gain < min_other {
                            min_other = gain;
                        }
                        if gain > max_other {
                            max_other = gain;
                        }
                        count += 1;
                        taken += 1;
                    }
                }
            }
        }
        {
            debug_assert!(slotted <= self.candidate.len());
            let group = &mut self.group[..];
            // SAFETY: slotted <= candidate.len(); a candidate is a label,
            // below group.len().
            for &c in unsafe { self.candidate.get_unchecked(..slotted) } {
                debug_assert!((c as usize) < group.len());
                unsafe { group.get_unchecked_mut(c as usize).slot = -1 };
            }
        }
        status?;

        // The best set: the mandatory labels and the best prefix of the
        // positive gains, or the single best label. `same`: it is v's row.
        let kept = self.top_positive(count, self.cap - mandatory)?;
        if mandatory == 0 && kept == 0 {
            let mut best_gain = f64::NEG_INFINITY;
            for i in 0..count {
                if single < 0
                    || scored_before(
                        self.value[i],
                        self.candidate[i],
                        best_gain,
                        self.candidate[single as usize],
                    )
                {
                    single = i as i64;
                    best_gain = self.value[i];
                }
            }
            if fresh_allowed && (single < 0 || best_gain < 0.0) {
                uses_fresh = true;
                best_gain = 0.0;
            }
            best_score = best_gain;
            same = !uses_fresh && k == 1 && single == 0;
        } else {
            let mut prefix = 0.0f64;
            let mut best = 0.0f64; // j = 0: the mandatory labels alone, worth 0
            for j in 1..=kept {
                prefix += self.top[j - 1].gain;
                let score = prefix * self.inv_sqrt[mandatory as usize + j];
                if (mandatory == 0 && j == 1) || score > best {
                    best = score;
                    j_best = j as Id;
                }
            }
            best_score = best;
            if best_score > 0.0 {
                // Each computed prefix score is positive, at most best_score,
                // and within (j + 4) roundings of its exact value.
                e_pre = add_up(mul_up(gamma_k(self.cap as f64 + 6.0), best_score), 2.0 * ETA);
            }
            let mut s = mandatory + j_best == k;
            let mut j = 0;
            while s && j < j_best as usize {
                s = self.top[j].index < k;
                j += 1;
            }
            same = s;
        }

        let x = VisitRecord {
            k,
            mandatory,
            single,
            j_best,
            count,
            uses_fresh,
            gn,
            old_term,
            eg,
            es_coef,
            es_floor,
            current,
            abs_current,
            e_pre,
            max_support,
            max_mass,
            min_other,
            max_other,
        };
        if probe {
            let (_, gain) = self.bounds(&x);
            let (d, e) = difference(best_score, current, gain);
            return Ok(gain_up(d, e));
        }

        // `coarse` bounds both full bounds and the rounding of the
        // difference (native/README.md, "The coarse bound").
        let mut d = best_score - current;
        let mut e;
        let coarse = {
            let other = if max_other > -min_other { max_other } else { -min_other };
            let abs_gain = if other > abs_current { other } else { abs_current };
            let own_mass = (1.0 + self.theta) * own_hi + old_term;
            let own_em = self.theta * own_hi + 3.0 * U * own_mass;
            let mass = if own_mass > max_mass { own_mass } else { max_mass };
            let support = if own_support_max > max_support { own_support_max } else { max_support };
            let em = own_em
                + (self.theta + KAPPA * (1.0 + self.theta) + 4.0 * U) * (mass + own_em)
                + self.tiny_mass
                + 2.0 * ETA;
            let b = U * abs_gain
                + 8.0 * ETA
                + es_coef * support
                + es_floor
                + gn.abs() * em
                + eg * (mass + em);
            let root_k = self.inv_sqrt[ku] * k as f64;
            2.0 * (e_pre
                + self.sqrt_cap_up * b
                + root_k * (b + 1.125 * U * abs_gain)
                + 6.0 * U * current.abs()
                + U * d.abs()
                + 32.0 * ETA)
        };
        // The bound recorded with a visit that does not move v: the full one
        // (>= 0), or minus a coarse one (< 0) that the slack replaces by the
        // full one if it matters.
        if same {
            let t = d + coarse;
            let bound = if t > 0.0 { -up(t) } else { 0.0 };
            self.queue.settle_bound(v, bound);
            return Ok(0.0);
        }
        if d > coarse && self.trace.is_none() {
            e = 0.0; // certainly an improvement: d > coarse >= the full bound
        } else if !(d + coarse > 0.0) {
            self.queue.settle_bound(v, 0.0); // certainly no improvement, nor any gain
            return Ok(0.0);
        } else {
            let (mv, gain) = self.bounds(&x);
            (d, e) = difference(best_score, current, mv);
            if !(d > e) {
                (d, e) = difference(best_score, current, gain);
                let bound = gain_up(d, e);
                self.queue.settle_bound(v, bound);
                return Ok(0.0);
            }
        }

        // The new row, sorted.
        self.chosen.clear();
        if mandatory > 0 {
            for i in 0..ku {
                let c = self.pool[r.start + i];
                if self.group[c as usize].members == 1 {
                    self.chosen.push(c);
                }
            }
        }
        if mandatory > 0 || kept > 0 {
            for j in 0..j_best as usize {
                let label = self.top[j].label;
                self.chosen.push(label);
            }
        } else if !uses_fresh {
            let c = self.candidate[single as usize];
            self.chosen.push(c);
        }
        sort_ids(&mut self.chosen);
        if uses_fresh {
            let fresh = self.fresh_label()?;
            self.chosen.push(fresh); // the only label: still sorted
        }
        let chosen_size = self.chosen.len();

        // Masses: leave, rescale, join (rows are sorted: merge them).
        {
            let new_term = nv * self.inv_sqrt[chosen_size];
            let (mut a, mut b) = (0usize, 0usize);
            while a < ku || b < chosen_size {
                let own_a = if a < ku { self.pool[r.start + a] } else { 0 };
                let chosen_b = if b < chosen_size { self.chosen[b] } else { 0 };
                if b >= chosen_size || (a < ku && own_a < chosen_b) {
                    self.leave(own_a, old_term)?;
                    a += 1;
                } else if a >= ku || chosen_b < own_a {
                    self.join(chosen_b, new_term);
                    b += 1;
                } else {
                    if chosen_size != ku {
                        self.rescale(own_a, old_term, new_term);
                    }
                    a += 1;
                    b += 1;
                }
            }
        }
        self.write_row(v, chosen_size as Id)?;
        self.queue.moved(v);
        if self.rebuild {
            self.rebuild_masses();
        }
        if let Some(trace) = self.trace.as_deref_mut() {
            trace.record_move(v as i64, d, e, &self.chosen);
        }
        debug_assert!(begin <= end && end <= self.graph.adj.len());
        // SAFETY: v's block of the well-formed adjacency; neighbours are
        // vertices, below n.
        for &u in unsafe { self.graph.adj.get_unchecked(begin..end) } {
            unsafe { self.queue.push_unchecked(u) };
        }
        Ok(0.0)
    }

    fn visit_any(&mut self, v: Id, probe: bool) -> Result<f64> {
        if self.graph.weighted {
            self.visit::<true>(v, probe)
        } else {
            self.visit::<false>(v, probe)
        }
    }

    /// Prefetches, in stages, what the visits of the next queued vertices
    /// will read: adjacency offsets and rows 8 visits ahead, adjacency blocks
    /// and own labels 4 ahead, the neighbours' rows 2 ahead and their labels
    /// 1 ahead. Only cache hints: no value depends on them. (A stage for the
    /// labels' records, one hint per label of up to 16 neighbours before
    /// each visit, cost more than it saved: DBLP top5000 M=3 and BA 300k M=3
    /// ran 6% and 4.5% faster without it.)
    #[inline(always)]
    fn prefetch_ahead(&self) {
        let (g, q) = (&self.graph, &self.queue);
        let neighbours = |u: Id| {
            let (b, e) = (g.start[u as usize] as usize, g.start[u as usize + 1] as usize);
            &g.adj[b..e.min(b + PREFETCH_DEGREE)]
        };
        if let Some(u) = q.peek(PREFETCH_FAR) {
            prefetch(&g.start, u as usize);
            prefetch(&self.row, u as usize);
        }
        if let Some(u) = q.peek(PREFETCH_MID) {
            prefetch(&g.adj, g.start[u as usize] as usize);
            prefetch(&self.pool, self.row[u as usize].start);
        }
        if let Some(u) = q.peek(PREFETCH_NEAR) {
            for &w in neighbours(u) {
                prefetch(&self.row, w as usize);
            }
        }
        if let Some(u) = q.peek(PREFETCH_ONE) {
            for &w in neighbours(u) {
                prefetch(&self.pool, self.row[w as usize].start);
            }
        }
    }

    fn run(&mut self, cb: &mut Callbacks) -> Result<()> {
        let weighted = self.graph.weighted;
        let ahead = self.prefetch;
        #[cfg(debug_assertions)]
        assert!(self.invariants());
        while let Some(v) = self.queue.next(cb)? {
            if ahead {
                self.prefetch_ahead();
            }
            if weighted {
                self.visit::<true>(v, false)?;
            } else {
                self.visit::<false>(v, false)?;
            }
        }
        #[cfg(debug_assertions)]
        assert!(self.invariants());
        Ok(())
    }

    /// The slack: the largest gain bound of the last visits, all made in the
    /// final state. Coarse bounds are replaced by full ones, largest first,
    /// until the next coarse bound cannot raise the maximum: the same value
    /// as recording every full bound.
    fn compute_slack(&mut self) -> Result<()> {
        let mut slack = 0.0f64;
        for s in &self.queue.settled {
            if s.bound > slack {
                slack = s.bound;
            }
        }
        // Counted first and reserved once, as C does (no growth by doubling
        // with the whole state alive).
        let count = self.queue.settled.iter().filter(|s| -s.bound > slack).count();
        let mut open: Vec<(f64, Id)> = Vec::new();
        open.try_reserve_exact(count).map_err(|_| Error::nomem())?;
        for (v, s) in self.queue.settled.iter().enumerate() {
            if -s.bound > slack {
                open.push((-s.bound, v as Id));
            }
        }
        // Descending; ties (equal coarse bounds) in vertex order, a total
        // order (vertices are distinct). Which of two tied vertices goes first
        // cannot change the maximum: a full bound never exceeds its coarse one.
        open.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        for &(gain, v) in &open {
            if !(gain > slack) {
                break;
            }
            let full = self.visit_any(v, true)?;
            if full > slack {
                slack = full;
            }
        }
        self.slack = slack;
        Ok(())
    }
}

/// The cover a run ended in: labels 0, 1, ... in order of first
/// appearance, each row sorted. Takes the rows and the pool alone, so the
/// rest of the run state can be released first.
fn result(row: &[RowRef], pool: &[Id], label_count: Id) -> Result<(Cover, i64)> {
    let n = row.len();
    let total: usize = row.iter().map(|r| r.size as usize).sum();
    let mut offset = filled(n + 1, 0i64)?;
    let mut labels = filled(total, 0i64)?;
    let mut relabel = filled(label_count as usize, -1 as Id)?;
    let mut next: Id = 0;
    for v in 0..n {
        let r = row[v];
        let base = offset[v] as usize;
        let out = &mut labels[base..base + r.size as usize];
        for (i, o) in out.iter_mut().enumerate() {
            let c = pool[r.start + i] as usize;
            if relabel[c] < 0 {
                relabel[c] = next;
                next += 1;
            }
            *o = relabel[c] as i64;
        }
        sort_ids(out);
        offset[v + 1] = offset[v] + r.size as i64;
    }
    Ok((Cover { offset, labels }, next as i64))
}

/// Sorts labels ascending: insertion for the short rows of a cover (as
/// hig_sort_ids), the library's sort past them.
fn sort_ids<T: Ord + Copy>(a: &mut [T]) {
    if a.len() > 16 {
        a.sort_unstable();
        return;
    }
    for i in 1..a.len() {
        let x = a[i];
        let mut j = i;
        while j > 0 && a[j - 1] > x {
            a[j] = a[j - 1];
            j -= 1;
        }
        a[j] = x;
    }
}

fn shared_labels(a: &[i64], b: &[i64]) -> i64 {
    let (mut i, mut j, mut shared) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            i += 1;
        } else if a[i] > b[j] {
            j += 1;
        } else {
            shared += 1;
            i += 1;
            j += 1;
        }
    }
    shared
}

/// (sum_edges 2 w |sigma_a & sigma_b| f_a f_b - gamma sum_c S_c^2) / 2W.
fn quality(graph: &Graph, gamma: f64, cover: &Cover, count: i64, inv_sqrt: &[f64]) -> Result<f64> {
    let mut mass = filled(count as usize, 0.0f64)?;
    let (mut q, mut total) = (0.0f64, 0.0f64);
    let weights = graph.edge_weights;
    graph.each_edge(|e, a, b| {
        let (ra, rb) = (cover.row(a as usize), cover.row(b as usize));
        let w = weights.map_or(1.0, |w| w[e]);
        let shared = shared_labels(ra, rb);
        total += w;
        if shared > 0 {
            q += 2.0 * w * shared as f64 * inv_sqrt[ra.len()] * inv_sqrt[rb.len()];
        }
    });
    for v in 0..graph.vertex_count as usize {
        let row = cover.row(v);
        let term = graph.node_weight(v as i64) * inv_sqrt[row.len()];
        for &c in row {
            mass[c as usize] += term;
        }
    }
    for &m in &mass {
        q = fma(-(gamma * m), m, q);
    }
    Ok(q / (2.0 * total))
}

/// The start rows (the caller's or the default): labels compacted, rows
/// sorted and duplicate-free, count limits checked. Returns the rows (one
/// Vec per vertex, internal ids) and the number of labels.
fn start_rows(options: &Options, n: Id, cap: Id, start: Option<&Cover>) -> Result<(Vec<usize>, Vec<Id>, i64)> {
    let nu = n as usize;
    let mut offset = filled(nu + 1, 0i64)?;
    let labels: Vec<i64>;
    match start {
        Some(start) => {
            if start.offset.is_empty() || start.vertex_count() != n as i64 {
                return Err(Error::inval(
                    "Initial membership list length does not equal the number of vertices.",
                ));
            }
            for v in 0..nu {
                let size = start.offset[v + 1] - start.offset[v];
                if size < 1 {
                    return Err(Error::inval(
                        "Initial overlapping membership vectors must be non-empty.",
                    ));
                }
                offset[v + 1] = offset[v] + size;
            }
            let mut l = filled(offset[nu] as usize, 0i64)?;
            for v in 0..nu {
                let row = start.row(v);
                for (i, &x) in row.iter().enumerate() {
                    l[offset[v] as usize + i] = x;
                    if x < 0 {
                        return Err(Error::inval(
                            "Initial overlapping membership indices must be non-negative.",
                        ));
                    }
                }
            }
            labels = l;
        }
        None => {
            // Singletons; under a count limit, K labels round-robin (with more
            // labels than vertices, the extra ones are spread over the rows).
            let n64 = n as i64;
            let target: i64 = if limit_set(options.n_communities) {
                options.n_communities
            } else if limit_set(options.max_total_communities) && options.max_total_communities < n64 {
                options.max_total_communities
            } else {
                n64
            };
            let extra = if target > n64 { target - n64 } else { 0 };
            for v in 0..nu {
                offset[v + 1] = offset[v] + 1 + extra / n64 + ((v as i64) < extra % n64) as i64;
            }
            let mut l = filled(offset[nu] as usize, 0i64)?;
            for v in 0..nu {
                let mut i = offset[v] as usize;
                let v64 = v as i64;
                l[i] = if target < n64 { v64 % target } else { v64 };
                i += 1;
                let mut c = n64 + v64;
                while c < target {
                    l[i] = c;
                    i += 1;
                    c += n64;
                }
            }
            labels = l;
        }
    }
    let mut ids = filled(labels.len(), 0 as Id)?;
    let distinct = compact_labels(&labels, &mut ids)?;
    // Each row sorted and deduplicated in place, packed to the front of ids
    // (a row is written at or before where it was read).
    let mut bounds = filled(nu + 1, 0usize)?;
    let mut kept = 0usize;
    for v in 0..nu {
        let (b, e) = (offset[v] as usize, offset[v + 1] as usize);
        sort_ids(&mut ids[b..e]);
        let mut size = 0usize;
        for i in b..e {
            let x = ids[i];
            if size == 0 || ids[kept + size - 1] != x {
                ids[kept + size] = x;
                size += 1;
            }
        }
        if size > cap as usize {
            return Err(Error::inval("Initial overlapping membership vector exceeds max_memberships."));
        }
        kept += size;
        bounds[v + 1] = kept;
    }
    ids.truncate(kept);
    if limit_set(options.max_total_communities) && distinct > options.max_total_communities {
        return Err(Error::inval("Initial cover exceeds max_total_communities."));
    }
    if limit_set(options.n_communities) && distinct != options.n_communities {
        return Err(Error::inval("Initial cover does not contain exactly n_communities labels."));
    }
    Ok((bounds, ids, distinct))
}

/// Local moving of a cover (`hig_cover` / `hig_cover_traced`). `start`: a
/// non-empty row of non-negative labels per vertex (a repeated label counts
/// once), or None for singletons (under a count limit, a round-robin start
/// that occupies K labels).
pub fn cover(
    graph: &Graph,
    options: &Options,
    start: Option<&Cover>,
    cb: &mut Callbacks,
    mut trace: Option<&mut Trace>,
) -> Result<CoverResult> {
    let mut options = *options;
    check_input(graph, &mut options, true)?;
    let n = graph.vertex_count as Id;
    let nu = n as usize;
    let gamma = options.resolution;
    let cap = options.max_memberships as Id;
    let isolation = options.allow_isolation;
    let exact = limit_set(options.n_communities);
    let max_total = options.max_total_communities;
    let sqrt_cap_up = up((cap as f64).sqrt());

    if let Some(start) = start {
        if start.offset.len() == nu + 1 {
            start.validate()?;
        }
    }
    let (bounds, rows, distinct) = start_rows(&options, n, cap, start)?;
    let adjacency = Adjacency::build(graph)?;

    // Rows into the pool, each at the front of its own capacity.
    let mut row = filled(nu, RowRef::default())?;
    let mut pool_live = 0usize;
    for v in 0..nu {
        let size = (bounds[v + 1] - bounds[v]) as Id;
        let capacity = row_capacity(cap, size);
        row[v] = RowRef { start: pool_live, size, capacity };
        pool_live += capacity as usize;
    }
    let mut pool: Vec<Id> = Vec::new();
    pool.try_reserve_exact(pool_live + pool_live / 2 + 1024).map_err(|_| Error::nomem())?;
    pool.resize(pool_live, 0);
    for v in 0..nu {
        let labels = &rows[bounds[v]..bounds[v + 1]];
        pool[row[v].start..row[v].start + labels.len()].copy_from_slice(labels);
    }
    drop(rows);
    drop(bounds);

    let mut inv_sqrt = filled(cap as usize + 1, 0.0f64)?;
    for k in 1..=cap as usize {
        inv_sqrt[k] = 1.0 / (k as f64).sqrt();
    }
    // Can a term n_u f_u or w f_u fall below DBL_MIN (and lose its relative
    // accuracy)? f_u >= 1 / sqrt(cap) > 1 / (2 sqrt(cap)).
    let smallest = 2.0 * DBL_MIN * 2.0 * sqrt_cap_up;
    let mut tiny_mass = 0.0;
    if adjacency.node_weighted {
        for &w in &adjacency.node_weight {
            if w > 0.0 && w < smallest {
                tiny_mass = n as f64 * ETA;
                break;
            }
        }
    }
    let tiny_support = adjacency.weighted && adjacency.weight.iter().any(|&w| w > 0.0 && w < smallest);

    let mut s = State {
        graph: adjacency,
        gamma,
        cap,
        isolation,
        exact,
        max_total,
        sqrt_cap_up,
        row,
        pool,
        pool_live,
        group: Vec::new(),
        tail: Vec::new(),
        theta: 0.0,
        theta_below: 0.0,
        rebuild: false,
        tiny_mass,
        tiny_support,
        label_count: distinct as Id,
        free_ids: Vec::new(),
        occupied: 0,
        inv_sqrt,
        use_heap: gamma < 0.0 || !isolation || limit_set(max_total) || exact,
        heap_ready: false,
        heap: Heap::new(gamma < 0.0),
        candidate: Vec::new(),
        value: Vec::new(),
        top: Vec::new(),
        chosen: filled(0, 0 as Id)?,
        queue: Queue::new(0, cb)?, // replaced below
        own_support: filled(cap as usize + 1, 0.0)?,
        slack: 0.0,
        trace: None,
        prefetch: false,
    };
    s.chosen.try_reserve_exact(cap as usize + 1).map_err(|_| Error::nomem())?;
    s.reserve_labels(distinct as usize + nu + 16)?;
    for v in 0..nu {
        let r = s.row[v];
        for i in 0..r.size as usize {
            let c = s.pool[r.start + i] as usize;
            s.group[c].members += 1;
        }
    }
    s.occupied = s.label_count;
    s.rebuild_masses(); // the masses, theta and the heap
    s.reserve_candidates(256)?;
    s.queue = Queue::new(n, cb)?;
    // row, start, the queue and records per vertex, the pool's labels, group,
    // tail and heap entries per label, the adjacency (and weights).
    let labels = s.group.len();
    let working_set = nu * 72 + s.pool.len() * 4 + labels * 48
        + s.graph.adj.len() * if s.graph.weighted { 12 } else { 4 };
    s.prefetch = wanted(working_set);
    if let Some(t) = trace.as_deref_mut() {
        for v in 0..nu {
            let r = s.row[v];
            t.record_start(&s.pool[r.start..r.start + r.size as usize]);
        }
    }
    s.trace = trace;
    s.run(cb)?;
    s.compute_slack()?;
    // The run state goes before the result is built: only the rows, the
    // pool and the 1/sqrt(k) table stay.
    let raw_slack = s.slack;
    let (row, pool) = (std::mem::take(&mut s.row), std::mem::take(&mut s.pool));
    let inv_sqrt = std::mem::take(&mut s.inv_sqrt);
    let label_count = s.label_count;
    drop(s);
    let (cover, count) = result(&row, &pool, label_count)?;
    drop(pool);
    drop(row);
    let q = quality(graph, options.resolution, &cover, count, &inv_sqrt)?;
    Ok(CoverResult {
        cover,
        count,
        quality: q,
        raw_slack,
        slack: normalized_slack(raw_slack, graph.total_edge_weight()),
    })
}
