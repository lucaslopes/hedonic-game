//! Fixed-cover stability audit over resolution [0, 1]
//! (native/core/check/audit.c), with the paper's plain-double arithmetic and
//! heuristic tolerance, not the detector's certificate.
//!
//! Every sum has a specified order: no fma, no reassociation, no reciprocal
//! multiplication for utilities, no positive-prefix shortcut. The two
//! current-utility coefficients are CPython 3.12's compensated `sum`, as the
//! paper's reference computes them.

use crate::error::{filled, push, Error, Result};
use crate::graph::{Adjacency, Graph, Id, ID_MAX};

/// The immutable prepared state: owns its adjacency, rows, intensities,
/// masses and lightest-label order; borrows nothing.
pub struct AuditState {
    adj: Adjacency,
    offset: Vec<i64>,
    labels: Vec<Id>,
    lightest: Vec<Id>,
    intensity: Vec<f64>,
    mass: Vec<f64>,
    label_count: i64,
    max_row: i64,
    utility_bound: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct AuditOptions {
    /// 0: the largest row; otherwise at least the largest row.
    pub max_memberships: i64,
    /// Add a fresh, zero-score label.
    pub allow_isolation: bool,
    /// true: relevant labels; false: the reference superset.
    pub prune: bool,
    pub atol: f64,
    pub rtol: f64,
}

impl Default for AuditOptions {
    fn default() -> Self {
        AuditOptions { max_memberships: 0, allow_isolation: false, prune: true, atol: 1e-10, rtol: 1e-9 }
    }
}

/// Which vertices a query visits, in order (repeats allowed).
#[derive(Clone, Copy, Debug)]
pub enum Selection<'a> {
    Range(i64, i64),
    List(&'a [i64]),
}

impl Selection<'_> {
    fn len(&self) -> i64 {
        match *self {
            Selection::Range(begin, end) => end - begin,
            Selection::List(list) => list.len() as i64,
        }
    }

    fn get(&self, j: i64) -> i64 {
        match *self {
            Selection::Range(begin, _) => begin + j,
            Selection::List(list) => list[j as usize],
        }
    }
}

/// Polled during long loops; true stops the call.
pub type Interrupt<'a> = Option<&'a mut dyn FnMut() -> bool>;

fn poll(interrupt: &mut Interrupt) -> Result<()> {
    if let Some(check) = interrupt.as_mut() {
        if check() {
            return Err(Error::interrupted("Stability audit interrupted."));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Default)]
struct Entry {
    a: f64,
    b: f64,
    score: f64,
    own: bool,
}

struct Work {
    support: Vec<f64>,
    seen: Vec<u8>,
    touched: Vec<Id>,
    entries: Vec<Entry>,
    order: Vec<usize>,
    roots: Vec<f64>,
    cuts: Vec<f64>,
    /// Windowed search: sorted a or b, per-prefix rounding margins, the
    /// window's cuts before they are placed.
    values: Vec<f64>,
    margin: Vec<f64>,
    kept: Vec<f64>,
    /// `sort_order`'s keys.
    keys: Vec<u128>,
    cur_a: f64,
    cur_b: f64,
    norm: f64,
}

/// Scores descending, ties by ascending index: the cells' order (`resort`
/// spells it out inline; the unit tests check the keys against it).
#[cfg(test)]
fn before(e: &[Entry], x: usize, y: usize) -> bool {
    e[x].score > e[y].score || (e[x].score == e[y].score && x < y)
}

/// A key that ascends as a finite score descends; -0 and +0 share one.
#[inline(always)]
fn descending(score: f64) -> u64 {
    let bits = if score == 0.0 { 0 } else { score.to_bits() };
    // Ascending order of the doubles as unsigned integers, then reversed.
    let ascending = if bits >> 63 != 0 { !bits } else { bits | 1 << 63 };
    !ascending
}

/// The intervals of the selected vertices, in one buffer: vertex j's are
/// `pairs[offset[j]..offset[j + 1]]` (`row(j)`), as hig_audit_intervals_t.
#[derive(Debug, Default)]
pub struct Intervals {
    pub offset: Vec<usize>,
    pub pairs: Vec<(f64, f64)>,
}

impl Intervals {
    /// The number of vertices.
    pub fn len(&self) -> usize {
        self.offset.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn row(&self, j: usize) -> &[(f64, f64)] {
        &self.pairs[self.offset[j]..self.offset[j + 1]]
    }

    pub fn rows(&self) -> impl Iterator<Item = &[(f64, f64)]> + '_ {
        self.offset.windows(2).map(|w| &self.pairs[w[0]..w[1]])
    }
}

/// The vertex being searched: its intervals are the store's from `begin`.
struct Row<'a> {
    pairs: &'a mut Vec<(f64, f64)>,
    begin: usize,
}

/// The `top` largest values first, in descending total order (the rest
/// after them in no order): the same `top` bits, in the same order, as a
/// full descending sort, since equal values under total_cmp are equal bits.
fn largest_first(values: &mut [f64], top: usize) {
    if top == 0 {
        return;
    }
    if top < values.len() {
        values.select_nth_unstable_by(top - 1, |x, y| y.total_cmp(x));
    }
    values[..top].sort_unstable_by(|x, y| y.total_cmp(x));
}

/// Add [lo, hi] to a vertex's row, merged into the last interval when it
/// starts within 1e-15 of its end (never into another vertex's: `begin`).
fn emit(row: &mut Row, lo: f64, hi: f64) -> Result<()> {
    if row.pairs.len() > row.begin {
        let last = row.pairs.last_mut().unwrap();
        if lo <= last.1 + 1e-15 {
            if hi > last.1 {
                last.1 = hi;
            }
            return Ok(());
        }
    }
    push(row.pairs, (lo, hi))
}

/// A point of the windowed search (a cut or a bisection point).
#[derive(Clone, Copy)]
struct Point {
    x: f64,
    /// The best set's deviation line: g(y) >= alpha - beta * y.
    alpha: f64,
    beta: f64,
    /// Max over prefixes k of utility_k + margin_k, at x.
    peak: f64,
}

struct Window {
    /// T + M: a deviation line above it excludes.
    limit: f64,
    atol: f64,
    rtol: f64,
    band: f64,
    cap: i64,
    top: usize,
}

/// Is every cell of [A, B] a whole-cell piece? (Certification.)
fn certified(w: &Work, win: &Window, a: &Point, b: &Point) -> bool {
    let ua = w.cur_a - a.x * w.cur_b;
    let ub = w.cur_a - b.x * w.cur_b;
    // The tolerance line ta + x * tb.
    let (mut ta, mut tb) = (win.atol, 0.0);
    if ua >= win.band && ub >= win.band {
        ta = win.atol + win.rtol * w.cur_a;
        tb = -win.rtol * w.cur_b;
    } else if ua <= -win.band && ub <= -win.band {
        ta = win.atol - win.rtol * w.cur_a;
        tb = win.rtol * w.cur_b;
    }
    a.peak - ua - (ta + a.x * tb) <= 0.0 && b.peak - ub - (ta + b.x * tb) <= 0.0
}

fn excluded(win: &Window, a: &Point, b: &Point) -> bool {
    (a.alpha - a.beta * a.x > win.limit && a.alpha - a.beta * b.x > win.limit)
        || (b.alpha - b.beta * a.x > win.limit && b.alpha - b.beta * b.x > win.limit)
}

/// Which cuts the window keeps: the nearest below wl, the nearest above
/// wr, the first and last in the core, and everything else in [wl, wr].
/// `floor`, `ceiling` and `inner_*` are the multipliers of the cheap
/// rejections (`window_cuts`).
struct Keeper {
    wl: f64,
    wr: f64,
    core_lo: f64,
    core_hi: f64,
    below: f64,
    above: f64,
    first: f64,
    last: f64,
    floor: f64,
    ceiling: f64,
    inner_lo: f64,
    inner_hi: f64,
}

impl Keeper {
    #[inline]
    fn keep(&mut self, cut: f64, kept: &mut Vec<f64>) -> Result<()> {
        if cut < self.wl {
            if cut > self.below {
                self.below = cut;
                if self.below > 0.0 {
                    self.floor = self.below * (1.0 - 1e-12);
                }
            }
        } else if cut > self.wr {
            if cut < self.above {
                self.above = cut;
                self.ceiling = self.above * (1.0 + 1e-12);
            }
        } else if cut >= self.core_lo && cut <= self.core_hi {
            if cut < self.first {
                self.first = cut;
            }
            if cut > self.last {
                self.last = cut;
            }
            self.inner_lo = self.first * (1.0 + 1e-12);
            self.inner_hi = self.last * (1.0 - 1e-12);
        } else {
            push(kept, cut)?;
        }
        Ok(())
    }
}

impl AuditState {
    /// Nonempty rows of non-negative contiguous labels (a repeated label
    /// counts once), a loopless graph with finite non-negative edge weights,
    /// unit node weights only. `offset` has n + 1 entries.
    pub fn prepare(
        graph: &Graph,
        offset: &[i64],
        labels: &[i64],
        mut interrupt: Interrupt,
    ) -> Result<AuditState> {
        let n = graph.vertex_count;
        let m = graph.edge_count;
        let mut weight_sum = 0.0f64;
        let mut largest: i64 = -1;
        if n < 0 || m < 0 {
            return Err(Error::inval("Invalid audit graph or cover."));
        }
        if n > (ID_MAX - 1) as i64 || m > i64::MAX / 2 {
            return Err(Error::overflow("Audit graph count overflow."));
        }
        if offset.len() as i64 != n + 1 || offset[0] != 0 {
            return Err(Error::inval("Missing or inconsistent audit input arrays."));
        }
        for v in 0..n as usize {
            if v & 4095 == 0 {
                poll(&mut interrupt)?;
            }
            if offset[v + 1] <= offset[v] {
                return Err(Error::inval("Audit rows must be non-empty with increasing offsets."));
            }
            if graph.node_weight(v as i64) != 1.0 {
                return Err(Error::inval("Stability audits support unit node weights only."));
            }
        }
        let total = offset[n as usize];
        if total > labels.len() as i64 {
            return Err(Error::inval("Audit row offset outside labels buffer"));
        }
        for e in 0..m {
            let (a, b) = (graph.end(2 * e), graph.end(2 * e + 1));
            let w = graph.edge_weight(e);
            if e & 4095 == 0 {
                poll(&mut interrupt)?;
            }
            if a < 0 || a >= n || b < 0 || b >= n || a == b {
                return Err(Error::inval("Audit edges must have distinct ends in the vertex range."));
            }
            if !w.is_finite() || w < 0.0 {
                return Err(Error::inval("Audit edge weights must be finite and non-negative."));
            }
            weight_sum += w;
        }
        // Bound every support/prefix sum, including repeated label contributions.
        let total_or_one = if total > 0 { total } else { 1 } as f64;
        if !(weight_sum <= (f64::MAX / 4096.0) / total_or_one) {
            return Err(Error::overflow("Audit payoff sums would overflow."));
        }
        let mut new_offset = filled(n as usize + 1, 0i64)?;
        let mut rows = filled(total as usize, 0 as Id)?;
        let mut intensity = filled(n as usize, 0.0f64)?;
        let mut max_row: i64 = 1;
        let mut used: usize = 0;
        for v in 0..n as usize {
            let begin = used;
            let length = (offset[v + 1] - offset[v]) as usize;
            poll(&mut interrupt)?;
            for j in 0..length {
                let c = labels[offset[v] as usize + j];
                if c < 0 || c >= total {
                    return Err(Error::inval("Audit labels must be contiguous non-negative integers."));
                }
                if c >= ID_MAX as i64 {
                    return Err(Error::overflow("Too many audit labels."));
                }
                rows[begin + j] = c as Id;
                if c > largest {
                    largest = c;
                }
            }
            rows[begin..begin + length].sort_unstable();
            for j in 0..length {
                let c = rows[begin + j];
                if j == 0 || c != rows[used - 1] {
                    rows[used] = c;
                    used += 1;
                }
            }
            new_offset[v + 1] = used as i64;
            if (used - begin) as i64 > max_row {
                max_row = (used - begin) as i64;
            }
            intensity[v] = 1.0 / ((used - begin) as f64).sqrt();
        }
        rows.truncate(used);
        let mut utility_bound = weight_sum * total_or_one;
        if n as f64 * max_row as f64 > utility_bound {
            utility_bound = n as f64 * max_row as f64;
        }
        if utility_bound < 1.0 {
            utility_bound = 1.0;
        }
        let label_count = largest + 1;
        let mut mass = filled(label_count as usize, 0.0f64)?;
        for v in 0..n as usize {
            if v & 4095 == 0 {
                poll(&mut interrupt)?;
            }
            for j in new_offset[v] as usize..new_offset[v + 1] as usize {
                mass[rows[j] as usize] += intensity[v];
            }
        }
        let mut ranked: Vec<(f64, Id)> = Vec::new();
        ranked.try_reserve_exact(label_count as usize).map_err(|_| Error::nomem())?;
        for c in 0..label_count as usize {
            if mass[c] == 0.0 {
                return Err(Error::inval("Audit labels must be contiguous from zero."));
            }
            ranked.push((mass[c], c as Id));
        }
        // (mass, label): a total order (masses are positive and finite).
        ranked.sort_unstable_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        let lightest: Vec<Id> = ranked.iter().map(|r| r.1).collect();
        poll(&mut interrupt)?;
        let adj = Adjacency::build(graph)?;
        poll(&mut interrupt)?;
        Ok(AuditState {
            adj,
            offset: new_offset,
            labels: rows,
            lightest,
            intensity,
            mass,
            label_count,
            max_row,
            utility_bound,
        })
    }

    pub fn vertex_count(&self) -> i64 {
        self.adj.n as i64
    }

    fn check_options(&self, o: &AuditOptions, sel: &Selection, interrupt: &mut Interrupt) -> Result<i64> {
        let cap = if o.max_memberships == 0 { self.max_row } else { o.max_memberships };
        if cap < self.max_row {
            return Err(Error::inval("max_memberships must be at least the largest audit row."));
        }
        if !o.atol.is_finite() || !o.rtol.is_finite() || o.atol < 0.0 || o.rtol < 0.0 {
            return Err(Error::inval("Audit tolerances must be finite and non-negative."));
        }
        // Mass prefix terms are <= n * cap; keep the tolerance coefficients finite.
        if o.atol > f64::MAX / 4096.0 || o.rtol > (f64::MAX / 4096.0) / self.utility_bound {
            return Err(Error::overflow("Audit tolerance coefficients would overflow."));
        }
        match *sel {
            Selection::Range(begin, end) => {
                if begin < 0 || end < begin || end > self.adj.n as i64 {
                    return Err(Error::inval("Invalid audit vertex range."));
                }
            }
            Selection::List(list) => {
                for (j, &v) in list.iter().enumerate() {
                    if j & 4095 == 0 {
                        poll(interrupt)?;
                    }
                    if v < 0 || v >= self.adj.n as i64 {
                        return Err(Error::inval("Audit vertex outside the graph."));
                    }
                }
            }
        }
        poll(interrupt)?;
        Ok(cap)
    }

    fn work(&self, cap: i64) -> Result<Work> {
        let k = self.label_count as usize;
        let top = if cap < self.label_count + 1 { cap } else { self.label_count + 1 } as usize;
        let mut roots = filled(top + 1, 0.0f64)?;
        for (j, r) in roots.iter_mut().enumerate().skip(1) {
            *r = (j as f64).sqrt();
        }
        Ok(Work {
            support: filled(k, 0.0)?,
            seen: filled(k, 0)?,
            touched: crate::error::with_capacity(k)?,
            entries: crate::error::with_capacity(k + 1)?,
            order: crate::error::with_capacity(k + 1)?,
            roots,
            cuts: Vec::new(),
            values: crate::error::with_capacity(k + 1)?,
            margin: filled(top + 1, 0.0)?,
            kept: Vec::new(),
            keys: Vec::new(),
            cur_a: 0.0,
            cur_b: 0.0,
            norm: 0.0,
        })
    }

    fn vertex(&self, v: i64, cap: i64, o: &AuditOptions, w: &mut Work, interrupt: &mut Interrupt) -> Result<()> {
        fn touch(w: &mut Work, c: Id) {
            let cu = c as usize;
            if w.seen[cu] == 0 {
                w.seen[cu] = 1;
                w.support[cu] = 0.0;
                w.touched.push(c);
            }
        }
        let vu = v as usize;
        let (mut correction_a, mut correction_b) = (0.0f64, 0.0f64);
        for &c in &w.touched {
            w.seen[c as usize] = 0;
        }
        w.touched.clear();
        for j in self.offset[vu] as usize..self.offset[vu + 1] as usize {
            let c = self.labels[j];
            touch(w, c);
            w.seen[c as usize] = 2;
        }
        // Loopless adjacency is in edge-id order: for a fixed vertex the two
        // contribution directions of the reference cannot interleave.
        for e in self.adj.start[vu] as usize..self.adj.start[vu + 1] as usize {
            let u = self.adj.adj[e] as usize;
            let weight = if self.adj.weighted { self.adj.weight[e] } else { 1.0 };
            if e & 4095 == 0 {
                poll(interrupt)?;
            }
            if weight == 0.0 {
                continue;
            }
            let term = weight * self.intensity[u];
            for j in self.offset[u] as usize..self.offset[u + 1] as usize {
                let c = self.labels[j];
                touch(w, c);
                w.support[c as usize] += term;
            }
        }
        // seen: own = 2, supported = 1; candidates get bit 4.
        let mut kept: i64 = 0;
        for i in 0..w.touched.len() {
            let c = w.touched[i] as usize;
            if !o.prune || w.seen[c] == 2 || w.support[c] > 0.0 {
                w.seen[c] |= 4;
                kept += 1;
            }
        }
        if o.prune {
            for j in 0..self.label_count as usize {
                let c = self.lightest[j];
                if w.seen[c as usize] & 4 == 0 {
                    touch(w, c);
                    w.seen[c as usize] |= 4;
                    break;
                }
            }
        } else {
            let k = self.label_count;
            let dense = k <= 256 || cap >= (k + 7) / 8;
            let take = if dense || cap >= k - kept - 1 { k } else { cap + kept + 1 };
            for j in 0..take as usize {
                let c = self.lightest[j];
                touch(w, c);
                w.seen[c as usize] |= 4;
            }
        }
        w.touched.sort_unstable();
        w.entries.clear();
        if o.allow_isolation {
            w.entries.push(Entry::default());
        }
        w.cur_a = 0.0;
        w.cur_b = 0.0;
        w.norm = ((self.offset[vu + 1] - self.offset[vu]) as f64).sqrt();
        for i in 0..w.touched.len() {
            let c = w.touched[i] as usize;
            let own = (w.seen[c] & 3) == 2;
            let a = w.support[c];
            let b = self.mass[c] - if own { self.intensity[vu] } else { 0.0 };
            if w.seen[c] & 4 == 0 {
                continue;
            }
            w.entries.push(Entry { a, b, score: 0.0, own });
            if own {
                // Python 3.12 sum(), used ONLY for the two current
                // coefficients; state, cell prefixes and regrets use +=.
                let t = w.cur_a + a;
                correction_a += if w.cur_a.abs() >= a.abs() { (w.cur_a - t) + a } else { (a - t) + w.cur_a };
                w.cur_a = t;
                let t = w.cur_b + b;
                correction_b += if w.cur_b.abs() >= b.abs() { (w.cur_b - t) + b } else { (b - t) + w.cur_b };
                w.cur_b = t;
            }
        }
        w.cur_a = (w.cur_a + correction_a) / w.norm;
        w.cur_b = (w.cur_b + correction_b) / w.norm;
        w.order.clear();
        w.order.extend(0..w.entries.len());
        Ok(())
    }

    /// Breakpoints: 0, 1, every pairwise score crossing in (0, 1), and the
    /// current utility's crossings of +1 and -1; sorted.
    fn cuts(&self, w: &mut Work, interrupt: &mut Interrupt) -> Result<()> {
        w.cuts.clear();
        w.cuts.push(0.0);
        w.cuts.push(1.0);
        let count = w.entries.len();
        for i in 0..count {
            poll(interrupt)?;
            for j in i + 1..count {
                let db = w.entries[i].b - w.entries[j].b;
                if j & 4095 == 0 {
                    poll(interrupt)?;
                }
                if db != 0.0 {
                    let gamma = (w.entries[i].a - w.entries[j].a) / db;
                    if gamma > 0.0 && gamma < 1.0 {
                        push(&mut w.cuts, gamma)?;
                    }
                }
            }
        }
        if w.cur_b != 0.0 {
            for level in [1.0, -1.0] {
                let gamma = (w.cur_a - level) / w.cur_b;
                if gamma > 0.0 && gamma < 1.0 {
                    push(&mut w.cuts, gamma)?;
                }
            }
        }
        poll(interrupt)?;
        // Values in [0, 1], no NaN and no -0: any sort gives the same array.
        w.cuts.sort_unstable_by(f64::total_cmp);
        poll(interrupt)
    }

    /// Stable insertion sort of the entries by score (descending, ties by
    /// index), carried from one cell to the next.
    fn order(w: &mut Work, gamma: f64, interrupt: &mut Interrupt) -> Result<()> {
        for entry in w.entries.iter_mut() {
            entry.score = entry.a - gamma * entry.b;
        }
        let mut moves: u64 = 0;
        for i in 1..w.order.len() {
            let x = w.order[i];
            if i & 4095 == 0 {
                poll(interrupt)?;
            }
            let sx = w.entries[x].score;
            let mut j = i;
            while j > 0 {
                let y = w.order[j - 1];
                let sy = w.entries[y].score;
                if !(sy < sx || (sy == sx && y > x)) {
                    break;
                }
                w.order[j] = y;
                j -= 1;
                moves += 1;
                if moves & 4095 == 0 {
                    poll(interrupt)?;
                }
            }
            w.order[j] = x;
        }
        Ok(())
    }

    /// The same order (scores descending, ties by ascending index) from
    /// whatever `w.order` holds: insertion sort while it stays cheap (a
    /// nearby point or cell), else a full sort. Scores are finite and ties
    /// go by index, so the order is unique and both give it.
    fn resort(w: &mut Work, gamma: f64, interrupt: &mut Interrupt) -> Result<()> {
        let count = w.entries.len();
        let mut budget: u64 = 64;
        let mut n = count;
        while n > 1 {
            budget += count as u64;
            n /= 2;
        }
        for entry in w.entries.iter_mut() {
            entry.score = entry.a - gamma * entry.b;
        }
        debug_assert!(Self::permutation(w));
        let (entries, order) = (&w.entries[..], &mut w.order[..count]);
        let mut moves: u64 = 0;
        for i in 1..count {
            let x = order[i];
            // SAFETY: `order` is a permutation of 0..count (`vertex` and
            // `sort_order` set it, and this sort and `order` only permute it;
            // debug-asserted above), so every entry index is in bounds, and
            // j - 1 < i < count.
            let sx = unsafe { entries.get_unchecked(x).score };
            let mut j = i;
            while j > 0 {
                let y = unsafe { *order.get_unchecked(j - 1) };
                let sy = unsafe { entries.get_unchecked(y).score };
                // before(x, y): scores descending, ties by ascending index.
                if !(sx > sy || (sx == sy && x < y)) {
                    break;
                }
                unsafe { *order.get_unchecked_mut(j) = y };
                j -= 1;
                moves += 1;
                if moves > budget {
                    order[j] = x;
                    return Self::sort_order(w, interrupt);
                }
            }
            order[j] = x;
            if i & 4095 == 0 {
                poll(interrupt)?;
            }
        }
        Ok(())
    }

    /// Is `w.order` a permutation of the entries' indices? (Debug checks.)
    fn permutation(w: &Work) -> bool {
        let mut seen = vec![false; w.entries.len()];
        w.order.len() == w.entries.len()
            && w.order.iter().all(|&x| x < seen.len() && !std::mem::replace(&mut seen[x], true))
    }

    /// The order `resort` gives, from scratch (scores already set).
    fn sort_order(w: &mut Work, interrupt: &mut Interrupt) -> Result<()> {
        poll(interrupt)?;
        // Integer keys in `before`'s order: the score's bits mapped to an
        // order that descends with the score (-0 as +0, so equal scores tie;
        // scores are finite), then the index ascending. Sorting them sorts
        // by `before` with no indirection in the comparisons.
        w.keys.clear();
        w.keys.try_reserve(w.entries.len()).map_err(|_| Error::nomem())?;
        w.keys.extend(w.entries.iter().enumerate().map(|(i, e)| ((descending(e.score) as u128) << 64) | i as u128));
        w.keys.sort_unstable();
        w.order.clear();
        w.order.extend(w.keys.iter().map(|&key| key as u64 as usize));
        poll(interrupt)
    }

    /// One cell [p, q] between consecutive cuts: the reference's prefix
    /// inequalities in the score order at its midpoint (`w.order` must hold
    /// it), and its stable piece. Both searches run exactly this.
    fn cell(w: &Work, o: &AuditOptions, cap: i64, p: f64, q: f64, row: &mut Row) -> Result<()> {
        let mid = 0.5 * (p + q);
        let (mut lo, mut hi) = (p, q);
        let (mut sum_a, mut sum_b) = (0.0f64, 0.0f64);
        let u_mid = w.cur_a - mid * w.cur_b;
        let count = w.entries.len() as i64;
        let top = if cap < count { cap } else { count } as usize;
        let (tau_a, tau_b);
        if u_mid.abs() <= 1.0 {
            tau_a = o.atol + o.rtol;
            tau_b = 0.0;
        } else {
            let sign = if u_mid > 0.0 { 1.0 } else { -1.0 };
            tau_a = o.atol + o.rtol * sign * w.cur_a;
            tau_b = -o.rtol * sign * w.cur_b;
        }
        for k in 1..=top {
            let entry = &w.entries[w.order[k - 1]];
            sum_a += entry.a;
            sum_b += entry.b;
            let c0 = sum_a / w.roots[k] - w.cur_a - tau_a;
            let c1 = -sum_b / w.roots[k] + w.cur_b - tau_b;
            if c1 > 0.0 {
                let x = -c0 / c1;
                if x < hi {
                    hi = x;
                }
            } else if c1 < 0.0 {
                let x = -c0 / c1;
                if x > lo {
                    lo = x;
                }
            } else if c0 > 0.0 {
                return Ok(());
            }
            if lo > hi {
                return Ok(());
            }
        }
        if lo > hi {
            return Ok(());
        }
        emit(row, lo, hi)
    }

    /// Exhaustive search: every cell of every pairwise crossing.
    fn vertex_exhaustive(
        &self,
        w: &mut Work,
        o: &AuditOptions,
        cap: i64,
        row: &mut Row,
        interrupt: &mut Interrupt,
    ) -> Result<()> {
        self.cuts(w, interrupt)?;
        for cell in 0..w.cuts.len().saturating_sub(1) {
            let (p, q) = (w.cuts[cell], w.cuts[cell + 1]);
            if cell & 255 == 0 {
                poll(interrupt)?;
            }
            if !(q > p) {
                continue;
            }
            Self::order(w, 0.5 * (p + q), interrupt)?;
            Self::cell(w, o, cap, p, q, row)?;
        }
        Ok(())
    }

    // ---- Windowed search ------------------------------------------------
    //
    // Exact arithmetic first. Every candidate set S of size k has an affine
    // utility U_S(x) = (a_S - x b_S) / sqrt(k); the current utility u(x) is
    // affine. The best response best(x) = max_S U_S(x) is convex, and so is
    // best_k(x), the best over sets of size k. Hence:
    //
    //  - Exclusion. For any S, g(x) = best(x) - u(x) >= U_S(x) - u(x), a
    //    line. The reference's tolerance never exceeds T = atol + rtol *
    //    max(1, |u(0)|, |u(1)|). A cell whose stable piece contains x needs
    //    g(x) <= T up to the rounding of its prefix sums and of its midpoint
    //    order, and M (below) is at least four times every such bound. So x
    //    where some line exceeds T + M lies in no piece. Newton steps along
    //    the best set's line at the current endpoint shrink [0, 1] to a
    //    window [wl, wr] outside which every x is excluded; most
    //    never-stable vertices end there, with no cut.
    //
    //  - Certification. Inside a range of cells [A, B] whose cells all take
    //    one tolerance line tol(x) (the reference picks it from u at the
    //    midpoint; atol alone is below every choice), best_k(x) - u(x) -
    //    tol(x) is convex. If it is at most -margin_k at A and at B, for
    //    every k, it is so on all of [A, B], and every prefix inequality of
    //    every cell there, whatever its midpoint order, holds with room for
    //    its rounding: no inequality moves lo or hi off the cell, and each
    //    cell's piece is the whole cell. The pieces then merge into [A, B],
    //    which is emitted in one step.
    //
    // Everything else, the cells near the ends of the stable set, runs the
    // reference's cell code (`cell`) on the reference's own cut values: the
    // cuts in the window, plus the nearest cut on either side of it. The
    // cells are those of the exhaustive search, in its order; the skipped
    // ones emit nothing or a whole-cell piece, so the output is the same,
    // bit for bit.

    /// Best-prefix line and certification peak at x (sorts `w.order`).
    fn point(w: &mut Work, top: usize, x: f64, interrupt: &mut Interrupt) -> Result<Point> {
        Self::resort(w, x, interrupt)?;
        let (mut sum_a, mut sum_b) = (0.0f64, 0.0f64);
        let (mut best, mut peak) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        let (mut alpha, mut beta) = (0.0f64, 0.0f64);
        let (entries, order) = (&w.entries[..], &w.order[..top]);
        let (roots, margin) = (&w.roots[..=top], &w.margin[..=top]);
        for k in 1..=top {
            debug_assert!(order[k - 1] < entries.len());
            // SAFETY: `order` is a permutation of the entries' indices (see `resort`).
            let entry = unsafe { entries.get_unchecked(order[k - 1]) };
            sum_a += entry.a;
            sum_b += entry.b;
            let utility = (sum_a - x * sum_b) / roots[k];
            if utility > best {
                best = utility;
                alpha = sum_a / roots[k];
                beta = sum_b / roots[k];
            }
            if utility + margin[k] > peak {
                peak = utility + margin[k];
            }
        }
        Ok(Point { x, alpha: alpha - w.cur_a, beta: beta - w.cur_b, peak })
    }

    /// Margins: margin_k >= 4x the rounding of a prefix-k inequality (sums
    /// of k non-negative terms of at most the top-k a and b, sqrt, divisions,
    /// current and tolerance terms), and of this file's own evaluation of it.
    fn margins(w: &mut Work, o: &AuditOptions, win: &mut Window) {
        let (cur_a, cur_b) = (w.cur_a.abs(), w.cur_b.abs());
        let fixed = cur_a + cur_b + o.atol + o.rtol * (cur_a + cur_b);
        let (mut top_a, mut top_b, mut all_a, mut all_b) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        w.values.clear();
        for entry in &w.entries {
            w.values.push(entry.a);
            all_a += entry.a;
        }
        largest_first(&mut w.values, win.top);
        for k in 1..=win.top {
            top_a += w.values[k - 1];
            w.margin[k] = top_a;
        }
        w.values.clear();
        for entry in &w.entries {
            w.values.push(entry.b);
            all_b += entry.b;
        }
        largest_first(&mut w.values, win.top);
        for k in 1..=win.top {
            top_b += w.values[k - 1];
            w.margin[k] = 8.0 * (k + 8) as f64 * f64::EPSILON * ((w.margin[k] + top_b) / w.roots[k] + fixed);
        }
        let shifted = (w.cur_a - w.cur_b).abs();
        let mut t = if cur_a > shifted { cur_a } else { shifted };
        t = o.atol + o.rtol * if t > 1.0 { t } else { 1.0 };
        win.limit = t + 8.0 * (w.entries.len() + 8) as f64 * f64::EPSILON * (all_a + all_b + fixed + t);
        // u at a cell's midpoint is within 4 eps (|cur_a| + |cur_b|) of exact.
        win.band = 1.0 + 1e-6 + 16.0 * f64::EPSILON * (cur_a + cur_b);
    }

    /// Newton steps from both ends of [0, 1] along best-set lines; None: no
    /// point of [0, 1] can be stable.
    fn window(w: &mut Work, win: &Window, interrupt: &mut Interrupt) -> Result<Option<(f64, f64)>> {
        let (mut wl, mut wr) = (0.0f64, 1.0f64);
        for side in 0..2 {
            for _ in 0..64 {
                let x = if side == 0 { wl } else { wr };
                let pt = Self::point(w, win.top, x, interrupt)?;
                if pt.alpha - pt.beta * x <= win.limit {
                    break;
                }
                if if side == 0 { !(pt.beta > 0.0) } else { !(pt.beta < 0.0) } {
                    return Ok(None);
                }
                let next = (pt.alpha - win.limit) / pt.beta;
                if side == 0 {
                    if !(next > wl) {
                        break;
                    }
                    wl = next;
                } else {
                    if !(next < wr) {
                        break;
                    }
                    wr = next;
                }
                if wl > wr {
                    return Ok(None);
                }
            }
        }
        Ok(Some((wl, wr)))
    }

    /// Every cut of the exhaustive search in [wl, wr], sorted, with the
    /// nearest cut below wl in front and the nearest above wr at the end,
    /// into `w.cuts`. With a core [A, B] (core_lo <= core_hi), cuts inside
    /// it are not kept: only the first and the last, at the returned index
    /// and the next (None: no cut in the core).
    fn window_cuts(
        w: &mut Work,
        wl: f64,
        wr: f64,
        core_lo: f64,
        core_hi: f64,
        interrupt: &mut Interrupt,
    ) -> Result<Option<usize>> {
        let mut k = Keeper {
            wl,
            wr,
            core_lo,
            core_hi,
            below: f64::NEG_INFINITY,
            above: f64::INFINITY,
            first: f64::INFINITY,
            last: f64::NEG_INFINITY,
            floor: 0.0,
            ceiling: f64::INFINITY,
            inner_lo: f64::INFINITY,
            inner_hi: f64::NEG_INFINITY,
        };
        w.kept.clear();
        k.keep(0.0, &mut w.kept)?;
        k.keep(1.0, &mut w.kept)?;
        if w.cur_b != 0.0 {
            for level in [1.0, -1.0] {
                let gamma = (w.cur_a - level) / w.cur_b;
                if gamma > 0.0 && gamma < 1.0 {
                    k.keep(gamma, &mut w.kept)?;
                }
            }
        }
        let (entries, kept) = (&w.entries, &mut w.kept);
        for (i, entry) in entries.iter().enumerate() {
            let (ai, bi) = (entry.a, entry.b);
            poll(interrupt)?;
            for (j, other) in entries.iter().enumerate().skip(i + 1) {
                let mut da = ai - other.a;
                let mut db = bi - other.b;
                if j & 4095 == 0 {
                    poll(interrupt)?;
                }
                // The exhaustive search's cut is da / db; negating both
                // leaves it unchanged. It lies in (0, 1) only if 0 < da < db
                // (db > 0). A product test that holds with room for its
                // rounding puts it at or below `below`, at or above `above`,
                // or in [first, last]: a repeat or a cut that changes nothing
                // kept here.
                if db < 0.0 {
                    da = -da;
                    db = -db;
                }
                if !(da > 0.0)
                    || !(da < db)
                    || da < k.floor * db
                    || da > k.ceiling * db
                    || (da > k.inner_lo * db && da < k.inner_hi * db)
                {
                    continue;
                }
                let gamma = da / db;
                if gamma > 0.0 && gamma < 1.0 {
                    k.keep(gamma, kept)?;
                }
            }
        }
        poll(interrupt)?;
        // Values in [0, 1], no NaN and no -0: any sort gives the same array,
        // and their bits ascend with them (an integer sort).
        w.kept.sort_unstable_by_key(|cut| cut.to_bits());
        poll(interrupt)?;
        w.cuts.clear();
        w.cuts.try_reserve(w.kept.len() + 4).map_err(|_| Error::nomem())?;
        if k.below > f64::NEG_INFINITY {
            w.cuts.push(k.below);
        }
        let mut core = None;
        if k.first <= k.last {
            // The core's first and last cut, between the kept ones below and above it.
            let split = w.kept.iter().position(|&cut| !(cut < core_lo)).unwrap_or(w.kept.len());
            w.cuts.extend_from_slice(&w.kept[..split]);
            core = Some(w.cuts.len());
            w.cuts.push(k.first);
            w.cuts.push(k.last);
            w.cuts.extend_from_slice(&w.kept[split..]);
        } else {
            w.cuts.extend_from_slice(&w.kept);
        }
        if k.above < f64::INFINITY {
            w.cuts.push(k.above);
        }
        Ok(core)
    }

    /// Cells first .. last - 1 one by one, as the exhaustive search runs them.
    fn cells(
        w: &mut Work,
        o: &AuditOptions,
        win: &Window,
        first: usize,
        last: usize,
        row: &mut Row,
        interrupt: &mut Interrupt,
    ) -> Result<()> {
        for cell in first..last {
            let (p, q) = (w.cuts[cell], w.cuts[cell + 1]);
            if !(q > p) {
                continue;
            }
            Self::resort(w, 0.5 * (p + q), interrupt)?;
            Self::cell(w, o, win.cap, p, q, row)?;
        }
        Ok(())
    }

    /// Cells first .. last - 1 (cuts first .. last), left to right.
    #[allow(clippy::too_many_arguments)]
    fn range(
        w: &mut Work,
        o: &AuditOptions,
        win: &Window,
        first: usize,
        last: usize,
        a: &Point,
        b: &Point,
        row: &mut Row,
        interrupt: &mut Interrupt,
    ) -> Result<()> {
        if last - first <= 8 {
            return Self::cells(w, o, win, first, last, row, interrupt);
        }
        poll(interrupt)?;
        if excluded(win, a, b) {
            return Ok(());
        }
        if certified(w, win, a, b) {
            if w.cuts[last] > w.cuts[first] {
                emit(row, w.cuts[first], w.cuts[last])?;
            }
            return Ok(());
        }
        let half = first + (last - first) / 2;
        let middle = Self::point(w, win.top, w.cuts[half], interrupt)?;
        Self::range(w, o, win, first, half, a, &middle, row, interrupt)?;
        Self::range(w, o, win, half, last, &middle, b, row, interrupt)
    }

    /// A certified core [A, B] of the window, for vertices with many cuts: a
    /// certified midpoint, then bisection towards both ends of the window.
    /// (inf, -inf): none.
    fn core(w: &mut Work, win: &Window, wl: f64, wr: f64, interrupt: &mut Interrupt) -> Result<(f64, f64)> {
        let none = (f64::INFINITY, f64::NEG_INFINITY);
        let centre = Self::point(w, win.top, 0.5 * (wl + wr), interrupt)?;
        if !certified(w, win, &centre, &centre) {
            return Ok(none);
        }
        let count = w.entries.len() as f64;
        let mut ends = [centre; 2];
        for side in 0..2 {
            let mut bad = if side == 0 { wl } else { wr };
            for _ in 0..48 {
                let x = 0.5 * (bad + ends[side].x);
                let gap = (ends[side].x - bad).abs();
                // Narrow enough when the band left holds few cuts (at most
                // about count^2 / 2 cross [0, 1]), or at the resolution of
                // the window.
                if !(gap > 1e-9 * (wr - wl)) || gap * count * count <= 256.0 || x == bad || x == ends[side].x {
                    break;
                }
                let probe = Self::point(w, win.top, x, interrupt)?;
                let ok = if side == 0 {
                    certified(w, win, &probe, &centre)
                } else {
                    certified(w, win, &centre, &probe)
                };
                if ok {
                    ends[side] = probe;
                } else {
                    bad = x;
                }
            }
        }
        if certified(w, win, &ends[0], &ends[1]) {
            Ok((ends[0].x, ends[1].x))
        } else {
            Ok(none)
        }
    }

    fn vertex_windowed(
        w: &mut Work,
        o: &AuditOptions,
        cap: i64,
        row: &mut Row,
        interrupt: &mut Interrupt,
    ) -> Result<()> {
        let count = w.entries.len() as i64;
        let mut win = Window {
            limit: 0.0,
            atol: o.atol,
            rtol: o.rtol,
            band: 0.0,
            cap,
            top: if cap < count { cap } else { count } as usize,
        };
        Self::margins(w, o, &mut win);
        let Some((wl, wr)) = Self::window(w, &win, interrupt)? else {
            return Ok(());
        };
        let (core_lo, core_hi) =
            if count >= 128 { Self::core(w, &win, wl, wr, interrupt)? } else { (f64::INFINITY, f64::NEG_INFINITY) };
        let core = Self::window_cuts(w, wl, wr, core_lo, core_hi, interrupt)?;
        let cuts = w.cuts.len();
        if cuts < 2 {
            return Ok(());
        }
        let a = Self::point(w, win.top, w.cuts[0], interrupt)?;
        let b = Self::point(w, win.top, w.cuts[cuts - 1], interrupt)?;
        let Some(core) = core else {
            return Self::range(w, o, &win, 0, cuts - 1, &a, &b, row, interrupt);
        };
        // Left of the core, the core as one piece (its cells are certified), right of it.
        let c = Self::point(w, win.top, w.cuts[core], interrupt)?;
        let d = Self::point(w, win.top, w.cuts[core + 1], interrupt)?;
        Self::range(w, o, &win, 0, core, &a, &c, row, interrupt)?;
        if w.cuts[core + 1] > w.cuts[core] {
            emit(row, w.cuts[core], w.cuts[core + 1])?;
        }
        Self::range(w, o, &win, core + 1, cuts - 1, &d, &b, row, interrupt)
    }

    /// The closed stable intervals in [0, 1] of each selected vertex.
    pub fn intervals(&self, o: &AuditOptions, sel: Selection, interrupt: Interrupt) -> Result<Intervals> {
        self.search(o, sel, interrupt, false)
    }

    /// The same intervals by the exhaustive search (every cell of every
    /// pairwise crossing): the reference the windowed search is tested on.
    pub fn intervals_exhaustive(
        &self,
        o: &AuditOptions,
        sel: Selection,
        interrupt: Interrupt,
    ) -> Result<Intervals> {
        self.search(o, sel, interrupt, true)
    }

    fn search(
        &self,
        o: &AuditOptions,
        sel: Selection,
        mut interrupt: Interrupt,
        exhaustive: bool,
    ) -> Result<Intervals> {
        let cap = self.check_options(o, &sel, &mut interrupt)?;
        let mut w = self.work(cap)?;
        let mut out = Intervals::default();
        out.offset.try_reserve_exact(sel.len() as usize + 1).map_err(|_| Error::nomem())?;
        out.offset.push(0);
        for j in 0..sel.len() {
            poll(&mut interrupt)?;
            self.vertex(sel.get(j), cap, o, &mut w, &mut interrupt)?;
            let mut row = Row { begin: out.pairs.len(), pairs: &mut out.pairs };
            if exhaustive {
                self.vertex_exhaustive(&mut w, o, cap, &mut row, &mut interrupt)?;
            } else {
                Self::vertex_windowed(&mut w, o, cap, &mut row, &mut interrupt)?;
            }
            out.offset.push(out.pairs.len());
        }
        Ok(out)
    }

    /// Grid regrets and tolerances, row-major (selected vertex x resolution).
    /// Regrets at most 1e-15 are clamped to zero.
    pub fn regrets(
        &self,
        o: &AuditOptions,
        sel: Selection,
        gammas: &[f64],
        regret: &mut [f64],
        tolerance: &mut [f64],
        mut interrupt: Interrupt,
    ) -> Result<()> {
        let cap = self.check_options(o, &sel, &mut interrupt)?;
        let ng = gammas.len();
        let nv = sel.len() as usize;
        if ng > 0 && nv.checked_mul(ng).is_none() {
            return Err(Error::overflow("Audit regret output size overflow."));
        }
        if regret.len() != nv * ng || tolerance.len() != nv * ng {
            return Err(Error::inval("Missing audit regret output arrays."));
        }
        for &g in gammas {
            if !g.is_finite() || !(0.0..=1.0).contains(&g) {
                return Err(Error::inval("Audit resolutions must be finite and in [0, 1]."));
            }
        }
        if ng == 0 {
            return Ok(());
        }
        let mut w = self.work(cap)?;
        for j in 0..nv {
            poll(&mut interrupt)?;
            self.vertex(sel.get(j as i64), cap, o, &mut w, &mut interrupt)?;
            let count = w.entries.len() as i64;
            let top = if cap < count { cap } else { count } as usize;
            for (g, &gamma) in gammas.iter().enumerate() {
                let out = j * ng + g;
                poll(&mut interrupt)?;
                Self::order(&mut w, gamma, &mut interrupt)?;
                let mut current = 0.0f64;
                for entry in &w.entries {
                    if entry.own {
                        current += entry.score;
                    }
                }
                current /= w.norm;
                let (mut prefix, mut best) = (0.0f64, f64::NEG_INFINITY);
                for k in 1..=top {
                    prefix += w.entries[w.order[k - 1]].score;
                    let utility = prefix / w.roots[k];
                    if utility > best {
                        best = utility;
                    }
                }
                let diff = best - current;
                regret[out] = if diff > 1e-15 { diff } else { 0.0 };
                tolerance[out] = o.atol + o.rtol * if current.abs() > 1.0 { current.abs() } else { 1.0 };
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `descending` orders finite doubles as `before` does: a larger score
    /// first, equal scores (-0 and +0 included) tied.
    #[test]
    fn descending_keys_order_scores() {
        let mut state = 0x2545f4914f6cdd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let special =
            [0.0, -0.0, f64::MIN_POSITIVE, -f64::MIN_POSITIVE, 5e-324, -5e-324, f64::MAX, -f64::MAX, 1.0, -1.0];
        let mut values: Vec<f64> = special.to_vec();
        while values.len() < 4000 {
            let bits = next();
            let x = match bits % 3 {
                0 => f64::from_bits(bits >> 2),
                1 => (bits % 7) as f64 - 3.0,
                _ => -f64::from_bits(bits >> 3),
            };
            if x.is_finite() {
                values.push(x);
            }
        }
        for &x in &values {
            for _ in 0..8 {
                let y = values[(next() % values.len() as u64) as usize];
                let (kx, ky) = (descending(x), descending(y));
                assert_eq!(kx < ky, x > y, "{x} {y}");
                assert_eq!(kx == ky, x == y, "{x} {y}");
            }
        }
        // And the full sort agrees with the insertion sort on ties.
        let entries: Vec<Entry> = values.iter().map(|&a| Entry { a, b: 0.0, score: a, own: false }).collect();
        let mut keys: Vec<u128> =
            entries.iter().enumerate().map(|(i, e)| ((descending(e.score) as u128) << 64) | i as u128).collect();
        keys.sort_unstable();
        let order: Vec<usize> = keys.iter().map(|&k| k as u64 as usize).collect();
        for pair in order.windows(2) {
            assert!(before(&entries, pair[0], pair[1]));
        }
    }
}
