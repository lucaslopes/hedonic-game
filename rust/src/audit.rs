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
    cur_a: f64,
    cur_b: f64,
    norm: f64,
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

    /// The closed stable intervals in [0, 1] of each selected vertex.
    pub fn intervals(
        &self,
        o: &AuditOptions,
        sel: Selection,
        mut interrupt: Interrupt,
    ) -> Result<Vec<Vec<(f64, f64)>>> {
        let cap = self.check_options(o, &sel, &mut interrupt)?;
        let mut w = self.work(cap)?;
        let mut out = Vec::new();
        out.try_reserve_exact(sel.len() as usize).map_err(|_| Error::nomem())?;
        for j in 0..sel.len() {
            poll(&mut interrupt)?;
            self.vertex(sel.get(j), cap, o, &mut w, &mut interrupt)?;
            self.cuts(&mut w, &mut interrupt)?;
            let mut row: Vec<(f64, f64)> = Vec::new();
            let count = w.entries.len() as i64;
            let top = if cap < count { cap } else { count } as usize;
            for cell in 0..w.cuts.len().saturating_sub(1) {
                let (p, q) = (w.cuts[cell], w.cuts[cell + 1]);
                let mid = 0.5 * (p + q);
                let (mut lo, mut hi) = (p, q);
                let (mut sum_a, mut sum_b) = (0.0f64, 0.0f64);
                let u_mid = w.cur_a - mid * w.cur_b;
                let mut stable = true;
                if cell & 255 == 0 {
                    poll(&mut interrupt)?;
                }
                if !(q > p) {
                    continue;
                }
                Self::order(&mut w, mid, &mut interrupt)?;
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
                    let entry = w.entries[w.order[k - 1]];
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
                        stable = false;
                        break;
                    }
                    if lo > hi {
                        stable = false;
                        break;
                    }
                }
                if !stable || lo > hi {
                    continue;
                }
                match row.last_mut() {
                    Some(last) if lo <= last.1 + 1e-15 => {
                        if hi > last.1 {
                            last.1 = hi;
                        }
                    }
                    _ => push(&mut row, (lo, hi))?,
                }
            }
            out.push(row);
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
