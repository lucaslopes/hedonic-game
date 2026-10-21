//! The visiting queue (native/core/shared/queue.c).
//!
//! A FIFO ring of vertices (each at most once), the random order they were
//! first queued in, and the settlement record: `settled[v].at` is the number
//! of moves made when a visit of v last found no improving move (-1 if v has
//! moved since, or was never visited), `settled[v].bound` what that visit
//! recorded. When the FIFO runs dry, the unsettled vertices are queued again
//! in a fresh random order; when none is left, every vertex is a best
//! response to the others and the run stops.

use crate::error::{filled, Error, Result};
use crate::graph::Id;
use crate::rng::{random_below, Rng};

/// Polled every this many visits.
pub const VISITS_PER_INTERRUPTION_CHECK: i64 = 1 << 14;

/// What a call reads from its caller while it runs: the random source and an
/// optional interruption check (true: stop).
pub struct Callbacks<'a> {
    pub rng: &'a mut dyn Rng,
    pub interrupt: Option<&'a mut dyn FnMut() -> bool>,
}

impl<'a> Callbacks<'a> {
    pub fn new(rng: &'a mut dyn Rng) -> Self {
        Callbacks { rng, interrupt: None }
    }

    #[inline]
    pub fn interrupted(&mut self) -> bool {
        match self.interrupt.as_mut() {
            Some(check) => check(),
            None => false,
        }
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Settled {
    pub at: i64,
    pub bound: f64,
}

pub struct Queue {
    n: usize,
    ring: Vec<Id>,
    queued: Vec<u8>,
    order: Vec<Id>,
    pub settled: Vec<Settled>,
    head: usize,
    count: usize,
    pub moves: i64,
    visits: i64,
}

impl Queue {
    /// Every vertex, in a random order (Fisher-Yates).
    pub fn new(n: Id, cb: &mut Callbacks) -> Result<Queue> {
        let n_us = n as usize;
        let mut q = Queue {
            n: n_us,
            ring: filled(n_us, 0)?,
            queued: filled(n_us, 0)?,
            order: (0..n).collect(),
            settled: filled(n_us, Settled { at: -1, bound: 0.0 })?,
            head: 0,
            count: 0,
            moves: 0,
            visits: 0,
        };
        let mut i = n - 1;
        while i > 0 {
            let j = random_below(cb.rng, i as u32 + 1)? as usize;
            q.order.swap(i as usize, j);
            i -= 1;
        }
        for i in 0..n_us {
            let v = q.order[i];
            q.push(v);
        }
        Ok(q)
    }

    #[inline(always)]
    pub fn push(&mut self, u: Id) {
        let u_us = u as usize;
        if self.queued[u_us] == 0 {
            let mut tail = self.head + self.count;
            if tail >= self.n {
                tail -= self.n;
            }
            self.ring[tail] = u;
            self.queued[u_us] = 1;
            self.count += 1;
        }
    }

    /// push(u) without bounds checks.
    ///
    /// # Safety
    /// u is a vertex id, below n.
    #[inline(always)]
    pub unsafe fn push_unchecked(&mut self, u: Id) {
        debug_assert!((u as usize) < self.queued.len());
        // SAFETY: u < n = queued.len(); u is not queued, so fewer than n are
        // and the tail position is below n = ring.len().
        let queued = unsafe { self.queued.get_unchecked_mut(u as usize) };
        if *queued == 0 {
            *queued = 1;
            let mut tail = self.head + self.count;
            if tail >= self.n {
                tail -= self.n;
            }
            debug_assert!(tail < self.ring.len());
            unsafe { *self.ring.get_unchecked_mut(tail) = u };
            self.count += 1;
        }
    }

    #[inline(always)]
    pub fn settle(&mut self, v: Id) {
        self.settled[v as usize].at = self.moves;
    }

    /// settle(v) without a bounds check.
    ///
    /// # Safety
    /// v is a vertex id, below n.
    #[inline(always)]
    pub unsafe fn settle_unchecked(&mut self, v: Id) {
        debug_assert!((v as usize) < self.settled.len());
        unsafe { self.settled.get_unchecked_mut(v as usize).at = self.moves };
    }

    #[inline(always)]
    pub fn settle_bound(&mut self, v: Id, bound: f64) {
        self.settled[v as usize] = Settled { at: self.moves, bound };
    }

    #[inline(always)]
    pub fn moved(&mut self, v: Id) {
        self.moves += 1;
        self.settled[v as usize].at = -1;
    }

    /// The vertex `ahead` places after the next one to visit, if queued
    /// (for prefetching only: the queue may change before its turn).
    #[inline(always)]
    pub fn peek(&self, ahead: usize) -> Option<Id> {
        if ahead < self.count {
            let mut at = self.head + ahead;
            if at >= self.n {
                at -= self.n;
            }
            Some(self.ring[at])
        } else {
            None
        }
    }

    /// The next vertex to visit; None when every vertex is settled.
    #[inline(always)]
    pub fn next(&mut self, cb: &mut Callbacks) -> Result<Option<Id>> {
        if self.count == 0 && !self.refill(cb)? {
            return Ok(None);
        }
        debug_assert!(self.head < self.ring.len());
        // SAFETY: head < n = ring.len() (it starts at 0 and wraps at n), and
        // the ring holds vertex ids, below n = queued.len() (push indexes
        // queued, checked, before it writes one).
        let v = unsafe { *self.ring.get_unchecked(self.head) };
        self.head = if self.head + 1 == self.n { 0 } else { self.head + 1 };
        self.count -= 1;
        debug_assert!((v as usize) < self.queued.len());
        unsafe { *self.queued.get_unchecked_mut(v as usize) = 0 };
        self.visits += 1;
        if self.visits % VISITS_PER_INTERRUPTION_CHECK == 0 {
            Self::check_interrupt(cb)?;
        }
        Ok(Some(v))
    }

    /// A new round: the unsettled vertices, in a fresh random order
    /// (Fisher-Yates over them only); false when every vertex is settled.
    #[cold]
    #[inline(never)]
    fn refill(&mut self, cb: &mut Callbacks) -> Result<bool> {
        self.head = 0;
        for i in 0..self.n {
            let u = self.order[i];
            if self.settled[u as usize].at != self.moves {
                self.push(u);
            }
        }
        if self.count == 0 {
            return Ok(false);
        }
        let mut i = self.count - 1;
        while i > 0 {
            let j = random_below(cb.rng, i as u32 + 1)? as usize;
            self.ring.swap(i, j);
            i -= 1;
        }
        Ok(true)
    }

    #[cold]
    #[inline(never)]
    fn check_interrupt(cb: &mut Callbacks) -> Result<()> {
        if cb.interrupted() {
            return Err(Error::interrupted("Interrupted."));
        }
        Ok(())
    }
}
