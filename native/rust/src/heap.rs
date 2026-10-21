//! The heap of communities by mass (native/core/shared/heap.c).
//!
//! Occupied communities ordered by mass, lightest first (`descending` false)
//! or heaviest first, ties by smaller id: the extreme community a vertex has
//! no edge to is found by walking it best-first and skipping the vertex's
//! candidates. `pos[c]` is c's index in `heap` or -1.

use crate::error::{Error, Result};
use crate::graph::Id;

/// One community (or label): the high half of its mass, its number of
/// members, and its slot in the current visit's accumulator (-1 outside a
/// visit).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Group {
    pub mass: f64,
    pub members: Id,
    pub slot: Id,
}

impl Group {
    pub const EMPTY: Group = Group { mass: 0.0, members: 0, slot: -1 };
}

/// The rest of a mass: its low half and a bound of its accumulated error.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Tail {
    pub lo: f64,
    pub err: f64,
}

#[derive(Default)]
pub struct Heap {
    heap: Vec<Id>,
    pos: Vec<Id>,
    pub descending: bool,
    frontier: Vec<usize>,
    /// The walk's state: None before its first step.
    started: bool,
}

impl Heap {
    pub fn new(descending: bool) -> Self {
        Heap { descending, ..Default::default() }
    }

    #[inline(always)]
    fn before(&self, groups: &[Group], a: Id, b: Id) -> bool {
        let (ma, mb) = (groups[a as usize].mass, groups[b as usize].mass);
        if ma != mb {
            return if self.descending { ma > mb } else { ma < mb };
        }
        a < b
    }

    #[inline(always)]
    fn place(&mut self, i: usize, c: Id) {
        self.heap[i] = c;
        self.pos[c as usize] = i as Id;
    }

    fn sift_up(&mut self, groups: &[Group], mut i: usize) {
        let c = self.heap[i];
        while i > 0 {
            let parent = (i - 1) / 2;
            if !self.before(groups, c, self.heap[parent]) {
                break;
            }
            let p = self.heap[parent];
            self.place(i, p);
            i = parent;
        }
        self.place(i, c);
    }

    fn sift_down(&mut self, groups: &[Group], mut i: usize) {
        let c = self.heap[i];
        let size = self.heap.len();
        loop {
            let mut child = 2 * i + 1;
            if child >= size {
                break;
            }
            if child + 1 < size && self.before(groups, self.heap[child + 1], self.heap[child]) {
                child += 1;
            }
            if !self.before(groups, self.heap[child], c) {
                break;
            }
            let h = self.heap[child];
            self.place(i, h);
            i = child;
        }
        self.place(i, c);
    }

    /// Room for ids below `ids`.
    pub fn reserve(&mut self, ids: usize) -> Result<()> {
        if ids > self.pos.len() {
            let extra = ids - self.pos.len();
            self.pos.try_reserve(extra).map_err(|_| Error::nomem())?;
            self.pos.resize(ids, -1);
            let more = ids.saturating_sub(self.heap.len());
            self.heap.try_reserve(more).map_err(|_| Error::nomem())?;
        }
        Ok(())
    }

    pub fn insert(&mut self, groups: &[Group], c: Id) {
        let i = self.heap.len();
        self.heap.push(c);
        self.pos[c as usize] = i as Id;
        self.sift_up(groups, i);
    }

    pub fn remove(&mut self, groups: &[Group], c: Id) {
        let i = self.pos[c as usize] as usize;
        let last = self.heap.pop().expect("remove from an empty heap");
        self.pos[c as usize] = -1;
        if i < self.heap.len() {
            self.place(i, last);
            self.sift_up(groups, i);
            let p = self.pos[last as usize] as usize;
            self.sift_down(groups, p);
        }
    }

    /// Empties the heap (its ids stay reserved).
    pub fn clear(&mut self) {
        for &c in &self.heap {
            self.pos[c as usize] = -1;
        }
        self.heap.clear();
    }

    pub fn update(&mut self, groups: &[Group], c: Id) {
        let i = self.pos[c as usize] as usize;
        self.sift_up(groups, i);
        let i = self.pos[c as usize] as usize;
        self.sift_down(groups, i);
    }

    // The walk keeps the heap indices it may visit next in a small heap of
    // its own: each step pops the best and offers its two children, so k
    // steps cost O(k log k) whatever the heap's size.

    #[inline(always)]
    fn frontier_before(&self, groups: &[Group], a: usize, b: usize) -> bool {
        self.before(groups, self.heap[a], self.heap[b])
    }

    fn frontier_push(&mut self, groups: &[Group], index: usize) -> Result<()> {
        if self.frontier.len() == self.frontier.capacity() {
            self.frontier.try_reserve(16).map_err(|_| Error::nomem())?;
        }
        let mut i = self.frontier.len();
        self.frontier.push(index);
        while i > 0 && self.frontier_before(groups, index, self.frontier[(i - 1) / 2]) {
            self.frontier[i] = self.frontier[(i - 1) / 2];
            i = (i - 1) / 2;
        }
        self.frontier[i] = index;
        Ok(())
    }

    fn frontier_pop(&mut self, groups: &[Group]) -> usize {
        let top = self.frontier[0];
        let last = self.frontier.pop().unwrap();
        let size = self.frontier.len();
        let mut i = 0;
        loop {
            let mut child = 2 * i + 1;
            if child >= size {
                break;
            }
            if child + 1 < size
                && self.frontier_before(groups, self.frontier[child + 1], self.frontier[child])
            {
                child += 1;
            }
            if !self.frontier_before(groups, self.frontier[child], last) {
                break;
            }
            self.frontier[i] = self.frontier[child];
            i = child;
        }
        if size > 0 {
            self.frontier[i] = last;
        }
        top
    }

    /// Best-first walk: begin, then next until it returns None.
    pub fn walk_begin(&mut self) {
        self.frontier.clear();
        self.started = false; // the root is offered by the first step
    }

    pub fn walk_next(&mut self, groups: &[Group]) -> Result<Option<Id>> {
        let i;
        if !self.started {
            self.started = true;
            if self.heap.is_empty() {
                return Ok(None);
            }
            i = 0;
        } else {
            if self.frontier.is_empty() {
                return Ok(None);
            }
            i = self.frontier_pop(groups);
        }
        let size = self.heap.len();
        if 2 * i + 1 < size {
            self.frontier_push(groups, 2 * i + 1)?;
        }
        if 2 * i + 2 < size {
            self.frontier_push(groups, 2 * i + 2)?;
        }
        Ok(Some(self.heap[i]))
    }
}
