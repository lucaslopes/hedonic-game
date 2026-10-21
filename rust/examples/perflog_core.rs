//! The core alone on a perflog input (rust/bench/perflog.py writes it): the
//! input check, the adjacency build and the whole call timed apart, and the
//! core's own heap (live peak, bytes and count of allocations) counted by a
//! global allocator. The C counterpart is rust/bench/perflog_core.c; both
//! print one JSON line. Benchmarks only.
//!
//!     cargo run --release --example perflog_core -- INPUT
use hedonic_core_rs::audit::{AuditOptions, AuditState, Selection};
use hedonic_core_rs::cover::{cover, Cover};
use hedonic_core_rs::graph::{check_input, Adjacency, Edges, Graph, Options};
use hedonic_core_rs::partition::partition;
use hedonic_core_rs::queue::Callbacks;
use hedonic_core_rs::rng::Mt19937;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static TOTAL: AtomicUsize = AtomicUsize::new(0);
static COUNT: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    let live = LIVE.fetch_add(by, Relaxed) + by;
    PEAK.fetch_max(live, Relaxed);
    TOTAL.fetch_add(by, Relaxed);
    COUNT.fetch_add(1, Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        grew(l.size());
        System.alloc(l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        grew(l.size());
        System.alloc_zeroed(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, size: usize) -> *mut u8 {
        LIVE.fetch_sub(l.size(), Relaxed);
        grew(size);
        System.realloc(p, l, size)
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Counts from here on: (live peak above the live bytes now, bytes, count).
struct Window(usize, usize, usize);

impl Window {
    fn open() -> Window {
        let live = LIVE.load(Relaxed);
        PEAK.store(live, Relaxed);
        Window(live, TOTAL.load(Relaxed), COUNT.load(Relaxed))
    }
    fn json(&self) -> String {
        format!(
            "{{\"peak_live_bytes\": {}, \"total_bytes\": {}, \"count\": {}}}",
            PEAK.load(Relaxed) - self.0,
            TOTAL.load(Relaxed) - self.1,
            COUNT.load(Relaxed) - self.2
        )
    }
}

/// Instructions retired by this process so far (proc_pid_rusage, macOS; 0 elsewhere).
#[cfg(target_os = "macos")]
fn instructions() -> u64 {
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
        fn getpid() -> i32;
    }
    let mut buf = [0u64; 64];
    // SAFETY: RUSAGE_INFO_V4 fills fewer than 64 words.
    if unsafe { proc_pid_rusage(getpid(), 4, buf.as_mut_ptr()) } != 0 {
        return 0;
    }
    buf[2 + 29]
}

#[cfg(not(target_os = "macos"))]
fn instructions() -> u64 {
    0
}

struct Reader(Vec<u8>, usize);

impl Reader {
    fn i64s(&mut self, k: usize) -> Vec<i64> {
        let out = self.0[self.1..self.1 + 8 * k].chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        self.1 += 8 * k;
        out
    }
    fn f64s(&mut self, k: usize) -> Vec<f64> {
        self.i64s(k).into_iter().map(|x| f64::from_bits(x as u64)).collect()
    }
}

fn fnv(values: impl Iterator<Item = u64>) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for v in values {
        for b in v.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn list(xs: &[f64]) -> String {
    format!("[{}]", xs.iter().map(|x| format!("{x:.9}")).collect::<Vec<_>>().join(", "))
}

fn main() {
    let path = std::env::args().nth(1).expect("perflog_core INPUT");
    let mut r = Reader(std::fs::read(path).unwrap(), 8);
    assert_eq!(&r.0[..8], b"HPLOG001");
    let h = r.i64s(16);
    let (mode, n, m, cap, has_ew, has_nw, start_kind) = (h[0], h[1], h[2] as usize, h[3], h[4], h[5], h[6]);
    let (isolation, max_total, n_communities, reps, prune, start_len) = (h[7] != 0, h[8], h[9], h[10], h[11] != 0, h[12]);
    let f = r.f64s(4);
    let (gamma, atol, rtol) = (f[0], f[1], f[2]);
    let words = r.i64s(625);
    let mut mt0 = Mt19937 { state: [0; 624], index: words[624] as usize };
    for i in 0..624 {
        mt0.state[i] = words[i] as u32;
    }
    let edges = r.i64s(2 * m);
    let ew = if has_ew != 0 { Some(r.f64s(m)) } else { None };
    let nw = if has_nw != 0 { Some(r.f64s(n as usize)) } else { None };
    let (mut flat, mut rows) = (None, None);
    if start_kind == 1 {
        flat = Some(r.i64s(n as usize));
    } else if start_kind == 2 {
        let offset = r.i64s(n as usize + 1);
        let labels = r.i64s(start_len as usize);
        rows = Some(Cover { offset, labels });
    }
    let g = Graph::new(n, Edges::Flat(&edges), ew.as_deref(), nw.as_deref()).unwrap();
    let options = Options { resolution: gamma, max_memberships: cap, max_total_communities: max_total, n_communities, allow_isolation: isolation };

    let (mut check_s, mut adjacency_s, mut call_s, mut extra_s) = (vec![], vec![], vec![], vec![]);
    let mut call_instructions: Vec<f64> = vec![];
    let (mut alloc_adjacency, mut alloc_call, mut result) = (String::new(), String::new(), String::new());
    for rep in 0..reps {
        if mode != 2 {
            let mut o = options;
            let t = Instant::now();
            check_input(&g, &mut o, mode == 1).unwrap();
            check_s.push(t.elapsed().as_secs_f64());
            let w = Window::open();
            let t = Instant::now();
            let a = Adjacency::build(&g).unwrap();
            adjacency_s.push(t.elapsed().as_secs_f64());
            if rep == 0 {
                alloc_adjacency = w.json();
            }
            drop(a);
        }
        let mut mt = mt0.clone();
        let mut cb = Callbacks::new(&mut mt);
        let w = Window::open();
        let i0 = instructions();
        let t = Instant::now();
        match mode {
            0 => {
                let out = partition(&g, &options, flat.as_deref(), &mut cb, None).unwrap();
                call_s.push(t.elapsed().as_secs_f64());
                if rep == 0 {
                    alloc_call = w.json();
                    let hash = fnv(out.membership.iter().map(|&x| x as u64));
                    result = format!(
                        "{{\"count\": {}, \"quality\": {:?}, \"slack\": {:?}, \"hash\": \"{:016x}\"}}",
                        out.count, out.quality, out.slack, hash
                    );
                }
            }
            1 => {
                let out = cover(&g, &options, rows.as_ref(), &mut cb, None).unwrap();
                call_s.push(t.elapsed().as_secs_f64());
                if rep == 0 {
                    alloc_call = w.json();
                    let c = &out.cover;
                    let hash = fnv(c.offset.iter().chain(c.labels.iter()).map(|&x| x as u64));
                    result = format!(
                        "{{\"count\": {}, \"quality\": {:?}, \"slack\": {:?}, \"hash\": \"{:016x}\"}}",
                        out.count, out.quality, out.slack, hash
                    );
                }
            }
            _ => {
                let c = rows.as_ref().unwrap();
                let state = AuditState::prepare(&g, &c.offset, &c.labels, None).unwrap();
                extra_s.push(t.elapsed().as_secs_f64());
                let o = AuditOptions { max_memberships: cap, allow_isolation: isolation, prune, atol, rtol };
                let selected: Vec<i64> = (0..n).collect();
                let t = Instant::now();
                let out = state.intervals(&o, Selection::List(&selected), None).unwrap();
                call_s.push(t.elapsed().as_secs_f64());
                if rep == 0 {
                    alloc_call = w.json();
                    let count = out.pairs.len();
                    let hash = fnv(out.pairs.iter().flat_map(|&(a, b)| [a.to_bits(), b.to_bits()]));
                    result = format!("{{\"count\": {count}, \"hash\": \"{hash:016x}\"}}");
                }
            }
        }
        call_instructions.push((instructions() - i0) as f64);
    }
    println!(
        "{{\"core\": \"rust\", \"mode\": {mode}, \"check_s\": {}, \"adjacency_s\": {}, \"prepare_s\": {}, \"call_s\": {}, \"call_instructions\": {}, \
         \"alloc_adjacency\": {}, \"alloc_call\": {}, \"result\": {}}}",
        list(&check_s),
        list(&adjacency_s),
        list(&extra_s),
        list(&call_s),
        format!("[{}]", call_instructions.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(", ")),
        if alloc_adjacency.is_empty() { "null".into() } else { alloc_adjacency },
        alloc_call,
        result
    );
}
