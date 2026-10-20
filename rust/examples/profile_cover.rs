//! A profiling loop (not a test): cover() on an edge list file, repeated.
//! cargo run --release --example profile_cover -- FILE GAMMA CAP REPS
use hedonic_core_rs::cover::cover;
use hedonic_core_rs::graph::{Edges, Graph, Options};
use hedonic_core_rs::partition::partition;
use hedonic_core_rs::queue::Callbacks;
use hedonic_core_rs::rng::Mt19937;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).unwrap();
    let flat: Vec<i64> = text.split_whitespace().map(|x| x.parse().unwrap()).collect();
    let n = flat.iter().max().unwrap() + 1;
    let gamma: f64 = args[2].parse().unwrap();
    let cap: i64 = args[3].parse().unwrap();
    let reps: usize = args[4].parse().unwrap();
    let g = Graph::new(n, Edges::Flat(&flat), None, None).unwrap();
    let o = Options { resolution: gamma, max_memberships: cap, allow_isolation: false, ..Options::default() };
    let start = std::time::Instant::now();
    let mut q = 0.0;
    for _ in 0..reps {
        let mut mt = Mt19937::python_seed(0);
        let mut cb = Callbacks::new(&mut mt);
        q += if cap == 1 {
            partition(&g, &o, None, &mut cb, None).unwrap().quality
        } else {
            cover(&g, &o, None, &mut cb, None).unwrap().quality
        };
    }
    println!("{} reps: {:.1} ms each (quality sum {q})", reps, start.elapsed().as_secs_f64() * 1e3 / reps as f64);
}
