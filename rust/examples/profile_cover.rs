//! A profiling loop (not a test): cover() on an edge list file, repeated.
//! cargo run --release --example profile_cover -- FILE GAMMA CAP REPS [w]
use hedonic_core_rs::cover::cover;
use hedonic_core_rs::graph::{Edges, Graph, Options};
use hedonic_core_rs::partition::partition;
use hedonic_core_rs::queue::Callbacks;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).unwrap();
    // Pairs "a b", or triples "a b w" with a fifth argument "w".
    let weighted = args.get(5).map(|a| a == "w").unwrap_or(false);
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let step = if weighted { 3 } else { 2 };
    let flat: Vec<i64> = tokens.chunks(step).flat_map(|t| [t[0].parse().unwrap(), t[1].parse().unwrap()]).collect();
    let weights: Vec<f64> = if weighted { tokens.chunks(3).map(|t| t[2].parse().unwrap()).collect() } else { Vec::new() };
    let n = flat.iter().max().unwrap() + 1;
    let gamma: f64 = args[2].parse().unwrap();
    let cap: i64 = args[3].parse().unwrap();
    let reps: usize = args[4].parse().unwrap();
    let g = Graph::new(n, Edges::Flat(&flat), if weighted { Some(&weights[..]) } else { None }, None).unwrap();
    let o = Options { resolution: gamma, max_memberships: cap, allow_isolation: false, ..Options::default() };
    let mut q = 0.0;
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let start = std::time::Instant::now();
        let mut mt = hedonic_core_rs::rng::Pcg32::new(42, 54);
        let mut cb = Callbacks::new(&mut mt);
        q += if cap == 1 {
            partition(&g, &o, None, &mut cb, None).unwrap().quality
        } else {
            cover(&g, &o, None, &mut cb, None).unwrap().quality
        };
        best = best.min(start.elapsed().as_secs_f64());
    }
    println!("{:.4} ms (fastest of {reps}; quality sum {q})", best * 1e3);
}
