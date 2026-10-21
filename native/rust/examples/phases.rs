//! Times the phases of a partition call apart (benchmarks only).
//! cargo run --release --example phases -- FILE GAMMA REPS   (lines "a b w")
use hedonic_core_rs::graph::{check_input, Adjacency, Edges, Graph, Options};
use hedonic_core_rs::partition::partition;
use hedonic_core_rs::queue::Callbacks;
use hedonic_core_rs::rng::Pcg32;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).unwrap();
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let flat: Vec<i64> = tokens.chunks(3).flat_map(|t| [t[0].parse().unwrap(), t[1].parse().unwrap()]).collect();
    let weights: Vec<f64> = tokens.chunks(3).map(|t| t[2].parse().unwrap()).collect();
    let n = flat.iter().max().unwrap() + 1;
    let gamma: f64 = args[2].parse().unwrap();
    let reps: usize = args[3].parse().unwrap();
    let setup_only = args.get(4).map(|a| a == "setup").unwrap_or(false);
    let g = Graph::new(n, Edges::Flat(&flat), Some(&weights), None).unwrap();
    let o = Options { resolution: gamma, max_memberships: 1, allow_isolation: false, ..Options::default() };
    let mut best = [f64::INFINITY; 3];
    for _ in 0..reps {
        let t = Instant::now();
        let mut oo = o;
        check_input(&g, &mut oo, false).unwrap();
        best[0] = best[0].min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let a = Adjacency::build(&g).unwrap();
        best[1] = best[1].min(t.elapsed().as_secs_f64());
        drop(a);
        if setup_only {
            continue;
        }
        let mut rng = Pcg32::new(42, 54);
        let mut cb = Callbacks::new(&mut rng);
        let t = Instant::now();
        partition(&g, &o, None, &mut cb, None).unwrap();
        best[2] = best[2].min(t.elapsed().as_secs_f64());
    }
    println!("check {:.4} ms, adjacency {:.4} ms, partition {:.4} ms", best[0] * 1e3, best[1] * 1e3, best[2] * 1e3);
}
