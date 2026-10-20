//! Prefetching on against off, interleaved in one process (benchmarks only).
//! cargo run --release --example ab_prefetch -- FILE GAMMA CAP ROUNDS
use hedonic_core_rs::cover::cover;
use hedonic_core_rs::graph::{Edges, Graph, Options};
use hedonic_core_rs::partition::partition;
use hedonic_core_rs::prefetch::set_override;
use hedonic_core_rs::queue::Callbacks;
use hedonic_core_rs::rng::Mt19937;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).unwrap();
    let flat: Vec<i64> = text.split_whitespace().map(|x| x.parse().unwrap()).collect();
    let n = flat.iter().max().unwrap() + 1;
    let gamma: f64 = args[2].parse().unwrap();
    let cap: i64 = args[3].parse().unwrap();
    let rounds: usize = args[4].parse().unwrap();
    let g = Graph::new(n, Edges::Flat(&flat), None, None).unwrap();
    let o = Options { resolution: gamma, max_memberships: cap, allow_isolation: false, ..Options::default() };
    let mut best = [f64::INFINITY; 2];
    let mut q = [0u64; 2];
    for _ in 0..rounds {
        for (k, on) in [false, true].into_iter().enumerate() {
            set_override(Some(on));
            let mut mt = Mt19937::python_seed(0);
            let mut cb = Callbacks::new(&mut mt);
            let t = std::time::Instant::now();
            let quality = if cap == 1 {
                partition(&g, &o, None, &mut cb, None).unwrap().quality
            } else {
                cover(&g, &o, None, &mut cb, None).unwrap().quality
            };
            best[k] = best[k].min(t.elapsed().as_secs_f64());
            q[k] = quality.to_bits();
        }
    }
    assert_eq!(q[0], q[1], "prefetching changed a result");
    println!("n={n} m={} off {:.1} ms, on {:.1} ms, off/on {:.2}", flat.len() / 2, best[0] * 1e3, best[1] * 1e3, best[0] / best[1]);
}
