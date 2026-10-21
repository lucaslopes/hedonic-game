//! The core from Rust, without Python: a partition, a cover and their audit on
//! Zachary's karate club. cargo run --release --no-default-features --example karate
//!
//! The visiting order is drawn from `Mt19937::python_seed(0)`, Python's
//! `random.Random(0)`, so the results are those of
//! `hedonic.arrays.detect(34, EDGES, resolution=0.1, seed=0)` (and of
//! `Game.community_hedonic`) bit for bit; the quality and slack print as hex.
use hedonic_core_rs::audit::{AuditOptions, AuditState, Selection};
use hedonic_core_rs::cover::cover;
use hedonic_core_rs::graph::{Edges, Graph, Options};
use hedonic_core_rs::partition::partition;
use hedonic_core_rs::queue::Callbacks;
use hedonic_core_rs::rng::Mt19937;

/// igraph's Graph.Famous("Zachary"), in its edge order.
const EDGES: [(i64, i64); 78] = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12), (0, 13), (0, 17),
    (0, 19), (0, 21), (0, 31), (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30), (2, 3), (2, 7),
    (2, 27), (2, 28), (2, 32), (2, 9), (2, 8), (2, 13), (3, 7), (3, 12), (3, 13), (4, 6), (4, 10), (5, 6), (5, 10),
    (5, 16), (6, 16), (8, 30), (8, 32), (8, 33), (9, 33), (13, 33), (14, 32), (14, 33), (15, 32), (15, 33), (18, 32),
    (18, 33), (19, 33), (20, 32), (20, 33), (22, 32), (22, 33), (23, 25), (23, 27), (23, 32), (23, 33), (23, 29),
    (24, 25), (24, 27), (24, 31), (25, 31), (26, 29), (26, 33), (27, 33), (28, 31), (28, 33), (29, 32), (29, 33),
    (30, 32), (30, 33), (31, 32), (31, 33), (32, 33),
];

fn main() -> Result<(), hedonic_core_rs::error::Error> {
    let flat: Vec<i64> = EDGES.iter().flat_map(|&(u, v)| [u, v]).collect();
    // Unit edge and vertex weights: None (Some(&weights) for others).
    let graph = Graph::new(34, Edges::Flat(&flat), None, None)?;
    // Python's defaults but the resolution: allow_isolation is false there
    // (Options::default() allows it), and resolution=None is the density.
    let options = Options { resolution: 0.1, allow_isolation: false, ..Options::default() };

    let mut rng = Mt19937::python_seed(0);
    let p = partition(&graph, &options, None, &mut Callbacks::new(&mut rng), None)?;
    println!("partition: {} communities, quality {:e} ({}), slack {:e}",
             p.count, p.quality, hex(p.quality), p.slack);
    println!("  membership {:?}", p.membership);

    let mut rng = Mt19937::python_seed(0);
    let cover_options = Options { max_memberships: 2, ..options };
    let c = cover(&graph, &cover_options, None, &mut Callbacks::new(&mut rng), None)?;
    println!("cover: {} communities, quality {:e} ({}), slack {:e}", c.count, c.quality, hex(c.quality), c.slack);
    println!("  rows of vertices 0-3: {:?}", (0..4).map(|v| c.cover.row(v)).collect::<Vec<_>>());

    // The audit of the cover: stable intervals of the resolution in [0, 1],
    // and regrets on a grid (row-major: vertex x resolution).
    let audit = AuditState::prepare(&graph, &c.cover.offset, &c.cover.labels, None)?;
    let audit_options = AuditOptions::default();
    let intervals = audit.intervals(&audit_options, Selection::Range(0, 4), None)?;
    println!("stable intervals of vertices 0-3: {:?}", intervals);
    let gammas = [0.05, 0.1, 0.5];
    let (mut regret, mut tolerance) = (vec![0.0; 4 * gammas.len()], vec![0.0; 4 * gammas.len()]);
    audit.regrets(&audit_options, Selection::List(&[0, 1, 2, 3]), &gammas, &mut regret, &mut tolerance, None)?;
    println!("regrets of vertices 0-3 at {:?}: {:?}", gammas, regret);
    Ok(())
}

/// Python's float.hex().
fn hex(x: f64) -> String {
    let bits = x.to_bits();
    let (sign, exponent, mantissa) = (bits >> 63, ((bits >> 52) & 0x7ff) as i64, bits & ((1u64 << 52) - 1));
    let sign = if sign == 1 { "-" } else { "" };
    if exponent == 0 && mantissa == 0 {
        return format!("{sign}0x0.0p+0");
    }
    let digits = format!("{mantissa:013x}");
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    let (lead, exponent) = if exponent == 0 { (0, -1022) } else { (1, exponent - 1023) };
    format!("{sign}0x{lead}.{digits}p{exponent:+}")
}
