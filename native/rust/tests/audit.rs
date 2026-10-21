//! The fixed-cover audit (native/tests/test_audit.c): hand cases,
//! validation, interruption, and the windowed search against the exhaustive
//! one, bit for bit, on random covers.

use hedonic_core_rs::audit::{AuditOptions, AuditState, Intervals, Selection};
use hedonic_core_rs::error::Kind;
use hedonic_core_rs::graph::{Edges, Graph};

fn prepare(
    n: i64,
    edges: &[i64],
    weights: Option<&[f64]>,
    nodes: Option<&[f64]>,
    offset: &[i64],
    labels: &[i64],
) -> Result<AuditState, Kind> {
    let graph = Graph::new(n, Edges::Flat(edges), weights, nodes).map_err(|e| e.kind)?;
    AuditState::prepare(&graph, offset, labels, None).map_err(|e| e.kind)
}

#[test]
fn hand_cases_validation_and_interruption() {
    let (edges, offsets, labels, weights) = ([0i64, 1], [0i64, 1, 2], [0i64, 1], [0.5f64]);
    let s = prepare(2, &edges, Some(&weights), None, &offsets, &labels).unwrap();
    let mut o = AuditOptions::default();
    let all = Selection::Range(0, 2);
    let rows = s.intervals(&o, all, None).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows.row(0), [(0.5 - (o.atol + o.rtol), 1.0)]);
    assert_eq!(rows.row(1).len(), 1);
    let (mut regret, mut tolerance) = ([0.0; 6], [0.0; 6]);
    s.regrets(&o, all, &[0.0, 0.5, 1.0], &mut regret, &mut tolerance, None)
        .unwrap();
    assert_eq!(&regret[..3], &[0.5, 0.0, 0.0]);
    assert!(tolerance.iter().all(|&t| t == o.atol + o.rtol));
    let mut stop = || true;
    assert_eq!(
        s.intervals(&o, all, Some(&mut stop)).unwrap_err().kind,
        Kind::Interrupted
    );
    o.max_memberships = -1;
    assert_eq!(s.intervals(&o, all, None).unwrap_err().kind, Kind::Inval);
    o.max_memberships = 0;
    o.atol = f64::NAN;
    assert_eq!(s.intervals(&o, all, None).unwrap_err().kind, Kind::Inval);
    o.atol = 1e-10;
    let err = s.regrets(
        &o,
        all,
        &[-0.1, 0.5, 1.0],
        &mut regret,
        &mut tolerance,
        None,
    );
    assert_eq!(err.unwrap_err().kind, Kind::Inval);
    assert_eq!(
        s.intervals(&o, Selection::Range(0, 3), None)
            .unwrap_err()
            .kind,
        Kind::Inval
    );

    // Repeated labels count once; a private label has exactly zero crowding.
    let s = prepare(
        2,
        &edges,
        Some(&weights),
        None,
        &[0, 3, 5],
        &[0, 0, 1, 0, 1],
    )
    .unwrap();
    o.max_memberships = 1;
    assert_eq!(s.intervals(&o, all, None).unwrap_err().kind, Kind::Inval);
    o.max_memberships = 2;
    s.intervals(&o, all, None).unwrap();

    let graph = Graph::new(2, Edges::Flat(&edges), Some(&weights), None).unwrap();
    let mut stop = || true;
    let err = AuditState::prepare(&graph, &offsets, &labels, Some(&mut stop));
    assert_eq!(err.err().unwrap().kind, Kind::Interrupted);
    for bad in [-1.0, f64::INFINITY] {
        assert_eq!(
            prepare(2, &edges, Some(&[bad]), None, &offsets, &labels).err(),
            Some(Kind::Inval)
        );
    }
    let w0 = Some(&[0.0][..]);
    assert_eq!(
        prepare(2, &[0, 0], w0, None, &offsets, &labels).err(),
        Some(Kind::Inval)
    );
    assert_eq!(
        prepare(2, &edges, w0, None, &offsets, &[0, 2]).err(),
        Some(Kind::Inval)
    );
    assert_eq!(
        prepare(2, &edges, w0, None, &[0, 0, 2], &labels).err(),
        Some(Kind::Inval)
    );
    assert_eq!(
        prepare(2, &edges, w0, Some(&[1.0, 2.0]), &offsets, &labels).err(),
        Some(Kind::Inval)
    );
    prepare(2, &edges, w0, Some(&[1.0, 1.0]), &offsets, &labels).unwrap();

    let s = prepare(2, &[], None, None, &offsets, &labels).unwrap();
    o.allow_isolation = true;
    let rows = s.intervals(&o, all, None).unwrap();
    assert_eq!(rows.rows().collect::<Vec<_>>(), [[(0.0, 1.0)], [(0.0, 1.0)]]);
}

struct Fuzz(u64);

impl Fuzz {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: i64) -> i64 {
        (self.next() % n as u64) as i64
    }
}

fn bits(rows: &Intervals) -> Vec<Vec<(u64, u64)>> {
    rows.rows()
        .map(|row| {
            row.iter()
                .map(|&(lo, hi)| (lo.to_bits(), hi.to_bits()))
                .collect()
        })
        .collect()
}

/// The windowed search against the exhaustive one, bit for bit, on random
/// covers: integer weights (ties, repeated cuts) and real ones, every option.
/// The same games as test_audit.c's (its xorshift and draws, in its order).
#[test]
fn windowed_matches_exhaustive() {
    const TOLERANCES: [(f64, f64); 5] = [
        (1e-10, 1e-9),
        (0.0, 0.0),
        (1e-3, 1e-2),
        (0.05, 0.0),
        (0.0, 0.1),
    ];
    let games: i64 = std::env::var("HIG_AUDIT_FUZZ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let mut f = Fuzz(0x9e3779b97f4a7c15);
    let (mut compared, mut intervals) = (0i64, 0usize);
    for game in 0..games {
        // Small games, then larger ones, then a few whose vertices have
        // hundreds of candidates (the windowed search's certified core).
        let n = 3 + f.below(if game < games / 2 {
            14
        } else if game % 16 != 0 {
            60
        } else {
            240
        });
        let label_range = 1 + f.below(n);
        let integer = f.below(2) != 0;
        let (mut edges, mut weights, mut rows, mut offset) =
            (Vec::new(), Vec::new(), Vec::new(), vec![0i64]);
        let p = 0.1 + 0.6 * f.below(100) as f64 / 100.0;
        for a in 0..n {
            for b in a + 1..n {
                if (f.below(1000) as f64 / 1000.0) < p {
                    edges.extend([a, b]);
                    weights.push(if integer {
                        (1 + f.below(3)) as f64
                    } else {
                        (1 + f.below(1000)) as f64 / 997.0
                    });
                }
            }
        }
        for _ in 0..n {
            let spread = if f.below(3) == 0 { 4 } else { 2 };
            let size = 1 + f.below(spread);
            for _ in 0..size {
                rows.push(f.below(label_range));
            }
            offset.push(rows.len() as i64);
        }
        // Contiguous labels: renumber the ones used.
        let mut map = vec![-1i64; 256];
        let mut next = 0;
        for label in rows.iter_mut() {
            if map[*label as usize] < 0 {
                map[*label as usize] = next;
                next += 1;
            }
            *label = map[*label as usize];
        }
        let weighted = f.below(4) != 0;
        let s = prepare(
            n,
            &edges,
            weighted.then_some(&weights[..]),
            None,
            &offset,
            &rows,
        )
        .unwrap();
        for _ in 0..8 {
            let t = f.below(5) as usize;
            let o = AuditOptions {
                atol: TOLERANCES[t].0,
                rtol: TOLERANCES[t].1,
                allow_isolation: f.below(2) != 0,
                prune: f.below(4) != 0,
                max_memberships: if f.below(2) != 0 { 0 } else { 4 + f.below(60) },
            };
            let fast = s.intervals(&o, Selection::Range(0, n), None).unwrap();
            let full = s
                .intervals_exhaustive(&o, Selection::Range(0, n), None)
                .unwrap();
            assert_eq!(bits(&fast), bits(&full), "game {game}, options {o:?}");
            compared += n;
            intervals += full.pairs.len();
        }
    }
    eprintln!("windowed audit: {compared} vertex audits, {intervals} intervals, identical to the exhaustive search");
}
