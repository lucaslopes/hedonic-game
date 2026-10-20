# hedonic-core-rs

A Rust port of `native/` (the C core and its CPython binding), built with
[PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs). It computes
the same results as the C core bit for bit: every fused multiply-add is an
explicit `fma::fma` (the instruction, or the soft fma ported from musl),
rustc never contracts `a * b + c`, and every sum runs in the C core's order.

| Rust | C |
|---|---|
| `fma.rs`, `arith.rs` | `softfma.h`, `dispatch.h`, `internal.h` |
| `graph.rs`, `rng.rs`, `queue.rs`, `heap.rs` | `shared/` |
| `partition.rs` | `partition/` |
| `cover.rs` | `cover/` |
| `audit.rs`, `verify.rs`, `trace.rs` | `check/` |
| `python/` (module `hedonic_core_rs`) | `binding/` (module `hedonic._core`) |

The module has the functions of `hedonic._core` with the same arguments,
results, exceptions and messages. `HEDONIC_CORE=rust` makes the package load
it in place of the C extension; `source_sha256` is the digest of this crate's
sources (setup.py's format), so records name the Rust core apart.

What is safer: the engines are safe Rust except the hot scans (the
neighbour scans of both engines, the partition's scoring and slot reset),
whose indexing invariants are stated and debug-asserted, and the prefetch
hints (`prefetch.rs`), which read nothing; allocation failures are errors,
not aborts; Python buffers are borrowed only while no Python code can run,
and copied for a generator that calls Python on each draw.

What is faster (the same visits, moves and bits as C, in less time):

- **The generator's state in place.** An exact `random.Random` is run
  natively by both cores, but C copies its state through `getstate` and
  `setstate` (two tuples of 625 ints, about 20 us a call). The Rust binding
  reads and writes the Mersenne Twister words in the `_random.Random`
  struct, after a probe at first use has checked that layout against
  `getstate`/`setstate` (otherwise, and on free-threaded builds, it uses the
  tuples). Small graphs run 1.5 to 2.4 times faster end to end.
- **Prefetching ahead of the queue.** The next visits are known (the FIFO
  queue), so both engines hint, in stages 8, 4, 2, 1 and 0 visits ahead, the
  adjacency, the neighbours' labels or rows and those labels' records.
  It pays only past the caches, so it runs when a game's working set
  exceeds 12 MB (the L2 of an M1 cluster): BA 300k runs 1.3 (partition) to
  1.6 (cover) times faster; below that it is off.
- **Less per visit.** The cover's best-prefix selection sorts in a
  persistent buffer (C's is the same; the first port copied a 512-byte
  array per visit); rows are packed in one buffer at the start; short rows
  are insertion-sorted; the slack's sort is unstable over a total order.

`python rust/bench/bench.py [--quick] [--reps N]` times `detect` of both
cores, interleaved, on the same calls (and checks the results are equal);
on an M1 Max under load, the geometric mean of C time / Rust time is 1.18,
with the in-cache mid-size games (BA 50k, SBM 20k, ER 100k) at parity
(0.90 to 1.03).

## Build and test

```bash
cd rust
cargo test --release                              # the C core's bit pins, soft fma vs the instruction
cargo test --release --features force-soft-fma    # the same with the soft fma everywhere
maturin develop --release                         # into the active venv
cd ..
HEDONIC_CORE=rust python tools/native_tests.py    # the pinned Python tests on the Rust core
python -m pytest -x -q rust/tests/test_differential.py   # C against Rust, bit for bit
```

`cargo run --release --example profile_cover -- EDGES GAMMA CAP REPS [w]` is a
profiling loop over an edge-list file (`a b` lines, or `a b w` with `w`);
`ab_prefetch` times prefetching off and on in one process and checks the
results agree.
