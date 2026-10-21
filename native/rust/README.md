# hedonic-core-rs

The core of hedonic 2.1, in Rust: best-response local moving for the CPM
hedonic game and the fixed-cover stability audit, built with
[PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs) into the
extension `hedonic._core`. `native/README.md` is its design. It is a port of
hedonic 2.0's C core and computes the same results bit for bit: every fused
multiply-add is an explicit `fma::fma` (the instruction, or the soft fma
ported from musl), rustc never contracts `a * b + c`, and every sum runs in
the C core's order. Its doc comments name the C file each module ports
(paths in hedonic 2.0's `native/`):

| Rust | C (hedonic 2.0) |
|---|---|
| `fma.rs`, `arith.rs` | `softfma.h`, `dispatch.h`, `internal.h` |
| `graph.rs`, `rng.rs`, `queue.rs`, `heap.rs` | `shared/` |
| `partition.rs` | `partition/` |
| `cover.rs` | `cover/` |
| `audit.rs`, `verify.rs`, `trace.rs` | `check/` |
| `python/` (module `hedonic._core`) | `binding/` (module `hedonic._core`) |
| `tests/pins.rs`, `tests/core.rs`, `tests/audit.rs` | `tests/test_core.c`, `tests/test_audit.c` |

The module has the functions of hedonic 2.0's extension with the same
arguments, results, exceptions and messages, plus `backend` (`"rust"`) and
`fma_path()`. `source_sha256` is the digest of this crate's sources
(`build.rs`: every `src/**/*.rs`, `build.rs` and `Cargo.toml`, each as path,
length and bytes, in path order, CRLF as LF), so records name the build of the
core by its source.

The stability audit is rc11's: the windowed search (exclusion by Newton
steps, certified ranges and a certified core, the reference's own cells
near the ends of each stable set) is `AuditState::intervals`, and the
exhaustive search it must equal bit for bit is `intervals_exhaustive`
(tests only). Its re-sorts read unchecked (the order is a permutation,
debug-asserted), and the full sort runs on packed integer keys (the
score's bits in descending order, then the index), not an index
comparator. On the manuscript's SNAP pairs (Hedonic covers at cap 3 and
uncapped, pruned and not) both cores audit every vertex to the same
intervals, Rust 1.07 to 1.47 times as fast as C (the smallest, Amazon's
top5000 pruned, ties at 12 ms; YouTube uncapped and unpruned, the longest,
254 s against 294 s).

**The package build.** The repository's `pyproject.toml` has maturin build
this crate into `hedonic._core` with the cargo feature `core-module` (the
module's name). The feature `python` alone builds the same module as
`hedonic_core_rs`, which no build of the package uses. CI
(`.github/workflows/rust.yml`) runs the cargo tests (release, soft fma,
debug, and x86_64 under Rosetta on macOS), then builds the package and runs
the pinned tests, the libm check and the whole test suite on every wheel
platform.

**A second core beside it.** `HEDONIC_CORE=<name>` makes the package load
`hedonic._core_<name>`, if it is installed beside `hedonic._core`, in its
place (`src/hedonic/__init__.py`), so every test runs on it unchanged. Built
as `hedonic._core_c`, hedonic 2.0's C core is compared with this one by
`tests/test_differential.py` (every result, trace, verifier report, audit
answer, error and the caller's generator state, bit for bit) and timed
against it by `bench/bench.py`; both skip (or stop) with the reason when it
is absent.

What is safer: the engines are safe Rust except the hot paths (the visits
of both engines: their scans, scoring, slot resets and requeueing; the
queue's next vertex; the adjacency's fill), whose indexing invariants are
stated and debug-asserted (the cover's in one place, on its `State`, and
checked whole before and after each run in debug builds), and the prefetch hints
(`prefetch.rs`), which read nothing. `Adjacency::build` checks every edge
end against the vertex range itself before its unchecked fill, so no
caller's check is relied upon; the visits rely on the shape it builds
(`Adjacency::well_formed`, asserted in debug builds). Allocation failures
are errors, not aborts; Python buffers are borrowed only while no Python
code can run, and copied for a generator that calls Python on each draw.

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
- **Less per visit.** The queue's next vertex is inlined, its refill and
  interruption check are out of line (it was a call returning through
  memory on every visit); the partition scores candidates with selects, not
  branches (as C does), and the visits of both engines read unchecked.
  The cover's best-prefix selection sorts in a persistent buffer (C's is
  the same; the first port copied a 512-byte array per visit); rows are
  packed in one buffer at the start; short rows are insertion-sorted; the
  slack's sort is unstable over a total order. Per call, the partition now
  retires fewer instructions than C: 59.6M against 60.4M (weighted SBM
  20k), 65.8M against 67.4M (SBM 20k), 127M against 129M (BA 50k); the
  cover 561M against 566M (BA 50k, cap 3; it was 683M).

`python native/rust/bench/bench.py [--quick] [--reps N]` times `detect` of both
cores (`hedonic._core` and `hedonic._core_c`), interleaved, on the same calls
(and checks the results are equal); on an M1 Max (load average 8), every
case is at least as fast in Rust:
the mid-size games (BA 50k, SBM 20k, ER 100k) 1.01 to 1.08, the small
graphs 1.5 to 2.5 (the generator in place), the large games 1.3 to 1.4
(prefetching); the geometric mean of C time / Rust time is 1.23 (1.35
under a heavy load, where the mid-size partitions gain 1.15 to 1.6).

## Build and test

```bash
cd native/rust
cargo test --release                              # the bit pins and the C core's tests (test_core.c, test_audit.c), soft fma vs the instruction
cargo test --release --features force-soft-fma    # the same with the soft fma everywhere
cd ../..
pip install -e ".[cli]" pytest                    # maturin builds the crate into hedonic._core (or: maturin develop --release)
python tools/native_tests.py                      # the pinned Python tests (HIG_PINNED=1, no skips)
python native/check_libm.py "$(python -c 'import hedonic._core as c; print(c.__file__)')"
python -m pytest -q                               # the whole suite
```

`cargo run --release --example profile_cover -- EDGES GAMMA CAP REPS [w]` is a
profiling loop over an edge-list file (`a b` lines, or `a b w` with `w`);
`ab_prefetch` times prefetching off and on in one process and checks the
results agree; `phases` times the input check, the adjacency build and the
partition apart.
