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
| `python/` (module `hedonic_core_rs`, or `hedonic._core`) | `binding/` (module `hedonic._core`) |

The module has the functions of `hedonic._core` with the same arguments,
results, exceptions and messages. `HEDONIC_CORE=rust` makes the package load
it in place of the C extension; `source_sha256` is the digest of this crate's
sources (setup.py's format), so records name the Rust core apart.

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

**Without the C.** The cargo feature `core-module` names the module
`hedonic._core`, and `tools/rust_pyproject.py` writes the repository's
`pyproject.toml` with maturin building `rust/` into it in place of
setuptools building `native/` (the `[project]` table is kept, so it is the
same distribution). CI (`.github/workflows/rust.yml`, job `rust_only`)
deletes `native/` and `setup.py`, builds and installs that package, and runs
the pinned tests and the whole test suite on every wheel platform.

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
  queue), so both engines hint, in stages 8, 4, 2, 1 (and, for the
  partition, 0) visits ahead, the adjacency, the neighbours' labels or rows
  and (partition) those labels' records.
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

`python rust/bench/bench.py [--quick] [--reps N]` times `detect` of both
cores, interleaved, on the same calls (and checks the results are equal);
on an M1 Max (load average 8), every case is at least as fast in Rust:
the mid-size games (BA 50k, SBM 20k, ER 100k) 1.01 to 1.08, the small
graphs 1.5 to 2.5 (the generator in place), the large games 1.3 to 1.4
(prefetching); the geometric mean of C time / Rust time is 1.23 (1.35
under a heavy load, where the mid-size partitions gain 1.15 to 1.6).

## Build and test

```bash
cd rust
cargo test --release                              # the C core's bit pins and tests (test_core.c, test_audit.c), soft fma vs the instruction
cargo test --release --features force-soft-fma    # the same with the soft fma everywhere
maturin develop --release                         # into the active venv
cd ..
HEDONIC_CORE=rust python tools/native_tests.py    # the pinned Python tests on the Rust core
python -m pytest -x -q rust/tests/test_differential.py   # C against Rust, bit for bit
```

hedonic with the Rust core alone, in a copy of the checkout:

```bash
python tools/rust_pyproject.py > pyproject.rust.toml && mv pyproject.rust.toml pyproject.toml
rm -rf native setup.py
pip install ".[cli]" pytest && python tools/native_tests.py && python -m pytest -q
```

`cargo run --release --example profile_cover -- EDGES GAMMA CAP REPS [w]` is a
profiling loop over an edge-list file (`a b` lines, or `a b w` with `w`);
`ab_prefetch` times prefetching off and on in one process and checks the
results agree; `phases` times the input check, the adjacency build and the
partition apart.

## Perflog: where time and memory go

`python rust/bench/perflog.py` (run it under
`.agents/skills/hedonic-native-core/scripts/guard.sh 16`, one heavy job at a
time) measures both cores, workload by workload, each measurement in a fresh
child process, and writes a markdown table (`--table`) and every number
(`--json`); `rust/bench/perflog-baseline.md` is the baseline and
`rust/bench/perflog-after.md` the same run after the Rust performance pass. The workloads
(`--list`, `--filter`): Zachary (100 calls a measurement), Barabasi-Albert
50k as a partition and as covers at M = 2, 5 and -1, a weighted SBM, node
weights, a seeded random start, the SNAP covered subgraphs of DBLP, Amazon
and YouTube (`top5000`, from the read-only pair pickles in
`~/Databases/Hedonic/Pickles`: partitions at the density, uncapped covers at
the paper queue's calibrated resolutions) and the stability audit of a DBLP
cover, capped and uncapped.

- **Time by layer.** Wrappers timestamp `Game.community_hedonic`,
  `hedonic._native.community_hedonic` and `_core.detect` (or `_prepare`,
  `_audit_prepare`, `_audit_intervals`), with retired instructions at each
  boundary (`proc_pid_rusage`), which a loaded machine does not blur. The
  core alone runs in a native driver on the very same input (edges in the
  capsule's order, weights, start, the Mersenne Twister words of the seeded
  generator), `rust/examples/perflog_core.rs` and `rust/bench/perflog_core.c`
  (the C core included as one translation unit), and must return the call's
  result (count, quality and a hash of the rows); the driver times the input
  check and the adjacency build apart. Whatever detect spends beyond the
  driver's call (and beyond the same call with `start=None`, the start rows'
  read) is the binding's: generator, buffers and result lists.
- **Memory.** `ru_maxrss` of each child (`wait4`), the growth of its high-water
  mark and of its peak physical footprint across the one call (the interval
  maximum of `proc_pid_rusage`, reset just before the call: on macOS freed
  pages stay in the footprint, so the high-water mark is the honest number),
  the Python objects the call allocates (tracemalloc, a child of its own) and
  the core's own heap, counted in the drivers by a global allocator (Rust)
  and by wrappers of `malloc`/`calloc`/`realloc`/`free` seen only by the
  driver's copy of the C core.
- **Profiles.** `perflog.py --profile NAME [--seconds 20] [--cores c,rust]`
  runs the workload in a loop and attaches macOS `sample`; the full call trees
  go to `rust/bench/profiles/*.sample.txt` (ignored by git) and the functions
  at the top of the stack to `*.top.txt`.
