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

What is safer: allocation failures are errors, not aborts; every input the
engines take is validated by the engine itself (`Adjacency::build` checks
every edge end, so no caller's check is relied upon), and the unchecked
indexing that remains is the part measured to pay (see "Unsafe code").
Python buffers and igraph's edges are read with the GIL held, and the
visits run on the engine's own state built from them (see "Threads").

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
  branches (as C does), and the visits of both engines read unchecked
  (where that pays: "Unsafe code").
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

## Unsafe code

Every `unsafe` block has a `// SAFETY:` comment, and where the check is
cheap a `debug_assert!` of it, which the debug test build (`cargo test`)
runs. What is left, and why:

| Where | What | Why |
|---|---|---|
| `partition.rs` | `get_unchecked` in the visit (labels of the neighbours, community records, candidate slots), the scoring loop, the slot reset and the requeue | 1.06 to 1.13 times faster partitions |
| `cover.rs` | the same in `collect`, the visit, the best-set insertion buffer (`top_positive`) and the requeue | 1.05 to 1.19 times faster covers |
| `audit.rs` | the audit's insertion sort and best-prefix walk over `order`, a permutation (debug-asserted) | 1.015 to 1.04 times faster audits |
| `fma.rs` | the FMA instruction, called only after runtime detection | the instruction |
| `prefetch.rs` | `prfm` / `_mm_prefetch`, which never access memory (compiled out under Miri) | 1.3 to 1.6 on large games |
| `python/` | the CPython C API: buffers, the `random.Random` struct in place, igraph's capsule, C's integer conversions | the binding itself |

The numbers come from `python rust/bench/ab_unsafe.py --group G` (under
`guard.sh`, arm64): it builds `examples/bench_core` from the crate and from
a copy whose files of group G index with bounds checks, runs the two
alternately (fastest of 60 runs of BA 50k, a weighted SBM 20k and ER 100k,
partitions, covers at cap 3 and their audits), checks that both compute
the same hashes, and prints checked / unchecked. The adjacency build
(including a spare-capacity fill with `set_len`) and the queue measured
0.94 to 1.05, inside the noise, so they are checked code now. All three
kept groups together: 1.04 to 1.07 on runs, 1.01 to 1.02 on the audit.

The CI job `miri` runs `cargo miri test` (no `python` feature; the random
games scaled down with `HIG_RANDOM_CASES`, `HIG_AUDIT_FUZZ`; the 10^6-case
fma tests and the 1500-vertex planted game skipped). Miri needs rustup's
nightly; this crate has been developed with Homebrew's stable Rust (no
rustup), so Miri has not run locally.

## Fuzzing

`rust/fuzz` is a cargo-fuzz crate (its own workspace and `Cargo.lock`) with
two targets, whose checks are plain functions in `fuzz/src/lib.rs`:

- `inputs`: arbitrary bytes as vertex counts, edge ends in both layouts,
  weights, options, partition starts, cover offsets and labels, verifier
  inputs and audit queries (prepare with an interruption, both interval
  searches, regrets). Every call must return Ok or Err; its release profile
  keeps debug assertions and overflow checks on.
- `runs`: small valid games (weights, count limits, starts, caps up to n).
  Every traced move's improvement must beat its bound, the verifier must
  certify no gain above the raw slack (tests/test_equilibrium.py's check),
  the result must be well formed, and the same seed must give the same bits.

```bash
cd rust/fuzz
cargo test --release                                  # the checks on pseudo-random inputs (HIG_FUZZ_CASES, default 20 000)
cargo +nightly fuzz run inputs -- -max_total_time=60  # libFuzzer with AddressSanitizer (CI: nightly, 60 s each)
```

On stable, libFuzzer still runs with coverage if the targets are built with
sancov (macOS also needs the cdylib's sancov symbols left undefined):

```bash
CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS="-Cpasses=sancov-module -Cllvm-args=-sanitizer-coverage-level=4 \
  -Cllvm-args=-sanitizer-coverage-inline-8bit-counters -Cllvm-args=-sanitizer-coverage-pc-table \
  -Cllvm-args=-sanitizer-coverage-trace-compares --cfg fuzzing -Clink-arg=-Wl,-undefined,dynamic_lookup" \
  cargo build --release --bins --target aarch64-apple-darwin --target-dir target/sancov
target/sancov/aarch64-apple-darwin/release/inputs -max_total_time=60 CORPUS_DIR
```

So built, 60 s per target ran 1.5 million (`inputs`) and 1.4 million
(`runs`) inputs without a failure, after 300 000 of each through
`cargo test`.

## x86_64: which fma

aarch64 always has the fused multiply-add. On x86_64, `fma::fma` asks once
(`is_x86_feature_detected!("fma")`, cached in an atomic) and calls the
instruction (`#[target_feature(enable = "fma")]`) or the soft fma. The
detection checks CPUID's FMA bit and that the OS saves the AVX state
(OSXSAVE, XCR0), as C's `hig_cpu_has_fma` does; under Miri it reports
nothing, so Miri runs the soft fma. Both give the same bits, so the choice
only changes the speed; `fma_path()` says which one runs.
`HIG_EXPECT_FMA=soft|instruction` makes the test `fma_dispatch_as_expected`
assert it. The CI job `x86_64_dispatch` runs the pins built for
`target-cpu=x86-64` and for `haswell` (instruction), with `force-soft-fma`,
and the baseline build under QEMU's Nehalem, a CPU without FMA, where the
dispatch must choose the soft fma. Locally there is no x86_64 standard
library (Homebrew's Rust has only aarch64-apple-darwin), so this runs in CI
only; macOS x86_64 runs under Rosetta 2 in `rust_core`.

## igraph's edges

`detect` and `_audit_prepare` read an `igraph.Graph`'s edges as the C
binding does (`native/binding/graph_edges.c`): `vcount()` and `ecount()`,
then the `igraph_t` head behind `Graph.__graph_as_capsule()`, borrowed only
if everything read from it agrees with the public API (vertex count,
undirected, `end - stor_begin` equal to the edge count in both vectors,
`end <= stor_end`, every end a vertex, edges 0, m/2 and m - 1 equal to
`graph.es[e].tuple`); otherwise `get_edgelist()` is copied. The Rust side
also requires aligned pointers, and reads the head after the probes
(`es[e].tuple` runs Python code), so nothing runs between reading and using
it. In a free-threaded interpreter without the GIL another thread could
change the graph at any moment, so there the edges are always copied.
`_edges_borrowed(graph)` (tests) says which path a graph takes.
`tests/test_graph_edges.py` forces each fallback through a subclass whose
capsule method raises, returns no capsule, or the capsule of a directed
graph, of other edges or of fewer vertices, and checks that results,
audits and the generator's state equal the plain graph's on both cores;
the CI job `igraph_versions` checks the capsule is still read on
python-igraph's newest release and (not blocking) its main branch.

A divergence from C: a generator that is called (not copied) can run any
Python code during the visits, for example add an edge. C keeps the edge
pointers it read at the start and reads them again for the quality (which
then may read freed memory); Rust reads the capsule afresh for `finish` and
raises `RuntimeError("The graph changed during the call.")` if the counts
changed.

## Threads

Each call runs in three steps (`PartitionCall`, `CoverCall`): `prepare`
builds the engine's own state from the borrowed inputs with the GIL held,
`run` draws the visiting order and runs the visits on that state only, and
`finish` computes the result from the inputs read again. For an exact
`random.Random` (the generator runs natively) `run` releases the GIL
(`Python::detach`); a generator that is called keeps it. The audit's
queries (`_audit_intervals`, `_audit_regrets`) run without the GIL on copies
of their inputs and write the outputs after re-attaching. Ctrl-C: the
detached visits re-attach to check signals at most every 10 ms, and only on
the main thread (where CPython runs signal handlers), so a handler that
raises stops the call, and the generator's state is still written back.

`tests/test_native_threads.py`: detections (partitions, covers, weighted),
plain `partition()` calls, `stable_intervals` and `regrets` from 8 threads
at once equal their serial results bit for bit, with each caller's
generator ending where its serial call leaves it; while a detection runs on
one thread, another keeps running Python code in the middle half of the
call (the C core lets it in only past 95% of the call); and a signal stops a
detached detection.

Both modules declare `gil_used = false`: the module keeps no Python-visible
mutable state (its caches are once-initialised, its counters atomics), and
on a free-threaded build the generator goes through
`getstate`/`setstate` (under its own lock) and igraph's edges through its
API. On CPython 3.13.6t (`uv python install 3.13t`, macOS arm64), the module
loads without enabling the GIL. python-igraph has no free-threaded wheels
and its sdist builds only with the limited API, which free-threaded builds
lack; built from its 1.0.0 sdist with the full API (and one fix to a call
of `igraph_vector_int_view` in that unused path), its import enables the
GIL. With the GIL so enabled and with `PYTHON_GIL=0` (forcing it off),
`HEDONIC_CORE=rust tools/native_tests.py` passed (35 tests),
`tests/test_native_threads.py` passed, and `rust/tests/test_differential.py`
passed against the C core (6 tests); `_edges_borrowed` is True with the GIL
and False without it.

## Errors

The functions raise what the C binding raises, type and message. Arguments
C parses as `n`, `L`, `d` and `p` go through `CArg` (PyNumber_Index and
PyLong_AsSsize_t, PyLong_AsLongLong, PyFloat_AsDouble, truth), not PyO3's
typed parameters, whose errors carry an `argument 'name': ` prefix and whose
i64 overflows say "C long"; integers read from sequences use
PyLong_AsLongLong too (`long_long`). `rust/tests/test_differential.py`
compares the plain functions' errors in one process;
`rust/tests/test_error_parity.py` runs 118 calls of `Game.community_hedonic`,
`Game.stable_intervals` and `Game.regrets` (52 detections, 25 bad covers
through both audit functions, 10 regret grids, 6 graphs and generators) in a
process on each core and compares every exception type and message
(and the results of the rows that are fine).

## Toolchain and supply chain

- `Cargo.lock` (and `fuzz/Cargo.lock`) is committed, and every cargo
  build, test, clippy and Miri run in CI is `--locked` (cargo-fuzz's own
  nightly run builds from the committed lock file too).
- `rust-version = "1.78"`: `Cargo.lock` is version 4, which cargo reads
  from 1.78; PyO3 0.27 and its build dependencies need 1.74, the crate's own
  code 1.65. The CI job `msrv` builds and tests with 1.78 exactly.
- `deny.toml` (CI job `supply_chain`, `cargo deny check --all-features`):
  any RustSec advisory or yanked version fails; licences are those of the
  dependency tree (MIT, Apache-2.0, Apache-2.0 WITH LLVM-exception,
  Unicode-3.0) and the crate's GPL-3.0-or-later; one version of each crate;
  crates.io only. The same job checks that `Cargo.toml`, `fuzz/Cargo.toml`,
  `pyproject.toml` here and the repository's `pyproject.toml` name one
  licence.
- `cargo clippy --all-targets -- -D warnings` is clean on every feature
  set (CI job `lints`); the lints allowed on purpose are listed, with the
  reason, at the top of `src/lib.rs`.
- `source_sha256` (`build.rs`) is the digest of `src/`, `build.rs`,
  `Cargo.toml` and `Cargo.lock`: a dependency bump changes the compiled
  core, so it changes the core's identity.

## Build and test

```bash
cd rust
cargo test --release                              # the C core's bit pins and tests (test_core.c, test_audit.c), soft fma vs the instruction
cargo test --release --features force-soft-fma    # the same with the soft fma everywhere
cargo test                                        # debug: every debug_assert of the unchecked paths
cargo clippy --all-targets --features python -- -D warnings
maturin develop --release                         # into the active venv
cd ..
HEDONIC_CORE=rust python tools/native_tests.py    # the pinned Python tests on the Rust core
python -m pytest -x -q rust/tests/                # C against Rust: results, errors and messages
python -m pytest -q tests/test_native_threads.py tests/test_graph_edges.py   # (also with HEDONIC_CORE=rust)
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
partition apart; `bench_core` times the engines and the audit on fixed
synthetic graphs (what `bench/ab_unsafe.py` compares).
