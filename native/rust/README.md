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
(`build.rs`: every `src/**/*.rs`, `build.rs`, `Cargo.toml` and `Cargo.lock`,
each as path, length and bytes, in path order, CRLF as LF), so records name the build of the
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
  queue), so both engines hint, in stages 8, 4, 2, 1 (and, for the
  partition, 0) visits ahead, the adjacency, the neighbours' labels or rows
  and (partition) those labels' records.
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

`python native/rust/bench/bench.py [--quick] [--reps N]` times `detect` of both
cores (`hedonic._core` and `hedonic._core_c`), interleaved, on the same calls
(and checks the results are equal); on an M1 Max (load average 8), every
case is at least as fast in Rust:
the mid-size games (BA 50k, SBM 20k, ER 100k) 1.01 to 1.08, the small
graphs 1.5 to 2.5 (the generator in place), the large games 1.3 to 1.4
(prefetching); the geometric mean of C time / Rust time is 1.23 (1.35
under a heavy load, where the mid-size partitions gain 1.15 to 1.6).

The performance pass of 2.1 (`bench/perflog-after.md`, "Perflog" below)
then sorts the long best-set selection of uncapped covers as integer keys
over the same total order, and releases the run state before the result is
built. Against 2.0's C core, in retired instructions (a loaded machine does
not blur them): uncapped covers 14.1G against 26.8G (BA 50k), 21.2G against
34.2G (YouTube top5000) and 3.5G against 4.4G (DBLP top5000), the uncapped
DBLP audit 7.45G against 8.96G, partitions equal; the core's heap peaks are
C's or lower.

## Perflog: where time and memory go

`python native/rust/bench/perflog.py` (run it under
`.agents/skills/hedonic-native-core/scripts/guard.sh 16`, one heavy job at a
time) measures the core, workload by workload, each measurement in a fresh
child process, and writes a markdown table (`--table`) and every number
(`--json`). Where hedonic 2.0's C core is built beside it (`native/c`,
`hedonic._core_c`, `HEDONIC_CORE=c`: the paper branch) it measures both;
`bench/perflog-baseline.md` is the comparison before the Rust performance
pass of 2.1 and `bench/perflog-after.md` the same run after it;
`bench/perflog-python.md` compares the Python layer before and after its own
pass (the same core, Rust only). The workloads
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
  generator), `examples/perflog_core.rs` and `bench/perflog_core.c` (the C
  core included as one translation unit), and must return the call's result
  (count, quality and a hash of the rows); the driver times the input check
  and the adjacency build apart. Whatever detect spends beyond the driver's
  call (and beyond the same call with `start=None`, the start rows' read) is
  the binding's: generator, buffers and result lists.
- **Memory.** `ru_maxrss` of each child (`wait4`), the growth of its high-water
  mark and of its peak physical footprint across the one call (the interval
  maximum of `proc_pid_rusage`, reset just before the call: on macOS freed
  pages stay in the footprint, so the high-water mark is the honest number),
  the Python objects the call allocates (tracemalloc, a child of its own) and
  the core's own heap, counted in the drivers by a global allocator (Rust)
  and by wrappers of `malloc`/`calloc`/`realloc`/`free` seen only by the
  driver's copy of the C core.
- **Profiles.** `perflog.py --profile NAME [--seconds 20] [--cores rust,c]`
  runs the workload in a loop and attaches macOS `sample`; the full call trees
  go to `bench/profiles/*.sample.txt` (ignored by git) and the functions at
  the top of the stack to `*.top.txt`.

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

The numbers come from `python native/rust/bench/ab_unsafe.py --group G` (under
`guard.sh`, arm64): it builds `examples/bench_core` from the crate and from
a copy whose files of group G index with bounds checks, runs the two
alternately (fastest of 60 runs of BA 50k, a weighted SBM 20k and ER 100k,
partitions, covers at cap 3 and their audits), checks that both compute
the same hashes, and prints checked / unchecked. The adjacency build
(including a spare-capacity fill with `set_len`) and the queue measured
0.94 to 1.05, inside the noise, so they are checked code now. All three
kept groups together: 1.04 to 1.07 on runs, 1.01 to 1.02 on the audit.

The CI job `miri` runs `cargo miri test`, one job per test target (no
`python` feature; the random games scaled down with `HIG_RANDOM_CASES`,
`HIG_AUDIT_FUZZ`; the 10^6-case fma tests and the 1500-vertex planted game
skipped). Miri perturbs, on purpose, the float operations Rust does not
promise to be exact (`powi`, `exp`, ...): the core uses none (`+ - * /`,
`sqrt` and its fma only), and the tests build their expected values without
them, so the pins hold under Miri bit for bit. Miri needs rustup's nightly:
`cargo +nightly miri test --target x86_64-unknown-linux-gnu --test pins`
runs the x86_64 path (the soft fma, since Miri reports no CPU feature) on
any host.

## Fuzzing

`native/rust/fuzz` is a cargo-fuzz crate (its own workspace and `Cargo.lock`) with
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
cd native/rust/fuzz
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

`detect_arrays` and `_audit_prepare_arrays` (the entry points of
`hedonic.arrays`) take no graph: a vertex count and two buffers of 64-bit
integers, edge e joining `first[e]` and `second[e]` (igraph's order, the
smaller end first, which `hedonic.arrays` writes), the core reading them as
the capsule's two vectors (`Edges::Split`); `detect_arrays` takes the start
as CSR buffers (`start_offsets`, n + 1 entries, and `start_labels`), checked
as `detect` checks its rows (`start_rows_from_csr` shares `read_start_rows`'
dedup, cap and contiguity checks and messages). Everything else is
`detect`'s (`run_detect`: options, `prepare`, the visits without the GIL for
an exact `random.Random`, `finish`, the result lists). The buffers are held
for the whole call (an exporter keeps an exported buffer in place) and
sliced, like `partition()`'s, only in `prepare` and `finish`; nothing
depends on igraph's struct layout, the pointer width or the GIL.
`tests/test_arrays.py` checks them against `Game`, bit for bit, and their
own argument checks. A second core without them (2.0's C core under
`HEDONIC_CORE=c`) is given an edge-list object instead, read through
`get_edgelist()`.

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
`tools/native_tests.py` passed (35 tests),
`tests/test_native_threads.py` passed, and `native/rust/tests/test_differential.py`
passed against the C core (6 tests); `_edges_borrowed` is True with the GIL
and False without it.

## Errors

The functions raise what the C binding raises, type and message. Arguments
C parses as `n`, `L`, `d` and `p` go through `CArg` (PyNumber_Index and
PyLong_AsSsize_t, PyLong_AsLongLong, PyFloat_AsDouble, truth), not PyO3's
typed parameters, whose errors carry an `argument 'name': ` prefix and whose
i64 overflows say "C long"; integers read from sequences use
PyLong_AsLongLong too (`long_long`). `native/rust/tests/test_differential.py`
compares the plain functions' errors in one process;
`native/rust/tests/test_error_parity.py` runs 118 calls of `Game.community_hedonic`,
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
cd native/rust
cargo test --release                              # the bit pins and the C core's tests (test_core.c, test_audit.c), soft fma vs the instruction
cargo test --release --features force-soft-fma    # the same with the soft fma everywhere
cargo test                                        # debug: every debug_assert of the unchecked paths
cargo clippy --all-targets --features python -- -D warnings
cd ../..
pip install -e ".[cli]" pytest                    # maturin builds the crate into hedonic._core (or: maturin develop --release)
python tools/native_tests.py                      # the pinned Python tests (HIG_PINNED=1, no skips)
python native/check_libm.py "$(python -c 'import hedonic._core as c; print(c.__file__)')"
python -m pytest -q tests/test_native_threads.py tests/test_graph_edges.py   # threads, igraph's edges
python -m pytest -q                               # the whole suite
```

`cargo run --release --example profile_cover -- EDGES GAMMA CAP REPS [w]` is a
profiling loop over an edge-list file (`a b` lines, or `a b w` with `w`);
`ab_prefetch` times prefetching off and on in one process and checks the
results agree; `phases` times the input check, the adjacency build and the
partition apart; `bench_core` times the engines and the audit on fixed
synthetic graphs (what `bench/ab_unsafe.py` compares).

## The Rust API

Without its `python` feature (the default: maturin turns it on) the crate is
a plain Rust library with no dependency, and `cargo build --no-default-features`
builds it. The engines take slices and return owned results:

| Item | What |
|---|---|
| `graph::Graph::new(n, Edges::Flat(&[u0, v0, u1, v1, ..]) or Edges::Split(&us, &vs), edge_weights, node_weights)` | the graph, borrowed; checked against its counts |
| `graph::Options { resolution, max_memberships, max_total_communities, n_communities, allow_isolation }` | `Options::default()`: gamma 1, a partition, no limits, isolation allowed |
| `partition::partition(&graph, &options, start, &mut callbacks, None)` | `PartitionResult { membership, count, quality, raw_slack, slack }` |
| `cover::cover(&graph, &options, start, &mut callbacks, None)` | `CoverResult { cover: Cover { offset, labels }, count, quality, raw_slack, slack }` |
| `audit::AuditState::prepare(&graph, &offset, &labels, interrupt)` | the fixed-cover audit; `.intervals(&AuditOptions, Selection, interrupt)` and `.regrets(..)` |
| `queue::Callbacks::new(&mut rng)` | the random source (`rng::Rng`: 32-bit words) and an optional interruption check |
| `rng::Mt19937::python_seed(seed)`, `rng::Pcg32::new(seed, stream)` | Python's `random.Random(seed)` (for a seed >= 0), and PCG32 |
| `error::Error { kind, message }` | `Kind::Inval`, `NoMem`, `Overflow`, `Interrupted`, `Callback` (the random source failed) |

The last argument of the engines is a `trace::Trace` (tests). The Python
defaults are not the crate's: `hedonic` passes `allow_isolation = false` and,
for `resolution=None`, the weighted density. `examples/karate.rs` runs a
partition, a cover and their audit on Zachary's karate club and prints the
bits `hedonic.arrays.detect(34, edges, resolution=0.1, seed=0)` returns:

```bash
cargo run --release --no-default-features --example karate
```

**What a crates.io release would need** (nothing here is published:
`publish = false`):

- **Its own name and a narrow surface.** Every module is `pub` today, the
  engines' internals included (`arith`, `fma`, `heap`, `queue::Queue`,
  `prefetch`, `verify`): a release would re-export `Graph`, `Edges`,
  `Options`, `partition`, `cover`, `Cover`, the audit, `Rng` and `Error` at the
  root and make the rest `pub(crate)` or `#[doc(hidden)]`, so its semver
  covers only those. `Id` is `i32`: graphs have fewer than 2^31 vertices.
- **One crate per artifact.** `crate-type = ["cdylib", "rlib"]` builds a
  shared library for every dependent; a release splits it into the core
  (`rlib`, no PyO3) and the CPython module (`cdylib`, depending on it), which
  maturin builds.
- **Metadata**: `repository`, `readme`, `documentation`, `keywords`,
  `categories`, `rust-version` (edition 2021), an `include` list without
  `bench/`, `tests/test_differential.py` and `pyproject.toml`, and crate-level
  docs with a doctest (the karate example).
- **The source digest.** `build.rs` hashes `src/`, `build.rs` and
  `Cargo.toml` into `SOURCE_SHA256`, and `cargo package` rewrites
  `Cargo.toml`, so a packaged crate names other sources than this checkout:
  records that name the core by that digest would need the package's digest
  (or an alias) as well.
- **Each of these edits changes the source digest** (any byte of `src/`,
  `build.rs` or `Cargo.toml`), so it belongs in a release that changes the
  digest anyway, or comes with an entry in `runtime.EQUIVALENT_CORES`.

## Platforms

hedonic 2.1 ships this crate as `hedonic._core` (`pyproject.toml`, maturin).
`.github/workflows/wheels.yml` builds it with cibuildwheel
(`.github/cibuildwheel.toml`) for CPython 3.12, 3.13, 3.14 and 3.14t
(free-threaded) and tests every wheel where it was built. Three checks prove
the bits there:

- `tools/core_pins.py` loads `hedonic._core` from the installed wheel without
  the package (so without igraph or NumPy) and runs the pins of
  `tests/pins.rs` (test_core.c's) through `partition()` and `cover()` over
  array buffers, with the generator run natively and called from Python; next
  to a checkout it also checks that `source_sha256` is this crate's digest.
- `tools/native_tests.py`: the pinned Python tests (HIG_PINNED=1, no skips).
- the library's suite (`tests/` without `tests/experiments`, which needs the
  cli extra; the sdist and verify jobs run the whole suite).

| Platform | Wheels | Tested in CI with |
|---|---|---|
| Linux x86_64, aarch64 (manylinux_2_28, musllinux_1_2) | cp312-cp314, cp314t | pins, pinned tests, suite (cp314t: core pins) |
| Linux i686 (manylinux_2_28) | cp312-cp314, cp314t | core pins in the wheel's container; pinned tests and suite of the cp314 wheel in Debian i386 (`foreign`) |
| Linux ppc64le, s390x, armv7l, riscv64 (manylinux, musllinux), under QEMU | cp312-cp314, cp314t | core pins in the wheel's container; pinned tests and suite of the cp314 manylinux wheel in Debian (`foreign`) |
| macOS arm64, x86_64 (11.0+; x86_64 under Rosetta 2) | cp312-cp314, cp314t | pins, pinned tests, suite (cp314t: core pins) |
| Windows AMD64, ARM64 | cp312-cp314, cp314t | pins, pinned tests, suite (cp314t: core pins) |
| Windows x86 (win32) | cp312-cp314, cp314t | pins, pinned tests, suite less the tests that score with SciPy (no win32 SciPy) |
| Pyodide 314 (wasm32-emscripten) | cp314 | pins, pinned tests (Pyodide's igraph 1.0 and NumPy) |
| WASI (wasm32-wasip1) | none (no CPython wheels) | `cargo test` under wasmtime |

The crate's own tests (`cargo test`, release and debug) run natively on
Linux x86_64 and aarch64, macOS (and x86_64 under Rosetta 2), Windows x64,
x86 and ARM64; cross-compiled under qemu-user on s390x, ppc64le, armv7,
riscv64 and i686 (`tools/cargo_cross_test.sh`); and on wasm32-wasip1.

Why each choice:

- **cibuildwheel, not maturin-action.** cibuildwheel tests each wheel in the
  environment it was built for (its manylinux or musllinux container, Rosetta
  for macOS x86_64, the ARM64 Windows runner, Pyodide's runtime) with one
  test command; maturin-action only builds, so every one of those
  environments would need a hand-written install-and-test job. It is also the
  C wheels' tool (wheels.yml), so the publish and verify jobs carry over. The
  price: each Linux container installs Rust (one `before-all` line), each
  CPython version builds the crate again (see abi3 below), and the QEMU
  architectures compile under emulation (slow: one job per architecture and
  libc, side by side). maturin-action would cross-compile them quickly, but
  their tests need emulation either way.
- **32-bit x86 is in.** The C wheels skip it because C compilers for i386
  compute doubles in x87 registers (extended precision, then double
  rounding). Rust's i686 targets require SSE2 and compute every `f64`
  operation there, so the bits are the platform's IEEE doubles: the cargo
  pins pass unchanged on i686 Linux under qemu-user. win32
  (i686-pc-windows-msvc, SSE2 as well) is checked by the workflow's `core`
  and `wheels` jobs.
- **Big-endian.** The engines make no byte-order assumption: on s390x the
  cargo pins pass unchanged. The binding had one: a buffer declared in the
  other byte order (`<q` on a big-endian machine) was accepted and read
  wrong; it now accepts only the target's own prefix. The generator's state
  read in place is checked against `getstate()` at first use on every
  platform, s390x included.
- **32-bit igraph.** igraph's integers are as wide as a pointer, so on 32-bit
  targets (i686, armv7l, win32, wasm32) the head of `igraph_t` is not the one
  the capsule fast path reads; there the binding copies the edges through
  `get_edgelist()` (the same edges).
- **Where igraph has no wheel** (32-bit Linux, ppc64le, s390x, armv7l,
  riscv64; NumPy neither) the wheel is tested without its dependencies by
  `core_pins.py`, and the `foreign` job installs the cp314 manylinux wheels
  into Debian forky's Python 3.14 next to Debian's python3-igraph 1.0,
  python3-numpy and python3-scipy (built for all these architectures) and
  runs the pinned tests and the suite there, under QEMU.
- **musllinux** uses rustup's toolchain where rustup has a musl host
  (x86_64, aarch64, ppc64le) and Alpine's `cargo` elsewhere (s390x, armv7l,
  riscv64; the crate builds with Rust 1.87, Alpine 3.22's). musllinux i686
  has neither and is skipped.
- **Free-threaded (cp314t).** Built everywhere; tested with `core_pins.py`
  only, because igraph has no free-threaded wheel and its sdist builds only
  against the limited API, which free-threaded CPython lacks (igraph 1.0.0
  fails to compile on 3.14t). The module does not yet declare
  `gil_used = false`, so importing it re-enables the GIL with a
  RuntimeWarning (`core_pins.py` reports it); the generator's in-place path
  is off there by design (the tuple path is used). cp313t is not built:
  cibuildwheel 4 dropped it.
- **Pyodide** before 314.0 ships igraph 0.11, below hedonic's `igraph>=1.0`:
  only cp314 is built. Its Rust is Pyodide's (`pyodide config get
  rust_toolchain`, 1.93.0 for 314.0.7, target wasm32-unknown-emscripten).
- **Not built:** CPython 3.15 and 3.15t (pyo3 0.27 supports up to 3.14; the
  config lists the versions, since cibuildwheel 4.3 builds 3.15 by default),
  PyPy and GraalPy (NumPy publishes no PyPy wheels), iOS and Android (no
  igraph), loongarch64 (no cibuildwheel image).

Releases: `tools/release_version.py` rewrites `[project] version` in
pyproject.toml, which maturin reads for the sdist and every wheel
(a tag `v2.1.0rc1` builds `hedonic-2.1.0rc1.tar.gz` and its wheels). The
sdist carries `native/rust/` byte for byte (Cargo.toml states `readme`, which
maturin's sdist would otherwise add), so a wheel built from it names the same
core (`source_sha256`).

## abi3

Not used: the wheels are built per CPython version. An abi3 build (pyo3's
`abi3-py312`, one wheel for 3.12 on) computes the same bits but was not shown
to cost under 1%, and it would save little.

Measured on macOS arm64 (M-series, CPython 3.12, a shared machine at load 5
to 10). Each figure is a seeded random-start detection through
`Game.community_hedonic(seed=s, max_communities=k)`, timed per call. The
variants run alternately, one process per round, and every variant gave the
same memberships in every round:

| Graph | Calls per round | Default | abi3, first run (4 rounds) | abi3, second run (6 rounds) |
|---|---|---|---|---|
| Zachary | 400 | 0.041 ms | +2.2% | +2.4% |
| Barabási 1k, m=4 | 100 | 0.64 ms | -0.2% | -0.1% |
| Barabási 10k | 10 | 6.8 ms | +2.3% | -2.1% |
| Barabási 100k | 2 | 87 ms | +5.0% | -3.6% |

The abi3 columns compare the fastest round with the default build's. Medians
over the rounds agree: +0.2% to +2.8% on every graph except Barabási 100k in
the first run (+9.6%, under load).

- **The cost.** It is a steady ~2% (about 1 µs) per call on small graphs: the
  limited API reaches reference counts and tuple items through function calls
  where the full API uses macros. On large graphs it is lost in a noise of
  ±4%.
- **The saving.** It would be small:
  - one wheel for cp312-cp314 per platform, against three;
  - free-threaded CPython has no limited API before 3.15, so cp314t is built
    per version anyway;
  - the core reads `random.Random`'s state in place, which the limited API
    does not promise. The first-use probe against `getstate()` keeps the bits
    either way, but the read relies on more than abi3 states.

A build that saves CI minutes is not worth a measurable per-call cost and a
weaker claim about compatibility. Worth revisiting with abi3t (CPython 3.15),
which would cover the free-threaded builds too.

To reproduce:
- add `abi3 = ["python", "pyo3/abi3-py312"]` to `[features]`;
- in `src/python/args.rs`, replace `PyTuple_GET_ITEM` and `PyTuple_SET_ITEM`
  (not in the limited API) with `PyTuple_GetItem` and `PyTuple_SetItem`;
- build both variants with `maturin build --release` (abi3:
  `--features abi3`);
- time them alternately in separate processes.
