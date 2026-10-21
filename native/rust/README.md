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
