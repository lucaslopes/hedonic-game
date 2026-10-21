---
name: hedonic-native-core
description: Develop, verify, measure and release the Rust core that hedonic compiles into hedonic._core (native/rust, the maturin build in pyproject.toml, hedonic._native, the wheel and release workflows), on stock igraph, in hedonic 2.1 (release/2.1, the Rust port of hedonic 2.0's C core, bit for bit). Covers the arm64 venv and maturin develop, checking that the loaded extension matches the checkout, the cargo tests (release, soft fma, debug, x86_64 under Rosetta), the libm check, the pinned Python tests, the bit-level fingerprint of the detector (the same as hedonic 1 and 2.0's C core), a second core beside it selected with HEDONIC_CORE, A/B timing and memory under a memory guard, how a source change invalidates run records, the cross-platform CI and the tag-driven TestPyPI/PyPI release. Local by default: never pushes, tags or publishes. Use for any change to native/, the build or the wheel and release workflows, for performance work on the detector, or to check that a build gives the same results as another.
---

# Hedonic native core

`hedonic` carries its own core in `native/rust`, a Rust crate that maturin
compiles into `hedonic._core`, next to stock `igraph` from PyPI.
`native/README.md` is the design (the game, a visit, certified moves, the
slack, determinism and platforms), `native/rust/README.md` the crate (what
each module ports from hedonic 2.0's C core, what is unsafe and why, what is
faster); this skill is how to work on it.

| Need | Go to |
|---|---|
| What must never happen | [Boundary](#boundary) |
| The release lines | [Lines](#the-lines) |
| Everything on one screen | [Quick start](#quick-start) |
| Rules for editing the crate | [Writing Rust here](#writing-rust-here) |
| Proving a change keeps the results | [Verification](#verification) |
| Another core beside this one | [A second core](#a-second-core-beside-it) |
| What a source change does to run records | [Identity](#identity-and-run-records) |
| Measuring speed and memory | [Performance](#performance-and-memory) |
| Wheels, tags, TestPyPI, PyPI | [CI and release](#ci-and-release) |
| Symptom -> cause -> fix | [Pitfalls](#pitfalls-observed) |

## Boundary

* No `git push`, tag, release, pull request or upload unless the user asks
  for that exact step in this session. Pushes are the user's: the permission
  classifier blocks them, so give the command instead.
* No destructive git (`reset --hard`, `checkout --`/`restore` of uncommitted
  work, `clean`, `rebase`, `stash`, force-push), and never in the user's main
  checkout (`~/Code/community/hedonic`): work in a worktree under
  `~/Code/community/opus-worktrees/`.
* `main` is public, `paper` is private and is never pushed or moved by you.
  A branch that descends from `paper` must not reach `origin`.
* Venvs that other sessions or runs use are read-only (in particular the
  igraph-fork venv `~/dev/venvs/hedonic-game-824a2b09`). Make your own.
* Frozen evidence and historical locks are never rewritten; `~/Databases` is
  read-only except for outputs you created.
* Never touch live tmux sessions (`hedonic-*`): they are the user's runs.
* Commits carry no `Co-Authored-By` or other authorship trailers.
* Heavy work (complete SNAP graphs, parallel checks, timing loops) runs under
  `scripts/guard.sh`, one job at a time, at most 4 to 6 build jobs
  (`CARGO_BUILD_JOBS`), never next to a timed run of the user's.
* Deleting a PyPI release or version is the user's alone, and permanent (see
  [CI and release](#ci-and-release)).

## The lines

| | hedonic 1 | hedonic 2.0 | hedonic 2.1 |
|---|---|---|---|
| branch | `release/1.x` | `release/2.x` | `release/2.1` |
| core | C (`native/`, `setup.py`) | C (`native/`, `setup.py`) | Rust (`native/rust`, maturin) |
| detects | partitions only | partitions and covers | partitions and covers |

The three compute the same partitions, and 2.0 and 2.1 the same covers and
audits, bit for bit: the same `scripts/bitcheck.py` lines, the same pins.
A change to what the core computes in one line is a change of the algorithm
for all of them; a refactor or a speedup must keep every line of
[Verification](#verification) as it was. C work (hedonic 1, 2.0) follows the
C build of those branches (`make -C native test ...`, `setup.py`); this
skill's commands are 2.1's.

Older history: the igraph-fork releases (0.1, 1.5) computed with
`lucas-igraph`; 0.0.4 and 0.0.5 predate the C core. Their results are
not this core's, so they are not a reference for bit comparisons.

## Quick start

```bash
# 0. A worktree, never the main checkout.
WT=~/Code/community/opus-worktrees/hedonic-native-<topic>
git -C ~/Code/community/hedonic worktree add -b claude/<topic> "$WT" <base-ref>

# 1. An arm64 venv of its own (uv's default 3.12 here is x86_64 under Rosetta).
V=~/dev/venvs/hedonic-native-<topic>
uv venv -p cpython-3.12-macos-aarch64-none "$V"
uv pip install --python "$V/bin/python" -e "$WT[cli]" pytest maturin   # maturin builds the crate
"$V/bin/python" -c "import platform; print(platform.machine())"     # arm64

# 2. After every Rust edit: the cargo tests, then rebuild in place and check the build.
cd "$WT/native/rust"
cargo test --release && cargo test --release --features force-soft-fma && cargo test
cd "$WT"
VIRTUAL_ENV="$V" "$V/bin/maturin" develop --release
"$V/bin/python" .agents/skills/hedonic-native-core/scripts/fresh.py  # exit 0 = built from this checkout

# 3. Python: pinned tests (fail on any skip), the libm check, then the whole suite.
"$V/bin/python" tools/native_tests.py
"$V/bin/python" native/check_libm.py "$("$V/bin/python" -c 'import hedonic._core as c; print(c.__file__)')"
"$V/bin/python" -m pytest -q -p no:cacheprovider

# 4. Bit-level fingerprint, compared with the base build (or with the C core of 2.0).
"$V/bin/python" .agents/skills/hedonic-native-core/scripts/bitcheck.py --covers
```

Run the venv's python directly. `uv run` syncs the project first, can reach
for the network, and creates a stray `.venv` in the worktree. In zsh, `uv`
may be a shell function that picks another venv: use `command uv`.

## Writing Rust here

* **One module per step**, as `native/README.md`'s map lists them; a module's
  doc comment says what it does and which C file of hedonic 2.0 it ports.
  `build.rs` hashes every `src/**/*.rs`, so a new module is in the digest by
  itself.
* **Rounding is fixed by the source.** rustc never contracts `a * b + c`
  and has no fast-math. A fused multiply-add is written as `fma::fma`: the
  instruction on aarch64 (`f64::mul_add`, which is `fmadd` there), on x86_64
  the instruction when the CPU has it (detected once) else the soft fma, and
  the soft fma elsewhere. Never call `f64::mul_add` anywhere else (on a
  target without the instruction it is libm's `fma`), nor `exp`, `ln`,
  `powi`, `powf` or any other `f64` function but `sqrt`, `abs`, `min`,
  `max` and comparisons (`native/check_libm.py`). Never sum floats with an
  iterator's `sum()` or reorder a sum: write the loop in the order the C core
  summed. Never add `-C target-cpu=native`, target features or
  fast-math-like flags to the build.
* **Sequence anything with side effects.** Rust evaluates left to right, but
  keep one random draw per statement anyway, as the C core does (its order is
  the reference).
* **Unsafe** only where `native/rust/README.md` lists it (the visits' hot
  paths, the generator's state, the FMA instruction), with its invariant
  stated and debug-asserted where it can be; `cargo test` (debug) checks the
  assertions.
* **Errors, not panics.** Every fallible step returns `Result`; allocation
  goes through `error.rs`'s helpers, so running out of memory is an error the
  binding raises as `MemoryError`.
* **Debug knobs from the environment** count as set when present, even
  empty: unset them, don't empty them, before a baseline.
* **Format by hand**, about 120 columns, as the surrounding code: never run
  `cargo fmt` or rustfmt on the crate.
* **Trajectories may change** only when intended and recorded: a changed
  pin is updated in Rust (`tests/pins.rs`) and Python
  (`tests/test_native_core.py`, `tests/test_native_backend.py`) together, and
  the commit says why. Every such change also changes the run identity, and
  breaks the equality with hedonic 2.0's C core.

## Verification

Cheapest first; run all of them before committing a change to `native/`, and
say in the commit message what ran.

1. **The build is the checkout's.** `scripts/fresh.py` compares
   `hedonic._core.source_sha256` (embedded by `build.rs`) with the digest of
   the crate's sources now, and checks that the loaded module is
   `hedonic._core` itself (not a core selected with `HEDONIC_CORE`). An
   editable install never recompiles by itself: `maturin develop --release`
   after every edit, then `fresh.py`.
2. **Cargo tests** in `native/rust`: `cargo test --release` (the bit pins,
   random games against the verifier, the audit's windowed search against
   the exhaustive one, the soft fma against the instruction and its hash on
   10^6 cases), `--features force-soft-fma` (the same with the soft fma
   everywhere), and `cargo test` (debug assertions of the unchecked paths).
   On Apple Silicon, also as x86_64 under Rosetta: `rustup target add
   x86_64-apple-darwin` and `cargo test --release --target
   x86_64-apple-darwin` (with and without `force-soft-fma`).
3. **The libm check**: `native/check_libm.py` on the built extension.
4. **Pinned Python tests**: `tools/native_tests.py` sets `HIG_PINNED=1` and
   fails on any skip. Without `HIG_PINNED` the pins are skipped and passing
   tests prove nothing about bits.
5. **Fingerprint**: `scripts/bitcheck.py` runs a fixed grid of 5616 partition
   calls (26 graphs x 6 resolutions x 2 isolation policies x 6 option sets x 3
   seeds; seconds) and prints `partitions 5616 <sha256>` over memberships,
   quality bits and slack; `--covers` adds the same grid at M = 2 and 4.
   `--records FILE` writes one line per call, so `diff` names the first
   differing call. Expected for a refactor: the base's lines. Reference,
   hedonic 1.0.0, 2.0.0 (C) and 2.1.0 (Rust):
   `partitions 5616 7d8e46424fe4c2d3d88304a2d9d7de1d2875a2b6c4d199643340ef9454a79546`,
   and for 2.0.0 and 2.1.0
   `covers-M2 5616 62f4eed8fd04f5bbbaed864fa6921b5968bd7ca497c08a1ab8d8cb17c9c2cd3a`,
   `covers-M4 5616 16ce4b13d80a9bbfd5164780502666ad4e221c79ee750f8806d0d910689cbc9b`.
   Never change the grid; add another one.
6. **Whole suite**, and for a release candidate a **fresh install**: build
   the sdist and a wheel from it (`python -m build --sdist`, then `pip
   install dist/*.tar.gz`, which needs a Rust toolchain), install into a new
   venv and run the tests from a copy that has `tests/`, `tools/` and
   `pyproject.toml` (the version test reads it; the source-digest test skips
   without `native/rust`, as in the CI's wheel tests).

To compare with an older build, extract it and build it in its own venv
(two builds of `hedonic._core` cannot share one): `git archive <rev> | tar
-x -C ~/tmp/old-<rev>`, `uv venv` and `pip install` there, and run
`bitcheck.py` with each venv's python. The script prints the module, the
extension's path and its digest on stderr: check them on each side.

## A second core beside it

`src/hedonic/__init__.py` loads `hedonic._core_<name>` in place of
`hedonic._core` when `HEDONIC_CORE=<name>` is set, for every reader of the
package (`from hedonic import _core`, `hedonic._native`, the experiments);
the default core's own backend name (`rust`) selects the default, and a
name with no such module is an `ImportError` naming the variable. A second
core is any build of the core whose extension module is named
`hedonic._core_<name>` (its init function `PyInit__core_<name>`), placed in
the package folder beside `hedonic._core` (`src/hedonic/` for an editable
install). Then:

* `HEDONIC_CORE=<name> python tools/native_tests.py` and `HEDONIC_CORE=<name>
  python -m pytest` run the pins and the suite on it (the identity test
  checks only the default core's digest, and skips);
* with `hedonic._core_c` (hedonic 2.0's C core built that way),
  `native/rust/tests/test_differential.py` compares every result, trace,
  verifier report, audit answer and error of the two, bit for bit
  (`HIG_DIFF_CASES` sets the count), and `native/rust/bench/bench.py` times
  them interleaved; both skip (or stop) with the reason when it is absent,
  and need `HEDONIC_CORE` unset;
* `bitcheck.py` under `HEDONIC_CORE=<name>` fingerprints that core.

The package never names a particular second core: nothing on the public line
may require one.

## Identity and run records

Records name the build by `runtime.build_identity()`: the results series
(`runtime.RESULTS_SERIES`, `2.0`; not the package version, so 2.1 keeps 2.0's
records) and `hedonic_core_source_sha256`. That digest is the core's results
class, not its bytes: `runtime.CORE_CLASSES` maps the triple of the core's
fingerprints (`partitions-v1`, `covers-v1`, `audit-v1`,
`hedonic.experiments.fingerprint`) to the digest the class's records name, so
hedonic 2.0's C core and this Rust core, which compute the same results, are
one class. `hedonic._core.source_sha256`, the digest `build.rs` embeds (every
`src/**/*.rs`, `build.rs`, `Cargo.toml` and `Cargo.lock` of the crate:
path, length and bytes, CRLF as LF, in path order), stays as provenance
(`hedonic_core_built_source_sha256`) and is the name of a core whose triple
is not in the table.

* **A source change that keeps every fingerprint keeps every record**: a
  comment, a refactor, a faster search, another language. Check it with the
  pins of `tests/experiments/test_fingerprint.py` (seconds), then add the new
  digest to the class's `members` with the release and the evidence
  (`EQUIVALENT_CORES` follows from the table) and say so in the commit.
  `tests/experiments/test_core_identity.py` checks every member.
* **A change that moves a fingerprint changes results**: the core is a new
  class, names its own digest, and every overlap record, calibration and
  start-quality key is redone on it (a running queue keeps its build). Say so,
  re-pin in the same commit, and add the new triple to `CORE_CLASSES` when it
  ships. Batch such edits; do not change behaviour while the user's runs
  depend on the build.
* **The disjoint store keys Hedonic's rows on the partition fingerprint**
  alone, so hedonic 1, 2.0 and 2.1 (different sources, same partitions) share
  one store. Each shard records the source that wrote it; before a run keeps
  rows of another source it replays a sample of them and refuses on any
  difference. If a change of yours moves `partitions-v1`, every store's
  Hedonic rows are recomputed.
* **Overlap runs can be replayed**: `hedonic run check --replay N` reruns the
  cheapest Hedonic units and a vertex sample of the cover audits with the
  loaded core, and records a clean pass in the root's `adopted.json`. Use it
  to show a new core reproduces finished runs before relying on its class.
* `tests/experiments/test_runs_build_identity.py` recomputes the digest from
  the checkout and compares it with the loaded core's.
* Timing is a result: a store or benchmark run on an x86_64 (Rosetta) venv
  is not comparable with arm64 timings, nor a Rust run with a C one; keep
  such stores apart.

## Performance and memory

* **A/B**: the older build in its own venv (above), or two cores in one
  process with `bench/bench.py` (`hedonic._core` against `hedonic._core_c`).
  Check each side's `hedonic._core.__file__` and `source_sha256`. Alternate
  old and new for at least 3-4 rounds, compare the **minimum** CPU seconds,
  and run both orders when the gap is within a few percent. Report ratios
  from the same start state, not single timings (a commit message once
  claimed 87 s -> 31 s where the same-state comparison was 38 s -> 31 s).
  `cargo run --release --example profile_cover` and `phases` profile the
  engines without Python.
* **Visiting order matters more than micro-optimizations**: with a fixed
  order, complete YouTube at its density took 1,100 s; reshuffling the
  vertices still to settle at each rescan brought it under 5 s, where a
  pruned rescan only reached 372 s. Measure on complete SNAP graphs, not
  only small ones.
* **Memory on macOS**: freed blocks stay in the footprint until deferred
  reclaim, so RSS gaps of tens of MB can be timing. Compare the live heap
  (an interposed malloc tracker via `DYLD_INSERT_LIBRARIES`, with the venv's
  python: SIP strips `DYLD_*` from system binaries).
* **Guard**: `scripts/guard.sh LIMIT_GB CMD...` kills the whole process tree
  when its summed RSS passes the limit (exit 99). A leak once took the
  machine past 64 GB and rebooted it; parallel jobs once took 28 GB next to
  the user's runs. Never load full LiveJournal or larger unguarded.

## CI and release

`.github/workflows/rust.yml` runs the cargo tests (release, soft fma, debug;
x86_64 under Rosetta on macOS) on Linux x86_64/aarch64, macOS and Windows
AMD64/ARM64, then builds the package and runs the pinned tests and the libm
check, and the whole suite with the `cli` extra. `wheels.yml` (cibuildwheel,
`.github/cibuildwheel.toml`) builds the sdist (and a wheel from it, tested)
and the wheels: CPython 3.12-3.14 and 3.14t; Linux x86_64, i686, aarch64 and,
under QEMU, ppc64le, s390x (big-endian), armv7l, riscv64 (manylinux and
musllinux); macOS arm64 and x86_64; Windows AMD64, x86 and ARM64; Pyodide
cp314. 32-bit x86 is in: Rust's i686 targets compute f64 in SSE2, not x87.
Every wheel runs `tools/core_pins.py` (the bit pins through the extension
alone, no igraph or NumPy needed); where igraph has a wheel also
`tools/native_tests.py` and the library suite; `foreign` tests the cp314
wheels of the QEMU architectures and i686 in Debian with its python3-igraph.
`cargo test` also runs under qemu-user (`tools/cargo_cross_test.sh TARGET`)
and on wasm32-wasip1. Then `publish` (tags only, after every job) and
`verify` (installs the uploaded version from its index on each platform and
runs the pinned tests and the suite). A wheel's test
installs only the library's dependencies: keep heavy dependencies in extras
and tests independent of build tools.

Releases are tags; `tools/release_version.py` checks the tag against
`pyproject.toml`'s version and writes the tag's version into its `[project]`
table, where maturin reads it:

| tag | version | index |
|---|---|---|
| `vX.Y.Z.devN` | trial | TestPyPI (environment `testpypi`) |
| `vX.Y.ZrcN` | candidate (`pip --pre`) | PyPI |
| `vX.Y.Z` | release | PyPI |

Try `.devN` on TestPyPI, then `rcN`, and tag the release on the commit of the
last good candidate, so candidate and release are the same bytes. Give the
user the push and tag commands; never push a tag yourself (a local tag may
exist only to mark a commit).

PyPI never lets a version number be reused, even after the release is
deleted: to withdraw one, **yank** it. Taken on PyPI: 0.0.4, 0.0.6-0.0.10,
0.1.0, 0.1.1 and 1.5.0 (the igraph-fork releases), and the 2.0 releases.
Check `https://pypi.org/pypi/hedonic/json` before choosing a new number.

## Pitfalls observed

| Symptom | Cause | Fix |
|---|---|---|
| A Rust edit has no effect, or results look like the old build | the editable install never recompiles | `maturin develop --release`, then `scripts/fresh.py` |
| `fresh.py` fails with the right digest | `HEDONIC_CORE` is set: another core is loaded | unset it |
| Wheels or timings are x86_64 | uv's default 3.12 is x86_64 (Rosetta) | `uv venv -p cpython-3.12-macos-aarch64-none`; check `platform.machine()` |
| A store refuses every method of another venv | the venvs' igraph extensions differ (one x86_64) | the same arm64 venv recipe for both |
| `hedonic._core.core_version` names another release | `CORE_VERSION` in `src/lib.rs` not bumped | it is the release (`2.1.0`); `test_native_backend` checks it |
| `import hedonic` picks the wrong build | `PYTHONPATH` left from an A/B run, another venv, or `HEDONIC_CORE` | `fresh.py` / the module and path `bitcheck.py` prints |
| `ImportError: HEDONIC_CORE='x': no module hedonic._core_x` | the second core is not built beside `hedonic._core` | build it there, or unset `HEDONIC_CORE` |
| Pins "pass" but nothing was checked | `HIG_PINNED` unset | `tools/native_tests.py` |
| A result differs on x86_64 only | an `f64::mul_add` outside `fma.rs`, or a libm function | `fma::fma`; `native/check_libm.py` |
| A refactor changes a quality bit | a reordered or iterator sum, or a new fma | the C core's operation order; explicit `fma::fma` only where it was |
| `cargo test --locked` fails | `Cargo.lock` out of date (a version or dependency change) | update the lock in the same commit |
| A finished run's records count as missing | the source digest changed (any byte) | expected: rerun on the new build, or keep the old build for that run |
| A baseline "with the knob off" is still affected | `HIG_X=` (empty) is still set | `unset HIG_X` |
| CI wheel test fails on a missing module | the test needs a build or experiment tool | skip when absent; the wheel test env has only the library's deps |
| Tests from a copied tree fail on the version | the copy lacks `pyproject.toml` / `tools/` | copy them too |
| A stray `.venv` appears in the worktree | `uv run` synced the project | use the venv's python |
| `ig.Graph.SBM(n, pref, sizes)` TypeError | igraph 1.0's signature is `SBM(pref, sizes)` | build the edges yourself (as `bitcheck.py` does) |
| zsh: `bad substitution`, unsplit `$args` | zsh does not word-split | run scripts with bash, or `${=args}` |
| The machine runs out of memory | unguarded or concurrent heavy jobs | `guard.sh`, one job at a time |

## Completion report

End with: branch and commits (`git log --oneline base..HEAD`); what changed,
per file; the checks that ran and their results (fresh build, cargo tests,
the libm check, pinned tests, the bitcheck lines, the suite, the
differential test when a second core was built); A/B ratios when
performance was the point; whether the source digest changed (and so which
runs must be redone); what was not checked (e.g. CI, because nothing was
pushed); and the user's next decision (push, tag, which index).
