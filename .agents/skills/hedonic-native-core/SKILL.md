---
name: hedonic-native-core
description: Develop, verify, measure and release the C core that hedonic compiles into hedonic._core (native/, setup.py, hedonic._native, the wheel workflow), on stock igraph, in either release line - hedonic 1 (partitions only, release/1.x) or hedonic 2 (partitions and covers, release/2.0.0). Covers the arm64 venv and the in-place rebuild, checking that the loaded extension matches the checkout, the C tests (sanitized, -O3, soft fma, x86_64 under Rosetta, libm), the pinned Python tests, the bit-level fingerprint of the detector, keeping the two lines' shared engines byte-identical, A/B timing and memory under a memory guard, how a source change invalidates run records, the cross-platform wheel CI and the tag-driven TestPyPI/PyPI release. Local by default: never pushes, tags or publishes. Use for any change to native/, setup.py or the wheel and release workflow, for performance work on the detector, or to check that a build gives the same results as another.
---

# Hedonic native core

`hedonic` carries its own C core in `native/`, compiled by `setup.py` into
`hedonic._core`, next to stock `igraph` from PyPI. `native/README.md` is the
design (the game, a visit, certified moves, the slack, determinism and
platforms); this skill is how to work on it.

| Need | Go to |
|---|---|
| What must never happen | [Boundary](#boundary) |
| The two release lines | [Lines](#the-two-lines) |
| Everything on one screen | [Quick start](#quick-start) |
| Rules for editing C | [Writing C here](#writing-c-here) |
| Proving a change keeps the results | [Verification](#verification) |
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
  `scripts/guard.sh`, one job at a time, at most 6 build jobs, never next to a
  timed run of the user's.
* Deleting a PyPI release or version is the user's alone, and permanent (see
  [CI and release](#ci-and-release)).

## The two lines

| | hedonic 1 | hedonic 2 |
|---|---|---|
| branch | `release/1.x` | `release/2.0.0` (local working branch may be named otherwise) |
| version in `pyproject.toml` | `1.0.0` | `2.0.0` |
| detects | partitions (`max_memberships=1`; anything else is refused) | partitions and covers |
| native extra | - | `core/cover/`, `core/check/audit.c`, `binding/audit.c`, `tests/test_audit.c` |
| experiments | the disjoint paper (SBM benchmark) | both papers |

The partition path is the same code in both lines: `core/shared/`,
`core/partition/`, `core/softfma.h`, `core/dispatch.h`, `unity_fma.c`,
`setup.py`, `tools/native_tests.py` and the binding's `graph_edges.c`,
`random.c` and `start_rows.c` are byte-identical. The other files differ only
by what covers need (declarations, branches, methods). Rules:

* A change to a shared file lands in **both** lines, as the same bytes. Check
  with `diff -r ../hedonic-1.x/native/core/shared native/core/shared` (and
  `partition`, and the files above) before committing either side.
* Both lines must print the same `partitions` line of `scripts/bitcheck.py`
  (see [Verification](#verification)).
* Cover-only work (engine, audit, overlap experiments) is hedonic 2 only.

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
uv pip install --python "$V/bin/python" -e "$WT[cli]" pytest setuptools
"$V/bin/python" -c "import platform; print(platform.machine())"     # arm64

# 2. After every C edit: rebuild in place, check the build, run the C tests.
cd "$WT"
"$V/bin/python" setup.py build_ext --inplace --force
"$V/bin/python" .agents/skills/hedonic-native-core/scripts/fresh.py  # exit 0 = built from this checkout
make -C native test test-release test-soft-fma test-x86_64

# 3. Python: pinned tests (fail on any skip), then the whole suite.
"$V/bin/python" tools/native_tests.py
"$V/bin/python" -m pytest -q -p no:cacheprovider tests

# 4. Bit-level fingerprint, compared with the base build (same line, or the other line).
"$V/bin/python" .agents/skills/hedonic-native-core/scripts/bitcheck.py [--covers]
```

Run the venv's python directly. `uv run` syncs the project first, can reach
for the network, and creates a stray `.venv` in the worktree.

## Writing C here

* **One step per file**, one folder per engine (`core/shared/`,
  `core/partition/`, `core/cover/`, `core/check/`); each header begins with
  the map of its files. A new core file goes into `native/unity.c` (the
  extension compiles only `unity.c`, `unity_fma.c`, `dispatch.c` and the
  binding) and is then covered by `setup.py`'s `SOURCE_GLOBS` and the
  Makefile's `core/*/*.c`. A new binding file is picked up by `setup.py`'s
  glob; declare it in `binding/binding.h`.
* **Rounding is fixed by the source.** `-ffp-contract=off` (MSVC
  `/fp:strict`); a fused multiply-add is written as `hig_fma`, which is the
  instruction where the target has it and the correctly rounded soft fma
  (`softfma.h`) elsewhere; on x86_64 without a baseline FMA both builds are
  compiled and `dispatch.c` picks one from CPUID. The core calls no libm
  function but `sqrt` and the exact `fabs` (`make check-libm`). Never add
  `-ffast-math`, `-march=native`, `-Ofast`, or reassociate a sum.
* **Sequence anything with side effects.** C leaves the order of evaluation
  of function arguments and operands unspecified: two random draws in one
  expression (`ldexp(draw(), -draw())`, `draw() * (next() & 1 ? 1 : -1)`)
  gave different results under GCC x86_64 and MSVC. One draw per statement.
* **Portable C11.** No VLAs, no POSIX headers in the core, `int64_t` rather
  than `long` (32 bits on Windows), `__builtin_*` only under a GCC/Clang
  guard. MSVC rejects `INFINITY`/`NAN` in static initializers (C2099): use
  bit patterns.
* **Errors and cleanup.** Every fallible function returns an error code; a
  workspace is a zero-initialized struct with an idempotent destroy.
* **Debug knobs from the environment** count as set when present, even
  empty (`getenv` returns `""`): unset them, don't empty them, before a
  baseline.
* **Trajectories may change** only when intended and recorded: a changed
  pin is updated in C (`tests/test_core.c`) and Python
  (`tests/test_native_core.py`, `tests/test_native_backend.py`) together, and
  the commit says why. Every such change also changes the run identity.

## Verification

Cheapest first; run all of them before committing a change to `native/`, and
say in the commit message what ran.

1. **The build is the checkout's.** `scripts/fresh.py` compares
   `hedonic._core.source_sha256` (compiled in by `setup.py`) with the digest
   of the checkout's sources now. setuptools' incremental build once skipped
   a rebuild after edits to the included engine files; `--force` and the
   `depends=` list prevent it, and `fresh.py` proves it.
2. **C tests**: `make -C native test test-release test-soft-fma test-x86_64`.
   `test` compiles every file separately under ASan/UBSan and runs the libm
   check; on a dispatch target each build is also forced (`HIG_TEST_FMA`).
   `test-x86_64` runs everything again as x86_64 under Rosetta.
3. **Pinned Python tests**: `tools/native_tests.py` sets `HIG_PINNED=1` and
   fails on any skip. Without `HIG_PINNED` the pins are skipped and passing
   tests prove nothing about bits.
4. **Fingerprint**: `scripts/bitcheck.py` runs a fixed grid of 5616 partition
   calls (26 graphs x 6 resolutions x 2 isolation policies x 6 option sets x 3
   seeds; seconds) and prints `partitions 5616 <sha256>` over memberships,
   quality bits and slack; `--covers` adds the same grid at M = 2 and 4
   (hedonic 2). `--records FILE` writes one line per call, so `diff` names
   the first differing call. Expected for a refactor: the base's lines.
   Reference, both lines at hedonic 1.0.0 / 2.0.0:
   `partitions 5616 7d8e46424fe4c2d3d88304a2d9d7de1d2875a2b6c4d199643340ef9454a79546`.
   Never change the grid; add another one.
5. **Whole suite**, and for a release candidate a **fresh install**: build
   the wheel, install it into a new venv **without setuptools**, and run the
   tests from a copy that has `tests/`, `tools/` and `pyproject.toml`
   (the version test reads it; the source-digest test skips without
   setuptools, as in the CI's wheel tests).

To compare with an older build, extract it and point `PYTHONPATH` at it:

```bash
OLD=~/tmp/old-<rev>; mkdir -p $OLD
git archive <rev> src native setup.py pyproject.toml | tar -x -C $OLD
(cd $OLD && "$V/bin/python" setup.py build_ext --inplace -q)
PYTHONPATH=$OLD/src "$V/bin/python" .agents/skills/hedonic-native-core/scripts/bitcheck.py
```

The script prints the extension's path and digest on stderr: check them on
each side.

## Identity and run records

Records name the build by `runtime.build_identity()`: the release series
(`1.0`, `2.0`) and `hedonic_core_source_sha256`, the digest of every file in
`setup.py`'s `SOURCE_GLOBS` (path, length and bytes, CRLF as LF). Since the
rounding is fixed by the source, the source digest, not the compiled bytes,
is what results depend on.

* **Any byte** of any file in `SOURCE_GLOBS` changes the digest, a comment
  included. A disjoint store, a benchmark unit or a calibration recorded
  under another digest no longer counts as done for the new build: a running
  queue keeps its build, and its records must be redone on the new one.
  Batch native edits; do not polish comments while the user's runs depend on
  the build.
* **The disjoint store has a second fingerprint**, of behaviour:
  `hedonic.experiments.fingerprint` runs 8,736 partition calls on the loaded
  build (`partitions-v1:0668d101...`, pinned in
  `tests/experiments/test_fingerprint.py`) and the store keys Hedonic's rows
  on it, so the two lines (different sources, same partitions) share one
  store. Each shard records the source that wrote it; before a run keeps rows
  of another source it replays a sample of them and refuses on any
  difference. If a change of yours moves the fingerprint, every store's
  Hedonic rows are recomputed: say so, and update the pin in the same commit.
* Overlap records (hedonic 2) keep the source identity: there a source change
  always means new records.
* `SOURCE_GLOBS` must cover every C file the extension compiles or includes:
  it once missed the engines under `native/core/*/`, so editing an engine did
  not change the digest. `tests/experiments/test_runs_build_identity.py`
  checks it against every `.c`/`.h` in `native/` outside `tests/`.
* Timing is a result: a store or benchmark run on an x86_64 (Rosetta) venv
  is not comparable with arm64 timings; keep such stores apart.

## Performance and memory

* **A/B in one venv** with the older build on `PYTHONPATH` (above). Check
  `hedonic._core.__file__` on each side. Alternate old and new for at least
  3-4 rounds, compare the **minimum** CPU seconds, and run both orders when
  the gap is within a few percent. Report ratios from the same start state,
  not single timings (a commit message once claimed 87 s -> 31 s where the
  same-state comparison was 38 s -> 31 s).
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

`.github/workflows/wheels.yml` runs on pushes to `main` and `release/**`, on
`v*` tags and on pull requests:

* `c_core` / `c_core_msvc`: `make test test-release test-soft-fma
  test-x86_64` under GCC and Clang (Linux x86_64 and arm64, macOS), and MSVC
  x64/ARM64 (release, as the wheel compiles it) with the libm check;
* `sdist`: builds the sdist, a wheel from it, and tests it;
* `wheels`: cibuildwheel, CPython 3.12-3.14, Linux x86_64/aarch64 (manylinux
  and musllinux), macOS, Windows AMD64/ARM64; every job runs the C tests and
  `tools/native_tests.py` against the installed wheel. Free-threaded builds
  and 32-bit x86 (x87 rounding) are skipped. A wheel's test installs only the
  library's dependencies: when the experiment dependencies were regular
  dependencies, musllinux tests built duckdb from source and ran for hours.
  Keep heavy dependencies in extras and tests independent of build tools
  (setuptools is absent there: dev9 failed on a test that needed it);
* `publish` (tags only, after every job) and `verify` (installs the
  uploaded version from its index on each platform and runs the pinned tests
  and the suite).

Releases are tags; `tools/release_version.py` checks the tag against
`pyproject.toml`'s version and writes the tag's version into the build:

| tag | version | index |
|---|---|---|
| `vX.Y.Z.devN` | trial | TestPyPI (environment `testpypi`) |
| `vX.Y.ZrcN` | candidate (`pip --pre`) | PyPI |
| `vX.Y.Z` | release | PyPI |

Try `.devN` on TestPyPI, then `rcN`, and tag the release on the commit of the
last good candidate, so candidate and release are the same bytes. Give the
user the push and tag commands; never push a tag yourself (a local tag like
`v2.0.0` may exist only to mark a commit).

PyPI never lets a version number be reused, even after the release is
deleted: to withdraw one, **yank** it. Taken on PyPI: 0.0.4, 0.0.6-0.0.10,
0.1.0, 0.1.1 and 1.5.0 (the igraph-fork releases). Check
`https://pypi.org/pypi/hedonic/json` before choosing a new number.

Locally: `make -C native CC=gcc-16 test` (Homebrew GCC); `setup.py` cannot use
gcc on macOS (Python's CFLAGS hold clang-only flags), so compile the
extension by hand if a GCC build must be tried in Python.

## Pitfalls observed

| Symptom | Cause | Fix |
|---|---|---|
| A C edit has no effect, or results look like the old build | stale `_core.*.so`: setuptools skipped the rebuild, or the editable install never recompiles | `setup.py build_ext --inplace --force`, then `scripts/fresh.py` |
| Wheels or timings are x86_64 | uv's default 3.12 is x86_64 (Rosetta) | `uv venv -p cpython-3.12-macos-aarch64-none`; check `platform.machine()` |
| A store refuses every method of another venv | the venvs' igraph extensions differ (one x86_64) | the same arm64 venv recipe for both |
| `hedonic._core.core_version` names the other line | `HIG_VERSION_STRING` in `hedonic_core.h` not bumped | it is the release (`1.0.0`, `2.0.0`); `test_native_backend` checks it |
| `import hedonic` picks the wrong build | `PYTHONPATH` left from an A/B run, or another venv | `fresh.py` / the paths `bitcheck.py` prints |
| Pins "pass" but nothing was checked | `HIG_PINNED` unset | `tools/native_tests.py` |
| Result differs on GCC x86_64 or MSVC only | unsequenced side effects (two draws in one expression) | one draw per statement |
| MSVC C2099 | `INFINITY`/`NAN` in a static initializer | bit patterns |
| A refactor changes a quality bit | contraction or reassociation | explicit `hig_fma`, original operation order |
| Editing an engine did not change the digest | `SOURCE_GLOBS` missed a folder | extend it; the build-identity test checks coverage |
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
per file; which line(s) it touches and whether the shared files still match;
the checks that ran and their results (fresh build, C tests, pinned tests,
the bitcheck lines, the suite); A/B ratios when performance was the point;
whether the source digest changed (and so which runs must be redone); what
was not checked (e.g. CI, because nothing was pushed); and the user's next
decision (push, tag, which index).
