---
name: link-python-igraph
description: Build and qualify an isolated local development stack that links a C igraph worktree, a python-igraph worktree and a Hedonic worktree (lucas-igraph 1.0.0.N / hedonic 1.0.N), then investigate, change and verify the Leiden/hedonic implementation with snapshot builds, differential and build-to-build equivalence harnesses, sanitizers, complete-DBLP benchmarks and a decision ledger. Development mode is the default and never pushes, tags, publishes or touches global installations; production handoff is a separately authorized mode. Use for native igraph or python-igraph changes, roadmap investigations, performance work, or any task that must run Hedonic against a local igraph build.
---

# Link Python igraph

This skill is self-contained: it tells an agent how to set up the linked
stack, how to change and verify native code without fooling itself, what
went wrong the last time (the 1.0.0.5 investigation) and how to stop. The
bundled scripts are in [`scripts/`](scripts/); the detailed case study and
harness guide are in [`references/`](references/).

| Need | Go to |
|---|---|
| The whole stack on one screen | [Quick start](#quick-start-macos-arm64-verified-2026-09) |
| What must never happen | [Boundary](#purpose-and-default-boundary), [Safety rules](#safety-rules) |
| Step-by-step setup and why each flag exists | [Development workflow](#development-workflow) |
| Proving a change is (or is not) behaviour-preserving | [Verification toolkit](#verification-toolkit) |
| Attributing a change to one commit | [Snapshot builds](#snapshot-builds-and-attribution) |
| Symptom -> cause -> fix | [Pitfalls](#pitfalls-observed) |
| What was found last time | [references/lessons-1.0.0.5.md](references/lessons-1.0.0.5.md) |
| Harness details and output formats | [references/harnesses.md](references/harnesses.md) |

## Purpose and default boundary

This skill prepares a **local development** stack for coordinated work across
the native `igraph` C core, `python-igraph` (distributed as `lucas-igraph`)
and `hedonic-game`: investigating what is feasible, rebuilding repeatedly,
running tests and experiments, and preserving the original checkouts while an
agent works in isolated worktrees.

The default mode is **development**, which is local-only:

* no `git push`, GitHub release, GitHub Actions run, pull request, tag, PyPI
  upload or credential handling;
* no installation into `/usr`, `/usr/local`, Homebrew, Conda, a global Python
  or another shared prefix; no `sudo`; no edits to shell startup files;
* no published `lucas-igraph` wheel in place of the local worktree;
* no rewrite, reset, deletion or cleanup of the user's source worktrees, and
  no destructive git commands (`reset --hard`, `checkout --`, `clean`,
  force-push);
* no network access unless authorized: install Python dependencies from the
  local uv cache (`uv pip install --offline`) or stop and ask;
* the stopping point is clean development worktrees with local commits and a
  local verification report. Wait for an explicit go-ahead before any
  release or publication step.

Production mode is described only in [Gated production mode](#gated-production-mode).
It is never inferred from a request to build, test, compare or run
experiments. Pushing, tagging and publishing are the business of the
`sibling-release-pipeline` skill, which must be invoked explicitly.

## Role in the roadmap

This skill is the local qualification harness for the Hedonic/Leiden
roadmap (`docs/plans/ideal_hedonic_1.0.0.5_roadmap.md` and its successors).
The roadmap is a living investigation, not a list of features that must all
ship. Use the linked worktrees to test each hypothesis and decide whether to
**accept**, **redesign**, **defer**, **reject** or **supersede** it:

```text
unexamined -> prototyped -> qualified -> accepted
                         -> redesigned | deferred | rejected | superseded
```

* `prototyped` needs executable code or a test fixture;
* `qualified` needs the relevant tests, proofs, differentials, benchmarks or
  resource measurements;
* `accepted` means the API, compatibility scope and release boundary are
  explicit -- not that anything was published.

A finding that a feature should not be implemented is a successful result.
Choose the order by dependencies and information value; commit locally and
often, keep rejected experiments as local commits or patches, never squash
or publish. Record every decision in a ledger (see
[Evidence and the decision ledger](#evidence-and-the-decision-ledger)).

### Manuscript and evidence gate

Before prioritizing, read the current manuscript in both forms and follow its
numbers to the saved evidence:

```text
MANUSCRIPT_TEX=~/Code/community/hedonic-game/docs/papers/overlapping_communities/manuscript_template/adapted_hedonic_overlap_main.tex
MANUSCRIPT_PDF=~/Code/community/hedonic-game/docs/papers/overlapping_communities/manuscript_template/build/adapted_hedonic_overlap_main.pdf
MANUSCRIPT_DATA=~/Code/community/hedonic-game/docs/papers/overlapping_communities/manuscript_template/tuning
MANUSCRIPT_GENERATED=~/Code/community/hedonic-game/docs/papers/overlapping_communities/manuscript_template/generated
```

The `.tex` source is authoritative for protocol, claims, parameter semantics
and TODOs; the PDF is the reader-facing snapshot; `tuning/*.jsonl`,
`tuning/covers/`, `generated/` and the SNAP ledgers are the numeric layer.
`tuning/final.jsonl` holds the runs that wrote the saved covers
(`memberships_path`, detection and CPU seconds, audit).

Engineering baseline on complete com-DBLP (317,080 vertices): registered
HOC-local (`M=8`, gamma = density) about F1 0.392 / ONMI 0.496; the
reference-selected tuned local point (`M=64`, `c=7000`, gamma ~ 0.146) about
F1 0.425 / ONMI 0.519 -- optimistic, because it was selected against the
reference. Interrupted rows are statuses, not zero scores. Before accepting a
change, reproduce the nearest baseline **byte for byte** on the unmodified
build (the local 1.0.0.4 build reproduces `cand_m64_c7000_seed0` row for row),
record the manifest, and compare.

## Coordinated version identities

| Repository | Target for suffix `N=5` | Rule |
|---|---:|---|
| igraph C core | `1.0.0.5` | four-component upstream identity |
| python-igraph (`lucas-igraph`) | `1.0.0.5` | same identity; its `vendor/source/igraph` gitlink must name the exact released C commit |
| hedonic-game | `1.0.5` | three components, same suffix (never `0.1.N`, never `1.0.0.N`) |

During development, feature branches and temporary version strings are
fine (`IGRAPH_VERSION` file `1.0.0.5-dev`, `python-igraph/src/igraph/version.py`
`(1,0,0,5)`, `hedonic` `pyproject.toml` `1.0.5` with `lucas-igraph==1.0.0.5`).
Do **not** regenerate `uv.lock` or repoint the vendored gitlink before the
C commit is published; both are production steps.

## Quick start (macOS arm64, verified 2026-09)

```bash
# 0. Names. DEV_ROOT must be outside the original checkouts.
DEV_ROOT=~/Code/community/opus-worktrees
SKILL=~/Code/community/hedonic-game/.skills/link-python-igraph   # wherever this skill is checked out
IGRAPH_SRC=$DEV_ROOT/igraph-1.0.0.5
PY_IGRAPH_SRC=$DEV_ROOT/python-igraph-1.0.0.5
HEDONIC_SRC=$DEV_ROOT/hedonic-1.0.5
IGRAPH_BUILD_NAME=opus-1005-dev                 # short, path-safe, new per hypothesis
C_BUILD=$HOME/dev/igraph_$IGRAPH_BUILD_NAME
C_PREFIX=$DEV_ROOT/_prefix/igraph-1.0.0.5

# 1. Worktrees from verified refs (see "Resolve refs" below).
git -C ~/Code/community/igraph worktree add -b dev/opus-igraph-1.0.0.5 $IGRAPH_SRC 1.0.0.4
git -C ~/Code/community/python-igraph worktree add -b dev/opus-python-igraph-1.0.0.5 $PY_IGRAPH_SRC 1.0.0.4
git -C ~/Code/community/hedonic-game worktree add -b dev/opus-hedonic-1.0.5 $HEDONIC_SRC <hedonic-ref>

# 2. C core: worktrees cannot run `git describe` through CMake, so name the version.
echo 1.0.0.5-dev > $IGRAPH_SRC/IGRAPH_VERSION    # untracked and ignored
cmake -S $IGRAPH_SRC -B $C_BUILD -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$C_PREFIX -DBUILD_SHARED_LIBS=ON -DIGRAPH_WARNINGS_AS_ERRORS:BOOL=OFF
cmake --build $C_BUILD --parallel --target igraph && cmake --install $C_BUILD

# 3. python-igraph linked to that prefix (editable).
cd $PY_IGRAPH_SRC && uv venv --seed .venv
uv pip install --offline --python .venv/bin/python setuptools wheel   # not seeded on 3.12+
. $SKILL/scripts/python_build_env.sh $C_PREFIX  # SDKROOT, -isysroot, rpath, pkg-config
.venv/bin/python -m pip install --no-index --no-deps --no-build-isolation -e .

# 4. Hedonic on top (editable, no dependency resolution against PyPI).
cd $HEDONIC_SRC && uv venv --seed .venv
uv pip install --offline --python .venv/bin/python setuptools wheel
.venv/bin/python -m pip install --no-index --no-deps --no-build-isolation -e $PY_IGRAPH_SRC
.venv/bin/python -m pip install --no-deps -e $HEDONIC_SRC
uv pip install --offline --python .venv/bin/python numpy pandas scipy ...   # declared extras, from cache

# 5. Prove the chain before running anything.
$HEDONIC_SRC/.venv/bin/python $SKILL/scripts/check_stack.py \
  --expect-prefix $C_PREFIX --expect-igraph-source $PY_IGRAPH_SRC \
  --expect-hedonic-source $HEDONIC_SRC --expect-c-version 1.0.0.5-dev
```

`uv venv --seed` installs only pip for Python 3.12+, so setuptools and wheel
come from the local uv cache before `--no-build-isolation`. If a step needs
the network (a package is not cached), stop and ask.

## Recommended local layout

```text
$DEV_ROOT/                      agent-owned root, outside the original checkouts
  igraph-1.0.0.5/               C worktree (branch dev/<agent>-igraph-1.0.0.5)
  python-igraph-1.0.0.5/        python-igraph worktree (+ .venv)
  hedonic-1.0.5/                Hedonic worktree (+ .venv)
  _prefix/igraph-<name>/        install prefixes: the dev build and every snapshot
  _scratch/igraph-<name>/       `git archive` sources of snapshots
  _venvs/base-<version>/        baseline environment (unmodified tag, non-editable)
  _runs/<topic>/                raw outputs: *.jsonl records, rows/, logs
~/dev/igraph_<name>/            CMake build directories (the one exception to DEV_ROOT)
```

Build directories must be `~/dev/igraph_{{name}}` with a short, path-safe
name; use a new name for a new hypothesis rather than reusing a stale tree.
Keep a *baseline* environment -- the unmodified release tag built into its
own prefix and venv (non-editable) -- next to the development one: every
"faster" or "better" claim is measured against it on the same machine.

## Safety rules

1. No `sudo`, no shell startup edits; export variables only in the current
   command or shell.
2. No `git reset --hard`, `checkout --`, `git clean`, force-push or broad
   `rm -rf`. Delete only explicitly owned paths after a read-only check;
   prefer a new build name to deleting a build tree.
3. Preserve dirty user state. A dirty original checkout is a reason to create
   a worktree, never to clean it. Never write into the original checkouts.
4. **Do not disturb the user's running jobs.** Before heavy runs, look at
   `ps` / `tmux ls`; keep at most two or three heavy runs of your own; never
   reinstall a library that a long run has loaded (use a snapshot prefix).
5. Keep `PKG_CONFIG_PATH`, `LDFLAGS`, loader variables and venv activation
   local to the development shell.
6. Never ask the user to paste credentials; development mode needs none.
7. A subgraph result is never reported as a complete-graph result; label
   every record with its scope (`complete`, `induced_subgraph`,
   `covered_induced`).

## Development workflow

### 1. Resolve refs and create worktrees

Resolve before creating anything; stop if a tag is missing or ambiguous.

```bash
git -C ~/Code/community/igraph rev-parse --verify 'refs/tags/1.0.0.4^{commit}'
git -C ~/Code/community/igraph for-each-ref refs/tags/1.0.0.4 --format='%(objecttype) %(refname)'
git -C ~/Code/community/python-igraph ls-tree 1.0.0.4 vendor/source/igraph   # gitlink = C commit?
git -C <repo> worktree list                                                   # reuse, never move
```

Check that the python-igraph tag's vendored gitlink equals the C tag commit;
record the three worktree paths, branches, HEADs and source refs. Worktrees
of a repository share its stash stack -- never use bare `git stash`.

### 2. Build the C core

* `IGRAPH_VERSION` file first (CMake's revision helper fails in worktrees).
* Release build for measurements; `BUILD_SHARED_LIBS=ON` so Python loads the
  prefix. `cmake --build ... --target igraph` is enough for the library;
  `make build_tests && ctest -j6` for the full suite (581 tests in 1.0.x);
  single test targets are `test_<name>` / `example_<name>` and the CTest
  names `test::<name>`. Rebuild a test target before running it -- a stale
  binary compared against a new `.out` file is a false failure.
* After changing C, `cmake --build` + `cmake --install` is all Python needs:
  the extension links the shared library dynamically. Rebuild the extension
  only when a header or the binding changes.
* Sanitizer build: a second tree, `-DCMAKE_BUILD_TYPE=Debug
  -DUSE_SANITIZER="Address;Undefined" -DBUILD_SHARED_LIBS=OFF
  -DIGRAPH_VERIFY_FINALLY_STACK=ON`; run with
  `ASAN_OPTIONS=abort_on_error=1:halt_on_error=1:detect_leaks=0
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. LSan is unavailable on
  macOS arm64: use `leaks --atExit -- ./test_x` on Release binaries.
  Debug builds also enable the Leiden debug oracles (label index vs linear
  scan, bookkeeping drift). Debug and Release produce different floating-point
  results; compare Release with Release.

### 3. Link python-igraph

* Source [`scripts/python_build_env.sh`](scripts/python_build_env.sh)
  `<prefix>`: it sets `PKG_CONFIG_PATH`, `IGRAPH_USE_PKG_CONFIG=1`, an rpath
  to the prefix, and on macOS `SDKROOT` plus `-isysroot`. Without the SDK
  flags, uv-managed Pythons point at the SDK of their build machine and the
  link fails with `library 'c++' not found`.
* Editable install once; later rebuild in place with
  `.venv/bin/python setup.py build_ext --inplace` (same environment).
* After bumping `version.py` or `pyproject.toml`, reinstall the editable
  package (`pip install --no-index --no-deps --no-build-isolation -e .`):
  editable installs keep the old distribution metadata, and code that reads
  `importlib.metadata.version("lucas-igraph")` sees the stale value.
* python-igraph surface changes: `src/_igraph/graphobject.c` kwlist *and*
  the `PyArg_ParseTupleAndKeywords` format string, module functions in
  `src/_igraph/igraphmodule.c`, Python wrappers in `src/igraph/community.py`,
  exports in `src/igraph/__init__.py` and `__all__`, tests in `tests/`.
  Full suite: `.venv/bin/python -m pytest tests -q -p no:cacheprovider`.

### 4. Link Hedonic

Install the local python-igraph and Hedonic editable with `--no-deps` (so pip
never replaces the development build), then the experiment extras from the
local cache. Verify with [`scripts/check_stack.py`](scripts/check_stack.py):
it reports the interpreter, the python-igraph source path, the C version
compiled into the extension, the libigraph the loader actually maps, rpaths,
distribution metadata and Hedonic's algorithm identity, and exits 1 on any
`--expect-*` mismatch. Run it in every new shell and before every campaign.

Hedonic obligations when its surface changes (see `AGENTS.md`):

* `hedonic-exp` CLI registry, `AGENTS.md` tables and examples, tests;
* `PYTHON=.venv/bin/python node web/scripts/generate-docs.mjs` after any
  change to `Game`, `hedonic.utils`, CLI `COMMANDS` or disjoint presets, and
  `web/scripts/crosscheck_native.py` (seeded) after native behaviour changes;
* experiment checks that mirror native semantics (the integrity grid's
  projection trace, `overlapping-dnn-rational --require-debug-trace`) must be
  updated in the same change as the native rule;
* frozen protocol locks and evidence are never rewritten to chase a new
  build; add a new candidate instead.

Expected Hedonic failures in development mode (classify, do not "fix"):
tests that assert the installed lucas-igraph equals a literal version or
`uv.lock`, tests that compare the runtime identity with `uv.lock` or an
installed distribution tree (an editable, unpublished build cannot match),
and frozen-lock checks whose bound files already drifted on the base branch.
Run the same suite on the base commit to separate old from new failures.

### 5. Qualify and experiment

A prototype is evidence only after proportionate qualification:

* native: ownership and error unwinding (every failure path under ASan with
  FINALLY verification), checked integer arithmetic (`IGRAPH_SAFE_ADD/MULT`),
  empty/degenerate/disconnected/high-multiplicity graphs, label-bank and cap
  boundaries, interruption, leaks; mutation tests (break the rule, watch the
  test fail);
* binding and facade: native/Python differentials, invalid-combination
  errors, seed replay across processes, loader checks;
* performance: detector-only wall and CPU time, peak RSS, scoring separately,
  timeouts and OOMs as statuses, audit/certificate status, and accuracy.
  On a shared machine, CPU seconds against a baseline run in the same session
  are the comparable quantity; historical wall times are only indicative.

Then run the [verification toolkit](#verification-toolkit) and write the
ledger entry.

### 6. Stop

Leave each worktree clean (`git status --short` empty; build artifacts are
ignored), local commits only, the ledger committed, and a report (see
[Completion report](#development-completion-report)). Do not push, tag,
publish or open a pull request.

## Verification toolkit

| Question | Tool | Notes |
|---|---|---|
| Does this interpreter load the stack I think? | `scripts/check_stack.py` | also flags stale editable metadata |
| Is a C change behaviour-preserving, bit for bit? | `scripts/leiden_equivalence.c` | 50,000 instances in ~2 min; directed, signed, in-weights, loops, simple interface, count limits, diagnostics, error paths; fork + timeout per instance |
| What changed between two Python-visible builds, and where? | `scripts/native_differential.py` | classifies exact / quality_only / renamed / different / status by instance class |
| Which commit caused it? | `scripts/snapshot_build.sh` | distinct version per snapshot; load with the loader variable |
| Memory safety, cleanup accounting, internal oracles | ASan+UBSan Debug build with `IGRAPH_VERIFY_FINALLY_STACK=ON`; link the equivalence harness statically against it | expect only the reference error codes and no `IGRAPH_EINTERNAL` |
| Leaks | `leaks --atExit -- ./test_x` (macOS), valgrind/LSan (Linux) | |
| Large-graph behaviour | complete-DBLP runs with canonical and labelled cover hashes | on the dev branch: `docs/plans/opus_1_0_5/bench/hoc_bench.py` |

Workflow for a refactor that must not change results:

1. Freeze a reference: `snapshot_build.sh <c-worktree> HEAD <agent>-ref`.
2. Record the harness outputs on the reference library (two seeds).
3. Change the code; rebuild; rerun; `cmp` must be silent. Any difference is
   a bug until explained -- in 1.0.0.5 the first difference was a
   floating-point fusion change (see Pitfalls).
4. Run the ASan battery, ctest, python-igraph tests, the Python differential,
   and at least one complete-DBLP configuration per phase touched.

Workflow for a change that should alter results: the same batteries, then
classify every difference by instance class and direction (quality higher,
lower, equal) and attribute it with snapshots. Details and output formats
are in [references/harnesses.md](references/harnesses.md).

## Snapshot builds and attribution

```bash
$SKILL/scripts/snapshot_build.sh $IGRAPH_SRC <commit> <agent>-<label>
DYLD_LIBRARY_PATH=$DEV_ROOT/_prefix/igraph-<agent>-<label>/lib \
  $HEDONIC_SRC/.venv/bin/python <harness> ...     # LD_LIBRARY_PATH on Linux
```

* Set the loader variable **on the Python command itself**. macOS strips
  `DYLD_*` when a SIP-protected binary starts a process (`/usr/bin/nohup`,
  `/usr/bin/env`, `/bin/sh`, `/usr/bin/time`), silently loading the rpath
  library instead. One attribution run in 1.0.0.5 was invalid for this
  reason; it was caught because every record stores
  `igraph._igraph.__igraph_version__`.
* Give snapshots distinct version strings (the script does); a copied prefix
  reports the same version as its origin and proves nothing.
* A python-igraph build only runs with a C library that exports every symbol
  it references: a 1.0.0.5 binding cannot load a 1.0.0.4 library. Keep a
  baseline venv for the old binding.

## Evidence and the decision ledger

Keep, in the Hedonic development worktree, `docs/plans/<agent>_<version>/`:

* `decision_ledger.md` -- setup and provenance (worktrees, refs, builds,
  prefixes, venvs, commits), a table of items with disposition, key evidence
  and commits, detailed records (hypothesis, prototype, commands, dataset and
  scale, accuracy, wall/CPU/RSS, correctness, failure mode, disposition),
  evidence by scale (fixtures / subgraphs / complete DBLP / other SNAP),
  tests, remaining defects, production steps not performed;
* `bench/` -- the harnesses used (record schema: config, stack identity,
  graph identity and scope, status, detector and scoring seconds, CPU, peak
  RSS, native quality, canonical and labelled cover hashes, metrics);
* `evidence/` -- small run records with the home directory written as `~`,
  summaries and a manifest with SHA-256 of large raw outputs kept under
  `$DEV_ROOT/_runs/`.

Numbers in the ledger come from records, not memory; re-derive them with a
script before writing.

## Pitfalls observed

| Symptom | Cause | Fix |
|---|---|---|
| CMake: `IGRAPH_VERSION` missing | worktree, no `git describe` | write the ignored `IGRAPH_VERSION` file |
| python-igraph link: `library 'c++' not found` | uv Python records a missing Xcode SDK | `python_build_env.sh` (SDKROOT, `-isysroot`) |
| `importlib.metadata` reports the old version | editable metadata not refreshed | reinstall the editable package offline |
| Attribution run shows no difference | `DYLD_*` stripped by `nohup`/`env`/`sh`/`/usr/bin/time` | set the variable on the python command; check the recorded C version |
| Snapshot and dev report the same C version | prefix copied, not built | `snapshot_build.sh` |
| SIGINT test "does not interrupt" | background jobs start with SIGINT ignored | install `signal.default_int_handler` in the probe |
| Python harness hangs on a small graph | igraph polls interruption every 2^13-2^14 visits, so SIGALRM cannot stop short loops | fork per instance with a wall-clock limit (both bundled harnesses do) |
| ctest failure right after changing code | stale test binary vs updated `.out` | build `test_<name>` / `build_tests` first |
| Identical covers, last-bit different qualities | backend-discretionary `llvm.fmuladd`: splitting, moving or reordering an `a*b+c` changes fusion; the released overlapping potential fuses the crowding term only outside its unrolled loop | keep decision-relevant FP loops' shape; verify with the equivalence harness; a platform-stable choice (`-ffp-contract=off` or explicit `fma`) is a separate, re-baselining decision |
| Hedonic test expects literal `1.0.0.3` | hard-coded release expectations | classify as environment; fix only when `uv.lock` is refreshed |
| `$c:src` expands oddly in zsh | `:s` is a zsh history modifier | write `${c}:src/...` |
| `git show <commit>:<path>` fails in a loop | same zsh modifier issue | quote and brace the variable |
| Complete-DBLP RSS looks identical before and after | the forked child inherits the parent's loaded graph | compare detector CPU and relative RSS changes only |
| Covers differ only by label ids | relabelling (e.g. reindexed start states) | compare canonical bodies before calling it a behaviour change |
| MINGW i686 fails, every 64-bit job passes | x87 80-bit intermediates: where the compiler rounds to double decides near-ties, and code-shape changes move that point | compile `leiden.c` with `-msse2 -mfpmath=sse` on 32-bit x86 (in igraph's `src/CMakeLists.txt`); never squash or tag before the i686 jobs pass |
| A release diff narrates `1.0.0.1`..`1.0.0.N-1` | development changelogs and comments written relative to the previous fork release | gate 2 of production mode: rewrite relative to 1.0.0 on `lucas-dev`, then squash |
| Python build uses an old igraph's headers | `cmake --install --prefix P` copies files but keeps the configure-time prefix inside `igraph.pc` | configure with `-DCMAKE_INSTALL_PREFIX=P`, or fix `prefix=`/`exec_prefix=` in the installed `igraph.pc`; check `pkg-config --cflags igraph` |
| `git worktree remove` refuses a python-igraph worktree | it has an initialized `vendor/source/igraph` submodule | check the submodule is clean (`git -C <wt>/vendor/source/igraph status`), then `--force` |

## Gated production mode

Production begins only when the invoker explicitly requests a release and
names the suffix, branch boundary and allowed external actions. The
`sibling-release-pipeline` skill owns the release mechanics; resolve its
current instructions before mutating anything and do not invoke it from this
skill. Whatever the mechanics, a release passes these gates **in order**, and
a failed gate sends the work back to `lucas-dev`, never forward to a squash or
tag:

1. **Consolidate on `lucas-dev`.** Every accepted change of the development
   worktrees is on the C and Python `lucas-dev` lines (linear, no merges after
   the `1.0.0` bump). Python's `lucas-dev` vendors the C `lucas-dev` tip.
2. **Review the diff as a single step from `1.0.0`.** A sibling release is
   read as `1.0.0 -> 1.0.0.N` with nothing in between. In
   `git diff <1.0.0-bump> lucas-dev`, remove narration of the fork's own
   history: `1.0.0.1`..`1.0.0.N-1` references, "again", "no longer", "as
   released", "earlier releases", "legacy"/"old names" aliases, "width
   12 -> 15" style notes, per-release changelog sections, and decoders for
   formats only an older fork release produced. Describe the code as it is
   and changes relative to igraph/python-igraph 1.0.0. The upstream parts of
   `CHANGELOG.md` must be untouched: the diff against the bump removes no
   upstream line (`git diff <bump> -- CHANGELOG.md | grep -c '^-[^-]'` is 0),
   and the release adds one `[1.0.0.N]` section. Keep data-format identifiers
   (schema numbers) monotone across published releases; only their prose
   changes.
3. **Green hosted CI on `lucas-dev` first.** Push `lucas-dev` (authorized) and
   wait until **every** job of **every** workflow passes on that exact commit,
   C first, then Python (whose submodule must resolve on GitHub). The local
   arm64 or x86-64 suite never substitutes for the matrix: the 1.0.0.5
   candidate passed 582/582 locally and on 64-bit Windows, yet failed the
   MINGW **i686** jobs, where x87 80-bit arithmetic resolved a near-tie
   differently after the disjoint mover was restructured. Treat a skipped or
   still-running job as not green.
4. **Squash only then.** Create the one release commit as the single child of
   the `1.0.0` bump with exactly the `lucas-dev` tree (Python: the same tree
   with `vendor/source/igraph` set to the C release commit). Verify parent,
   distance 1, tree equality and the gitlink. The release message follows the
   previous sibling's template but describes the release relative to 1.0.0.
5. **Green CI on the release commit, then tag.** Point `lucas`/`physica-a` at
   the release commit (authorized, `--force-with-lease` pinned to the observed
   remote commits) and wait for the full matrix on it. Create and push the
   `1.0.0.N` tag only after that; a published tag is immutable, and a failure
   discovered after tagging forces a re-tag. Delete the temporary
   `lucas-1.0.0.N` branch once the tag exists.
6. **Downstream.** Then python-igraph (vendoring the tagged C commit), the
   `lucas-igraph` PyPI upload, and Hedonic `1.0.N` against the published
   package, each separately authorized:

```text
C lucas-dev green -> C 1.0.0.N squash green -> C tag
  -> Python lucas-dev (vendors C lucas-dev) green
  -> Python 1.0.0.N squash (vendors the C tag commit) green -> Python tag
  -> lucas-igraph 1.0.0.N -> PyPI
  -> hedonic 1.0.N (regenerated uv.lock, version-bound tests) -> PyPI
```

Never publish an editable or development-prefix build, and never treat a
local C pointer as a substitute for the vendored gitlink.

## Development completion report

Report, and state explicitly that no push, tag, GitHub release, pull request,
PyPI upload or production handoff was performed:

* `DEV_ROOT`, worktree paths, branches, verified base refs and HEADs; the
  vendored gitlink; version strings of the three packages;
* build names, prefixes, venvs, `pkg-config` version and the loaded
  libigraph (`check_stack.py` output);
* local commits per repository, grouped by ledger item;
* dispositions: accepted / redesigned / deferred / rejected / superseded;
* evidence by scale, kept separate: fixtures, subgraphs, complete graphs,
  other networks -- never a subgraph result as a complete-graph result;
* tests, sanitizers, differentials, equivalence batteries, benchmarks, with
  their exact outcomes (including classified expected failures);
* remaining defects and open questions; production steps not performed;
* worktree cleanliness and where large untracked outputs live.
