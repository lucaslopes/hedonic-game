# AGENTS.md — Working on `hedonic`

Guidance for humans and coding agents editing this repository. `hedonic` is a
small library: one public class (`Game`), one detector
(`Game.community_hedonic`), and a C core compiled into `hedonic._core`.

## Public `main` and private `paper`

- `main` tracks `origin/main`, the public GitHub repository. Only
  public-ready code and documentation belong there.
- `paper` is a local private research branch (manuscripts, experiments,
  evidence). It is never pushed to `origin` and never merged into `main`.
  Bring public changes into it with `git merge origin/main` while on `paper`;
  move a change the other way by recreating it on `main`.

## Layout

```
native/                  # the C core (no igraph dependency), built into hedonic._core
├── README.md            # the design: game, visits, certified moves, termination, slack,
│                        #   prefix lemma, verifier, queue, determinism and platforms
├── include/hedonic_core.h  # C API: hig_partition, hig_cover, options, errors
├── core/
│   ├── internal.h       # what the engines share, certified (upward-rounded) arithmetic, hig_fma
│   ├── softfma.h        # correctly rounded fma in integer arithmetic (from musl, MIT licence)
│   ├── dispatch.h       # x86_64 without baseline FMA: two builds, chosen from CPUID
│   ├── shared/          # what both engines share (declared in internal.h)
│   │   ├── inputs.c     # errors, memory, input checks, adjacency, start labels
│   │   ├── random.c     # random numbers; which fma the build runs
│   │   ├── queue.c      # random visiting order + queue
│   │   └── heap.c       # communities by mass
│   ├── partition/       # the disjoint engine (hig_partition)
│   │   ├── partition.h  # its state, functions its files share
│   │   ├── visit.c      # masses, payoff bounds, a visit, the run
│   │   └── partition.c  # start, quality, entry points
│   ├── cover/           # the overlapping engine (hig_cover)
│   │   ├── cover.h      # its state, functions its files share
│   │   ├── rows.c       # labels and rows
│   │   ├── visit.c      # candidates, bounds, best set; the run, the slack
│   │   ├── start.c      # start and setup
│   │   └── cover.c      # result, quality, entry points
│   └── check/           # tests only
│       ├── check.c      # the independent verifier
│       └── hedonic_check.h  # traced runs and the verifier (not the API)
├── unity.c              # the core as one translation unit (how the extension builds it;
│                        #   HIG_SHARED is static there, `make test` compiles each file alone)
├── unity_fma.c, dispatch.c  # x86_64: the core again with the FMA instruction, and the entry
│                        #   points that pick a build from CPUID (empty on other targets)
├── binding/             # CPython binding: detect() for Game, partition()/cover() over plain
│                        #   buffers, an exact random.Random bridge
├── tests/test_core.c    # `make -C native test` (ASan/UBSan + libm check), `test-release` (-O3),
│                        #   `test-soft-fma`, `test-x86_64` (Rosetta); C bit pins from
│                        #   random.Random's order; soft fma against the instruction
├── tests/check_libm.py  # `make -C native check-libm`: the core calls no libm but sqrt, fabs
└── Makefile
setup.py                 # the one Extension (hedonic._core); everything else is pyproject.toml
tools/native_tests.py    # the pinned Python tests with HIG_PINNED=1 (what the wheel builds run)
src/hedonic/
├── __init__.py          # public API: Game
├── Game.py              # Game(igraph.Graph) + community_hedonic
├── _native.py           # argument buffers + VertexClustering/VertexCover around hedonic._core
├── _game/               # detection, membership state, evaluation bridge, helpers
├── metrics.py           # scores against a reference cover (SciPy optional, extra `metrics`)
└── utils.py             # seeded sampling for the random disjoint start
web/                     # educational site (React + Vite), see below
```

## The detector

`Game.community_hedonic` runs best-response local moving of the CPM hedonic
game until every vertex has been visited without a certainly improving move
since the last move (a move's computed gain must beat a rigorous bound of its
rounding error, which makes every move raise the exact potential and every
call terminate; results report `equilibrium_slack`, see `native/README.md`):

| Goal | Call | Returns |
|---|---|---|
| Partition | `max_memberships=1` (default) | `VertexClustering` |
| Overlapping cover | `max_memberships > 1` (`-1`: no practical cap) | `VertexCover` |

Parameters: `initial_membership` (`None`: the default start; a start with
more labels per vertex than the cap keeps each vertex's preferred ones), then
keyword-only `max_communities` (random disjoint start only),
`max_memberships`, `resolution` (default: weighted density), `allow_isolation`,
`edge_weights`, `seed`, `node_weights`, `max_total_communities`,
`n_communities`. The graph must be undirected without self-loops; parallel
edges add up. There is no iteration budget, round cap, refinement, aggregation
or multilevel phase; do not add them. Results carry `_hedonic_provenance` and
`_hedonic_native_quality`. The returned state is stored in `Game.memberships`
but never read back by itself: a call continues from it only when it is passed
as `initial_membership`.

Invariants:

1. One public detector API, `Game.community_hedonic`, for both modes.
2. The detector is this package's purpose-built C core; no pure-Python port
   and no other backend. The core never reads an igraph graph; the binding
   passes it the edge list.
3. Dependencies stay `igraph>=1.0` and NumPy. Nothing checks igraph's
   version at import.
4. Results are bit-identical across platforms: rounding is fixed by the
   source (`-ffp-contract=off`, `/fp:strict`, every fused multiply-add an
   explicit `hig_fma`: the instruction or the core's soft fma, never libm's;
   no libm call but `sqrt`: `make -C native check-libm`), the x86_64 FMA and
   soft builds must agree bit for bit, and the pins in
   `native/tests/test_core.c`, `tests/test_native_core.py` and
   `tests/test_native_backend.py` (the same results) must pass unchanged. If
   a change alters a pinned result, it changes the algorithm: say so and
   re-pin deliberately, in C and Python together.
5. Every call returns a certified equilibrium. `tests/test_equilibrium.py`
   replays traced runs in exact arithmetic (every accepted move raises the
   exact potential; no deviation beats the reported slack; the slack is 0 in
   exact arithmetic) and runs the C verifier; keep it passing, and keep the
   bounds, the termination argument and the prefix lemma of
   `native/README.md` true when touching the move rule. C code stays plain
   C11: no VLAs, `long double`, `_Atomic`, `complex.h` or `stdatomic.h`, and
   GNU builtins only behind compiler checks with an MSVC or portable
   equivalent (the FMA instruction, clz, CPUID, the dispatcher's atomics).
   MSVC builds it with `/fp:strict`.
6. Scoring logic lives in `hedonic.metrics` only.

## Checks

```bash
python -m pip install -e ".[metrics]" pytest   # or: uv pip install -e ".[metrics]" pytest
python setup.py build_ext --inplace            # after editing native/
make -C native test test-release test-soft-fma test-x86_64   # C tests, libm check, x86_64 (Rosetta)
python tools/native_tests.py                   # pinned Python tests (fails on any skip)
python -m pytest
```

`.github/workflows/wheels.yml` runs the C tests (GCC, Clang, MSVC) and builds
and tests the wheels on Linux, macOS and Windows; every wheel job also runs
the C tests and `tools/native_tests.py` (HIG_PINNED=1, no skips), and
publishing needs every job. Branch pushes only build and test; a tag publishes,
and sets the version (`tools/release_version.py`; `pyproject.toml` holds the
release): `v2.0.0.devN` goes to TestPyPI, `v2.0.0rcN` (a pre-release) and
`v2.0.0` to PyPI, all by trusted publishing (environments `testpypi`, `pypi`).
After each upload the `verify` jobs install that version from its index on
every platform and run the pinned tests and the test suite. Try a release as
`.devN`, then `rcN`, and tag the release on the commit of the last good
candidate.

## Timing and memory of the experiments

One clock serves every method of the overlap benchmark (`hedonic.experiments.timing`,
`timing_scope = "detector-compute-v1"`). It starts after the graph is loaded and converted, the input files are
written, the imports are done and the temporary folders exist, and stops when the detector returns its raw result.
Validation, canonicalisation, scoring, audits and saving come after it.

- Executables (CoDeSEG, BigCLAM, FOX, NEO-K-Means, OSLOM, SVI, template commands) are timed exec-to-exit by
  `DetectorClock.process`; their own file read and write are counted (inseparable). Harness-shipped or generated code
  (the CDlib/DER/SLPA/link workers, the NcGame wrapper, the Octave and R scripts) times itself around the detector
  call and reports `detector_seconds`; start-up, imports and JSON transport are recorded as
  `isolation_overhead_seconds` and never counted. A worker that reports nothing is a failed record, not a guess.
- Peak memory: `detector_peak_rss_bytes` is the detector process's own `ru_maxrss` (`wait4`, never `RUSAGE_CHILDREN`,
  which keeps the largest child ever waited for). An in-process detector can only report the harness peak when the
  clock stopped (`detector_peak_rss_scope = "harness_process_lifetime"`), comparable only under `--isolate`. A
  worker's peak includes its interpreter and CDlib baseline (about 200 MB).
- A new adapter times its detector through `timing.active()` and spawns nothing with `subprocess.run`
  (`tests/experiments/test_timing_fairness.py` checks both). `unit_seconds` is the whole unit, for bookkeeping.
- The disjoint store's `duration` is the detector call only (`store.TIMING_SCOPE = "detector-call-v1"`, part of the
  spec, so a store timed another way is refused). Hedonic-only studies and calibration probes keep their own
  wall-clock for budget accounting; they are not method comparisons.

## Web explainer (`web/`)

An educational React + TypeScript + Vite site for the disjoint model, deployed
by `.github/workflows/pages.yml`. Run everything from the root:

```bash
npm install
npm run dev            # story at /, documentation at /docs/
npm run check          # docs:generate + typecheck + lint + test + build
```

- `web/src/model/` is the mathematics of a four-vertex teaching model. Keep it
  DOM-free and faithful to the paper, and extend its Vitest suite with any
  change. It never replaces `community_hedonic`.
- Re-run `npm run docs:generate` (and commit `web/src/generated/*.json`) when
  `Game` or `hedonic.utils` change.
