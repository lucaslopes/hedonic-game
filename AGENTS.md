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
│                        #   verifier, queue, determinism and platforms
├── include/hedonic_core.h  # C API: hig_partition, options, errors
├── core/
│   ├── internal.h       # what the engine shares, certified (upward-rounded) arithmetic, hig_fma
│   ├── softfma.h        # correctly rounded fma in integer arithmetic (from musl, MIT licence)
│   ├── dispatch.h       # x86_64 without baseline FMA: two builds, chosen from CPUID
│   ├── common.c         # errors, input checks, adjacency, random order + queue, mass heap
│   ├── partition.c      # the engine (hig_partition)
│   ├── check.c          # the independent verifier (tests only)
│   └── hedonic_check.h  # traced runs and the verifier (tests only, not the API)
├── unity.c              # the core as one translation unit (how the extension builds it)
├── unity_fma.c, dispatch.c  # x86_64: the core again with the FMA instruction, and the entry
│                        #   points that pick a build from CPUID (empty on other targets)
├── binding/             # CPython binding: detect() for Game, partition() over plain
│                        #   buffers, an exact random.Random bridge
├── tests/test_core.c    # `make -C native test` (ASan/UBSan + libm check), `test-release` (-O3),
│                        #   `test-soft-fma`, `test-x86_64` (Rosetta); C bit pins from
│                        #   random.Random's order; soft fma against the instruction
├── tests/check_libm.py  # `make -C native check-libm`: the core calls no libm but sqrt
└── Makefile
setup.py                 # the one Extension (hedonic._core); everything else is pyproject.toml
tools/native_tests.py    # the pinned Python tests with HIG_PINNED=1 (what the wheel builds run)
src/hedonic/
├── __init__.py          # public API: Game
├── Game.py              # Game(igraph.Graph) + community_hedonic
├── _native.py           # argument buffers + VertexClustering around hedonic._core
├── _game/               # detection, membership state, evaluation bridge, helpers
├── metrics.py           # scores against a reference partition or cover (SciPy optional, extra `metrics`)
└── utils.py             # seeded sampling for the random disjoint start
web/                     # educational site (React + Vite), see below
```

## The detector

`Game.community_hedonic` runs best-response local moving of the CPM hedonic
game until every vertex has been visited without a certainly improving move
since the last move (a move's computed gain must beat a rigorous bound of its
rounding error, which makes every move raise the exact potential and every
call terminate; results report `equilibrium_slack`, see `native/README.md`),
and returns the partition as a `VertexClustering`.

Parameters: `initial_membership`, `max_communities` (random disjoint start
only), `resolution` (default: density), `allow_isolation`, and keyword-only
`edge_weights`, `seed`, `node_weights`, `max_total_communities`,
`n_communities`. There is no iteration budget, round cap, refinement,
aggregation or multilevel phase; do not add them. Results carry
`_hedonic_provenance` and `_hedonic_native_quality`, and the returned state is
stored in `Game.memberships`.

Invariants:

1. One public detector API, `Game.community_hedonic`.
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
   bounds and the termination argument of `native/README.md` true when
   touching the move rule. C code stays plain C11: no VLAs, `long double`,
   `_Atomic`, `complex.h` or `stdatomic.h`, and GNU builtins only behind
   compiler checks with an MSVC or portable equivalent (the FMA instruction,
   clz, CPUID, the dispatcher's atomics). MSVC builds it with `/fp:strict`.
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
publishing needs every job. A `v*` tag publishes them to PyPI.

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
