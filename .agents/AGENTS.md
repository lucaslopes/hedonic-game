# AGENTS.md — Working on `hedonic`

Guidance for humans and coding agents editing this repository. `hedonic` 1.x
is a small library (one public class, `Game`, and one detector,
`Game.community_hedonic`, which returns a partition at a Nash equilibrium of
the CPM hedonic game, in a C core compiled into `hedonic._core`) plus the
command line that reproduces the disjoint paper (Felipe, Avrachenkov &
Menasché, Physica A 680:130989, 2025: the SBM benchmark). Overlapping covers,
the SNAP benchmark, its baselines and studies are hedonic 2: do not add them
here.

## Public `main` and private `paper`

- `main` tracks `origin/main`, the public GitHub repository. Only
  public-ready code and documentation belong there.
- `paper` is a local private branch. It is never pushed to `origin` and never
  merged into `main`. Bring public changes into it with `git merge
  origin/main` while on `paper`; move a change the other way by recreating it
  on `main`. Nothing on `main` may name or require a file that exists only
  on `paper`.

## Agent files and skills

Everything for coding agents lives in `.agents/`: this file
(`.agents/AGENTS.md`; the root `AGENTS.md` is a symlink to it, because tools
read `AGENTS.md` from the repository root) and the repository's skills in
`.agents/skills/<name>/SKILL.md`.

- **`hedonic-native-core`** is the path for any change to the C core
  (`native/`, `setup.py`, `hedonic._native`, the wheel and release workflow):
  the arm64 venv and in-place rebuild, `scripts/fresh.py` (the loaded
  extension is built from this checkout), the C and pinned Python tests,
  `scripts/bitcheck.py` (the detector's bit-level fingerprint), keeping the
  partition engines byte-identical between hedonic 1 (`release/1.x`) and
  hedonic 2, what a source change does to run records, timing and memory
  under `scripts/guard.sh`, and the tag-driven TestPyPI/PyPI release. Local
  only: it never pushes, tags or publishes.
- `clean-worktree-commits` (a user-level skill, not in this repository) turns
  a dirty checkout into reviewable local commits before a release.

## Layout

```
native/                  # the C core (no igraph dependency), built into hedonic._core
├── README.md            # the design: game, visits, certified moves, termination, slack,
│                        #   verifier, requeueing, data structures, determinism and platforms
├── include/hedonic_core.h  # C API: hig_partition, options, errors
├── core/
│   ├── internal.h       # shared declarations, certified (upward-rounded) arithmetic, hig_fma
│   ├── softfma.h        # correctly rounded fma in integer arithmetic (from musl, MIT licence)
│   ├── dispatch.h       # x86_64 without baseline FMA: two builds, chosen from CPUID
│   ├── shared/          # helpers of the engine (declared in internal.h)
│   │   ├── inputs.c     # errors, memory, input checks, adjacency, start labels
│   │   ├── random.c     # random numbers; which fma the build runs
│   │   ├── queue.c      # random visiting order + queue
│   │   └── heap.c       # communities by mass
│   ├── partition/       # the engine (hig_partition)
│   │   ├── partition.h  # its state, functions its files share
│   │   ├── visit.c      # masses, payoff bounds, a visit, the run
│   │   └── partition.c  # start, quality, entry points
│   └── check/
│       ├── check.c      # the independent verifier (tests only)
│       └── hedonic_check.h  # traced runs and the verifier (not the API)
├── unity.c              # the core as one translation unit (how the extension builds it;
│                        #   HIG_SHARED is static there, `make test` compiles each file alone)
├── unity_fma.c, dispatch.c  # x86_64: the core again with the FMA instruction, and the entry
│                        #   points that pick a build from CPUID (empty on other targets)
├── binding/             # CPython binding (binding.h lists its files): detect() for Game,
│                        #   partition() over plain buffers, an exact random.Random bridge
├── tests/test_core.c    # `make -C native test` (ASan/UBSan + libm check), `test-release` (-O3),
│                        #   `test-soft-fma`, `test-x86_64` (Rosetta); C bit pins
├── tests/check_libm.py  # `make -C native check-libm`: the core calls no libm but sqrt, fabs
└── Makefile
setup.py                 # the one Extension (hedonic._core, embeds source_sha256); the rest is pyproject.toml
tools/native_tests.py    # the pinned Python tests with HIG_PINNED=1 (what the wheel builds run)
tools/release_version.py # tag -> version and index (TestPyPI for .devN, PyPI otherwise)
src/hedonic/
├── __init__.py          # public API: Game (and hedonic.metrics, imported lazily)
├── Game.py              # Game(igraph.Graph): community_hedonic, memberships, evaluate_against, ...
├── _native.py           # argument buffers + VertexClustering around hedonic._core
├── _game/               # detection.py, memberships.py, evaluation.py, helpers.py
├── metrics.py           # scores against a reference partition or cover
├── utils.py             # seeded sampling for the random start
└── experiments/         # the command line (the `cli` extra)
    ├── CLI.py           # `hedonic` and `hedonic-exp`; COMMANDS is the registry
    ├── config.py        # DATA_ROOT, DISJOINT_RUNS_ROOT, SYNTHETIC_DIR, the archived-V1020 guard
    ├── resources.py     # shipped configs (package data)
    ├── runtime.py       # the identity every record carries (release series, digest of the core's sources)
    ├── leiden.py        # stock igraph CPM Leiden, seeded like community_hedonic (comparison only)
    ├── timing.py        # the multi-method benchmark clock of hedonic 2; no 1.x runner uses it
    ├── configs/         # hedonic-run.example.toml, queue-disjoint.toml
    ├── disjoint/
    │   ├── sbm_sweep.py     # seeded SBM graphs, the methods, the sweep (store / v1020 / legacy layouts)
    │   ├── store.py         # Parquet result store: shards, DuckDB views, identities, progress, verify/replay
    │   ├── front.py         # `hedonic run disjoint`: presets, wizard, plan, worker pipeline
    │   ├── paper_stats.py   # every statistic of the paper, from a store
    │   ├── data_loader.py   # JSON layouts -> CSV
    │   └── reproduce.py     # end-to-end: sweep -> table -> figures
    ├── plots/paper_figures.py  # the paper's five figures
    └── overlapping/     # (historical name) the run machinery shared by every `hedonic` verb:
        ├── runmanager.py    # durable runs: tmux/process launch, worker, memory guard, viewer, verbs
        ├── runqueue.py      # `hedonic run queue`: items, file-based skip/resume, report, overview
        ├── queuemodel.py, queuetui.py  # the queue as a tree, and its interactive view
        ├── parallel.py      # forked workers, flock slots, free-memory floor
        ├── tui.py           # dependency-free prompts (numbered fallback off a TTY)
        ├── common.py        # run names, seed lists, durations, run environment
        ├── info.py          # `hedonic show disjoint`
        ├── guide.py         # `hedonic guide [TOPIC]`
        ├── selfupdate.py    # `hedonic update [--yes]`
        └── userconfig.py    # ~/.config/hedonic/config.toml
tests/                   # library tests; tests/experiments/ for the command line
web/                     # educational site (React + Vite), see below
```

## The detector

`Game.community_hedonic` runs best-response local moving of the CPM hedonic
game until every vertex has been visited without a certainly improving move
since the last move (a move's computed gain must beat a rigorous bound of its
rounding error, which makes every move raise the exact potential and every
call terminate; results report `equilibrium_slack`, see `native/README.md`).
It returns an `igraph.VertexClustering`.

Parameters: `initial_membership` (`None`: singletons, a random start with
`max_communities`, or a feasible start under a count limit; a start with
several labels per vertex keeps each vertex's preferred one), then
keyword-only `max_communities` (random start only), `max_memberships`
(only `1`; anything else raises `ValueError` naming hedonic 2), `resolution`
(default: weighted density), `allow_isolation`, `edge_weights`, `seed`,
`node_weights`, `max_total_communities`, `n_communities`. Removed 1.x-fork
keywords (`n_iterations`, `local_move_only`, `beta`, ...) and extra
positional arguments raise a `TypeError` that says so. The graph must be
undirected without self-loops; parallel edges add up. There is no iteration
budget, round cap, refinement, aggregation or multilevel phase; do not add
them. Results carry `_hedonic_provenance` and `_hedonic_native_quality`. The
returned state is stored in `Game.memberships` but never read back by itself:
a call continues from it only when it is passed as `initial_membership`.
`Game.evaluate_against` scores it through `hedonic.metrics`.

Invariants:

1. One public detector API, `Game.community_hedonic`, for partitions.
2. The detector is this package's purpose-built C core; no pure-Python port
   and no other backend. The core never reads an igraph graph; the binding
   passes it the edge list. Full CPM Leiden, for comparison only, is
   `hedonic.experiments.leiden.leiden` (stock igraph).
3. The library's modules import only igraph (`>=1.0`) and NumPy, and SciPy
   inside `hedonic.metrics` for one-to-one matching. Nothing checks igraph's
   version at import (a test sets the old `HEDONIC_REQUIRE_IGRAPH_C` and
   checks that it is ignored).
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
   replays traced runs (`hedonic._core._trace_partition`) in exact arithmetic
   (every accepted move raises the exact potential; no deviation beats the
   reported slack; the slack is 0 in exact arithmetic) and runs the C
   verifier (`_verify_partition`); keep it passing, and keep the bounds and
   the termination argument of `native/README.md` true when touching the move
   rule. C code stays plain C11: no VLAs, `long double`, `_Atomic`,
   `complex.h` or `stdatomic.h`, and GNU builtins only behind compiler checks
   with an MSVC or portable equivalent. MSVC builds it with `/fp:strict`.
6. Scoring logic lives in `hedonic.metrics` only.
7. The library installs without the experiment dependencies; the command
   line is optional (the `cli` extra) and never imported by `hedonic`.

## Dependencies

From `pyproject.toml` (Python >= 3.12):

- `pip install hedonic`: `igraph>=1.0`, `numpy`.
- `metrics` extra: `scipy>=1.10` (one-to-one matching in `hedonic.metrics`).
- `cli` extra: SciPy, pandas, Matplotlib, PyArrow, DuckDB, tqdm (all but
  SciPy only off Windows). `experiments` is an alias of `cli`. The CLI's
  entry points refuse to start without it and name the extra
  (`CLI.CLI_MODULES` lists its import names), and refuse on Windows.
- Console scripts: `hedonic = "hedonic.experiments.CLI:hedonic_main"`,
  `hedonic-exp = "hedonic.experiments.CLI:main"`. Shipped configs are package
  data (`hedonic.experiments/configs/*.toml`, also in `MANIFEST.in`).

## The `hedonic` front door

```bash
hedonic                                # in a terminal: the home screen (status + menu); elsewhere the help
hedonic detect [--list]                # one network x one method x one seed, in this terminal
hedonic guide [TOPIC]                  # start, detect, disjoint, runs, config, update
hedonic show disjoint [--json]         # presets, grids, methods, estimates
hedonic run disjoint [--preset smoke|paper] [flags] [--dry-run]
hedonic run queue disjoint [--dry-run|--report] [--root DIR]
hedonic run list|status|attach|logs|stop|resume|check [NAME]
hedonic config [show|path|set KEY VALUE|unset KEY|example]
hedonic update [--yes]
hedonic-exp list                       # the experiment modules' own commands
```

Contracts to keep when changing it:

- **Hedonic 2 verbs.** `hedonic setup`, `hedonic pickle` and the overlap run
  verbs (`CLI.HEDONIC_2_RUN_VERBS`: `exp`, `paper`, `spectrum`, `calibrate`,
  `audit`, ...) and the overlap `hedonic show` topics exit 2 with a message
  naming hedonic 2; a registered run of a removed kind is refused the same
  way. Keep them failing loudly, never silently.
- **Home screen (`experiments/home.py`, the same file as hedonic 2's).** A
  bare `hedonic` opens it only when stdin and stdout are a terminal and
  `HEDONIC_PLAIN_VIEW` is unset; otherwise it prints `HEDONIC_HELP`. It asks
  the CLI which commands exist (`home._commands`: `TOP_LEVEL`, the run verbs,
  the bundled `queue-*.toml`) and offers only those, so this line shows
  Detect, the disjoint queue's plan, the disjoint benchmark, runs, presets,
  the guide, settings and the update check. Every item is a `hedonic` argv,
  printed before it runs; views come back to the menu, and a wizard the user
  backed out of (`tui.note_cancelled`) brings the menu back too.
- **Nothing starts unasked.** With no arguments and no terminal, `hedonic run
  disjoint` prints the plan and `not started: no terminal to confirm in`;
  flags or `--yes` still run directly. Every last question (`tui.confirm`)
  has **No** pre-selected. Backing out of a wizard exits 0 with `nothing
  started`; Ctrl-C is 130. Wizard values are checked at the prompt and asked
  again. `hedonic run resume` on a completed run only says where its results
  are (`--again` relaunches it).
- **`hedonic detect` (`experiments/detect.py`, the same file as hedonic 2's).**
  One network (`sbm`, `lfr`, `famous`, `file`: `SOURCE:name=value,...`) x one
  method (`hedonic`, `leiden`, `louvain`, `spectral`, `label_propagation`,
  `infomap`, `onepass`, `mirror`) x one seed, in the terminal, scored against
  the network's ground truth; `result.json` goes to `<data root>/Detect/`
  unless `--no-save`. hedonic 2 adds `overlapping/detect_overlap.py` (SNAP
  pairs, overlapping LFR, covers, the baselines); this file loads it only when
  it exists. `tests/experiments/test_detect.py` is shared too.
- **Durability.** Every run (`kind` `disjoint` or `queue`) executes detached
  in tmux session `hedonic-<name>`, or, without tmux, a detached process; on
  macOS under `caffeinate -i`. The worker ignores SIGINT and writes
  `progress.json`; the terminal shows a viewer whose Ctrl-C never stops the
  run. `stop` is explicit; `resume` relaunches with the same arguments and
  reuses finished work. `--foreground` runs the worker in the terminal,
  `--detach` returns at once, `--json` waits and prints the statistics.
  Liveness is the worker pid, checked against its command line (a recycled
  pid or an idle tmux pane is never trusted). An unexpected worker error is
  `failed` with its reason; a vanished worker is `interrupted`. Run names are
  `[A-Za-z0-9][A-Za-z0-9_-]{0,63}`; `NAME` arguments accept an exact name or a
  unique fragment.
- **Memory guard.** Every worker watches free memory
  (`runmanager.start_memory_guard`, `parallel.memory_floor`, default 10%,
  `HEDONIC_MEMORY_FLOOR`); below the floor it kills its process tree, records
  `failed` with the reason and exits 75; finished work is kept.
- **Registry.** `~/.cache/hedonic/runs/` (`HEDONIC_RUNS_DIR`), one entry per
  run with absolute paths. Tests point it, and `HEDONIC_CONFIG`, at temporary
  folders.
- **Disjoint runs.** `hedonic run disjoint` (`disjoint/front.py`) has presets
  `smoke` (16 graphs, seconds) and `paper` (50,000 graphs), every grid value
  as a flag, a wizard without arguments, and `--dry-run` with plan, time and
  disk estimates and what the store already holds per method. The worker runs
  `sbm_sweep` into the store (`--workers`, default CPUs - 2, one numerical
  thread each), then the logical-cell table, the five figures and `stats/`.
  Output: `<disjoint_output_root>/<name>` unless `--output-dir`.
- **Queue.** `hedonic run queue FILE|NAME` (`runqueue.py`) runs its items one
  after another, each an ordinary durable run started with `--detach` once
  the previous one completed. The shipped queue `disjoint`
  (`configs/queue-disjoint.toml`, queue name `disjoint-queue`) is
  `disjoint-smoke` then `disjoint-paper`. `{root}` in an item is `--root DIR`,
  by default `<disjoint_output_root>/<queue name>`. Each item is judged from
  its files on disk, twice (before anything starts, and again at its turn):
  complete is skipped, partial resumes, missing starts. `--adopt-root` is
  required to write into a root holding runs this hedonic's queues did not
  start. `hedonic run attach` on a queue opens the tree view (`queuetui`;
  `HEDONIC_PLAIN_VIEW=1` or a pipe gives the text view, which `--report`
  prints). The tree models (`queuemodel`) feed only the view and never change
  a run's counts or state.
- **User config.** `~/.config/hedonic/config.toml` (`HEDONIC_CONFIG`
  overrides the path; `XDG_CONFIG_HOME` is respected). `[defaults]` accepts
  `disjoint_output_root` (the parent of disjoint runs and queues) and
  `threads` (validated and kept, but no 1.x run reads it). Keep machine paths
  out of the repository; the template is `configs/hedonic-run.example.toml`
  (`hedonic config example`).
- **Paths.** `config.DATA_ROOT` = `~/Databases/Hedonic` (`HEDONIC_DATA_DIR`);
  disjoint runs under `Disjoint/` (`HEDONIC_DISJOINT_OUTPUT_DIR`); the
  archived synthetic results under `PHYSA/Synthetic_Networks/V1020`
  (`HEDONIC_SYNTHETIC_DIR`); `hedonic-exp` outputs and relative
  `artifacts/...` paths are anchored at `DATA_ROOT/Overlap`
  (`HEDONIC_OUTPUT_DIR`). Never hard-code a home path; import paths from
  `config`. Writers refuse the archived V1020 tree and its descendants
  (`config.ensure_not_archived_v1020`).
- **`hedonic run check`.** Replays a sample of a disjoint run's rows from
  their seeds (`--sample` graphs, `--rows` per graph) and compares them.
- **`hedonic guide` / `hedonic update`.** `guide.TOPICS` is user
  documentation: change it in the same commit as any flag or verb it names
  (tests check that the queues and configs it names exist and that it and the
  help texts name nothing removed). `update` makes a single GET to pypi.org,
  never installs without `--yes`, and never touches an editable install.
- **Identity.** A Hedonic record names its build by `runtime.build_identity()`:
  the release series (`major.minor`) and `hedonic._core.source_sha256`, the
  digest of the C sources `setup.py` embeds at build time (the compiled
  bytes' SHA-256 stays in `runtime_versions()` as provenance). A change of
  the Python layer that changes results bumps the minor version. The store's
  per-method identity (`store.method_identity`: call, parameters, restarts,
  `ROW_CODE`, the runtime fields of `METHOD_RUNTIME_FIELDS`) decides reuse; a
  store holding another identity is refused before any write, and `--rerun` /
  `--reuse-from` resolve it. The store's `duration` is the detector call only
  (`store.TIMING_SCOPE = "detector-call-v1"`, part of the spec). There is no
  igraph pin and no protocol lock file.
- **Two fingerprints (disjoint store).** The store's Hedonic rows are keyed on
  what the core computes, not on its bytes: `hedonic.experiments.fingerprint`
  runs a fixed grid of 8,736 partition calls (`GRID = "partitions-v1"`; seconds,
  cached per compiled core in `~/.cache/hedonic/fingerprints`,
  `HEDONIC_FINGERPRINT_CACHE`) and hashes memberships, quality and slack bits;
  `store.runtime_identity()` carries it as `hedonic_partition_fingerprint` next
  to `hedonic_core_source_sha256`, and `METHOD_RUNTIME_FIELDS["Hedonic"]` keys on
  it, so hedonic 1 and hedonic 2 (other sources, the same partitions) share a
  store. Each shard records the `source` that wrote a method's rows
  (`SOURCE_FIELDS`); `disjoint-store` lists the graphs per source. Before a run
  keeps rows of a source other than the loaded one, `store.check_adoption`
  replays a sample of them with this build (5 shards x 10 rows per source) and
  refuses on any difference (`--rerun hedonic` recomputes those rows); a source
  it accepted is recorded in `manifest["adopted"]` and not replayed again.
  A store of the same build written before the fingerprint stays current
  (`store.identity_aliases`). The pinned fingerprint is in
  `tests/experiments/test_fingerprint.py`: changing it means every store's
  Hedonic rows are recomputed. Other records (overlap studies) keep the source
  identity.

## CLI registry (`experiments/CLI.py`) — keep it in sync

`hedonic-exp` subcommands are registered in `CLI.COMMANDS`, the single source
of truth:

| Subcommand | Module | Purpose |
|---|---|---|
| `smoke` | (built in) | Isolated check, no data: `disjoint --smoke` into a temporary folder (`--keep-output`) |
| `disjoint` | `disjoint.sbm_sweep` | SBM sweeps: presets `sbm`/`sbm-smoke` (Parquet store), `v1020`/`v1020-smoke` (archived JSON layout); `--preflight`, `--resume`, `--workers`, `--partitions`, `--rerun`, `--reuse-from` |
| `disjoint-store` | `disjoint.store` | Summarize, `--sql` (DuckDB over `runs`/`cells`), `--progress`, `--export` the logical-cell table, `--verify` (replay rows from their seeds) |
| `disjoint-stats` | `disjoint.paper_stats` | Every statistic of the paper's Empirical Analysis from a store or its cell table (`summary.md`, `paper_stats.json`, `tables.tex`, CSVs) |
| `disjoint-load` | `disjoint.data_loader` | JSON results (V1020 or legacy layout) -> gzipped CSV (`--simple`) |
| `plots` | `plots.paper_figures` | The five paper figures from a store, cell table or CSV (`--smoke` draws a synthetic table) |
| `reproduce-disjoint` | `disjoint.reproduce` | End-to-end sweep -> table/CSV -> figures (default preset `sbm-smoke`; `--plots-only`, `--preflight`) |
| `list` | meta | List subcommands |

`hedonic-exp disjoint` and `reproduce-disjoint` with an `sbm`/`sbm-smoke`
preset print a one-line notice naming `hedonic run disjoint`
(`CLI._run_equivalent`); the `v1020` presets and custom grids have no durable
equivalent. `RUN_EQUIVALENTS` is empty in 1.x.

### Agent rule: always update the CLI when experiments change

Any change to the experiment surface updates the CLI and this file in the
same change:

| Change | Required updates |
|---|---|
| Add a runnable module with `main(argv=None)` | Register it in `CLI.COMMANDS`; this table; a test |
| Remove or rename an experiment | Drop/rename the entry; fix help examples, `guide.TOPICS` and this table |
| Change argparse flags or defaults | `hedonic-exp <cmd> --help` still works; update examples here, in `CLI.py` and in `guide.TOPICS` |
| Add a preset | A flag on the module and, if users run it durably, a `front.PRESETS` entry |
| Move `main` / a module | Fix `Command.module` / `Command.attr` |
| Change an entry point | Keep both `[project.scripts]` strings accurate |

Checklist:

1. [ ] Module exposes `main(argv=None)` (import-safe, no work at import time).
2. [ ] Registered in `CLI.COMMANDS` (or a `hedonic` verb in `CLI.hedonic_main`).
3. [ ] `hedonic-exp <name> --help` works.
4. [ ] This file, `guide.TOPICS` and the help texts updated; none advertises a removed feature.
5. [ ] Test added or updated under `tests/experiments/`.

Do not add a second entrypoint module; extend `CLI.py`.

## Disjoint pattern (`disjoint/sbm_sweep.py`)

1. `generate_graph` builds an SBM as a pure function of `(n_communities,
   community_size, p_in, difficulty, seed)` under `SBM_GENERATOR_ID`
   (`hedonic-sbm-pcg64-threshold-v1`): each vertex pair consumes one raw
   PCG64 output keyed by a `SeedSequence` of the id, the seed and the exact
   decimal parameters. No global RNG is read or written, so graphs are
   regenerated, never stored; `graph_fingerprint` is a SHA-256 identity.
   Golden fingerprints are pinned in `tests/experiments/test_disjoint_sweep.py`;
   if sampling must change, bump the id rather than editing them.
2. Ground truth from the block labels; noisy starts at each noise level and
   partition seed.
3. `METHODS` maps a name to `method_call_name` + parameters: GroundTruth,
   Mirror, OnePass, Spectral, Leiden (stock igraph CPM Leiden through
   `experiments.leiden.leiden`) and Hedonic (`community_hedonic`). Leiden and
   Hedonic run `--n_runs` seeded restarts; the others once per cell.
4. Every call runs under a seed derived from the row's keys, so rows replay on
   the same build.
5. Layouts: `store` (`manifest.json` + one atomic Parquet shard per graph,
   one row per detector run; `--partitions hash|full`; an exclusive `flock`
   on `.writer.lock`), `v1020` (`resultados/.../partition_*.json`, as the
   archived PHYSA V1020 tree) and `legacy`.
6. Presets: `sbm` (the paper: 50,000 graphs of 1,020 vertices, hashes only,
   ~3.1 GB), `sbm-smoke` (16 graphs, full partitions), `v1020` (the archived
   five-seed grid, JSON layout; refuses the archive folder), `v1020-smoke`,
   and `--smoke` (a legacy mini run).
7. Figures (`plots`): `gt_robustness`, `noise`, `n_communities`,
   `acc_robustness`, `acc_efficiency`.

When adding a method, extend `METHODS` and give it an identity; do not
re-implement Leiden or hedonic moves in an experiment.

### V1020 reproduction (archived JSON layout)

```bash
hedonic-exp disjoint --preset v1020 --output_root artifacts/disjoint/v1020 --preflight
hedonic-exp reproduce-disjoint --preset v1020 --output_root artifacts/disjoint/v1020
hedonic-exp reproduce-disjoint --plots-only \
  --data ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/resultados_ari.csv.gzip \
  --output_root artifacts/disjoint/v1020 --max_rows 50000
```

Step by step: `hedonic-exp disjoint --preset v1020 ...`, then
`hedonic-exp disjoint-load --results_folder ROOT/resultados --simple`, then
`hedonic-exp plots --data ROOT/resultados.csv.gzip --output_dir ROOT/figures`.
Never write into the archived `.../Synthetic_Networks/V1020` (the guards
refuse it).

## Checks

```bash
python -m pip install -e ".[cli]" pytest       # or: uv pip install -e ".[cli]" pytest
python setup.py build_ext --inplace            # after editing native/
make -C native test test-release test-soft-fma test-x86_64   # C tests, libm check, x86_64 (Rosetta)
python tools/native_tests.py                   # pinned Python tests (fails on any skip)
python -m pytest                               # library and tests/experiments
hedonic-exp smoke                              # the disjoint pipeline, no data
```

`tests/experiments/` needs the `cli` extra and no data. `.github/workflows/wheels.yml`
runs the C tests (GCC, Clang, MSVC) and builds and tests the wheels on Linux,
macOS and Windows; every wheel job also runs `tools/native_tests.py`
(HIG_PINNED=1, no skips), and publishing needs every job. Branch pushes only
build and test; a tag publishes and sets the version
(`tools/release_version.py`; `pyproject.toml` holds the release):
`v1.0.0.devN` goes to TestPyPI, `v1.0.0rcN` (a pre-release) and `v1.0.0` to
PyPI, by trusted publishing (environments `testpypi`, `pypi`). After each
upload the `verify` jobs install that version from its index and run the
pinned tests and the test suite.

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
- Documentation pages are `web/content/docs/*.md`; the public site never reads
  private paper material.
- Re-run `npm run docs:generate` (and commit `web/src/generated/*.json`) when
  `Game`, `hedonic.utils`, `hedonic.metrics` or `pyproject.toml` change; it
  parses the sources with `ast` only (`web/scripts/extract_api.py`).
- `web/scripts/crosscheck_native.py` (needs the Python package) refreshes the
  fixture that checks native results against the browser model's sinks.
