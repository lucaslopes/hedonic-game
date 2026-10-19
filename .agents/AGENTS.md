# AGENTS.md — Working on `hedonic`

Guidance for humans and coding agents editing this repository. `hedonic` is a
small library (one public class, `Game`, one detector,
`Game.community_hedonic`, and a fixed-cover stability audit, all in a C core
compiled into `hedonic._core`) plus the experiment command line that ships
with it (`hedonic`, `hedonic-exp`).

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
- **`snap-networks`** is the path for the SNAP ground-truth networks
  (`~/Databases/Hedonic/Networks`, read-only): `network:cover` pairs, the
  loader's ID alignment, the covered-induced subgraph, detection with
  `community_hedonic` and scoring with `hedonic.metrics`, the `hedonic run exp`
  / `show` / `pickle` verbs, an inventory of the pairs (Table 1 sizes) and
  `scripts/inspect_snap_network.py`, a bounded read-only look at the archive.
  It is overlap-only, so it exists on hedonic 2, not on `release/1.x`.
- `clean-worktree-commits` (a user-level skill, not in this repository) turns
  a dirty checkout into reviewable local commits before a release.

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
│   └── check/
│       ├── audit.c      # the fixed-cover stability audit (Game.stable_intervals, Game.regrets)
│       ├── check.c      # the independent verifier (tests only)
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
├── _audit.py            # argument buffers of the C audit
├── stability.py         # aggregates of stable intervals (stable fraction, measures, Nash window)
├── metrics.py           # scores against a reference cover
├── utils.py             # seeded sampling for the random disjoint start
└── experiments/         # the command line: see "Experiments" below
    ├── CLI.py           # `hedonic` and `hedonic-exp`; COMMANDS is the registry
    ├── config.py        # DATA_ROOT, RUNS_ROOT, NETWORKS_DIR, codeseg_cache_dir()
    ├── resources.py     # shipped configs and worker scripts (package data)
    ├── runtime.py       # the identity every record carries (release series, digest of the core's sources)
    ├── configs/         # run configs and the queues (queue-overlap.toml, queue-disjoint.toml, queue-all.toml)
    ├── workers/         # scripts external methods run in their own environments
    ├── disjoint/        # SBM sweep, Parquet result store, `hedonic run disjoint`
    ├── plots/           # figures
    └── overlapping/     # SNAP loading, methods, studies, durable runs and queues
tests/                   # library tests; tests/experiments/ for the command line
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
3. The library's modules import only igraph (`>=1.0`) and NumPy, and SciPy
   inside `hedonic.metrics` for one-to-one matching. Those two are the only
   dependencies of `pip install hedonic`; SciPy is the `metrics` extra, and
   the command line's (SciPy, pandas, Matplotlib, NetworkX, PyArrow, DuckDB,
   tqdm) are the `cli` extra (`experiments` is its 1.x name). The CLI's entry
   points refuse to start without it and name the extra (`CLI.CLI_MODULES`
   must list the `cli` extra; a test checks). Nothing checks igraph's version
   at import.
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

## The stability audit

`Game.stable_intervals` and `Game.regrets` audit a fixed cover over
resolution [0, 1] (`native/core/check/audit.c`, `native/README.md`); the
tolerance is the paper's `1e-10 + 1e-9 * max(1, |utility|)`, not the
detector's certified slack. Every experiment that audits a cover (cover
audits and the benchmark's `--robustness` stage, the post-hoc analysis, the
ground-truth spectrum, the calibration's label-free audit) goes through
`hedonic.experiments.overlapping.stability`: `prepare()` builds the C state
(engine `"c"`, the default) or the pure-Python reference (`"python"`,
`robustness.FractionalState`, for tests); both return the same floats, bit
for bit, and `tests/experiments/test_studies_audit_engines.py` checks every
consumer on both. `--workers` forks processes that split vertex ranges over
the read-only C state. The reference spells out CPython 3.12's compensated
`sum` for the two current-utility coefficients, so it does not depend on the
interpreter. Do not add a second audit implementation.

## Experiments

```bash
hedonic                                # in a terminal: the home screen (status + menu); elsewhere the help
hedonic guide                          # state-aware tour of the commands
hedonic setup --all [--dry-run]        # SNAP pairs and the eight baselines' runtimes
hedonic run exp [--dry-run]            # benchmark front door (wizard in a terminal)
hedonic run queue overlap --root DIR     # the paper's queue, in file order (two phases); --dry-run, --report
hedonic run queue disjoint             # the disjoint paper (smoke, then the 100-seed grid)
hedonic run list|status|attach|logs|stop|resume|check
hedonic show methods|networks|covers|metrics
hedonic-exp list                       # analysis and utility commands
```

- **Home screen (`experiments/home.py`).** A bare `hedonic` opens it only when
  stdin and stdout are a terminal and `HEDONIC_PLAIN_VIEW` is unset; otherwise
  (and for `--help`, always) it prints `HEDONIC_HELP`, so scripts are
  unchanged. The header reads local state only (pair files present,
  `info.method_status`, the run registry; a queue's progress is not counted).
  Every menu item is a `hedonic` argv that is printed before
  `CLI.hedonic_main` runs it: views (`back=True`: lists, `show`, `setup
  --dry-run`, `run queue overlap --dry-run`, `config show`, `update`) return to
  the menu, runs and the wizards hand the terminal over; every wizard ends
  with a yes/no (`quickstart.confirm_start`) and resuming asks first. Keep
  each item a real command, never a screen-only action;
  `tests/experiments/test_home.py` drives it in a pty. The cursor returns to
  the item chosen last, and a command the user backed out of (q, Esc, Quit:
  `tui.note_cancelled`, read by `tui.take_cancelled`) brings the menu back
  instead of ending the program.
- **Nothing starts unasked.** With no arguments and no terminal (a pipe, a
  script, `</dev/null`), `hedonic run`, `hedonic run exp` and `hedonic run
  disjoint` print the plan and `not started: no terminal to confirm in`; flags
  or `--yes` still run directly, so scripts and queues are unchanged. Every
  last question (`tui.confirm`, the wizards' `Start?`, resume from the home
  screen) has **No** pre-selected, so Enter alone never starts a run. Backing
  out of a wizard (q, Esc, Quit) is a choice: exit 0, `nothing started`;
  Ctrl-C is 130. Typed wizard values are checked at the prompt and asked
  again with the reason (`tui.text(validate=)`), never left to fail at the
  end. `hedonic run resume` on a completed run only says where its results
  are (`--again` relaunches it to re-check its records); the queue's own
  resume path is unchanged. Machine folders follow one rule: flag >
  environment (`HEDONIC_NETWORKS_DIR`, `HEDONIC_OUTPUT_DIR`) > user config >
  default.
- **`hedonic detect` (`experiments/detect.py`, `overlapping/detect_overlap.py`).**
  One network x one method x one seed, run in the terminal and scored
  against the network's ground truth; `result.json` goes to
  `<data root>/Detect/<source>-<method>-<stamp>` unless `--no-save`. A
  network is `SOURCE:name=value,...` (`sbm`, `lfr`, `famous`, `file`), a
  method `METHOD:name=value,...` (`hedonic`, `leiden`, `louvain`,
  `spectral`, `label_propagation`, `infomap`, `onepass`, `mirror`); `--list`
  prints both registries with every parameter, and the run prints its own
  command line. `detect.py` is the same file in hedonic 1 and 2. hedonic 2
  adds `detect_overlap.py`, which `detect._load_overlap` registers when it
  exists: sources `snap` (a `network:cover` pair from disk, covered-induced
  by default, `max_nodes`, `download=yes` to fetch) and `lfr_overlap`
  (the controlled-overlap generator), `max_memberships` on `hedonic` (a
  cover above 1), the eight baselines through the benchmark's own
  `codeseg_reproduction._run_method` at their screened settings (unavailable,
  with the `hedonic setup` command, until their runtime is ready), and cover
  scores. A detection never downloads or builds by itself unless asked.
  `tests/experiments/test_detect.py` covers both halves.
- **Everything comes from the package.** Configs resolve through
  `resources.config_path` (a path that exists wins, then
  `hedonic/experiments/configs/`), worker scripts through
  `resources.worker_path`. Never reach the repository with
  `Path(__file__).parents[N]`, and never require a file outside the package.
- **Paths.** Everything defaults to one folder, `~/Databases/Hedonic`
  (`config.DATA_ROOT`, `HEDONIC_DATA_DIR`): SNAP files under `Networks/`
  (`HEDONIC_NETWORKS_DIR`), overlapping runs under `Overlap/`
  (`config.RUNS_ROOT`, `HEDONIC_OUTPUT_DIR`; the overlap queue in
  `Overlap/overlap-queue`), disjoint runs under `Disjoint/`
  (`config.DISJOINT_RUNS_ROOT`, `HEDONIC_DISJOINT_OUTPUT_DIR`), pickles under
  `Pickles/`. Tests point these at temporary folders. External
  runtimes live in `config.codeseg_cache_dir()`: `DIR/codeseg` for a `--cache-dir DIR` flag (the
  same for `hedonic setup` and `hedonic run exp`), else
  `HEDONIC_CODESEG_CACHE`, else the user config's `cache_dir/codeseg`, else
  `~/.cache/hedonic/codeseg`. Machine settings are in
  `~/.config/hedonic/config.toml` (`hedonic config`). Never hard-code a home
  path.
- **Runtimes.** `hedonic setup` downloads pairs and builds or installs the
  baselines; an unavailable method's message names `hedonic setup --methods
  NAME`. CoDeSEG has no licence file and is never vendored; CDlib runs only
  in its locked environment, never in the hedonic process.
- **Identity.** A record's identity names Hedonic by `runtime.build_identity()`:
  the release series (`major.minor`, so `2.0.0.devN`, `rcN` and patch
  releases share it) and `hedonic._core.source_sha256`, the digest of the C
  sources that `setup.py` embeds at build time (rounding is fixed by the
  source, so every wheel of one source computes the same results). The full
  version and the compiled bytes' SHA-256 stay in `runtime_versions()` as
  provenance only. A change of the Python layer that changes results bumps
  the minor version. Every study checks it before reusing work: a
  calibration search (`build`), a start-quality key (`code=`), the
  spectrum's runtime identity, a benchmark unit's `config_digest`, the
  controlled study's environment, and the disjoint store's per-method
  identity (`store.method_identity`: call, parameters, runs, `ROW_CODE`,
  runtime fields; a store holding another identity is refused before any
  write, `--rerun` / `--reuse-from` resolve it). There is no igraph version
  pin and no protocol lock file.
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
- **Hedonic calibration, no cap.** `hedonic run calibrate --cap unlimited`
  searches the resolution only at `max_memberships = -1` (no cap: the
  detector's cap is the vertex count), on a log grid over [1e-6, 0.5] at
  factor-4 spacing before golden section (`calibration_search.WIDE_SCAN`),
  because the F1 profile can peak two to three decades below the density.
  The overlap queue uses it: the cap is the one setting that never reads the
  reference. A record of an uncapped Hedonic unit carries the vertex count as
  `max_memberships` and `max_memberships_source = "unlimited"`; the robustness
  audit then plays the uncapped game. A search counts only under its own cap
  mode (`calibrate.same_cap_mode`).
- **One queue per paper, and queues of queues.** The shipped queues are
  `overlap` (`queue-overlap.toml`, run `overlap-queue`), `disjoint` and `all`
  (both, one after the other); `paper` is refused with a message naming them.
  An item `run = ["queue", FILE|NAME, ("--root", DIR)]` is a whole queue
  (`runqueue._queue_item`): it runs under the inner queue's name, is started
  with `hedonic run queue ... --detach` (never `resume`, so it re-judges its
  items), is complete when every inner item is (`_queue_state`), adds the inner
  requirements to the setup check, and the tree hangs the inner tree under it
  (keys prefixed `("in", NAME)`). A queue that contains itself is refused.
- **Time limits are per single run** (`limits.py`, `queuelimits.py`). A queue
  file's `[limits]` (`timeout`, `max_eta`: seconds or `"90s"`, `"30m"`,
  `"2h"`, `"1h30m"`; `"none"` removes one) and an item's `limits = {...}`
  override the limits inherited from `hedonic run queue --timeout/--max-eta`
  or from the queue that runs this one (passed down as those flags). The
  item's row keeps its merged `limits`. `timeout` reaches the runner as
  `--limit-seconds`, which only lowers its own per-run timeout and is never
  part of an identity: the benchmark's detector run (`codeseg_reproduction`,
  a timer around an in-process detector, the program's own timeout
  otherwise), one probe of `calibration_search` (a forked child), one
  evaluation of `baseline_calibration` and one start-quality condition. A run
  it stopped is `status: "timeout"` with `limit_seconds` and stands only under
  a limit no larger (`limits.stands`; `unit_integrity.final`). Never measure a
  limit against a sum of runs: a calibration of thirty short probes is thirty
  runs. The spectrum's `timeout_seconds` is in every condition key, so it is
  not lowered (said in `--dry-run`); `controlled`, `disjoint`, `posthoc` and
  `audit` take no limit. `max_eta` is judged by the worker at an item's (or a
  pair's) turn from `queuelimits.estimates`: one run per pair, measured on the
  pair, else the same runs on other pairs scaled by edges, else the method's
  planning constant, bounded by the run's timeout; a pair with no estimate is
  never skipped. A skipped benchmark, start-quality or spectrum pair leaves
  `--networks`/`--pairs`, a skipped baseline-calibration pair moves to
  `--transfer`; the Hedonic calibration is never skipped on an estimate.
- **The overlap queue goes method by method, in two phases** (`schedule =
  "file"`). Phase 1 is DBLP, Amazon and YouTube (`pairs`, each network's
  `top5000` and `all` covers together): Hedonic first (`synthetic-hedonic`,
  `calibration-hedonic`, then `gt-robustness`, `leveraging-gt` and
  `benchmark-hedonic` pair by pair), then each baseline: `calibration-M`
  (adopting the finished searches of `baseline-calibration-rc1`) and
  `benchmark-M`. Phase 2 repeats every method, Hedonic first, on
  LiveJournal/top5000 and Friendster/top5000 (the `-large` items: the largest
  covered subgraphs, last so a slow method there cannot hold up the rest;
  `gt-robustness-large` lists all eight pairs in the `gt-robustness` folder,
  so one coverage report describes them). `leveraging-gt` caps every vertex
  at the reference's largest multiplicity (`start-quality --cap reference`). Every
  benchmark item writes into `{root}/benchmark` (`--shared-folder`: the
  folder itself, not OUTPUT_DIR/NAME), so finished units are reused
  and each item waits for the earlier ones (an item depends on every earlier
  item whose folder its arguments name). Each phase ends with its robustness
  audit (`audit`, `audit-large`: `hedonic run audit --source {root}/benchmark
  --skip-missing` into `{root}/benchmark/robustness`, the ledger the old
  inline stage wrote): the benchmark items run `--no-robustness`, so every
  detector run of a phase (accuracy and time) is in before any audit starts.
  `tests/experiments/test_queue_skip.py` pins the order.
- **Answers first, audits and scores after.** A benchmark unit run with
  `--save-covers` keeps its answer (`overlapping/cover_store.py`): the cover,
  Hedonic's labelled rows in `<runner>.labels.json.gz` (older records keep
  them inline in `detector.final_memberships`), one `vertex_ids.json.gz` per
  seed folder (SNAP id of each analysis vertex) and the analysis graph's
  SHA-256 in `dataset_report.graph_sha256`; the seal covers the labels too.
  The audit and `hedonic-exp overlapping-rescore` (`cover_rescore.py`: the
  runner's own `codeseg_reproduction.score_cover` with other settings;
  `--check` reproduces each record) refuse a loaded graph whose hash differs.
  An audit counts only while it describes the cover on disk
  (`cover_audit.done`: its cover's or labels' hash, or its digest, against the
  unit's seal); a unit that ran again is audited again and the ledger's last
  record of a key wins. Read audit records through `done`, never by key alone.
- **The queue view nests by method.** When a queue's items study more than one
  method, `hedonic run attach` groups its runs under one node per method
  (`runqueue.item_method`: Hedonic for its own verbs, else the single method of
  `--methods`), then experiment > pair > loops. A method's row (and the root)
  counts runs finished of runs but fills its bar and percent with the mean of
  its runs' fractions (`queuemodel.fraction`, `Node.share`), so two runs at 7/8
  and 6/8 read 0/2 runs and 81%. A Hedonic calibration search
  counts its evaluations (`queuebuild.decorate_calibration`, read from
  `ledger/evals*.jsonl`) and lists one row per probed setting (γ, F1, communities,
  seeds, ★ chosen); a search under way is priced from its own pace, and the run's
  ETA is its longest search, since they run side by side.
  A baseline calibration counts the searches it will adopt (`--adopt-from`,
  copied only when it starts) as done, and a benchmark waiting for its
  `--tuned` file is counted with the adopted files (`runqueue._preview`); it
  stays `awaiting` and is judged with the real file at its turn.
- **Fixed-setting baselines load nothing.** CoDeSEG, NcGame and Ego-Splitting
  take the authors' setting on every pair (`bc.FIXED_METHODS`): without
  `--sensitivity` their per-pair record is written without loading `G[S]`.
- **Baseline calibration.** `hedonic run calibrate-baselines` takes
  `--method-budget M=N[:MIN]` (the overlap queue: ANGEL at 3 settings, 60 min
  per pair), `--transfer PAIRS` (not searched: each searched method takes the
  nearest calibrated pair's setting, same network first, else nearest
  density, K scaled by n for BigCLAM and NEO-K-Means), `--transfer-missing`
  (the same for a search that ended without a setting) and `--adopt-from DIR`
  (copy an earlier calibration's finished searches and its ledgers; the
  source is never changed). Every transferred or adopted entry names its
  source; report them as such.
- **Durable runs.** Every `hedonic run` kind runs detached (tmux, else a
  detached process), writes `progress.json`, survives a closed terminal, and
  resumes from its files. `--foreground` runs the same worker in the
  terminal and registers it under its `--name`. A queue judges each item, and each pair of a
  by-pair item, from the files on disk; never reorder queue items to change
  the pair order (that is `pairs` in the queue file). The overlap queue's pairs
  are the manuscript's (Table 1's "in paper" marks): DBLP, Amazon and YouTube
  with both covers in phase 1, LiveJournal and Friendster `top5000` in phase 2.
- **Calibrated resolutions.** The benchmark, the start curve and the spectrum
  read `hoc_tuned.json` through `--tuned`. The spectrum takes only each
  pair's gamma and adds it to that pair's detector and audit grids
  (`gt_spectrum.detector_resolutions` / `audit_resolutions`; the cap stays the
  cover's multiplicity); a pair the file does not name is an error, and the
  key `tuned_resolutions` enters the scientific grid only when a file is
  read, so studies without one keep their hash. A queue item whose file is
  missing or unfinished is `awaiting`.
- **Friendster.** Its edge file (9.4 GB compressed) is never built whole:
  every covered-induced loader (`codeseg_reproduction._load_dataset`,
  `gt_spectrum.load_job`, `cover_audit.load_graph`, `snap_pickle`) streams it
  when `ground_truth_data.stream_covered_wanted` says so.
- **Memory.** Detector runs whose time or memory is a result run one at a
  time; parallel steps use `overlapping/parallel.py` (forked workers, a
  free-memory floor) and plan their workers with `overlapping/capacity.py`.
  Check a new heavy step under a memory guard before running it for real.
- **CLI registry.** Every runnable `main(argv=None)` is registered in
  `CLI.COMMANDS` (import-safe, `--help` works, a test names it). Commands
  that `hedonic run` performs durably stay registered in `hedonic-exp` with a
  deprecation notice (`RUN_EQUIVALENTS`). `overlapping/guide.py` (`TOPICS`) is
  user documentation: change it with any flag or verb it names. No help
  text advertises a removed feature.
- **Replication.** `overlapping/replication.py` (`REPLICATION`) is the one
  table of how each overlap method behaves under a seed: `deterministic` (one
  run per setting is the result), `seeded` (the seed reaches the program) or
  `replicas` (stochastic, no seed reaches it); `calibration_kind` overrides it
  for the baseline calibration's own settings. `baseline_calibration`
  re-exports it, and the benchmark plans from it: `quickstart.build_plan`
  runs a deterministic method only at the lowest seed of the plan unless the
  run asks `replication = "all"` (`--replication all`). Everything that
  counts or reads units goes through that plan (the scan, `_final_check`,
  `runqueue.data_state`/`unit_counts`, `queuebuild.benchmark`,
  `cover_audit.conditions` for the stage and `hedonic run audit`, the tables,
  `hedonic run check`, `plots/benchmark_tradeoffs.py`), and the setting never
  enters a unit's arguments, digest or folder. Classify a new method there,
  from runs at the benchmark's own settings (threads included): CoDeSEG
  repeats at one thread but not at all cores, so the benchmark gives it no
  `threads` (the runner's default, one thread).
- **Hedonic only through `Game.community_hedonic`.** The method's key is
  `hedonic` (runner `hedonic_local`). The 1.x multilevel phase and the
  multi-phase variants are refused with a message saying they are not
  available in hedonic 2.0; full CPM Leiden for comparison is
  `hedonic.experiments.leiden.leiden` (stock igraph).
- **Tests.** `tests/experiments/` runs without the SNAP data; an autouse
  fixture undoes `extra_methods.install()` after every test.

## Checks

```bash
python -m pip install -e ".[cli]" pytest       # or: uv pip install -e ".[cli]" pytest
python setup.py build_ext --inplace            # after editing native/
make -C native test test-release test-soft-fma test-x86_64   # C tests, libm check, x86_64 (Rosetta)
python tools/native_tests.py                   # pinned Python tests (fails on any skip)
python -m pytest                               # library and tests/experiments
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
  which keeps the largest child ever waited for). On Linux exec carries the parent's high-water mark into the child, so
  there the peak is the child's `VmHWM` sampled from `/proc` every 10 ms (`peak_rss_source = "proc_vmhwm_sampled"`). An in-process detector can only report the harness peak when the
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
