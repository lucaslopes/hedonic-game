# Changelog

## 2.1.0

- The core is now written in Rust (`native/rust/`, compiled by maturin into
  `hedonic._core`); hedonic 2.0's C core leaves the package. Nothing it
  computes changes: every call returns the same memberships, quality bits and
  `equilibrium_slack` as 2.0, on every platform, and the stability audit the
  same intervals and regrets. The bit pins of 2.0's C tests hold unchanged in
  the crate's tests (`cargo test`) and in the pinned Python tests. Rounding is
  still fixed by the source: every fused multiply-add is the instruction or
  the core's own correctly rounded soft fma (x86_64 picks one at run time),
  and the core calls no libm function but `sqrt` and `fabs`.
- It is faster: about 1.2 times as fast as the C core on the benchmark's
  geometric mean, 1.5 to 2.5 times on small graphs (the random generator's
  state is read in place) and 1.3 to 1.6 on large ones (prefetching ahead of
  the queue); see `native/rust/README.md`.
- To compare the two releases, install `hedonic==2.0.*` and `hedonic==2.1.*`
  in two environments and run the same calls (README, "Comparing hedonic 2.0
  (C) and 2.1 (Rust)"). `hedonic._core.backend` is `"rust"`. A second build of
  the core installed beside it as `hedonic._core_<name>` is used in its place
  with `HEDONIC_CORE=<name>`, so the same program runs on either.
- Building from source needs a Rust toolchain instead of a C compiler.
- Records name the Rust core by the digest of its sources
  (`hedonic_core_source_sha256`, embedded by `native/rust/build.rs`) and the
  release series `2.1`.

## 2.0.0

- `hedonic` computes communities with its own purpose-built C core
  (`native/`, compiled into `hedonic._core`): best-response local moving of
  the CPM hedonic game, for partitions (`max_memberships=1`) and overlapping
  covers (`max_memberships > 1`). Every call runs until every vertex has been
  visited without a certainly improving move since the last move, so the
  result is an equilibrium; there is no iteration budget, no round cap and no
  tolerance. Every payoff the core compares carries a rigorous bound of its
  rounding error, and a vertex moves only when its gain exceeds it, so every
  move raises the exact potential and every call terminates
  (`native/README.md`).
- Results report `equilibrium_slack` in `_hedonic_provenance`: no unilateral
  deviation raises a vertex's exact payoff by more than the slack times the
  total edge weight. For a partition it is 0 when the arithmetic is exact
  (for instance unit weights at resolution 0, 1/2 or 1): an exact Nash
  equilibrium; other weights often give 0 too. A cover's payoffs divide by
  square roots, so its slack stays a tiny positive bound (`1e-15` to `1e-17`
  of the total weight, smaller on larger graphs).
- When the queue of vertices to visit runs dry, the vertices not yet settled
  are visited again in a fresh random order. A fixed order made complete
  YouTube at its density take 11 to 18 minutes; it now takes under 5 s.
- The fixed-cover stability audit (`Game.stable_intervals`, the
  experiments' robustness audits) no longer visits every cell between
  pairwise score crossings. The best response is convex in the resolution,
  so best-set lines bound each vertex's stable window, a convexity test
  certifies whole ranges of cells as stable, and only the cells near the
  ends of the stable set are evaluated. The intervals are the same, bit for
  bit (the exhaustive search remains as a test oracle). On complete YouTube
  with an uncapped Hedonic cover the exact stage takes 17 to 32 s instead of
  48 to 63 minutes.
  Records keep naming the previous core (`runtime.EQUIVALENT_CORES`), so
  finished runs are reused; `hedonic_core_built_source_sha256` records the
  core's own source digest.
- Records name the core by what it computes, not by its bytes (2.0.0rc12),
  so a finished run stays valid on any core that computes the same results:
  a later C core, or the Rust core of 2.1. Three fingerprints hash fixed
  grids of partitions (`partitions-v1`), covers (`covers-v1`) and stability
  audits (`audit-v1`) bit for bit; they are computed once per compiled core
  (about 5 s) and cached. `runtime.CORE_CLASSES` maps a core's three
  fingerprints to the source digest its class's records name; the rc5 to
  rc11 cores are one class, and every identity (the benchmark's units, the
  calibrations, the start-quality keys, the spectrum, the controlled study,
  the disjoint store) is the same, byte for byte, as under rc11. A core with
  other fingerprints names its own digest, so its records start afresh. The
  Python layer is named by its results series (`2.0`), not the package
  version, so `2.0.x` and a 2.1 that computes the same results keep every
  record. The ground-truth spectrum, which also hashes its own source files,
  reads an equivalence table of code digests kept outside those files
  (`hedonic.experiments.spectrum_code`): an edit that touches only identity
  or plumbing is entered there and its records stay current.
  - `hedonic run check --replay N [NAME | --root DIR]` replays a run's N
    cheapest Hedonic units and N cover audits (on a vertex sample) with the
    loaded core and records a clean pass in the run's `adopted.json`;
    `--dry-run` lists what it would replay and which cores wrote the units.
- The method is called `hedonic` everywhere in the command line: the
  calibration writes `hedonic_tuned.json`, the start-quality config is
  `start-quality-hedonic` and the Hedonic-only profile is `paper-hedonic`
  (`hedonic run paper-hedonic`). Files and names of earlier releases are
  still read: a folder's `hoc_tuned.json` stands in for its
  `hedonic_tuned.json`, `start-quality-hoc-local` resolves to the renamed
  config, and `paper-hoc` is refused with its new name.
- The library needs only the standard `igraph>=1.0` package and NumPy.
  Importing `hedonic` does not check igraph's version, so it installs next to
  any igraph 1.x. The package also depends on SciPy, pandas, Matplotlib,
  NetworkX, PyArrow, DuckDB and tqdm for its experiment command line. The
  experiments' robustness audits run on the C audit, so there is no Numba
  extra.
- `Game.community_hedonic(initial_membership, *, max_communities,
  max_memberships, resolution, allow_isolation, edge_weights, seed,
  node_weights, max_total_communities, n_communities)`: everything after
  `initial_membership` is keyword-only. The 1.x keywords `n_iterations`,
  `local_move_only`, `beta`, `debug_trace`, `phase_policy` and
  `ensure_equilibrium`, and 1.x-style positional calls, raise a `TypeError`
  that says what replaces them.
- Every call starts afresh, from its default start (singletons, a random
  start with `max_communities`, or a feasible start under a count limit)
  unless `initial_membership` is given; an igraph `VertexClustering` or
  `VertexCover` is accepted as a start. The result is stored in
  `Game.memberships`; pass `initial_membership=g.memberships` to continue
  from it.
- A start with more labels per vertex than `max_memberships` keeps each
  vertex's preferred labels (ranked by the payoff of each community on the
  start), so a cover can start a partition; `_hedonic_provenance` counts the
  `projected_vertices`.
- The graph must be undirected and free of self-loops (the error names
  `g.simplify(multiple=False, loops=True)`); parallel edges count as one edge
  with their summed weight. The default resolution is the weighted density,
  the total edge weight over the number of vertex pairs.
- `seed=` draws the visiting order from `random.Random(seed)`; without it the
  order comes from Python's `random` module.
- Community-count limits: `max_total_communities` (at most K) and
  `n_communities` (exactly K), enforced at every move.
- `hedonic.metrics` and `Game.evaluate_against` score a result against a
  reference cover or partition (a flat reference may use any labels); the
  scores built on a one-to-one matching use SciPy (the `metrics` extra is
  kept for older install lines). `restrict_to_reference=True` (and
  `hedonic.metrics.restrict_cover`) scores against a reference that labels
  only part of the graph; Omega is exact on graphs with at most
  `omega_sample_size` (100,000) vertex pairs. `hedonic.metrics` is reachable
  from `import hedonic` and loaded on first use.
- A start whose labels have gaps is renumbered in label order rather than
  refused.
- Wheels for Linux, macOS and Windows; the C core's rounding is fixed by the
  source and it calls no libm function but `sqrt` and the exact `fabs` (the fused multiply-add is
  the instruction or its own correctly rounded soft fma; x86_64 wheels pick
  a build with or without the instruction from CPUID), and every wheel
  job runs the C tests and the Python tests against results pinned bit for
  bit.
- `pip install hedonic` is the library alone (igraph and numpy). The
  experiment command line ships with the package, and its dependencies are
  the `cli` extra: `pip install "hedonic[cli]"` (also `hedonic[experiments]`)
  installs `hedonic` (runs) and `hedonic-exp` (analysis and utility
  commands), with the run configs and the external-method workers as package
  data.
  - `hedonic run queue overlap` runs the overlapping-communities paper's
    experiments on eight SNAP graph/cover pairs, method by method, writing under
    `--root DIR` (default `OUTPUT_DIR/overlap-queue`); `--dry-run` prints the
    plan and what is already done, and a started queue checks that its data
    and methods are prepared. `hedonic run queue disjoint` runs the disjoint
    paper's, and `hedonic run queue all` both; a queue item can be a whole
    queue (`run = ["queue", NAME|FILE]`).
  - A queue file's `[limits]` (and an item's `limits = {...}`, or
    `hedonic run queue --timeout D --max-eta D`) bound each single run:
    `timeout` stops one detector run, calibration probe or start-quality
    condition that is still going, and `max_eta` skips a pair whose single
    run is estimated longer. Neither is ever applied to a total of runs, and
    neither changes a record's identity: a stopped run is recorded as a
    `timeout` and runs again under a larger limit.
  - A memory watchdog guards every detector run of the experiments (2.0.0rc12):
    while a benchmark run, a calibration probe or evaluation, a start-quality
    condition or a controlled-study run is going, its parent reads the
    machine's free memory twice a second (`HEDONIC_MEMORY_FLOOR`, default 10%
    free), and below the floor kills that run's whole process tree; a detector
    running in-process is interrupted from a profiling-timer signal. The run
    is recorded as `status: "memory"` with the floor and the free memory seen,
    and stands like a timeout: a queue does not run it again unless the floor
    is lower or `hedonic run ... --rerun-memory` asks for it. Tables and
    `hedonic run check` say "out of memory". Before, the guard looked only
    before a pair started, so two Clique Percolation evaluations side by side
    (LiveJournal/top5000 and Friendster/top5000) exhausted a 64 GB machine
    and restarted it. The guards of a whole run (a pool of forked workers,
    the run's worker), which record nothing, now act at once only below half
    the floor and otherwise after 2 s below it, so the run's own watchdog
    stops and records it first. The ground-truth spectrum keeps its own
    per-run guard.
  - `--alone PAIRS` (`hedonic run calibrate`, `calibrate-baselines`) runs the
    calibrations of those pairs one at a time, never beside another pair; the
    overlap queue's phase-2 calibrations use it for LiveJournal/top5000 and
    Friendster/top5000.
  - `--out-of-memory METHOD@PAIR,...` (`calibrate-baselines`, `hedonic run
    exp`) records a search or a benchmark unit as out of memory without
    loading the pair or starting anything, citing the evidence; such a search
    gets no setting and is never transferred. The overlap queue declares
    Clique Percolation on LiveJournal/top5000 (every evaluation, k = 3, 4 and
    5, was killed by the OS for memory) and Friendster/top5000 (larger).
  - The queue tree times a finished benchmark unit by its whole span
    (detection, the robustness audits of its cover, scoring), not by its
    detection alone, and a running unit's ETA includes the audit profiles
    still to come (a Hedonic unit audits its `cover`, then its `labelled`
    profile).
  - The queue view redraws only the rows that changed, inside a
    synchronized update, and writes nothing when nothing changed: over SSH
    the full repaint five times a second showed half-drawn frames and the
    selected row's highlight flickered. Closing the view always prints how
    to reopen or stop the queue (a state check on the view's own thread
    used to swallow it).
  - A baseline calibration in the queue view counts the settings of the
    method's own `--method-budget`, not the run's `--max-evals`, and prices
    what is left within the method's minutes per pair: the view showed ANGEL
    at 0/8 settings (its budget was 3) and an ETA of six hours.
  - A benchmark that prepares a runtime on first use (`hedonic run exp`)
    writes its setup manifest where it reads it: it passed the runtime
    folder as `--cache-dir`, which setup takes as the folder above it, so
    the manifest went to `codeseg/codeseg/` and the method was reported
    unavailable (the overlap queue's clique percolation benchmark recorded
    no run).
  - ANGEL's calibration walks its pre-registered grid (phi 0.2 to 0.8) from
    the largest phi down, and a setting stopped at the evaluation timeout
    ends the walk; ties still go to the smaller phi. A smaller phi merges
    more and costs more (DBLP/top5000: 27 s at 0.6, over an hour at 0.2), so
    the ascending walk spent its budget of 3 settings on the expensive end:
    three pairs ended without a setting, and the two that finished chose the
    edge of what was tried. The overlap queue gives it all 5 settings and
    240 minutes per pair. A search recorded under the earlier rule is
    searched again (`SEARCH_VERSIONS`), reusing its ledger, and is not
    adopted from another calibration.
  - `hedonic setup [--pairs P,...] [--methods M,...] [--all] [--dry-run]`
    downloads SNAP pairs and builds or installs the external baselines,
    after checking the system tools they need.
  - `hedonic run disjoint --preset smoke|paper` reproduces the disjoint
    paper.
  - Runs are detached and resumable (`hedonic run list|status|attach|logs|
    stop|resume|check`); `hedonic guide` is a tour of the commands.
  - The overlap benchmark runs each method as often as its replication kind
    says (one table, `overlapping/replication.py`, shared with the baseline
    calibration): a deterministic method (CoDeSEG, which the benchmark runs
    at one thread as the calibration does, CliquePerc, FOX, BigCLAM, NcGame,
    NEO-K-Means, the disjoint oracle) runs once per pair, at the lowest seed
    of the plan, because every other seed repeats its cover; Hedonic, ANGEL
    and Ego-Splitting (seeded) run at every seed. The paper's benchmark has
    288 units instead of 720. The plan, `--dry-run` and
    its estimate, the scan, the queue's counts and tree, the robustness stage
    and `hedonic run audit`, the result tables (a single run has no ± sd and
    reads `1/1`), `hedonic run check` and the joint plots follow it.
    `--replication all` (or `replication = "all"` in a config) runs every
    method at every seed, to audit determinism; it changes no unit's identity,
    so both settings share their records.
  - Records carry the hedonic version and the SHA-256 of the C core as their
    runtime identity. Hedonic is the method `hedonic`; the multilevel phase
    (`hoc_multilevel`, multi-phase variants) is not available and is refused
    with that message.
  - The robustness audits (`hedonic run audit`, the benchmark's
    `--robustness` stage, `hedonic run posthoc`, the ground-truth spectrum
    and the calibration's label-free audit) run on the C audit behind
    `Game.stable_intervals` and `Game.regrets`; `--workers` forked processes
    split the vertices and share its read-only prepared state. It returns the
    floats of the pure-Python reference (kept for tests), so the records keep
    their format and values; the spectrum's regret aggregates, which 1.x
    computed through a separate best-response path, can differ from 1.x in
    the last bit. The complete DBLP/top5000 spectrum audit (131 resolutions,
    two policies) takes 4 s instead of 2 minutes.
  - A benchmark unit run with `--save-covers` keeps its whole answer, so it
    can be audited and scored again without its detector: the cover, Hedonic's
    labelled memberships in `<runner>.labels.json.gz` (no longer inline in the
    record), the SNAP id of every analysis vertex (`vertex_ids.json.gz`, one
    per seed folder) and the analysis graph's SHA-256 in the record's
    `dataset_report.graph_sha256`. The gzips carry no time stamp, so the same
    answer is the same bytes, and the unit's seal covers the labels too. The
    audit refuses a loaded graph whose hash differs from the record's.
  - `hedonic-exp overlapping-rescore --source RUN --networks PAIRS` scores the
    saved covers again with the runner's own scoring and other settings
    (paper filter, Omega, singleton mode, matching weight, NF1); `--check`
    reproduces every record's scores and exits 1 on any difference.
  - A robustness audit counts only while it describes the cover on disk (its
    cover's or labels' hash, or its digest, against the unit's seal), and the
    ledger's last record of a key wins. Before, a unit that ran again kept the
    audit of its earlier cover.
  - The overlap queue's benchmark items run `--no-robustness`; each phase
    ends with an `audit` item that audits every saved cover, so all detector
    runs of a phase finish before any audit starts. Its rows are grouped as
    "Robustness audit" in the queue view, with per-unit progress and ETA.
  - `hedonic show table` lists Orkut's `top5000` cover with SNAP's reported
    counts of its induced subgraph (731,514 vertices, 21,992,510 edges), so
    Table 1 does not rescan the 117-million-edge graph; the networks are
    ordered D, A, Y, F, L, O and the cover descriptions are shorter.
  - Data defaults to `~/hedonic-data` and runs to `~/hedonic-runs`
    (`HEDONIC_DATA_DIR`, `HEDONIC_OUTPUT_DIR`, `HEDONIC_NETWORKS_DIR` and
    `~/.config/hedonic/config.toml` override them). `hedonic setup` and every
    run find the external methods in one folder: `HEDONIC_CODESEG_CACHE`,
    else the user config's `cache_dir/codeseg`, else
    `~/.cache/hedonic/codeseg`.
  - Configs resolve from the package (a file that exists wins), so a run
    needs no checkout: for example `hedonic run posthoc --config amazon-full`
    and `hedonic run spectrum --config full`. There is no protocol lock file.
