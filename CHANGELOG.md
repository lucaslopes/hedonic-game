# Changelog

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
  - `hedonic run queue paper` runs the overlapping-communities paper's six
    experiments on eight SNAP graph/cover pairs, pair by pair, writing under
    `--root DIR` (default `OUTPUT_DIR/paper-queue`); `--dry-run` prints the
    plan and what is already done, and a started queue checks that its data
    and methods are prepared.
  - `hedonic setup [--pairs P,...] [--methods M,...] [--all] [--dry-run]`
    downloads SNAP pairs and builds or installs the external baselines,
    after checking the system tools they need.
  - `hedonic run disjoint --preset smoke|paper` reproduces the disjoint
    paper.
  - Runs are detached and resumable (`hedonic run list|status|attach|logs|
    stop|resume|check`); `hedonic guide` is a tour of the commands.
  - The overlap benchmark runs each method as often as its replication kind
    says (one table, `overlapping/replication.py`, shared with the baseline
    calibration): a deterministic method (CliquePerc, FOX, BigCLAM, NcGame,
    NEO-K-Means, the disjoint oracle) runs once per pair, at the lowest seed
    of the plan, because every other seed repeats its cover; Hedonic, ANGEL
    and Ego-Splitting (seeded) and CoDeSEG (its runs differ at the
    benchmark's thread count, so they are replicas) run at every seed. The
    paper's benchmark has 360 units instead of 720. The plan, `--dry-run` and
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
  - Data defaults to `~/hedonic-data` and runs to `~/hedonic-runs`
    (`HEDONIC_DATA_DIR`, `HEDONIC_OUTPUT_DIR`, `HEDONIC_NETWORKS_DIR` and
    `~/.config/hedonic/config.toml` override them). `hedonic setup` and every
    run find the external methods in one folder: `HEDONIC_CODESEG_CACHE`,
    else the user config's `cache_dir/codeseg`, else
    `~/.cache/hedonic/codeseg`.
  - Configs resolve from the package (a file that exists wins), so a run
    needs no checkout: for example `hedonic run posthoc --config amazon-full`
    and `hedonic run spectrum --config full`. There is no protocol lock file.
