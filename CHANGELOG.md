# Changelog

## 1.0.0

- `hedonic` computes partitions with its own purpose-built C core
  (`native/`, compiled into `hedonic._core`): best-response local moving of
  the CPM hedonic game (`max_memberships=1`). Every call runs until every
  vertex has been visited without a certainly improving move since the last
  move, so the result is an equilibrium; there is no iteration budget, no
  round cap and no tolerance. Every payoff the core compares carries a
  rigorous bound of its rounding error, and a vertex moves only when its gain
  exceeds it, so every move raises the exact potential and every call
  terminates (`native/README.md`). It is the partition path of hedonic 2,
  byte for byte, and returns the same partitions bit for bit.
- Overlapping covers (`max_memberships > 1`), the exact stability audit
  (`Game.stable_intervals`, `Game.regrets`) and the overlapping experiments
  are hedonic 2. A call with `max_memberships != 1` raises an error that says
  so.
- Results report `equilibrium_slack` in `_hedonic_provenance`: no unilateral
  deviation raises a vertex's exact payoff by more than the slack times the
  total edge weight. It is 0 when the arithmetic is exact (for instance unit
  weights at resolution 0, 1/2 or 1): an exact Nash equilibrium; other
  weights often give 0 too.
- When the queue of vertices to visit runs dry, the vertices not yet settled
  are visited again in a fresh random order.
- The library needs only the standard `igraph>=1.0` package and NumPy.
  Importing `hedonic` does not check igraph's version, so it installs next to
  any igraph 1.x.
- `Game.community_hedonic(initial_membership, *, max_communities,
  max_memberships, resolution, allow_isolation, edge_weights, seed,
  node_weights, max_total_communities, n_communities)`: everything after
  `initial_membership` is keyword-only. The keywords of the igraph-fork
  releases (0.1, 1.5) `n_iterations`, `local_move_only`, `beta`,
  `debug_trace`, `phase_policy` and `ensure_equilibrium`, and positional
  calls, raise a `TypeError` that says what replaces them.
- Every call starts afresh, from its default start (singletons, a random
  start with `max_communities`, or a feasible start under a count limit)
  unless `initial_membership` is given; an igraph `VertexClustering` or
  `VertexCover` is accepted as a start. The result is stored in
  `Game.memberships`; pass `initial_membership=g.memberships` to continue
  from it.
- A start with more than one label per vertex (a cover) keeps each vertex's
  preferred label (ranked by the payoff of each community on the start);
  `_hedonic_provenance` counts the `projected_vertices`.
- The graph must be undirected and free of self-loops (the error names
  `g.simplify(multiple=False, loops=True)`); parallel edges count as one edge
  with their summed weight. The default resolution is the weighted density,
  the total edge weight over the number of vertex pairs.
- `seed=` draws the visiting order from `random.Random(seed)`; without it the
  order comes from Python's `random` module.
- Community-count limits: `max_total_communities` (at most K) and
  `n_communities` (exactly K), enforced at every move.
- `hedonic.metrics` and `Game.evaluate_against` score a result against a
  reference partition or cover (a flat reference may use any labels); the
  scores built on a one-to-one matching use SciPy (the `metrics` extra).
  `restrict_to_reference=True` (and `hedonic.metrics.restrict_cover`) scores
  against a reference that labels only part of the graph. `hedonic.metrics`
  is reachable from `import hedonic` and loaded on first use.
- A start whose labels have gaps is renumbered in label order rather than
  refused.
- Wheels for Linux, macOS and Windows; the C core's rounding is fixed by the
  source and it calls no libm function but `sqrt` and the exact `fabs` (the
  fused multiply-add is the instruction or its own correctly rounded soft
  fma; x86_64 wheels pick a build with or without the instruction from
  CPUID), and every wheel job runs the C tests and the Python tests against
  results pinned bit for bit.
- `pip install hedonic` is the library alone (igraph and numpy). The
  experiment command line ships with the package, and its dependencies are
  the `cli` extra: `pip install "hedonic[cli]"` (also `hedonic[experiments]`)
  installs `hedonic` (runs) and `hedonic-exp` (analysis and utility
  commands), with the run configs as package data.
  - `hedonic run disjoint --preset smoke|paper` reproduces the disjoint
    paper; `hedonic run queue disjoint` runs it as a queue.
  - Runs are detached and resumable (`hedonic run list|status|attach|logs|
    stop|resume|check`); `hedonic guide` is a tour of the commands.
  - Records carry the hedonic version and the SHA-256 of the C core as their
    runtime identity.
  - The overlapping verbs (`hedonic setup`, `hedonic run exp|paper|spectrum|
    audit|calibrate|...`) are hedonic 2 and print a message that says so.
