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
  equilibrium. A cover's payoffs divide by square roots, so its slack stays
  a tiny positive bound (about `1e-16` of the total weight).
- When the queue of vertices to visit runs dry, the vertices not yet settled
  are visited again in a fresh random order. A fixed order made complete
  YouTube at its density take 11 to 18 minutes; it now takes under 5 s.
- Dependencies are the standard `igraph>=1.0` package and NumPy. Importing
  `hedonic` does not check igraph's version, so it installs next to any
  igraph 1.x.
- `Game.community_hedonic(initial_membership, *, max_communities,
  max_memberships, resolution, allow_isolation, edge_weights, seed,
  node_weights, max_total_communities, n_communities)`: everything after
  `initial_membership` is keyword-only. The 1.x keywords `n_iterations`,
  `local_move_only`, `beta`, `debug_trace`, `phase_policy` and
  `ensure_equilibrium`, and 1.x-style positional calls, raise a `TypeError`
  that says what replaces them.
- Every call starts from its default start (singletons) unless
  `initial_membership` is given. The result is stored in `Game.memberships`;
  pass `initial_membership=g.memberships` to continue from it.
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
  reference cover or partition (a flat reference may use any labels); only
  the scores built on a one-to-one matching use SciPy
  (`pip install "hedonic[metrics]"`).
- Wheels for Linux, macOS and Windows; the C core's rounding is fixed by the
  source and it calls no libm function but `sqrt` and the exact `fabs` (the fused multiply-add is
  the instruction or its own correctly rounded soft fma; x86_64 wheels pick
  a build with or without the instruction from CPUID), and every wheel
  job runs the C tests and the Python tests against results pinned bit for
  bit.
