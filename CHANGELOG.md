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
  total edge weight. It is 0 when the arithmetic is exact (an exact Nash
  equilibrium).
- Dependencies are the standard `igraph>=1.0` package and NumPy. Importing
  `hedonic` does not check igraph's version, so it installs next to any
  igraph 1.x.
- `Game.community_hedonic(initial_membership, max_communities,
  max_memberships, resolution, allow_isolation, *, edge_weights, seed,
  node_weights, max_total_communities, n_communities)`; everything after
  `allow_isolation` is keyword-only.
- `seed=` draws the visiting order from `random.Random(seed)`; without it the
  order comes from Python's `random` module.
- Community-count limits: `max_total_communities` (at most K) and
  `n_communities` (exactly K), enforced at every move.
- `hedonic.metrics` and `Game.evaluate_against` score a result against a
  reference cover or partition; one-to-one matching uses SciPy
  (`pip install "hedonic[metrics]"`).
- Wheels for Linux, macOS and Windows; the C core's rounding is fixed by the
  source and it calls no libm function but `sqrt` and the exact `fabs` (the fused multiply-add is
  the instruction or its own correctly rounded soft fma; x86_64 wheels pick
  a build with or without the instruction from CPUID), and every wheel
  job runs the C tests and the Python tests against results pinned bit for
  bit.
