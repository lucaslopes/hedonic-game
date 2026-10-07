# Changelog

## 1.5.0

Hedonic follows the coordinated `igraph 1.0.0.N` + `python-igraph 1.0.0.N`
identity as `hedonic 1.N.x`: `x` is 0 for the coordinated release and grows for
Hedonic-only fixes that keep the same `lucas-igraph`. This release is N = 5.
It supersedes the version number 1.0.5 used during development; nothing was
released under it.

- Add the `web/` educational explainer for the disjoint model (React + Vite,
  deployed by `.github/workflows/pages.yml`) and its npm workspace.
- Depend on `lucas-igraph==1.0.0.5` (faster overlapping local moving, global
  community-count limits, and the fixes in its changelog). `uv.lock`
  resolves it.
- `Game.community_hedonic` gains keyword-only `node_weights`,
  `max_total_communities`, `n_communities` and `debug_trace`;
  `max_memberships=-1` removes the practical per-vertex cap.
- `seed` now makes the whole call deterministic: it runs the call under
  `random.Random(seed)` as igraph's generator and restores the caller's
  generator, including on exceptions. Bindings without the RNG getter reject
  seeded calls before changing the caller's generator. Results for calls that
  pass `seed` differ from 0.1.x.
- Every result carries `_hedonic_provenance` (identity, runtime versions,
  effective parameters, count mode, stopping rule). The algorithm identity is
  `community_hedonic/lucas-igraph-1.0.0.5`; the frozen 1.0.0.3 producer string
  remains available for historical ledgers.
- The DBLP/CoDeSEG runner no longer derives the Hedonic cap from community
  sizes; it uses the pre-registered reference-free cap 8 or
  `--hedonic-max-memberships`.
- Trace checks follow the 1.0.0.5 multilevel guard (compare each token
  proposal with the iteration's local-moving cover; a tie may be kept when it
  occupies fewer labels, at most n times per call):
  `overlapping-dnn-rational --require-debug-trace`
  and the integrity grid's projection fixture. The integrity grid's `auto`
  version requirement is now "1.0.0.4 or later" instead of exactly 1.0.0.4.
- Version 1.0.0.5 trace audits require the new pre-rollback label counts and
  check the full tie budget. Durable integrity runs fingerprint the loaded
  native core as well as the binding, and reject a changed provider at
  completion; hidden native symbols fail with an explicit provenance error.
- Correct initial token-count provenance when an exact overlapping default
  start requests more communities than vertices.
- `debug_trace` works for partitions and covers, with or without count limits:
  `True`/`"full"` records accepted moves (partitions on every aggregation
  level), overlapping projections and counters; `"counters"` records only the
  counters. The trace envelope carries `seed_context` and the algorithm
  identity. lucas-igraph 1.0.0.5 restores the igraph 1.0.0
  `igraph_community_leiden()` symbol and python-igraph's 1.0.0 call shape;
  the facade already passes every argument by keyword and is unaffected.
- Validation errors name the conflicting option (`n_communities` above the
  vertex capacity, a start state that violates a count limit, an invalid
  `debug_trace`). `_hedonic_provenance` records `debug_trace` and whether the
  initialization-only `max_communities` applied (`max_communities_effect`).
- `Game.community_hedonic(phase_policy=...)`: `"direct"` (default) or the
  opt-in `"disjoint_then_overlap"` warm start, which runs the disjoint game
  with the same options and passes its partition as `initial_membership`
  to the overlapping run. The stage-1 partition is attached as
  `_hedonic_disjoint_stage` and summarized in provenance. Keep it opt-in:
  on complete DBLP it was faster but less accurate than direct overlap.
- `Game(graph)` and `to_igraph()` preserve directionality and vertex, edge
  and graph attributes; directed inputs are rejected by the detector instead
  of being silently converted (prepared as 0.1.2, never released).
- Repeated labels within an overlapping start row are collapsed; the start
  incidence trace still records them.
- The README is a tutorial again, with a section on the 1.0.0.5 options and on
  how an omitted `initial_membership` continues from `Game.memberships`.
- The released-wheel verifier and release guide expect `lucas-igraph==1.0.0.5`.
- New `hedonic run spectrum` / `hedonic-exp overlapping-gt-spectrum`: audits the
  supplied overlapping covers found in the local SNAP cache over a resolution
  grid and runs seeded local moving from the exact ground-truth cover, with its
  own protocol lock; nothing is downloaded without `--provision`.
- New `hedonic guide` (state-aware tour), `hedonic update` (PyPI check; installs
  only with `--yes`, never changes a development checkout) and
  `hedonic show covers`. `hedonic run` gains `--json`, clearer statuses and a
  resume that also continues failed runs. Per-user defaults live in
  `~/.config/hedonic/config.toml` (`hedonic config`).
- The public test suite covers the API, CLI front door, SNAP benchmark and
  release tooling; research-protocol tests stay in the private research tree.

## 0.1.1

- Update the native dependency to `lucas-igraph==1.0.0.4` and regenerate the
  dependency lock. The `Game` public API remains unchanged.
- Refresh the package README with installation and disjoint/overlapping examples.
- Test and build on public `main`; publish the exact successful Actions artifacts
  after tagging, with metadata, commit identity, and index hash verification.
- Preserve historical protocol locks and evidence identities at their original
  versions; a runtime dependency update does not reinterpret saved results.
