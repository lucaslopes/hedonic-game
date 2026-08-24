# SNAP overlapping benchmark

Run the saved-SNAP overlapping benchmark through the public experiment CLI:

```bash
uv sync --extra experiments

hedonic-exp overlapping-benchmark --profile smoke \
  --output_dir /tmp/hedonic-snap-smoke

hedonic-exp overlapping-benchmark \
  --datasets amazon,dblp,livejournal,youtube,wikipedia \
  --cover top5000 --profile standard \
  --output_dir ~/Databases/Hedonic/Networks/SNAP_BENCHMARK_CLI
```

The input root defaults to `~/Databases/Hedonic/Networks`; set
`HEDONIC_NETWORKS_DIR` or pass `--data_root` to override it. Raw inputs are
never modified. Normalized graph and cover caches are stored outside the
repository by default.

The supported collections are Amazon, DBLP, LiveJournal, YouTube, and
Wikipedia category lists. Wikipedia exposes only its `all` cover; requesting a
global `top5000` run records that dataset as an intentional skip.

## Profiles and controls

- `smoke` uses built-in small graphs and requires no archive.
- `standard` uses deterministic ground-truth-informed induced subgraphs capped
  at 3,000 vertices.
- `full` removes the node cap and should be run with explicit timeouts and
  `--resume`.

The default method set contains the three multi-phase hedonic variants plus
CPM and DEMON. `hedonic_local` remains available through `--methods`.

`--resolutions auto` evaluates each graph's density. Use `--seeds 0-4`,
`--omega`, and `--omega_sample_size` for repeated runs and sampled Omega.
`--dry-run` validates selected data without launching detectors.

## Methods

Hedonic variants call `Game.community_hedonic` with `n_iterations=-1` and a
membership capacity derived from the supplied cover. CPM uses NetworkX clique
percolation, and DEMON uses the optional `demon` package. All methods consume
the same loop-free undirected analysis graph after the loader's normalized
projection.

Use `hedonic-exp overlapping-benchmark --list-methods` for installed
dependencies and exact method parameters.

## Output and resumability

Each run writes a manifest, per-condition JSON records, JSONL/CSV summaries,
logs, and plots below the caller-supplied output directory. Records distinguish
`completed`, `timeout`, `memory_limit`, `oom`, `failed`,
`skipped_unsupported`, and `skipped_not_scalable`.

`--resume` reuses only compatible completed records (or explicit unsupported
dependency decisions). Changes to protocol, method, dataset, seed, resolution,
membership capacity, timeout, memory, or Omega settings make prior records
incompatible rather than silently mixing results.

The public checkout intentionally does not contain manuscript files or
historical result ledgers. A private research checkout may run a separate
manuscript orchestration protocol using the same public detector code and an
external configuration, but those materials are not part of this repository.
