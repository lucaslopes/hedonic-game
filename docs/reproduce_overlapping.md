# Reproducing the public overlapping experiments

Use the public CLI for smoke checks, DBLP subgraphs, full-graph diagnostics,
resolution sweeps, and the saved-SNAP benchmark. Inputs and output directories
are external to the source tree unless an ignored `artifacts/` path is chosen.

```bash
uv sync --extra experiments

uv run hedonic-exp overlapping-small
uv run hedonic-exp overlapping-subgraph --levels 1 --n_communities 5 \
  --methods leiden,hedonic_v1 \
  --output /tmp/hedonic-overlap-subgraph.json
uv run hedonic-exp overlapping-resolution --smoke \
  --output_dir artifacts/overlapping/resolution_f1/smoke
uv run hedonic-exp overlapping-benchmark --profile smoke \
  --output_dir artifacts/overlapping/snap_benchmark/smoke
```

For DBLP, set `HEDONIC_DBLP_DIR` or pass the module's data-path option. For
SNAP networks, set `HEDONIC_NETWORKS_DIR` or pass `--data_root`; raw archives
are never modified. All native Leiden calls request `n_iterations=-1`, and
equilibrium-sensitive hedonic runs use the native isolation-enabled cleanup
pipeline.

The public repository contains reusable detectors, metrics, loaders, and
smoke fixtures. Manuscript sources, private result ledgers, and full-paper
orchestration remain in the separate private research checkout.
