# Reproducing the disjoint synthetic experiments

The public checkout exposes the synthetic SBM sweep, aggregation, and figure
commands through `hedonic-exp`. Archived datasets are external inputs; keep
them read-only and write new output to a separate directory.

```bash
uv sync --extra experiments

# Fast structural check
uv run hedonic-exp disjoint --smoke \
  --output_root artifacts/disjoint/smoke

# Full preset (large; choose an external output root)
uv run hedonic-exp disjoint --preset v1020 \
  --output_root ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI

# Aggregate and plot an existing sweep
uv run hedonic-exp disjoint-load \
  --results_folder /path/to/resultados \
  --output /path/to/resultados_ari.csv.gzip
uv run hedonic-exp plots \
  --data /path/to/resultados_ari.csv.gzip \
  --output_dir /path/to/figures
```

Set `HEDONIC_SYNTHETIC_DIR` or use the module's `--output_root`/`--data`
options to point at external data. The CLI refuses to overwrite the archived
V1020 tree. Publication manuscripts, private ledgers, and their orchestration
configs are maintained outside this public checkout.
