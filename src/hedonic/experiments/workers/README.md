# External-method workers

Scripts that run an external community-detection method outside the hedonic
process. They are package data, located with
`hedonic.experiments.resources.worker_path` / `WORKER_DIR`.

- `slpa_env/`: the locked uv project for CDlib (ANGEL, Ego-Splitting, SLPA,
  DER, link clustering). `hedonic run` materialises it under the cache
  directory and runs `cdlib_worker.py` (and the other workers) with its
  interpreter, so CDlib's igraph never enters the hedonic process.
- `ncgame_worker.py`: runs the CoDeSEG authors' NcGame script (`NCG.py`) from
  the pinned CoDeSEG checkout and writes its cover.
