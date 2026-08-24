# AGENTS.md — public `hedonic` checkout

This checkout contains the reusable library and public experiment adapters.
Manuscripts, private evidence ledgers, and manuscript-only protocol files are
maintained in a separate private research checkout and must not be added here.
The public benchmark identity lock is source-controlled because it validates
the reusable benchmark code; it contains no manuscript or result files.

## Package layout

```text
src/hedonic/
├── Game.py                 # Game(igraph.Graph) and community_hedonic
├── utils.py
└── experiments/            # optional experiment CLI and adapters
```

The public API is:

```python
from hedonic import Game
```

Do not add generated datasets, result ledgers, manuscript sources, or local
machine paths to the repository. Use environment variables or home-relative
paths for external data.

## Development

Use `uv` for dependency management, testing, and builds:

```bash
uv sync --extra experiments
uv run pytest -q
uv build --no-sources
```

The native dependency is pinned in `pyproject.toml` and `uv.lock`. Update both
through `uv` when changing dependencies. `Game.community_hedonic` is the
single public entry point for disjoint and overlapping detection; do not add a
second Leiden implementation.

## Experiments

Experiment modules must expose import-safe `main(argv=None)` functions and be
registered in `src/hedonic/experiments/CLI.py`. Generated outputs belong under
the ignored `artifacts/` directory or an explicitly supplied external path.
Small deterministic fixtures belong under `tests/`, not under `artifacts/`.

When a test depends on an optional external dataset or private historical
ledger, make the dependency explicit and skip with a clear message when the
fixture is unavailable. Never silently replace a missing result with a new
measurement.

## Public/private boundary

The following are intentionally excluded by `.gitignore`:

- `docs/papers/`
- `artifacts/evidence/`
- manuscript-only TOML protocol locks (the public benchmark lock is retained)

Before committing, check that no private paths or absolute user directories
are staged:

```bash
git diff --cached --name-only
git grep -nE 'docs/papers|artifacts/evidence|/Users/' -- ':!docs/papers/**'
```
