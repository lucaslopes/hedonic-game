---
title: Reproducibility
summary: How to make a detector run reproducible, and how this site is generated, tested and deployed.
group: Project
order: 10
---

## Reproducible detector runs

`community_hedonic` visits the vertices in a random order. Pass `seed` and the whole call runs under `random.Random(seed)`: the returned labels are then a function of the graph, the weights, the parameters, the start state and the seed. Without a seed the order is drawn from Python's `random` module (python-igraph's default generator), so `random.seed(...)` before the call makes it repeatable too.

```python
first = graph.community_hedonic(initial_membership=None, resolution=0.1, seed=7)
again = graph.community_hedonic(initial_membership=None, resolution=0.1, seed=7)
assert first.membership == again.membership
```

Pass `initial_membership=None` explicitly when repeating a call: omitting it continues from the memberships the previous call stored in `Game.memberships`. Each result also carries a `_hedonic_provenance` dictionary with the package and igraph versions and the effective parameters of the call, which is worth saving next to the result.

## The explainer

The site is a static React + TypeScript application built with Vite. It has no server and loads no runtime CDN scripts: fonts, KaTeX and the D3 zoom modules are installed from npm and bundled, so it works offline after `npm install`.

| Command (repository root) | What it does |
| --- | --- |
| `npm install` | Installs the `web/` workspace. |
| `npm run dev` | Development server with hot reload. |
| `npm run docs:generate` | Regenerates `web/src/generated/*.json` from the Python source and `web/content/references.bib`. |
| `npm run test` | Runs the Vitest suite: model, layout, docs rendering and components. |
| `npm run typecheck` / `npm run lint` | TypeScript and ESLint checks. |
| `npm run build` | Type-checks and writes the static site to `web/dist/`. |
| `npm run check` | All of the above, in CI order. |

The mathematical model is plain TypeScript in `web/src/model/` with no browser dependencies, which is what the unit tests exercise: Bell-number enumeration, canonical identity, one-move adjacency, the friend/stranger deltas, the Familiarity Index, utility gains, orientation and ties at thresholds, sinks, and deterministic walks for every rule.

## Generated API reference

`npm run docs:generate` runs `web/scripts/extract_api.py`, which parses the Python files with the standard-library `ast` module — it never imports the package, so it needs neither igraph, NumPy, the compiled core nor a virtual environment. It records signatures, defaults, annotations, numpydoc parameter descriptions and source line numbers of `Game`, `hedonic.utils` and `hedonic.metrics`, and the version and dependencies from `pyproject.toml`. Source links point at the exact commit when the site is built by GitHub Actions and at `main` otherwise.

## Deployment

`.github/workflows/pages.yml` installs the locked dependencies, regenerates the documentation data, runs the tests and type checks, builds with `BASE_PATH` set to the repository sub-path reported by `actions/configure-pages`, and deploys `web/dist/` to GitHub Pages. Every URL in the site is relative to Vite's base, so the same build works at `/` locally and under `/<repository>/` on Pages (`npm run preview:pages` reproduces the sub-path locally).

## Cross-checking the detector

`web/scripts/crosscheck_native.py` runs `Game.community_hedonic` on the explainer's graph for several resolutions and both values of `allow_isolation`, and stores the results in a test fixture. The Vitest suite verifies that every native result with isolation allowed is a sink of the browser model. Re-run the script (it needs the installed Python package) after a change to the detector.
