---
title: Reproducibility
summary: How this site is generated, tested and deployed, and how the disjoint experiments of the paper are reproduced.
group: Project
order: 10
---

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

`npm run docs:generate` runs `web/scripts/extract_api.py`, which parses the Python files with the standard-library `ast` module — it never imports the package, so it needs neither lucas-igraph nor a virtual environment. It records signatures, defaults, annotations, numpydoc parameter descriptions, source line numbers, the `COMMANDS` registry of the CLI and the disjoint sweep presets. Source links point at the exact commit when the site is built by GitHub Actions and at `main` otherwise.

## Deployment

`.github/workflows/pages.yml` installs the locked dependencies, regenerates the documentation data, runs the tests and type checks, builds with `BASE_PATH` set to the repository sub-path reported by `actions/configure-pages`, and deploys `web/dist/` to GitHub Pages. Every URL in the site is relative to Vite's base, so the same build works at `/` locally and under `/<repository>/` on Pages (`npm run preview:pages` reproduces the sub-path locally).

## The disjoint experiments of the paper

The synthetic benchmark of the extended version is driven by the `hedonic-exp` command-line tool. Install the optional dependencies first:

```bash
uv sync --extra experiments
hedonic-exp smoke
```

Methods compared by the SBM sweep:

```sweep-methods
```

Disjoint commands of the CLI registry:

```cli-commands smoke disjoint plots reproduce-disjoint
```

A complete run is `hedonic-exp reproduce-disjoint --preset v1020 --output_root artifacts/disjoint/v1020` (very large; add `--preflight` first). The guide `docs/reproduce_disjoint.md` in the repository describes each step. The full registry, including the overlapping experiments, is:

```cli-commands
```

## Cross-checking the native detector

`web/scripts/crosscheck_native.py` runs `Game.community_hedonic` on the explainer's graph for several resolutions and both values of `allow_isolation`, and stores the results in a test fixture. The Vitest suite verifies that every native result with isolation allowed is a sink of the browser model. Re-run the script (it needs the Python package) after a native release.

```api-constants
```
