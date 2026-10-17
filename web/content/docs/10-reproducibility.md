---
title: Reproducibility
summary: How this site is generated, tested and deployed.
group: Project
order: 10
---

## The explainer

The site is a static React + TypeScript application built with Vite. It has no server and loads no runtime CDN scripts: fonts, KaTeX and the D3 zoom modules are installed from npm and bundled, so it works offline after `npm install`.

| Command (repository root) | What it does |
| --- | --- |
| `npm install` | Installs the `web/` workspace. |
| `npm run dev` | Development server with hot reload. |
| `npm run docs:generate` | Regenerates `web/src/generated/*.json` from the repository metadata and `web/content/references.bib`. |
| `npm run test` | Runs the Vitest suite: model, layout, docs rendering and components. |
| `npm run typecheck` / `npm run lint` | TypeScript and ESLint checks. |
| `npm run build` | Type-checks and writes the static site to `web/dist/`. |
| `npm run check` | All of the above, in CI order. |

The mathematical model is plain TypeScript in `web/src/model/` with no browser dependencies, which is what the unit tests exercise: Bell-number enumeration, canonical identity, one-move adjacency, the friend/stranger deltas, the Familiarity Index, utility gains, orientation and ties at thresholds, sinks, and deterministic walks for every rule.

## Deployment

`.github/workflows/pages.yml` installs the locked dependencies, regenerates the documentation data, runs the tests and type checks, builds with `BASE_PATH` set to the repository sub-path reported by `actions/configure-pages`, and deploys `web/dist/` to GitHub Pages. Every URL in the site is relative to Vite's base, so the same build works at `/` locally and under `/<repository>/` on Pages (`npm run preview:pages` reproduces the sub-path locally).
