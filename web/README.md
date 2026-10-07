# Hedonic game — interactive explainer

An educational, scrollytelling website for the disjoint hedonic game / Constant
Potts Model behind the `hedonic` Python package, plus its documentation. It is a
static React + TypeScript + Vite application deployed to GitHub Pages.

```bash
# from the repository root
npm install
npm run dev            # http://localhost:5173
npm run build          # static site in web/dist/
npm run test           # Vitest: model, layout, components, docs pipeline
npm run docs:generate  # regenerate API reference + citation data
npm run check          # docs:generate, typecheck, lint, test, build
npm run preview:pages  # build and serve under /hedonic-game/, like GitHub Pages
```

The frontend lives in `web/` because `docs/` already holds the Python
documentation and the research manuscripts. The root `package.json` is an npm
workspace that forwards every script, so nothing has to be run from inside
`web/`.

## What it shows

1. **The dilemma** — an agent compares two communities on a balance scale:
   friends are weights worth `1 − γ`, strangers are balloons that lift by `γ`.
2. **The resolution** — the dial γ tips the scale; the Familiarity Index
   `F = Δd / (Δd + Δd̂)` is where it balances.
3. **The decision tree** of the SRC abstract, highlighting the path for the
   current counts.
4. **The metagraph** of the four-vertex example (Figure 1(a)): two choices, then
   all 15 partitions, all 52 single-vertex moves, the paper's clear/frustrated
   typing, and orientation by γ.
5. **The better-response walk** to a sink, with the quality staircase and a
   stability certificate.
6. **Explore** — a free, zoomable lab with an inspector and a table view.
7. **Use the package** — links into the documentation page (`/docs/`).

The story is available in English and Brazilian Portuguese; the technical
documentation is in English.

## Source layout

```
web/
├── index.html, docs/index.html   two entry pages (story, documentation)
├── content/
│   ├── docs/*.md                 documentation pages (Markdown + front matter)
│   └── references.bib            citation data shown on the site
├── plugins/markdown-docs.ts      build-time Markdown → HTML (KaTeX, API blocks)
├── scripts/
│   ├── extract_api.py            Python AST → src/generated/api.json
│   ├── generate-docs.mjs         runs the extractor and the BibTeX reader
│   ├── bib.mjs                   dependency-free BibTeX reader
│   └── crosscheck_native.py      optional: record native detector results
└── src/
    ├── model/                    the mathematics (pure TypeScript, no DOM)
    ├── viz/                      balance scale, decision tree, metagraph (SVG)
    ├── walk/                     walk controls, status, chart, certificate
    ├── story/                    the scrollytelling chapters and Explore lab
    ├── docs/                     documentation page shell
    ├── components/               resolution control, tooltips, header, …
    ├── i18n/                     en.ts (source) and pt.ts (typed translation)
    ├── generated/                output of docs:generate (committed)
    └── styles/                   design tokens and base styles
```

## The model (`src/model/`)

- `partitions.ts` — every partition as a restricted growth string (RGS), so
  relabelling communities never creates a new partition; the count is the
  Bell number (`B4 = 15`).
- `moves.ts` — every single-vertex move: join another community or found a new
  one (a lone vertex joining another community removes its old one). Each move
  records the mover, origin, destination, friends/strangers before and after,
  and `Δd`, `Δd̂`.
- `tradeoff.ts` — `ΔU = Δd − γ(Δd + Δd̂)`, the Familiarity Index (null for a
  zero denominator), the clear / frustrated / indifferent classification and
  the orientation at γ. Gains within `1e-9` of zero are ties.
- `decisionTree.ts` — the paper's decision tree, proven equivalent to the
  classification by the tests.
- `metagraph.ts` — metanodes (id, blocks, sizes, extremes, BFS distances,
  layer) and metaedges (grouped moves, deltas, `F`, which end has more friends
  and which fewer strangers), plus `edgeState(edge, γ)`.
- `quality.ts` — `Φγ(π) = Σ_k [m_k − γ·C(n_k, 2)]`; each move changes it by
  exactly the mover's gain (exact potential game).
- `dynamics.ts` — strictly improving moves only, sinks, basins, stability
  certificates, and three reproducible rules: best response (default),
  round-robin, and seeded random.

For the example graph the tests pin the facts the story relies on: 52 edges
(37 clear, 9 frustrated, 6 indifferent; 12 of them omitted from the paper's
figure), thresholds 1/3, 1/2 and 2/3, and the sinks for every γ.

## Documentation pipeline

`npm run docs:generate` writes three committed files to `src/generated/`:

- `api.json` — signatures, defaults, annotations, numpydoc parameter
  descriptions and line numbers of `Game`, `hedonic.utils`, the `hedonic-exp`
  command registry and the disjoint sweep presets. `extract_api.py` parses the
  sources with the standard-library `ast` module and never imports the package.
- `site.json` — package version, native dependency pin and repository URL for
  the story page.
- `references.json` — citation cards from `content/references.bib`. Only DOI,
  arXiv and URL fields present in the bibliography become links.

Docs pages are Markdown with a small front matter (`title`, `summary`,
`group`, `order`). Besides `$…$` math and `[@key]` citations they accept
generated blocks: ` ```api Game.community_hedonic``` `, ` ```api-members``` `,
` ```api-constants``` `, ` ```package-info``` `, ` ```cli-commands disjoint``` `,
` ```sweep-methods``` ` and ` ```references``` `. Links can use `repo:path#L1`
(GitHub source), `story:#anchor` (explainer section) and `docs:#page`.

## Conventions

- **Faithful mathematics.** Every formula comes from the SRC abstract or
  arXiv:2509.03834; educational simplifications are labelled in the interface
  and in `content/docs/11-limitations.md`.
- **Accessibility.** Every visual has a textual equivalent (verdicts, live
  regions, node labels, the Explore table). Colour is always paired with a
  shape, dash or label. The metagraph has one tab stop and arrow-key
  navigation. Animations respect `prefers-reduced-motion` and the in-page
  motion toggle.
- **Offline.** No runtime CDN: fonts (Fontsource), KaTeX and D3 come from npm.
- **Base path.** Every URL is relative to Vite's `base`; the Pages workflow
  sets `BASE_PATH` to the repository sub-path.
- **Translations.** Add strings to `src/i18n/en.ts`; the `Dictionary` type
  makes a missing Portuguese key a type error.

## Deployment

`.github/workflows/pages.yml` runs `npm ci`, `docs:generate`, type-check, lint,
tests and the build on pull requests and on pushes to `main`, and deploys
`web/dist/` to GitHub Pages from `main`. Enable it once in the repository
settings (Pages → Source: GitHub Actions).
