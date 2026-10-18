---
title: Getting started
summary: Install the package, detect communities in a few lines, and run this explainer locally.
group: Guide
order: 1
---

## Install the package

`hedonic` is published on PyPI. It supports Python 3.12 and newer, depends only on `igraph` and `numpy`, and ships its detector as a compiled C extension (`hedonic._core`) inside the wheel.

```bash
python -m pip install hedonic
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add hedonic
```

`Game.evaluate_against` and `hedonic.metrics` score a result against a reference cover. Their one-to-one matching scores need SciPy, available as an optional extra:

```bash
python -m pip install "hedonic[metrics]"
```

The command line that reproduces the papers' experiments (`hedonic run ...`, macOS and Linux) has its own extra:

```bash
python -m pip install "hedonic[cli]"
```

```package-info
```

## Detect communities

Wrap any undirected `igraph.Graph` in a `Game` and call `community_hedonic`. One method covers both kinds of output:

```python
import igraph as ig
from hedonic import Game

graph = Game(ig.Graph.Famous("Zachary"))

# Disjoint partition: every vertex belongs to exactly one community.
partition = graph.community_hedonic(
    resolution=graph.density(),
    max_memberships=1,
    seed=0,
)
print(partition.membership)

# Overlapping cover: a vertex may belong to up to four communities.
cover = graph.community_hedonic(
    resolution=graph.density(),
    max_memberships=4,
    seed=0,
)
print(cover)
```

Every call runs until no vertex can improve by moving, so the result is an equilibrium of the game. `community_hedonic` returns an `igraph.VertexClustering` when `max_memberships=1` and an `igraph.VertexCover` when the cap is larger; the same `seed` on the same inputs returns the same result. See [community_hedonic](docs:#community-hedonic) for every parameter.

## Reproduce the explainer's network

The interactive story uses the four-vertex graph of Figure 1(a) of the paper: a triangle 0–1–2 plus vertex 3 hanging off vertex 0. The same graph in Python:

```python
import igraph as ig
from hedonic import Game

g = Game(ig.Graph(n=4, edges=[(0, 1), (0, 2), (0, 3), (1, 2)]))
for gamma in (0.2, 0.5):
    part = g.community_hedonic(
        resolution=gamma,
        max_memberships=1,
        allow_isolation=True,  # allow "leave to found a new community", as in the explainer
    )
    print(gamma, part.membership)
```

With hedonic 2.0.0 this prints, for every seed, `[0, 0, 0, 0]` (the grand coalition) for $\gamma = 0.2$ and `[0, 0, 0, 1]` (the partition {0,1,2}{3}) for $\gamma = 0.5$ — the same sinks the [explainer](story:#walk) finds. The [better-response dynamics](docs:#better-response) page explains why, and why `allow_isolation` matters.

## Run this explainer locally

The website lives in `web/` and is driven from the repository root:

```bash
npm install
npm run dev
```

Other scripts: `npm run build` (static site in `web/dist/`), `npm run test` (model and component tests), `npm run docs:generate` (regenerate the API reference from the Python source) and `npm run check` (everything, in CI order). See [Reproducibility](docs:#reproducibility) for details.
