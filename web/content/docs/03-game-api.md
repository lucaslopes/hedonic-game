---
title: The Game API
summary: Every public member of hedonic.Game, generated from the Python source.
group: Python API
order: 3
---

The package exposes a single public name. `Game` subclasses `igraph.Graph` (stock `igraph` from PyPI), so every igraph method remains available, and adds the hedonic detector plus a few membership helpers. The tables and signatures below are generated from [`src/hedonic/Game.py`](repo:src/hedonic/Game.py) and [`src/hedonic/__init__.py`](repo:src/hedonic/__init__.py) by `npm run docs:generate`; they never need to be edited by hand.

```python
from hedonic import Game
```

## Members

```api-members
```

## Constructor

```api Game.__init__
```

## Membership state

`community_hedonic` stores its result in `Game.memberships` (one list of labels per vertex) but never reads it back by itself: every call starts from singletons unless `initial_membership` is given. Pass `initial_membership=g.memberships` to continue from the last equilibrium.

```api Game.memberships
```

```api Game.validate_membership
```

## Conversion and evaluation

```api Game.to_igraph
```

```api Game.evaluate_against
```

```api Game.evaluate_accuracy
```

## Detection

The detector has its own page: [community_hedonic](docs:#community-hedonic).

## Scoring against a reference: `hedonic.metrics`

`Game.evaluate_against` is a facade over `hedonic.metrics.evaluate_cover`, which can also be called directly with two covers (lists of vertex lists). The module is imported only when it is used; its one-to-one matching scores need SciPy (`pip install "hedonic[metrics]"`).

```python
from hedonic.metrics import evaluate_cover, partition_to_cover_lists

# reference_cover: a list of vertex lists, e.g. [[0, 1, 2], [2, 3, 4]]
scores = evaluate_cover(partition_to_cover_lists(partition), reference_cover, graph.vcount())
print(scores["f1"])  # equals graph.evaluate_against(reference_cover) after the detection call
```

```api evaluate_cover
```

Every public function of the module:

```api-functions hedonic.metrics
```

Module constants:

```api-constants hedonic.metrics
```

## Utilities

`hedonic.utils` holds a small seeded helper:

```api sample_uniform_ints
```
