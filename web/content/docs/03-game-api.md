---
title: The Game API
summary: Every public member of hedonic.Game, generated from the Python source.
group: Python API
order: 3
---

The package exposes a single public name. `Game` subclasses `igraph.Graph` (from lucas-igraph), so every igraph method remains available, and adds the hedonic detector plus a few membership helpers. The tables and signatures below are generated from [`src/hedonic/Game.py`](repo:src/hedonic/Game.py) and [`src/hedonic/__init__.py`](repo:src/hedonic/__init__.py) by `npm run docs:generate`; they never need to be edited by hand.

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

`community_hedonic` stores its result in `Game.memberships` (one list of labels per vertex), so a later call without `initial_membership` continues from that equilibrium. Pass `initial_membership=None` to start again from singletons.

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

## Module constants and utilities

```api-constants
```

`hedonic.utils` holds small helpers used by the detector:

```api sample_uniform_ints
```
