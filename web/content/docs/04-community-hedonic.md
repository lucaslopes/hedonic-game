---
title: community_hedonic
summary: The one detector for disjoint partitions and overlapping covers — parameters, defaults and how they map to the explainer.
group: Python API
order: 4
---

`Game.community_hedonic` is the primary method for every experiment in this repository. It delegates to `community_leiden` from lucas-igraph with the flags that make it a hedonic local-moving run by default. Experiment code should call it rather than `community_leiden` directly, so the defaults stay consistent.

```api Game.community_hedonic
```

## Typical calls

```python
# Disjoint: the hedonic local-moving phase, run to the native no-change stop.
part = g.community_hedonic(resolution=g.density(), max_memberships=1, n_iterations=-1)

# Full Leiden (refinement + aggregation) with the same objective.
leiden = g.community_hedonic(resolution=g.density(), local_move_only=False, n_iterations=-1)

# Warm start from a known partition: one label per vertex, contiguous from 0.
tracked = g.community_hedonic(initial_membership=list(previous.membership), n_iterations=-1)

# Overlapping cover, warm-started from the disjoint result.
cover = g.community_hedonic(max_memberships=4, initial_membership=list(part.membership), n_iterations=-1)
```

## How the parameters map to the explainer

| Parameter | In the explainer | Notes |
| --- | --- | --- |
| `resolution` | the dial $\gamma$ | Defaults to `graph.density()` when omitted. The explainer's statements assume $0 \le \gamma \le 1$. |
| `max_memberships=1` | disjoint partitions | The explainer only covers the disjoint model. |
| `local_move_only=True` | one agent moves at a time | The native local-moving phase; the explainer's walk is a simplified, fully enumerated analogue. |
| `n_iterations=-1` | "walk until a sink" | Negative values run to the native no-change stopping condition. Positive values are budgets and may stop early. |
| `allow_isolation` | "leave to found a new community" | **Off by default.** The explainer's metagraph always includes these moves; see below. |
| `initial_membership` | the walk's start | `None` = singletons; omitted = reuse `Game.memberships`. |

## `allow_isolation` and the grand coalition

In the metagraph every vertex may also *leave* to found a new, singleton community. In the package these moves into an empty community are controlled by `allow_isolation`, which is `False` by default. Without them no vertex can ever leave the grand coalition, because there is no other community to join.

Recorded with hedonic 1.0.5 and lucas-igraph 1.0.0.5 on the explainer's four-vertex graph (`web/scripts/crosscheck_native.py`; the same for every native seed tried and for hedonic 0.1.1 with lucas-igraph 1.0.0.4):

| $\gamma$ | start | `allow_isolation=True` | `allow_isolation=False` |
| --- | --- | --- | --- |
| 0.2 | singletons | {0,1,2,3} | {0,1,2,3} |
| 0.2 | grand coalition | {0,1,2,3} | {0,1,2,3} |
| 0.5 | singletons | {0,1,2}{3} | {0,1,2}{3} |
| 0.5 | grand coalition | {0,1,2}{3} | {0,1,2,3} |
| 1.0 | singletons | {0}{1}{2}{3} | {0}{1}{2}{3} |
| 1.0 | grand coalition | one of the seven sinks (varies between runs) | {0,1,2,3} |

With isolation allowed, every result is a sink of the browser model; the test `nativeCrosscheck.test.ts` checks this against the recorded fixture. At $\gamma = 1$ seven partitions tie as sinks and the native vertex order decides which one is returned.

## Return value

A `VertexClustering` when `max_memberships == 1`, otherwise a `VertexCover`. The returned object also carries provenance attributes such as `_hedonic_algorithm_identity`, `_hedonic_token_preflight` and, for negative `n_iterations`, `_hedonic_raw_memberships`. They are implementation details for the experiment layer rather than a stable public API.
