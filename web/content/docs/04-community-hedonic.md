---
title: community_hedonic
summary: The one detector for disjoint partitions and overlapping covers — parameters, defaults and how they map to the explainer.
group: Python API
order: 4
---

`Game.community_hedonic` is the package's one detector, for disjoint partitions and overlapping covers alike. Every vertex is an agent; one vertex at a time, in random order, it moves to the community (or, for a cover, the set of communities) it prefers. After a move its neighbours are visited again, and the run stops only when every vertex has been visited without moving since the last move, so the result is always an equilibrium of the game. The arithmetic is floating point, so every payoff the detector compares carries a rigorous bound of its rounding error, and a vertex moves only when its gain exceeds that bound: every move raises the exact potential, which guarantees that the run stops. The result reports `equilibrium_slack` in `_hedonic_provenance`, a bound (relative to the total edge weight) of what any vertex could still gain by deviating; it is exactly 0 when the arithmetic is exact, for instance with unit weights at resolution 0, 1/2 or 1. The computation runs in the package's own C core, `hedonic._core`.

```api Game.community_hedonic
```

## Typical calls

```python
# Disjoint partition, run until no vertex wants to move; the seed fixes the visiting order.
part = g.community_hedonic(resolution=g.density(), max_memberships=1, seed=0)

# Warm start from a known partition: one label per vertex, contiguous from 0.
tracked = g.community_hedonic(initial_membership=list(previous.membership), seed=0)

# Overlapping cover, warm-started from the disjoint result.
cover = g.community_hedonic(max_memberships=4, initial_membership=list(part.membership), seed=0)

# At most five occupied communities in the result and in every state visited.
capped = g.community_hedonic(initial_membership=None, max_total_communities=5, seed=0)
```

## How the parameters map to the explainer

| Parameter | In the explainer | Notes |
| --- | --- | --- |
| `resolution` | the dial $\gamma$ | Defaults to `graph.density()` when omitted. The explainer's statements assume $0 \le \gamma \le 1$. |
| `max_memberships=1` | disjoint partitions | The explainer only covers the disjoint model. |
| (always) | one agent moves at a time, until a sink | The detector moves one vertex at a time and stops only when no vertex can improve; the explainer's walk is a simplified, fully enumerated analogue. |
| `seed` | the walk's random rule | Fixes the random visiting order, so the same inputs and seed return the same result. |
| `allow_isolation` | "leave to found a new community" | **Off by default.** The explainer's metagraph always includes these moves; see below. |
| `initial_membership` | the walk's start | `None` = singletons; omitted = reuse `Game.memberships`. |

## `allow_isolation` and the grand coalition

In the metagraph every vertex may also *leave* to found a new, singleton community. In the package these moves into an empty community are controlled by `allow_isolation`, which is `False` by default. Without them no vertex can ever leave the grand coalition, because there is no other community to join.

Recorded with hedonic 2.0.0 on the explainer's four-vertex graph (`web/scripts/crosscheck_native.py`; every seed tried gives the same result except where noted):

| $\gamma$ | start | `allow_isolation=True` | `allow_isolation=False` |
| --- | --- | --- | --- |
| 0.2 | singletons | {0,1,2,3} | {0,1,2,3} |
| 0.2 | grand coalition | {0,1,2,3} | {0,1,2,3} |
| 0.5 | singletons | {0,1,2}{3} | {0,1,2}{3} |
| 0.5 | grand coalition | {0,1,2}{3} | {0,1,2,3} |
| 1.0 | singletons | {0}{1}{2}{3} | {0}{1}{2}{3} |
| 1.0 | grand coalition | one of the seven sinks (depends on the seed) | {0,1,2,3} |

With isolation allowed, every result is a sink of the browser model; the test `nativeCrosscheck.test.ts` checks this against the recorded fixture. At $\gamma = 1$ seven partitions tie as sinks and the random vertex order, fixed by `seed`, decides which one is returned.

## Return value

A `VertexClustering` when `max_memberships == 1`, otherwise a `VertexCover`. The returned memberships are also stored in `Game.memberships`, so a later call without `initial_membership` continues from them, and `Game.evaluate_against` scores them. The result carries a `_hedonic_provenance` dictionary (versions and effective parameters of the call, and the `equilibrium_slack` described above); it is a record for reproducibility rather than a stable public API.
