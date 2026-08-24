# Native Leiden plan: complete the candidate set when isolation is disabled

Status: implemented upstream and adopted here. `lucas-igraph==1.0.0.3`
contains the native candidate-set fix described below. This repository now
uses that release; the Python two-pass cleanup is no longer required. Keep the
native regression and independent regret audit as safeguards for future
releases.

## 1. Scope and source of truth

The hedonic repository does not contain a copy of the native source. The
binding currently used by it is built from the sibling `igraph-overlap` fork;
the inspected source is:

```text
../igraph-overlap/src/community/leiden.c
```

The inspection was made at commit
`5259c84b861c68a243a6a249849ab33bda8407a1` (`overlap`, 2026-07-08). The fix
was implemented and tested in that native fork, released as
`lucas-igraph==1.0.0.3`, and is now pinned here. A Python cleanup pass may
remain as an audit tool, but it must not be the mechanism that repairs an
incomplete native best-response search.

The affected paths are:

- disjoint local moving:
  `igraph_i_community_leiden_fastmovenodes` (around lines 58--190);
- overlapping local moving:
  `igraph_i_community_leiden_ov_fastmovenodes` (around lines 1394--1790);
- the overlapping iteration/driver, which forwards `allow_isolation` to
  phase 1 and repeats while `changed` is true (around lines 1975--2195).

## 2. Defect to fix

For the disjoint mover, a vertex is removed from its current cluster and then
the candidate list is built from communities incident to its neighbors. When
`allow_isolation == false`, the current code does not add a recyclable empty
cluster and does not add any other existing, non-neighbor cluster either
(lines 133--157). Consequently, a smaller existing community can be absent
from the comparison even though its CPM penalty is lower:

```text
gain(v -> c) = edge_weight(v, c)
               - resolution * node_weight(v) * cluster_weight(c)
```

For a community with no neighbor of `v`, the first term is zero. At a
non-negative resolution, the smallest eligible community is therefore the
best omitted non-neighbor target. The native `changed == false` stopping rule
only proves that no *currently enumerated* candidate improves the objective;
it cannot prove a best response while this candidate is missing.

The overlapping mover has the same gap. It collects `v`'s own communities and
communities of direct neighbors (lines 1529--1563), and only adds one empty
community when isolation is enabled (lines 1515--1527 and 1565--1569). An
existing non-neighbor community can be the best SUBSTITUTE/ADD target because
its gain is

```text
g(v, c) = L_v^c - resolution * node_weight(v) * comm_mass(c)
```

with `L_v^c == 0` for an omitted non-neighbor. In overlap mode the primary
size is `comm_mass` (fractional mass `S_c`), not `comm_tokens`; they coincide
only in the unit-weight disjoint case.

This is a candidate-set correctness problem, not a failure of the negative
`n_iterations` stopping condition. A negative iteration count still means
“repeat until the native routine reports no change”; it does not make an
omitted action visible.

## 3. Correctness rule for the native patch

Assume the current CPM implementation's standard domain:

- edge weights are non-negative;
- node weights are non-negative; and
- `resolution_parameter >= 0`.

After own and neighbor candidates have been marked, and when
`allow_isolation == false`, add **one active existing community that is not
already a candidate and has minimum community mass**:

```text
disjoint:   mass(c) = cluster_weights[c]
overlap:    mass(c) = comm_mass[c]
active(c):  nb_nodes_per_cluster[c] > 0, or comm_tokens[c] > 0
```

The candidate key is `(mass(c), community_id(c))`. The first component fixes
the omitted-action bug; the second component gives a reproducible answer when
several communities have exactly the same minimum mass.

For every omitted non-neighbor, `g(v,c) = -resolution * node_weight(v) *
mass(c) <= 0`. Hence a minimum-mass representative has the largest possible
gain among all omitted non-neighbors. In disjoint mode this is immediately the
exact missing comparison. In overlapping mode, the prefix objective also does
not require every tied representative under these assumptions: once one
maximum, non-positive omitted gain is in the sorted prefix, adding another
equal non-positive copy cannot improve the normalized prefix score. An
exhaustive tiny-graph oracle must nevertheless test this claim, including
`max_memberships > 1`; if a future utility permits negative resolution or
signed weights, this pruning rule is invalid and all eligible communities must
be enumerated (or the API must reject those parameters).

The selected community is not forced into the cover. The usual strict
`gain > current_gain` / `score > current_score` rule remains in force. If no
strict improvement exists, the vertex stays where it is, including when the
best omitted target is tied with the current action.

## 4. Native implementation plan

### 4.1 Add a shared “smallest omitted active community” helper

Implement a small internal helper (one disjoint specialization and one
overlap specialization is acceptable) with this contract:

```text
find_min_omitted_active(mass, active, candidate_marker, current_memberships)
    -> community id, or “none”
```

The helper must:

1. ignore empty/recyclable IDs;
2. ignore IDs already marked in the current visit's candidate set;
3. ignore the current disjoint cluster after the vertex has been removed;
4. in overlap mode, ignore every current membership (already marked by the
   own-membership pass);
5. compare mass first and ID second; and
6. return at most one ID for the first correctness patch.

Call it **after** direct-neighbor collection and **before** gain calculation.
If it returns an ID, append it to `neighbor_clusters` (disjoint) or `cand`
(overlap), mark it in the same scratch structure, and let the existing gain,
strict-improvement, sorting, and prefix code handle it. Do not special-case
the new candidate in the scoring code.

The overlap helper must use `comm_mass` for the comparison. The mass is the
fractional mass currently stored in the native bookkeeping; it is the same
`W_v^c` quantity used by the gain formula for an external community. Do not
silently substitute the integer token count for weighted or overlapping
instances.

### 4.2 Preserve the isolation-enabled path

When `allow_isolation == true`, retain the current one recyclable empty
candidate. An empty community has mass zero and therefore dominates any
non-neighbor existing community for non-negative resolution; adding both is
redundant and would increase the candidate list. The new scan must therefore
run only in the `false` branch.

Update the native comments so they state the complete rule explicitly:

```text
allow_isolation = true  -> one recyclable empty candidate
allow_isolation = false -> one smallest omitted active existing candidate
```

### 4.3 Candidate storage and bounds

The existing overlap allocation is
`maxdeg * max_memberships + max_memberships + 1` (line 1500), which already
has room for own memberships, neighbor communities, and one extra candidate.
Rename the comment to say “one empty **or smallest-omitted-existing**
candidate”, keep an explicit `ncand < cand_cap` assertion, and add a unit test
that exercises the bound. The disjoint `neighbor_clusters` vector is sized to
the number of vertices and needs no new asymptotic allocation.

Every scratch marker set for the fallback candidate must be cleared in the
same loop as all other candidates. A candidate already discovered through a
neighbor must not be appended a second time.

### 4.4 Make the lookup efficient on large covers

The correctness reference implementation may use a linear scan over active
community IDs. Before merging the native fix, benchmark that scan because a
scan for every visited vertex can become `O(|V||C|)` on a cover with many
communities.

The preferred production structure is a native min-heap (or equivalent
ordered active-community index) keyed by `(mass, community_id)`:

- initialize one entry for every active community;
- on a mass change, increment a per-community generation and push a new entry;
- lazily discard stale or inactive entries;
- while looking up the fallback, temporarily skip entries whose IDs are in
  the current candidate marker, then restore those valid skipped entries;
- select the first valid, unseen active entry; and
- periodically rebuild the heap when stale entries exceed a bounded multiple
  of active entries.

This gives amortized `O(log C)` mass updates and a lookup proportional to the
number of small communities excluded by the current neighbor set, rather than
an unconditional full-cover scan. The initial linear scan should remain as a
simple test oracle and a fallback if the heap cannot be maintained safely.
The heap must be updated at exactly the same points as `cluster_weights` /
`comm_mass` and active-count changes, including source communities that become
empty and newly activated IDs.

### 4.5 Equal-size ties and the friends-of-friends idea

The default native policy should be deterministic `(minimum mass, lowest
community ID)`. Equal-minimum communities have identical immediate utility for
the current vertex, so choosing an arbitrary ID is mathematically valid for
the current best-response value; deterministic ID order is preferable for
reproducibility and avoids extra graph traversal.

Friends-of-friends may be evaluated as a **secondary trajectory heuristic**,
never as a replacement for the mass key. A possible definition for an
unweighted graph is the weighted number of length-two paths from `v` into a
candidate community:

```text
fof(v, c) = sum_{u in N(v)} w(v,u)
                     * sum_{x in N(u) ∩ C_c} w(u,x)
```

If this option is ever added, it must:

1. run only among equal minimum-mass candidates;
2. use a bounded two-hop work budget so hubs cannot cause an unbounded visit;
3. fall back to community ID on budget exhaustion or a numerical tie; and
4. be recorded as a tie policy, because it can change the later trajectory
   even though it cannot improve the current vertex's immediate score.

Do **not** expose a new Python parameter for friends-of-friends in the first
correctness patch. First land the native candidate-set fix with deterministic
ID ties and use the exhaustive oracle below to decide whether a trajectory
heuristic is worth its cost.

## 5. Tests required in the native fork

Add C-level tests and a small exhaustive reference checker. For each tiny
graph, enumerate every admissible disjoint label or overlapping membership set
under the same cap, and compare the native local mover's chosen score with the
true best response.

Minimum cases:

1. **Disjoint omitted smaller target.** Give `v` no edge to a smaller existing
   community, start it in a larger community, use positive resolution, and
   verify that the native mover sees and takes the smaller target whenever it
   is a strict improvement.
2. **Overlapping SUBSTITUTE/ADD.** Use `max_memberships >= 2` and an initial
   cover where the best target is a non-neighbor community omitted by the old
   collector. Compare the native score with exhaustive ADD, REMOVE, and
   SUBSTITUTE actions.
3. **Equal minimum masses.** Create two or more omitted communities with the
   same minimum mass. Verify that a fixed RNG seed produces the same selected
   ID, that the immediate objective is unchanged if the tied ID is exchanged,
   and that no duplicate candidate is generated when one tied community is a
   direct-neighbor candidate.
4. **Overlap tied-prefix regression.** Include several equal minimum targets
   and `max_memberships > 1`; verify the one-representative optimization
   against exhaustive prefix scoring under non-negative resolution. Keep this
   test as a guard against future changes to the utility.
5. **Edge cases.** Test a singleton source, a graph with no active omitted
   community, isolated vertices, self-loops as configured by the binding, and
   all communities already present in the neighbor candidate set.
6. **Parameter-domain guard.** Test negative resolution and signed weights.
   Either enumerate all active communities in those modes or fail explicitly;
   never apply the minimum-mass shortcut silently outside its proof domain.
7. **Isolation regression.** With `allow_isolation == true`, verify that the
   existing empty-candidate behavior and outputs are unchanged.
8. **Stopping behavior.** With `n_iterations < 0`, verify that the queue still
   stops only after no strict improvement remains, now including the newly
   visible omitted target.

The binding-level regression in this repo calls `Game.community_hedonic` with
`allow_isolation=False`, `ensure_equilibrium=False`, and a crafted initial
cover. The result should reach the native expected local equilibrium without a
second Python call. Keep the independent regret audit as a test oracle, not as
the repair path.

## 6. Release, benchmark, and acceptance criteria

1. Implement the native patch and tests in `igraph-overlap`.
2. Run the native test suite, the exhaustive tiny-graph oracle, and the
   hedonic Python tests against the rebuilt wheel.
3. Compare wall-clock time, number of candidate evaluations, moves, and final
   objective on smoke graphs and representative SNAP/DBLP covers for:
   `allow_isolation=False` and `allow_isolation=True`, disjoint and overlap,
   local-only and full multi-phase modes.
4. Require no regression in the isolation-enabled path and a bounded overhead
   for the disabled path. If the linear reference scan is too costly, land
   the heap before release rather than hiding the cost in Python.
5. Publish a new native package version and update the hedonic dependency and
   lockfile to the exact release/hash. Record the native protocol version in
   experiment manifests so old two-pass results cannot be mixed with fixed
   native results.
6. Only after the release is validated, simplify `community_hedonic`'s
   `ensure_equilibrium` handling. It may continue to provide an optional
   full-Leiden-plus-audit pipeline, but it must no longer be required to repair
   a missing `allow_isolation=False` candidate.

## 7. Decisions to keep explicit in review

- **Primary key:** community mass, not raw node count in overlap mode.
- **Tie default:** lowest stable community ID; ties are not forced moves.
- **Friends-of-friends:** optional bounded trajectory heuristic only; not part
  of the equilibrium certificate.
- **Data structure:** linear scan for the correctness oracle, maintained
  min-heap/ordered index for production if profiling justifies it.
- **Validity domain:** minimum-mass pruning is certified only for non-negative
  weights and non-negative resolution.
- **Algorithmic guarantee:** the patch makes the `allow_isolation=False`
  local candidate set complete for the omitted non-neighbor class under those
  assumptions. Full multi-phase projection still needs the normal subsequent
  native no-change cycle and an independent audit when a mathematical
  equilibrium certificate is required.

## 8. Adoption validation for lucas-igraph 1.0.0.3

The dependency was upgraded with:

```bash
uv add 'lucas-igraph==1.0.0.3'
```

The following checks passed against the installed wheel:

- a disjoint fixture where an isolated vertex must move from a larger current
  community to a smaller non-neighbor community;
- the analogous overlapping fixture with `max_memberships=2`;
- randomized disjoint and overlapping regret audits for local-only and full
  Leiden with `allow_isolation=False`;
- the focused and full `CommunityHedonic` test group; and
- the protocol, controlled-overlap, and SNAP benchmark test groups (the old
  v2 controlled ledger is explicitly skipped as historical 1.0.0.2 evidence).

`Game.community_hedonic(..., ensure_equilibrium=True)` now makes one native
call, forces `n_iterations=-1`, and preserves that native membership as
provenance. It no longer rejects `allow_isolation=False`, performs label-bank
re-encoding, or invokes a second Python cleanup call.
