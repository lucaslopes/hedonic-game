# The C core

`native/` is the detector of `hedonic`: best-response local moving for the
CPM hedonic game, disjoint and overlapping, written for this package. It has
no dependency beyond the C library and is compiled into `hedonic._core`.

```
include/hedonic_core.h   the API: hig_partition, hig_cover, options, errors
core/internal.h          what the engines share, certified arithmetic, hig_fma
core/softfma.h           a correctly rounded fma in integer arithmetic (from musl;
                         MIT licence)
core/dispatch.h          which builds a target compiles (x86_64: two)
core/shared/             what both engines share (declared in internal.h):
  inputs.c               errors and memory, input checks, adjacency, start labels
  random.c               random numbers; which fma the build runs
  queue.c                the random visiting order and the queue
  heap.c                 the heap of communities by mass
core/partition/          the disjoint engine
  partition.h            its state and what its files share
  visit.c                masses, payoff bounds, a visit, the run
  partition.c            the start, the quality, the entry points
core/cover/              the overlapping engine (unit-l2 memberships)
  cover.h                its state and what its files share
  rows.c                 labels and rows
  visit.c                candidates, bounds, the best set; the run, the slack
  start.c                the start and the setup
  cover.c                the result, the quality, the entry points
core/check/              tests only
  check.c                the independent verifier
  hedonic_check.h        traced runs and the verifier (not the API)
unity.c                  the core as one translation unit (the extension's build)
unity_fma.c, dispatch.c  x86_64 only: the core again with the FMA instruction, and
                         the entry points that choose a build from CPUID (empty
                         on other targets)
binding/                 the CPython module hedonic._core
tests/test_core.c        make test (ASan/UBSan), make test-release (-O3),
                         make test-soft-fma, make test-x86_64 (Rosetta)
tests/check_libm.py      make check-libm: no libm call but sqrt, fabs
```

## The game

Vertex `v` has weight `n_v >= 0`, `gamma` is the resolution and `W` the total
edge weight. In a partition, `v` in community `C` earns
`w(v, C \ v) - gamma n_v N(C \ v)`, where `N` sums vertex weights. In a cover,
`v` holds a set `sigma_v` of at most `M` labels with intensity
`f_v = 1/sqrt(|sigma_v|)` on each; label `c` has mass `S_c` (the sum of
`n_u f_u` over its holders) and is worth

    g_c(v) = sum_{u ~ v holding c} w_uv f_u  -  gamma n_v (S_c - n_v f_v [c in sigma_v])

to `v`, which earns `U_v(sigma) = sum_{c in sigma} g_c(v) / sqrt(|sigma|)`.
Both games are exact potential games; the returned quality is the potential
divided by `W` (partitions: `(2 sum_internal w - gamma sum_C N(C)^2) / 2W`;
covers: `(sum_edges 2 w |sigma_a & sigma_b| f_a f_b - gamma sum_c S_c^2) / 2W`).

## A visit

A visit computes the vertex's exact best response over every feasible action.

* **Partition.** Remove `v` from its community, gather `w(v, c)` for the
  communities of its neighbours in a sparse accumulator (a slot field on each
  community), and score staying, each of them, an empty community (when
  isolation is allowed and the count limits permit one more), and the best
  community `v` has no edge to, whose payoff is `-gamma n_v N(c)`: the
  heaviest when `gamma n_v < 0`, else the lightest, needed only when every
  other action loses.
* **Cover.** Score `v`'s own labels and its neighbours' labels the same way.
  The best set of at most `M` labels is a prefix of the labels sorted by
  `g_c`: the best `j` positive gains (`j` maximizing their sum over
  `sqrt(j)`), or, if no gain is positive, the single best label, where a
  fresh label (payoff 0) competes when isolation and the count limits allow.
  Labels `v` has no support from enter only when they can win: the `M`
  heaviest when `gamma n_v < 0`, the lightest when nothing else reaches 0.
  A label `v` holds alone is worth exactly 0; under `n_communities` it is
  kept, so the count never changes.

The communities "with no edge to `v`" come from a binary heap of occupied
communities ordered by mass (heaviest first when `gamma < 0`), walked
best-first with a small frontier heap, so a visit never scans all
communities. The heap is kept only when it can matter: `gamma < 0`,
isolation off, or a count limit. Even then, it is allocated and built only
when a visit first needs that walk; preceding moves update the masses and
member counts without maintaining a heap. Once built, it is maintained on
every move as usual, including mass rebuilds and new cover labels. Building
it from the current occupied communities yields the same mass/id ordering
as maintaining it from the start: the walk's answers, ties, payoff
arithmetic, certificates and random draws are unchanged. Runs that never
need an unsupported community avoid all heap allocation and maintenance.

## Certified moves

Every quantity a visit compares is computed in double precision, so a
computed improvement can differ from the exact one. The core never moves on
a computed improvement alone: each candidate payoff comes with a rigorous
bound of its rounding error, and a vertex moves only when

    computed improvement - (bound of the best action + bound of staying) > 0.

**The bounds.** They use the standard model of IEEE 754 binary64 with
rounding to nearest (`u = 2^-53`, `eta` the smallest subnormal): a sum is off
by at most `u` times its rounded value, a product or `fma` by at most that
plus `eta / 2` (`fma` here and below is the core's `hig_fma`: see
"Determinism and platforms"). Every bound is itself computed with operations
rounded upward (`core/internal.h`: `hig_up(x) = fma(x, 4u, x) >= x (1 + u)^2`,
`hig_add_up`, `hig_mul_up`), so it is never below the quantity it bounds,
and it is exactly 0 when every input to it is 0. The ingredients:

* *Edge weights to a community* (`w(v, C)` of a partition, a label's
  support `sum w_uv f_u` in a cover): a sum of at most `d_v` terms, off by
  `gamma_{d_v + 2}` times the sum of their absolute values
  (`gamma_k = k 1.125 u`, an upper bound of `(1 + u)^k - 1`). With unit
  weights a partition's sums are integers below `2^53`: exact, bound 0.
* *Masses.* Community masses (partitions: the sum of `n_u`; covers: of
  `n_u f_u`) are double-double sums `hi + lo`, updated with TwoSum, which is
  exact except for one rounding per update; each mass carries `err`, the sum
  of those roundings (exact when they are 0, so unit-weight partitions never
  accumulate any). `theta` is a running maximum of `(err + |lo|) / hi` over
  all masses. When a mass's error exceeds `16 u` of it (which only
  cancellation does: a heavy member leaves and light ones remain), every mass
  is summed afresh from the members (`hig_*_rebuild`, O(n + incidences)),
  and `theta` restarts. A cover's terms `n_u f_u` are themselves
  approximations of `n_u / sqrt(|sigma_u|)` (relative error at most
  `kappa = 5u`); when a term or a support term can underflow, `n eta` and
  `d_v eta` are added.
* *The crowding product* `gamma n_v M`: `gamma n_v` is a product (its error
  is computed exactly with `fma` when the product is not tiny), `M` has the
  error above. A payoff `r = fl(w - gamma n_v M)` gets the bound
  `u |r| + eta + e_w + |gamma n_v| e_M + e_{gamma n_v} (|M| + e_M)`; when
  `gamma n_v M` is not tiny, `w - gamma n_v M` is known exactly as a sum of
  doubles (`fma` residual and TwoSum), so a payoff that was computed exactly
  gets the bound 0.
* *Covers.* `v`'s own labels get individual bounds; the other candidate
  labels share one bound `b_max` computed from the largest support, mass and
  `|gain|` among them. The value of a prefix of `j` labels is off by at most
  `(j b_max) / sqrt(j)` plus `(cap + 6) 1.125 u` of the value (the prefix
  sum's roundings and `1/sqrt(j)`); `v`'s current payoff likewise from its
  labels' bounds.
* *The difference* `a - b` of two computed payoffs is formed with TwoSum
  (`d` plus an exact residual); `e` = (bound of `a`) + (bound of `b`) +
  |residual|, rounded up. Both `d` and `e` are doubles, so `d > e` decides
  `d - e > 0` exactly, and the exact improvement is at least `d - e`.

**The coarse bound.** Most visits are decided far from the bound, so a
visit first computes a cheap `coarse` (plain double arithmetic, from the
largest `|payoff|`, weight sum, mass and mass error of its candidates, and
`theta`): every term of the full bounds is replaced by one that dominates it
and the sum is doubled, which covers the upward roundings of the full bounds
and the roundings and underflows of computing `coarse` itself (all terms are
non-negative). For partitions `coarse` is at least four times every payoff's
bound, for covers at least the full bounds of the move and of the gain,
including the difference's residual. A computed improvement above `coarse`
(partitions: `3 coarse`) is then certainly above the full `e`, and the
vertex moves without computing it; a deviation below `-coarse`
(`-3 coarse`) certainly loses and records nothing. Only the rest (well
under 1% of the visits on DBLP and Amazon) computes the full bounds, in an
out-of-line function, and decides exactly as above. The decisions are those
of the full bounds in every case, so the trajectory and the result are the
same as computing them always (a trace computes them always).

**Termination.** Both games are exact potential games: a unilateral move
changes the potential by exactly the mover's change in payoff (partitions:
`Phi = sum_internal w - (gamma / 2) sum_C N(C)^2`; covers:
`Phi = sum_edges w |sigma_a & sigma_b| f_a f_b - (gamma / 2) sum_c S_c^2`).
Every accepted move has an exact improvement of at least `d - e > 0`, so the
exact potential strictly increases with every move, no state is ever
visited twice, and since the states are finite (at most `n^n` partitions,
finitely many covers with at most `M` of `n M` labels per vertex) the run
stops. Rebuilding masses changes no state. There is no cap on rounds or
visits, no tolerance parameter and no divergence error. Inputs are bounded
so that nothing overflows: `sum |w| <= DBL_MAX / 4096` and
`|gamma| max(N, 1)^2 M <= DBL_MAX / 4096` (otherwise `HIG_EOVERFLOW`).

## The equilibrium certificate and its slack

All vertices are queued once in a random order (a Fisher-Yates shuffle with
unbiased bounded draws). The queue is a
FIFO ring, each vertex in it at most once. After a move, the vertex's
neighbours are queued (for partitions, those outside its new community; see
"Requeueing" below). Each visit that does not move stamps the vertex with the
current number of moves. When the FIFO empties, every vertex whose stamp is
older than the last move is queued again, in a fresh random order (a
Fisher-Yates shuffle of those vertices only; these shuffles and the first are
the only uses of the random source); when none is, every vertex has been
visited, without moving, after the last move, in the final state, and the
call returns.

A fresh order at each rescan matters. A community can sit at the edge of its
threshold: on complete YouTube at its density (1,134,890 vertices), one
community of about 156,750 members holds many members with exactly two edges
inside it, for whom staying and leaving differ by a few units of `gamma`, so
each member that joins makes one of them leave and each that leaves lets an
outsider join. These are effects of a mass, not of an edge, so only a rescan
finds them. Rescanning in one fixed order advanced this exchange by a few
moves per rescan, over hundreds of rescans of every vertex (650 to 1,100 s
for the call, in two runs); a fresh order at each rescan settles it in a few (4.5 s, against 6 to
8 s for the same game played by lucas-igraph's local moving, which shuffles
every pass).

A visit that does not move records an upper bound of the exact gain of every
feasible deviation (with a count limit, every deviation the limit allows):
for a deviation with computed improvement `d` and error bound `e`, the exact
gain is at most `d + e`, rounded up (0 when `d + e <= 0`); the visit keeps
the largest. Deviations that certainly lose are discarded by a coarse bound
first; the ones a visit does not score are dominated by one that it does:

* partitions: the communities `v` has no edge to are worth `-gamma n_v M`,
  so the heaviest (`gamma n_v < 0`) or lightest one, found by the heap walk
  and bounded with `theta M` (every other one's mass is within `theta` of
  its computed value, and the heap orders computed values), stands for all
  of them; when there is no walk, `gamma n_v >= 0` and they are all worth
  at most 0, which is scored;
* covers: by the prefix lemma below, every set of labels is worth at most
  the best prefix value of the computed gains plus `e_pre` plus
  `sqrt(cap) b_max`, which the visit bounds; labels without support enter as
  the walk's proxies like a partition's communities (with `gamma n_v < 0`,
  the `cap` heaviest: a set of unscored labels is worth no more than the
  same number of the heaviest).

The run's raw slack is the largest recorded bound since the last move (a
move resets it to 0), i.e. over the visits of the final sweep, all made in
the final state. Partitions keep it as they go. A cover visit records its
bound with the vertex's stamp (the queue's settlement record), in full or,
when the vertex's best set is its own row (most visits), as the coarse upper
bound `d + coarse`. At the end the slack is the largest full bound; then the
coarse ones above it are replaced by full ones, largest first (by visiting
the vertex again without moving it: the state is the one its last visit
saw), until the next coarse one is not above the running maximum. This is
the same value as recording every full bound (bit for bit), for a few dozen
extra visits on DBLP and Amazon. **Claim:** at return, no feasible unilateral deviation
raises any vertex's exact payoff by more than the slack, and each recorded
bound is at most twice the error bound of the comparison it came from (a
deviation the core did not take has `d <= e`, so `d + e <= 2e`). For a
partition the slack is exactly 0 when every error bound is 0, as with unit
weights at resolution 0, 1/2 or 1: the result is then an exact Nash
equilibrium. Other weights carry nonzero bounds even when they are integers
(the bound does not test whether a sum was exact), yet the slack is still 0
when every vertex's best deviation is certainly worse (`d + e <= 0`), which
is common with integer weights. A cover's intensities `1/sqrt(k)` are
rounded, so a cover's slack is never 0 (normalized, about `1e-15` on
Zachary's karate club and smaller on larger graphs). `hig_partition` and `hig_cover`
return it divided by `|W|` (normalized like the quality, rounded up, plus
`eta`; NaN if `W = 0`), and the Python layer reports it as
`equilibrium_slack` in `_hedonic_provenance`. On complete DBLP and Amazon
(seeds 0-2) it is about `1e-21` (normalized) for partitions at the density
and at 0.1462, and `3e-18` to `1e-17` for covers with `M = 8` at 0.1462; the
verifier finds no certainly improving deviation in any of these results.

## The best set is a prefix

**Lemma.** Let `g_1 >= g_2 >= ... >= g_L` be real numbers and
`1 <= cap <= L`. Then

    max { sum_{i in S} g_i / sqrt(|S|) : S non-empty, |S| <= cap }
        = max { (g_1 + ... + g_k) / sqrt(k) : 1 <= k <= cap }.

*Proof (exchange).* Fix `k = |S|`. If `S` is not `{1, ..., k}`, it contains
some `i > k` and misses some `j <= k`; replacing `i` by `j` changes the sum
by `g_j - g_i >= 0` and keeps the size. At most `k` exchanges reach the
prefix, so among sets of size `k` the prefix has the largest sum, and,
dividing by the same `sqrt(k) > 0`, the largest value. Maximizing over `k`
gives the claim. []

How the cover engine uses it. Ties in the order are broken by label id, so
the prefix is a definite set. Only positive gains extend a positive prefix
(adding `g <= 0` to a positive sum lowers its value), and when no gain is
positive the single best label is optimal (`sum_S g <= |S| g_1 <=
sqrt(|S|) g_1` for `g_1 <= 0`). A fresh label is a gain of exactly 0 and a
label `v` must keep under `n_communities` is a fixed member worth 0; the same
exchange applies to the remaining labels with sizes shifted. With errors:
if every computed gain is within `b_max` of the exact one, every set of
`j <= cap` labels is worth at most its computed value plus `sqrt(j) b_max`,
hence at most the best computed prefix value plus `sqrt(cap) b_max`; this is
the bound the slack uses. `tests/test_equilibrium.py` checks the lemma
exhaustively in exact arithmetic (every multiset of up to 6 gains from
`{-3, -2, -1, -1/2, 0, 1/2, 1, 2, 3}` and every cap: 27,027 cases) and on
random rationals.

## Verifying a result

`core/check/check.c` (`hig_verify_partition`, `hig_verify_cover`; tests only,
declared in `core/check/hedonic_check.h`) checks a result independently of the
engines: no queue, no record of past visits, masses summed afresh, and every
feasible deviation of every vertex scored with its own bound (every
neighbour community or label, an empty community or fresh label, and the
proxies of the rest as above), giving the largest certified lower bound and
the largest upper bound of a deviation's gain. `make test` runs it on every
random game of `tests/test_core.c` and requires its certified gain to stay
at or below the slack the engine reported. The Python tests reach it as
`hedonic._core._verify_partition` / `_verify_cover`, next to the traced runs
`_trace_partition` / `_trace_cover` (start state, every accepted move with
its improvement and bound, raw slack) that `tests/test_equilibrium.py`
replays in exact arithmetic (Fractions for partitions, Decimal with at least
80 significant digits and a rigorous margin for covers).

## Requeueing

After a move of `v` to community `C`, the partition engine queues the
neighbours of `v` outside `C`. Those inside `C` may also have changed their
mind (their payoff for staying moved by `w_uv - gamma n_u n_v`); the final
sweep still visits them, so either rule gives a certified equilibrium. The
alternative, queueing every neighbour, was measured on complete DBLP and
Amazon and on two synthetic graphs (an SBM with 10,000 vertices and 93,892
edges, an Erdos-Renyi graph with 100,000 vertices and 509,965 edges): the C
core alone, seeds 0-2 summed, the best of three interleaved rounds, Apple M1
Max under background load:

| Graph, resolution | neighbours outside `C` (kept) | every neighbour |
|-------------------|------------------------------:|----------------:|
| DBLP, density     | 1.020 s | 1.034 s |
| DBLP, 0.1462      | 1.155 s | 1.199 s |
| Amazon, density   | 1.077 s | 1.109 s |
| Amazon, 0.1462    | 1.272 s | 1.298 s |
| SBM, density      | 0.0225 s | 0.0224 s |
| SBM, 0.1462       | 0.0405 s | 0.0439 s |
| ER, density       | 0.156 s | 0.149 s |
| ER, 0.1462        | 0.152 s | 0.154 s |

Queueing every neighbour visits more vertices for the same equilibria
(qualities within 0.003 of each other, either direction) and is 1-4% slower
on the complete graphs; the rule kept is the faster one there, and the two
are within noise on the synthetic graphs.

## Data structures

32-bit vertex and community ids (`n <= 2^31 - 2`), 64-bit CSR offsets. The
adjacency keeps edge order within each vertex and stores weights only when
some weight differs from 1. A community is 16 bytes (mass, members, slot)
plus the low half of its mass and the bound of its accumulated error (16
bytes, in their own array: a move reads both); empty ids are reused from a
stack. A vertex's settlement stamp and recorded bound sit together (16
bytes), because every visit writes both and the runs are bound by memory
latency. A cover's
rows live in one pool with power-of-two capacities (at most `M`), relocated
when they grow and compacted when the pool's waste exceeds its live size.

For unit edge weights, a partition visit accumulates support for its own
community in a local scalar, avoiding its sparse-accumulator lookup on
every internal edge. The additions still follow CSR edge order: the first
`0 + 1` is exactly the former assignment of `1`, and each later partial sum
is identical by induction. Thus the summation and its error bound are
unchanged. That community is appended to the candidate array for the rare
bound paths to recall; scoring and the heap walk already exclude it. Every
alternative keeps its encounter order and tie-breaking. Weighted edges
retain the original sparse accumulation, including signed-zero behavior.

## Determinism and platforms

No threads, fixed summation orders, `-ffp-contract=off` (`/fp:strict` on
MSVC), double everywhere (no `long double`), and only operations that IEEE
754 requires to be correctly rounded: `+ - * /`, `sqrt` and the fused
multiply-add. Given the inputs and the random words, results are therefore
identical on every platform the wheels are built for; the one exception
would be x87 double rounding on 32-bit x86, which is not a wheel target.
Assumed, as for any floating-point code: the default environment (rounding
to nearest, subnormals not flushed to zero).

**No libm.** `make check-libm` (part of `make test`, and of every CI job)
checks that the compiled core calls no libm function but `sqrt` and `fabs` (and no
`float` or `long double` variant). Functions such as `exp`, `log` or `pow`
have no rounding requirement and differ between C libraries. `sqrt` is
correctly rounded by IEEE 754 and is a hardware instruction on every target
(compilers inline it; a libm `sqrt` computes the same bits). `fabs` is
exact (it clears the sign bit); MSVC calls it under `/fp:strict` where other
compilers inline it, with the same bits. The fused
multiply-add is also correctly rounded by IEEE 754, but on x86_64 without
`-mfma` (the wheels' baseline) `fma()` is a libm call, which would make the
results rest on each C library's `fma` (glibc, musl, macOS, the Windows
UCRT) being right. The core never calls it. Every fused multiply-add is
`hig_fma` (`core/internal.h`):

* the instruction, where the target guarantees one at compile time: arm64
  with GCC or Clang (`__builtin_fma`, which is `fmadd`), x86_64 with `-mfma`
  or MSVC `/arch:AVX2` (`__FMA__`, `__AVX2__`), and the FMA build below;
* otherwise `hig_soft_fma` (`core/softfma.h`), a correctly rounded fma in
  integer arithmetic: the exact 106-bit product of the significands (four
  32x32-bit products), the addend aligned with a sticky bit, an exact
  128-bit sum, and one rounding to nearest-even done on the integer bits
  (normal, subnormal and overflowing results), the double assembled from its
  bits. It uses no `long double`, no libm and nothing of the floating-point
  environment (it raises no exception flags); the only floating-point
  operations left are on special operands (zeros, infinities, NaNs), where
  `x * y` and `+` are exact or give the IEEE result. MSVC on arm64 takes
  this path too, because whether its `fma()` is the instruction is not
  documented.

**x86_64 dispatch.** On x86_64 targets without a baseline FMA (`core/dispatch.h`)
the core is compiled twice: `unity.c` with the soft fma, its entry points
named `hig_x86soft_*`, and `unity_fma.c` (which includes `unity.c`) with every
function given `target("fma")` (GCC: `#pragma GCC target`, Clang:
`#pragma clang attribute`; MSVC: the `_mm_fmadd_sd` intrinsic, which needs no
`/arch`), named `hig_x86fma_*`. `dispatch.c` defines the public functions:
at the first call it reads CPUID (FMA, AVX and OSXSAVE in leaf 1, and XGETBV:
the OS saves the XMM and YMM state) and from then on calls the FMA build or
the soft one. Both compute the same correctly rounded `fma`, contraction is
off in both, and nothing else changes how an operation rounds, so the two
builds return the same results bit for bit; the choice only changes the
speed. `setup.py` lists the same three files on every platform: elsewhere
`unity_fma.c` and `dispatch.c` compile to nothing, so a cross or universal2
build decides per target in the preprocessor. `HIG_FORCE_SOFT_FMA` builds
the soft fma everywhere (tests); `hig_fma_path()` and `hig_fma_select()`
(`core/check/hedonic_check.h`, tests only) report and force the build.

**Tests.** `tests/test_core.c` compares `hig_soft_fma` with the instruction
(arm64; x86_64 when the CPU has FMA) on 10^6 adversarial cases (random bit
patterns and doubles, cancellation to the product's own rounding error,
exact zeros, signed zeros and underflowing products, subnormal products and
results, overflow, special operands, exact halfway cases built from integer
products, exponent gaps up to 1100, addends near half an ulp) and checks a
hash of its results against the one recorded on arm64, so a target without
the instruction checks the same results; `HIG_FMA_CASES` sets the count. On
arm64 (Apple M1 Max) 10^9 such cases gave 0 differences from `fmadd`, bit
for bit (NaNs compared as NaN), among them 19.5 million exact ties (248,000
with subnormal results) and 114 million zero, 101 million subnormal and 99
million infinite results. (musl 1.2.5, the source, returns `+0` for
`fma(x, y, +0)` when `x y` underflows to `-0`; IEEE 754 and the instruction
give `-0`, as `hig_soft_fma` does.) `make test-soft-fma` runs the whole C
suite with the soft fma, and `make test-release` and `make test` on an x86_64
target run it as dispatched and with each build forced (`HIG_TEST_FMA`); the
pins are the same in every case. `make test-x86_64` does all of this as
x86_64 under Rosetta: Rosetta 2 executes FMA and AVX2 code but reports
neither in CPUID unless `ROSETTA_ADVERTISE_AVX=1` (macOS 15 and later), so
the tests run once dispatched to the soft build and once to the instruction.

`test_core.c` pins results bit for bit from the seeded order of Python's
`random.Random(seed)` (an MT19937 equivalent), and the same results are
pinned in the Python tests. Sorts break every tie by id. The result is
relabelled `0, 1, ...` in order of first appearance; cover rows are sorted.

## Licence

GPL-3.0-or-later, like the package. The core was written for `hedonic`; it
contains no code from igraph. `core/softfma.h` is adapted from musl libc
1.2.5's `src/math/fma.c` (Copyright 2005-2020 Rich Felker, et al.) and keeps
musl's MIT licence notice; the changes (the integer rounding and assembly
of the result instead of `scalbn` and the floating-point environment, integer
normalization of subnormals, the sign of an underflowing product plus zero,
C portability) are released under the same MIT licence, which is compatible
with the GPL. The binding's `binding/random.c` implements the MT19937 step
(Matsumoto and Nishimura) as CPython's `Modules/_randommodule.c` does, to
reproduce `random.Random` draws.
