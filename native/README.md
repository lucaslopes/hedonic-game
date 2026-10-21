# The core

`native/` is the detector of `hedonic`: best-response local moving for the
CPM hedonic game, disjoint and overlapping, written for this package, and the
fixed-cover stability audit. It is compiled into the extension
`hedonic._core`. This document is its design: the game, a visit, certified
moves, termination, the slack, the prefix lemma, the verifier, the data
structures, determinism and the audit. They hold for every implementation of
the core; the names below are those of the Rust crate, `native/rust/`, which
is the core of hedonic 2.1. hedonic 2.0 shipped the same core in C, and the
Rust core computes its results bit for bit (`native/rust/README.md` maps each
module to the C file it ports).

```
README.md                this design
check_libm.py            no libm call but sqrt and fabs, in any build of the core
rust/                    the core in Rust (the crate hedonic-core-rs)
  Cargo.toml, build.rs   the crate; build.rs embeds the digest of its sources
                         (hedonic._core.source_sha256)
  src/fma.rs             the fused multiply-add: the instruction, or a correctly
                         rounded soft fma in integer arithmetic (from musl; MIT)
  src/arith.rs           certified (upward-rounded) arithmetic
  src/error.rs           errors; allocation failures as errors, not aborts
  src/graph.rs           inputs: checks, adjacency, start labels
  src/rng.rs             random words: random.Random's MT19937, PCG32, unbiased
                         bounded draws
  src/queue.rs           the random visiting order, the queue, the settlement record
  src/heap.rs            the heap of communities by mass
  src/prefetch.rs        cache hints ahead of the queue (no effect on any result)
  src/partition.rs       the disjoint engine: masses, payoff bounds, a visit, the run
  src/cover.rs           the overlapping engine (unit-l2 memberships): rows,
                         candidates, the best set, the run, the slack
  src/audit.rs           the fixed-cover stability audit
  src/verify.rs          the independent verifier (tests only)
  src/trace.rs           traced runs (tests only)
  src/python/            the CPython module hedonic._core (PyO3)
  tests/pins.rs          cargo test: the bit pins, the soft fma against the instruction
  tests/core.rs          random games against the verifier, errors, limits, callbacks,
                         the soft fma's hash
  tests/audit.rs         the audit, and its windowed search against the exhaustive one
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
plus `eta / 2` (`fma` here and below is the core's own `fma::fma`: see
"Determinism and platforms"). Every bound is itself computed with operations
rounded upward (`arith.rs`: `up(x) = fma(x, 4u, x) >= x (1 + u)^2`,
`add_up`, `mul_up`), so it is never below the quantity it bounds,
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
  is summed afresh from the members (`rebuild_masses`, O(n + incidences)),
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
`|gamma| max(N, 1)^2 M <= DBL_MAX / 4096` (otherwise an overflow error,
`Kind::Overflow`, Python's `OverflowError`).

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
Zachary's karate club and smaller on larger graphs). `partition` and `cover`
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

`verify.rs` (`verify_partition`, `verify_cover`; tests only) checks a result
independently of the
engines: no queue, no record of past visits, masses summed afresh, and every
feasible deviation of every vertex scored with its own bound (every
neighbour community or label, an empty community or fresh label, and the
proxies of the rest as above), giving the largest certified lower bound and
the largest upper bound of a deviation's gain. `cargo test` runs it on every
random game of `tests/core.rs` and requires its certified gain to stay
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
Max under background load (measured on the C core of hedonic 2.0, whose visits
the Rust core repeats move for move):

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

No threads, fixed summation orders, no contraction of `a * b + c` into a
fused multiply-add (rustc never contracts, and there is no fast-math), `f64`
everywhere, and only operations that IEEE 754 requires to be correctly
rounded: `+ - * /`, `sqrt` and the fused multiply-add (`abs`, `min`, `max`
and comparisons are exact). Given the inputs and the random words, results
are therefore identical on every platform the wheels are built for; the one
exception would be x87 double rounding on 32-bit x86, which is not a wheel
target. Assumed, as for any floating-point code: the default environment
(rounding to nearest, subnormals not flushed to zero). Sums are written in a
fixed order and never left to an iterator adaptor whose order or
association could change (`sum()` over floats is not used).

**No libm.** `check_libm.py` (a step of the CI's Rust job) checks that the
compiled extension calls no libm function but `sqrt` and `fabs` (and no
`float` or `long double` variant). Functions such as `exp`, `log` or `pow`
(Rust's `f64::exp`, `ln`, `powf`, and `powi`, whose rounding is
unspecified) have no rounding requirement and differ between C libraries;
the core uses none of them. `f64::sqrt` is correctly rounded by IEEE 754
and is an instruction on every target; `f64::abs` clears the sign bit. The
fused multiply-add is also correctly rounded by IEEE 754, but `f64::mul_add`
on a target without the instruction (x86_64 without `-C
target-feature=+fma`, the wheels' baseline) is a call to the C library's
`fma`, which would make the results rest on each C library's `fma` (glibc,
musl, macOS, the Windows UCRT) being right. The core never calls it. Every
fused multiply-add is `fma::fma`:

* the instruction where the target guarantees it: aarch64 (`f64::mul_add`,
  which is `fmadd` there and never a call);
* on x86_64, the instruction (`_mm_fmadd_sd` in a function compiled with
  `#[target_feature(enable = "fma")]`) when the CPU has it, else the soft
  fma. The choice is made at the first call (`is_x86_feature_detected!`,
  which reads CPUID and checks that the OS saves the AVX state) and kept in
  an atomic: one build, dispatched at run time;
* otherwise, and everywhere with the cargo feature `force-soft-fma`,
  `soft_fma`, a correctly rounded fma in integer arithmetic: the exact
  106-bit product of the significands, the addend aligned with a sticky
  bit, an exact 128-bit sum, and one rounding to nearest-even done on the
  integer bits (normal, subnormal and overflowing results), the double
  assembled from its bits. It uses no libm and nothing of the floating-point
  environment (it raises no exception flags); the only floating-point
  operations left are on special operands (zeros, infinities, NaNs), where
  `x * y` and `+` are exact or give the IEEE result.

The instruction and the soft fma compute the same correctly rounded `fma`,
so the choice only changes the speed. `fma_path()` (`hedonic._core.fma_path()`)
reports it: `instruction`, `soft`, or on x86_64 `dispatch: instruction` /
`dispatch: soft`.

**Tests.** `tests/pins.rs` compares `soft_fma` with the instruction (aarch64;
x86_64 when the CPU has FMA), and `tests/core.rs` runs it on 10^6
adversarial cases (random bit patterns and doubles, cancellation to the
product's own rounding error, exact zeros, signed zeros and underflowing
products, subnormal products and results, overflow, special operands, exact
halfway cases built from integer products, exponent gaps up to 1100, addends
near half an ulp) and checks a hash of its results against the one recorded
on arm64, so a target without the instruction checks the same results. On
arm64 (Apple M1 Max) 10^9 such cases gave 0 differences from `fmadd`, bit
for bit (NaNs compared as NaN), among them 19.5 million exact ties (248,000
with subnormal results) and 114 million zero, 101 million subnormal and 99
million infinite results. (musl 1.2.5, the source, returns `+0` for
`fma(x, y, +0)` when `x y` underflows to `-0`; IEEE 754 and the instruction
give `-0`, as `soft_fma` does.) `cargo test --release --features
force-soft-fma` runs the whole suite with the soft fma, and on macOS the CI
runs both again as x86_64 under Rosetta 2, where the CPU reports no FMA
unless `ROSETTA_ADVERTISE_AVX=1`; the pins are the same in every case.

`tests/pins.rs` pins results bit for bit from the seeded order of Python's
`random.Random(seed)` (an MT19937 equivalent, `rng.rs`), and the same
results are pinned in the Python tests (`tests/test_native_core.py`,
`tests/test_native_backend.py`; `tools/native_tests.py` runs them with
`HIG_PINNED=1`). They are the pins of hedonic 2.0's C core, unchanged. Sorts
break every tie by id. The result is relabelled `0, 1, ...` in order of
first appearance; cover rows are sorted.

## Licence

GPL-3.0-or-later, like the package. The core was written for `hedonic`; it
contains no code from igraph. The soft fma (`fma.rs`) is adapted from musl
libc 1.2.5's `src/math/fma.c` (Copyright 2005-2020 Rich Felker, et al.) and
keeps musl's MIT licence notice; the changes (the integer rounding and
assembly of the result instead of `scalbn` and the floating-point
environment, integer normalization of subnormals, the sign of an
underflowing product plus zero, the port) are released under the same MIT
licence, which is compatible with the GPL. `rng.rs` implements the MT19937
step (Matsumoto and Nishimura) as CPython's `Modules/_randommodule.c` does,
to reproduce `random.Random` draws.

## Fixed-cover stability audit

`AuditState::prepare`, `AuditState::intervals` and `AuditState::regrets` in
`audit.rs` (in Python, the extension's `_audit_prepare`, `_audit_intervals`
and `_audit_regrets`) audit an existing labelled cover, without changing it or running the
detector. In Python, use `Game.stable_intervals` and `Game.regrets`; the
module `hedonic.stability` supplies grid fractions, linear/log measures and
the Nash-window intersection. An interval is closed and lies in [0, 1].

The audit follows the overlapping paper's numerical definition of stability:
`best_utility - current_utility <= atol + rtol * max(1, abs(current_utility))`,
with defaults `atol=1e-10`, `rtol=1e-9`. This heuristic tolerance is independent
of the detector's rigorous rounding bounds and `equilibrium_slack`. The word
"exact" here refers to resolving every affine cell instead of sampling a
grid; it does not assert an exact-arithmetic equilibrium certificate.

Preparation copies and sorts each row, removes repeated labels, computes
`1.0 / sqrt(row_length)`, sums masses in vertex order, and orders labels by
(mass, label). The core's labels must be contiguous non-negative integers; the Python
facade also renumbers gaps in numeric label order, as detector starts do.
Rows must be non-empty. Duplicate community bodies remain distinct labels.
Callers must explicitly complete uncovered vertices, for example with private
labels. The cap defaults to the largest row and may exceed it, but cannot be
smaller. Only unit node weights are supported: the core rejects non-unit
node weights. Python's audit has no node-weight parameter.
Undirected loopless graphs, parallel edges, isolated vertices, and zero edge
weights are supported. Python defaults to an existing `weight` edge attribute,
then to unit weights. Audits accept finite non-negative edge weights only.

The prepared state is immutable and owns its adjacency, rows, intensities,
masses and lightest-label order; it borrows no input memory. A query selects
a range of vertices or an ordered list of vertex indices (repeats allowed,
`Selection`); intervals come back one list per selected vertex, and regrets
and tolerances one row per selected vertex. `AuditOptions` holds the cap,
isolation, the tolerances and `prune` (`prune` off selects the paper's larger
candidate superset for cross-checks). Prepared state can be shared by
concurrent readers or inherited by forked processes; each query owns its
scratch space.

There is no global (vertex, label) support table. At each visit, the adjacency
is traversed in edge-id order, skipping zero-weight edges. Each contribution
is `weight * neighbor_intensity`, added to a sparse accumulator. For loopless
graphs this is exactly the paper's contribution order: an edge contributes in
only one direction to any given vertex. Own labels, labels with positive
support, the lightest remaining label, and optionally a fresh zero-score
label are sorted by numeric label (fresh first). A private label held alone
has zero crowding through the ordinary mass subtraction.

**Arithmetic contract.** Masses, supports, cell prefix coefficients, and grid
score prefixes use sequential binary64 additions. Utilities divide by
`sqrt(k)`; they never multiply by its reciprocal. No audit expression uses
FMA, and contraction remains disabled. The supplied paper reference runs on
CPython 3.12, whose built-in `sum` uses compensation: *only* `vertex_affine`'s
`cur_a` and `cur_b` use that sum. The audit reproduces those two compensated
sums explicitly, independently of the host Python version. This is a narrow
exception to sequential summation, needed for bit-identical interval endpoints
with that reference. A Python 3.11 execution of the paper code can therefore
produce different endpoint bits. Grid regrets still sum evaluated scores
sequentially, as the reference's grid kernel does.

The intervals are defined by the reference's exhaustive search: the cuts
are all pairwise score crossings and the current-utility crossings of +1 and
-1, sorted with 0 and 1, and every nonempty cell between consecutive cuts is
visited. Score ties use ascending candidate index. In the cell's midpoint
order each prefix supplies the reference's linear inequality, with its
original operation order; surviving pieces are merged with the absolute gap
`1e-15`. Grid regrets evaluate scores first, then take the best prefix, and
clamp differences at most `1e-15` to zero. The two paths may disagree within
rounding of an interval endpoint.

`AuditState::intervals` returns that search's intervals, bit for bit, without
visiting most of its cells. It rests on convexity. Every candidate set S of
size k has an affine utility `U_S(x) = (a_S - x b_S) / sqrt(k)`, and the
current utility `u(x)` is affine, so the best response `best(x) = max_S
U_S(x)`, and `best_k(x)` over sets of size k, are convex; the stable set of
the disjoint game is likewise an interval bounded by Familiarity Indices.

- *Window.* For any S, `best(x) - u(x)` is at least the line `U_S(x) -
  u(x)`. The tolerance never exceeds `T = atol + rtol * max(1, |u(0)|,
  |u(1)|)`, and a point of a stable piece has `best - u <= T` up to the
  rounding of its cell's sums and midpoint order. `M` is at least four times
  every such bound, so a point where a line exceeds `T + M` lies in no piece.
  Newton steps along the best set's line at either end shrink [0, 1] to a
  window; a vertex that is never stable usually ends here, with no cut at
  all.
- *Certified core.* On a range [A, B] whose cells share one tolerance line
  (the reference picks it from `u` at the cell midpoint; `atol` alone is
  below every choice), `best_k - u - tol` is convex for each k. If it is at
  most `-margin_k` at A and at B for every k, where `margin_k` is four times
  the rounding of a prefix-k inequality, every prefix inequality of every
  cell inside holds with room for its rounding, whatever the cell's midpoint
  order: no inequality moves `lo` or `hi` off the cell, each cell's piece is
  the whole cell, and the pieces merge into one. A certified range is
  emitted in one step. For vertices with many candidates, a core found by
  bisection from a certified midpoint is never enumerated: only its first
  and last cut are kept.
- *Cells.* The rest, the cells near the ends of the stable set, run the
  exhaustive search's cell code on its own cut values: every cut in the
  window, the nearest one on either side of it, recursive halving of the
  cell range with the two tests above, and single cells below eight.

The cells visited are cells of the exhaustive search, in its order, and the
ones skipped emit nothing or a whole-cell piece, so the merged output is the
same. The exhaustive search stays in the core as
`AuditState::intervals_exhaustive` (tests only); `cargo test`
(`tests/audit.rs`) compares the two bit for bit on random covers with every option
and several tolerances.

With n vertices, m edges, I membership incidences and K occupied labels,
preparation uses O(n + m + I) storage. Query scratch uses O(K + B_max), where
B_max is the largest visited vertex's breakpoint count (at most quadratic in
its candidate count), plus the returned output. Accumulators are cleared only
at touched labels. Support work for vertex v is the sum of its neighbors' row
lengths. For L candidates the exhaustive search costs O(L²) cuts times O(L) per
cell, cubic in L. The windowed search scans the O(L²) pairs once, with no
division for most, and evaluates O(L log L) probes: a few Newton steps, a
bisection, and the halving near the ends of the stable set; it stores and
sorts only the cuts near those ends. On complete YouTube with an uncapped
Hedonic cover (hubs with over 10,000 candidates), with 8 workers, the exact
stage of one cover's audit went from 48 and 63 minutes (the cover and
labelled profiles) to 17 and 32 seconds. Counts and offsets are 64-bit,
allocation products are checked, and out-of-range input or arithmetic that
could overflow returns a core error. The interruption callback runs throughout
queries, including breakpoint generation and sorting; the binding checks
Python's signals with it, as the detector does.

`tests/audit.rs` checks the audit under every cargo test configuration.
`tests/test_stability_audit.py` compares interval tuples and regret/tolerance
bits with an independent Python/NumPy port in `tests/_audit_reference.py`,
and checks the labelled detector equilibrium without changing detector pins.
