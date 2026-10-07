# Case study: the 1.0.0.5 development investigation (2026-09-27)

What one agent (Opus) did with this skill, what it found, which tool found
each problem, and what to do differently. The full record is the decision
ledger on the local branch `dev/opus-hedonic-1.0.5`:
`docs/plans/opus_1_0_5/decision_ledger.md` (with `bench/` and `evidence/`).

## Setup that worked

| Repository | Worktree (under `~/Code/community/opus-worktrees/`) | Branch | Base |
|---|---|---|---|
| igraph C | `igraph-1.0.0.5` | `dev/opus-igraph-1.0.0.5` | tag `1.0.0.4` = `240c75b70` |
| python-igraph | `python-igraph-1.0.0.5` | `dev/opus-python-igraph-1.0.0.5` | tag `1.0.0.4` = `9fbd3547` (gitlink `240c75b70`) |
| hedonic-game | `hedonic-1.0.5` | `dev/opus-hedonic-1.0.5` | `codex/codeseg-reproduction` = `6f1d5d9` |

Builds: `~/dev/igraph_opus-base-1004` (unmodified tag), `~/dev/igraph_opus-1005-dev`
(Release), `~/dev/igraph_opus-1005-asan` (Debug, ASan+UBSan, FINALLY
verification); snapshots `_prefix/igraph-{abc,preC2,c2b,opus-e4db9bdb4}`.
Baseline venv `_venvs/base-1004` (tag build, non-editable Hedonic snapshot).

## What changed (native)

* Performance: amortized mass reconciliation, reuse of the last no-move sweep
  as the certificate, sorting only positive gains, a mass-ordered label index
  (both movers), a growable candidate buffer. Complete DBLP tuned run
  51.9 s -> ~9-10 s with a byte-identical cover; isolation-disabled runs
  700-790 s -> 2-5 s; YouTube 11.6-16.3x, Wikipedia 6.2x (CPU).
* Features: global count limits (`igraph_community_leiden_with_constraints`,
  at most K / exactly K), `get_random_number_generator()` in python-igraph,
  and seeds / node weights / count limits / provenance in `Game`.
* `leiden.c` restructured into documented sections (bit-identical on 50,000
  instances).

## Defects found, and the tool that found each

| Id | Defect | Found by |
|---|---|---|
| NT | 1.0.0.4 multilevel token stage loops forever on exact ties (K8, gamma = 1) | Python differential with a per-instance timeout |
| ZB | zero-budget disjoint call left `nb_clusters`/`quality` uninitialized | reading the budget loop, then a test |
| Diag | `debug_trace` raised false mismatches on 173/300 karate seeds (relative margin on O(m + L)-term sums) | running the diagnostic entry point on ordinary graphs |
| C1 | zero-resolution omitted candidate missing for signed disjoint weights | a review witness turned into a test |
| C2 | multilevel guard compared the proposal with the iteration's *start*, keeping proposals worse than the local-moving state | review, confirmed on DBLP (multilevel scored below local moving at high gamma) |
| C2-tie | strict C2 rejected tied proposals that merge duplicate labels; duplicate-body outputs rose 2,785 -> 4,282 of 12,000 | classifying differential changes by *kind*, then counting duplicate bodies |
| QB | a diagnostic refactor moved the last bit of every overlapping quality (split `q -= g*S*S`) | the "quality_only" class of the differential |
| LL | upstream: the last aggregation level of disjoint Leiden was never written back; `n_iterations < 0` looped forever with signed weights | timeouts in the C equivalence harness (first run) |
| MG | exact-count mandatory label gain not zeroed when already in front | reading the code while restructuring |
| FP | results depend on backend-discretionary fused multiply-add; the released potential fuses the crowding term only outside its unrolled loop body | equivalence harness after moving a loop into a helper |
| Checks | Hedonic trace checks encoded the old guard; the integrity grid pinned exactly 1.0.0.4 | the full Hedonic suite after the native change |
| Web | the explainer's native cross-check was unseeded (fixture flipped between runs) | regenerating web data, then 200-seed sweep |
| Unwind | disjoint mover returned `IGRAPH_INTERRUPTED` with live FINALLY entries | restructuring into workspace structs |

Rejected or deferred after prototyping: continuing multilevel after a
rejected proposal (same covers, 4.9x slower), disjoint-first-then-overlap as
a default (F1 0.380 vs 0.425), parallel conflict-free batches (median batch
4-5 vertices, lower equilibrium), a native unlimited cap (facade maps `-1`
to `n` instead), token-graph redesign.

## Process lessons

1. **Build the equivalence harness first.** It would have found LL and the
   floating-point fragility on day one, and it turns every refactor into a
   `cmp` of two text files.
2. **Every record carries the loaded C version**, and every snapshot has its
   own version string. That is what exposed the invalid `nohup` run.
3. **Classify differences, never just count them**: exact / quality-only /
   renamed / different / status, by instance class, and the direction of the
   quality change. QB and C2-tie were invisible in a single "differs" count.
4. **Codegen experiments in isolation lie.** A function compiled alone fused
   differently from the same function inlined in the library; only the
   in-library harness is authoritative.
5. **Re-run the whole downstream suite after a semantic native change.**
   Hedonic experiment checks mirror native rules; the web fixtures mirror
   native results.
6. **Classify pre-existing failures on the base commit** before touching
   anything, and keep the list in the ledger.
7. **Historical baselines ran on a loaded machine.** Compare detector CPU
   seconds, and prefer a same-session run of the unmodified build.
8. **Measure what users see.** Complete DBLP, not a subgraph, for claims about
   complete DBLP; covered-induced detection is a separate estimand and is
   labelled as such.
9. **Do not disturb the user's jobs.** A LiveJournal tuning run owned the
   machine; LiveJournal was not rerun, and the original checkout was never
   written.

## Production steps left undone

Push/tag/publish of all three repositories; repointing the python-igraph
gitlink; regenerating Hedonic's `uv.lock`; refreshing version-bound Hedonic
tests; regenerating frozen candidates (DNN rational artifact, integrity
grid) under 1.0.0.5; reporting LL upstream; the web Vitest run; LiveJournal.

## Compatibility boundary (restored in the final 1.0.0.5 line)

* `igraph_community_leiden()` is the igraph 1.0.0 symbol again (11 arguments,
  upstream candidate set, no certificate sweep). Removing the fork parameters
  is not enough: the shared disjoint mover completes the candidate set with
  omitted clusters (negative resolution, signed vertex weights), which changed
  1,331 of 20,000 upstream-shaped results until a private
  `complete_best_response` option switched it off for the 1.0.0 symbol.
  Compare against an upstream snapshot, not only against the fork's previous
  build.
* Fork controls enter only through `igraph_community_leiden_with_constraints()`
  and `igraph_community_leiden_with_diagnostics()`; python-igraph keeps the
  1.0.0 positional call shape, makes fork keywords keyword-only, and reaches
  the 1.0.0 C symbol when only 1.0.0 keywords are used.
* Diagnostics work for partitions and covers, with count limits; move trace
  width 15, counters schema 3. The Python differential against a pre-restoration
  build differs exactly on old-shape disjoint calls with negative resolution.

