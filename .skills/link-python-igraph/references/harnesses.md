# Harnesses

All paths are relative to `.skills/link-python-igraph/`. The harnesses only
read the installed stack and write their own output files.

## check_stack.py -- which stack does this interpreter load?

```bash
$VENV/bin/python scripts/check_stack.py \
  --expect-prefix "$C_PREFIX" --expect-igraph-source "$PY_IGRAPH_SRC" \
  --expect-hedonic-source "$HEDONIC_SRC" --expect-c-version 1.0.0.5-dev
```

Prints JSON (interpreter; `igraph.__version__`; `__igraph_version__`;
source path; extension file, its linked libigraph and rpaths; the libigraph
the loader maps; distribution metadata; Hedonic path and algorithm identity;
loader variables) and exits 1 on a mismatch. A binding that cannot load its
library (for example a 1.0.0.5 extension on a 1.0.0.4 library via
`DYLD_LIBRARY_PATH`) is reported as a failed import, not a crash. A warning
flags editable metadata that differs from `igraph.__version__`.

## snapshot_build.sh -- one commit, one prefix, one version string

```bash
scripts/snapshot_build.sh <igraph-worktree> <commit> <name> [dev-root]
```

`git archive` into `<dev-root>/_scratch/igraph-<name>`, Release build in
`~/dev/igraph_<name>`, install into `<dev-root>/_prefix/igraph-<name>`,
`igraph_version()` = `<base>-<name>`. Refuses to reuse any of the three paths.
About 20 s on an M1 Max. Load it with `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH`
set on the Python (or harness) command itself.

## leiden_equivalence.c -- bit-for-bit comparison of two C builds

```bash
P=$C_PREFIX
cc -O1 -I$P/include/igraph scripts/leiden_equivalence.c -L$P/lib -ligraph \
   -Wl,-rpath,$P/lib -o leiden_equivalence
./leiden_equivalence 20000 20260927 5 > new.txt                  # count seed timeout
DYLD_LIBRARY_PATH=$REF/lib ./leiden_equivalence 20000 20260927 5 > ref.txt
cmp ref.txt new.txt
./leiden_equivalence 312 20260927 5 311                          # rerun only instance 311
LEIDEN_EQ_VERBOSE=1 ./leiden_equivalence 312 20260927 5 311      # print its inputs (simple family)
```

**Interface generations.** One binary can serve two libraries only while the
Leiden signatures agree. lucas-igraph 1.0.0.5 final restored the 11-argument
igraph 1.0.0 `igraph_community_leiden()` and widened
`igraph_community_leiden_with_diagnostics()` (count limits, `membership`,
optional `counters`), so build one binary per prefix, each against its own
headers; the source selects the interface from
`IGRAPH_LEIDEN_TRACE_SCHEMA_VERSION` and, in a final build, hashes only the
12 legacy move-trace columns so the outputs still compare line by line.
Loading a library with different Leiden signatures through
`DYLD_LIBRARY_PATH` is undefined behaviour, not a comparison.

```bash
cc -O1 -I$REF/include/igraph scripts/leiden_equivalence.c -L$REF/lib -ligraph \
   -Wl,-rpath,$REF/lib -o eq_ref                                   # earlier interface
LEIDEN_EQ_SELFCHECK=1 ./leiden_equivalence 20000 20260927 60 > new.txt 2> self.err
grep -c SELFCHECK-MISMATCH self.err                              # must be 0
cc -O1 -DLEIDEN_EQ_BASE_API -I$UPSTREAM/include/igraph scripts/leiden_equivalence.c \
   -L$UPSTREAM/lib -ligraph -Wl,-rpath,$UPSTREAM/lib -o eq_base      # igraph 1.0.0 API only
./eq_base 20000 20260927 20 > upstream.txt
DYLD_LIBRARY_PATH=$C_PREFIX/lib ./eq_base 20000 20260927 20 > fork_base.txt
```

`LEIDEN_EQ_SELFCHECK=1` (final interface) re-runs every instance through the
other entry points with the same random state: diagnostics (full and
counters-only) must reproduce the constrained result, and a disjoint call
with the igraph 1.0.0 defaults and a non-negative resolution must match the
fork-base symbol. `-DLEIDEN_EQ_BASE_API` compiles only the igraph 1.0.0 entry
points (the 11-argument symbol and the simple interface), so one binary
compares a fork build with upstream. Expected fork-base differences from
upstream 1.0.0: zero-budget outputs (set and renumbered), the last-level
write-back fix (changed results and upstream timeouts), and last-bit
qualities from contraction (`-ffp-contract=off` on both sides removes them).

Families, in rotation: disjoint via `igraph_community_leiden_with_constraints`
(directed 30%, signed weights, out/in weights, loops, count limits, random
starts, list output), the simple interface (modularity, CPM, ER), overlapping
covers (weights with zeros, node weights, caps 2..n, count limits, random
covers with duplicate labels), and the diagnostic entry point (trace
matrices hashed). Line format:

```text
<index> <family> ok h=<fnv64 of result> nb=<clusters> q=<hex float>
<index> <family> error=<igraph_error_t>
<index> timeout | crashed
```

Instance parameters come from an independent splitmix64 PRNG; igraph's RNG
is reseeded per instance, so outputs do not depend on how much randomness an
earlier instance consumed. Each instance runs in a forked child with a
wall-clock limit (the parent replays generation without calling igraph).

Under sanitizers (static Debug build with `IGRAPH_VERIFY_FINALLY_STACK=ON`):

```bash
cd $ASAN_BUILD
c++ -x c -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer -g -O1 \
    -DIGRAPH_STATIC -I $IGRAPH_SRC/include -I include \
    $SKILL/scripts/leiden_equivalence.c -c -o eq_asan.o
c++ -fsanitize=address,undefined eq_asan.o src/libigraph.a <dependencies from
    tests/CMakeFiles/test_community_leiden.dir/link.txt> -o leiden_equivalence_asan
ASAN_OPTIONS=abort_on_error=1:halt_on_error=1:detect_leaks=0 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ./leiden_equivalence_asan 20000 20260927 60
```

Expect the reference error codes (only `4` = `IGRAPH_EINVAL` for invalid
inputs), no `38` (`IGRAPH_EINTERNAL`: a debug oracle or drift check fired),
no `crashed`, empty stderr. Debug covers differ from Release in the last bits
(different floating-point codegen); do not compare them with `cmp`.

## native_differential.py -- what changed, and where

```bash
python scripts/native_differential.py run --count 6000 --seed 0 --out a.jsonl
DYLD_LIBRARY_PATH=$REF/lib python scripts/native_differential.py run --count 6000 --seed 0 --out b.jsonl
python scripts/native_differential.py compare b.jsonl a.jsonl --seed 0
python scripts/native_differential.py dump 5012 --seed 1       # one instance as JSON
python scripts/native_differential.py identity                 # which igraph runs
```

Uses the public python-igraph API (`Graph.community_leiden`, CPM,
`node_weights`, `max_memberships`). Kinds in `compare`:

| Kind | Meaning |
|---|---|
| exact | same rows and same quality (or the same error) |
| quality_only | same rows, quality differs (usually last bits: FP codegen) |
| renamed | same community bodies, different label ids |
| different | different covers |
| status | one side errored or timed out |

With `--seed`, non-exact instances are grouped by class
(`M=1|M>1 / local|multilevel / budget<0|=0|>0`). The generator is frozen, so
batteries from different sessions compare directly.

## Complete-graph benchmarks (Hedonic development branch)

`docs/plans/opus_1_0_5/bench/hoc_bench.py` on `dev/opus-hedonic-1.0.5` runs
`Game.community_hedonic` on a SNAP network in a forked child and writes one
JSON record per run: config, stack identity (C version, loader path),
graph identity and scope, status, detector wall and CPU seconds, peak RSS,
native quality, canonical and labelled cover hashes, metrics (filtered and,
with `--unfiltered`, unfiltered), and optionally an independent regret audit
(`--audit`) and the saved rows (`--save-rows`). Options for estimands:
`--covered-induced` (detect on the reference-covered vertices; label the
result as a subgraph), `--subgraph-hops/--subgraph-target` (BFS subgraphs).
It reuses the manuscript's `tuning/hoc_tuning.py` loader and scorer, so its
numbers are comparable with `tuning/final.jsonl`.
