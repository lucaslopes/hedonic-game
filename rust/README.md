# hedonic-core-rs

A Rust port of `native/` (the C core and its CPython binding), built with
[PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs). It computes
the same results as the C core bit for bit: every fused multiply-add is an
explicit `fma::fma` (the instruction, or the soft fma ported from musl),
rustc never contracts `a * b + c`, and every sum runs in the C core's order.

| Rust | C |
|---|---|
| `fma.rs`, `arith.rs` | `softfma.h`, `dispatch.h`, `internal.h` |
| `graph.rs`, `rng.rs`, `queue.rs`, `heap.rs` | `shared/` |
| `partition.rs` | `partition/` |
| `cover.rs` | `cover/` |
| `audit.rs`, `verify.rs`, `trace.rs` | `check/` |
| `python/` (module `hedonic_core_rs`) | `binding/` (module `hedonic._core`) |

The module has the functions of `hedonic._core` with the same arguments,
results, exceptions and messages. `HEDONIC_CORE=rust` makes the package load
it in place of the C extension; `source_sha256` is the digest of this crate's
sources (setup.py's format), so records name the Rust core apart.

What is safer: the engines are safe Rust except one documented loop (the
cover's candidate scan, whose invariants are debug-asserted); allocation
failures are errors, not aborts; Python buffers are borrowed only while no
Python code can run, and copied for a generator that calls Python on each
draw.

## Build and test

```bash
cd rust
cargo test --release                              # the C core's bit pins, soft fma vs the instruction
cargo test --release --features force-soft-fma    # the same with the soft fma everywhere
maturin develop --release                         # into the active venv
cd ..
HEDONIC_CORE=rust python tools/native_tests.py    # the pinned Python tests on the Rust core
python -m pytest -x -q rust/tests/test_differential.py   # C against Rust, bit for bit
```

`cargo run --release --example profile_cover -- EDGES GAMMA CAP REPS` is a
profiling loop over an edge-list file.
