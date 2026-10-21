//! The hedonic core in Rust: a port of hedonic 2.0's C core (its native/) that
//! computes the same results bit for bit. Every fused multiply-add is an
//! explicit [`fma::fma`] (the instruction or the soft fma, never libm's), rustc
//! never contracts `a * b + c`, and every sum runs in the C core's order, so the
//! pins of its tests/test_core.c hold here unchanged (tests/pins.rs).
//!
//! The engines take plain slices and return owned results; the CPython module
//! (feature `python`) is a thin layer over them with the binding's semantics.

// Lints the port keeps on purpose (`cargo clippy --all-targets --features
// python -- -D warnings` is otherwise clean, and CI runs it):
// - `!(a <= b)` is the C core's comparison, false for NaN on purpose;
// - index loops and long argument lists mirror native/'s loops and calls, so
//   each line can be read against its C;
// - constants are spelled as in the C sources (ETA's 17 digits);
// - a variable declared before the branches that set it, as in C;
// - Mt19937::next is random.c's name (the struct is no iterator).
#![allow(
    clippy::neg_cmp_op_on_partial_ord,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::needless_late_init,
    clippy::should_implement_trait
)]

pub mod arith;
pub mod audit;
pub mod cover;
pub mod error;
pub mod fma;
pub mod graph;
pub mod heap;
pub mod partition;
pub mod prefetch;
pub mod queue;
pub mod rng;
pub mod trace;
pub mod verify;

#[cfg(feature = "python")]
mod python;

/// The version of the core's algorithm (the C core's `HIG_CORE_VERSION`).
pub const CORE_VERSION: &str = "2.1.0";

/// SHA-256 over this crate's sources (path, length and bytes of each, in path
/// order, line endings as LF), the format of hedonic 2.0's setup.py `source_digest`.
pub const SOURCE_SHA256: &str = env!("HIG_RUST_SOURCE_SHA256");
