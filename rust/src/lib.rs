//! The hedonic core in Rust: a port of native/ (the C core) that computes the
//! same results bit for bit. Every fused multiply-add is an explicit
//! [`fma::fma`] (the instruction or the soft fma, never libm's), rustc never
//! contracts `a * b + c`, and every sum runs in the C core's order, so the
//! pins of native/tests/test_core.c hold here unchanged.
//!
//! The engines take plain slices and return owned results; the CPython module
//! (feature `python`) is a thin layer over them with the binding's semantics.

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
pub const CORE_VERSION: &str = "2.0.0";

/// SHA-256 over this crate's sources (path, length and bytes of each, in path
/// order, line endings as LF), the format of setup.py's `source_digest`.
pub const SOURCE_SHA256: &str = env!("HIG_RUST_SOURCE_SHA256");
