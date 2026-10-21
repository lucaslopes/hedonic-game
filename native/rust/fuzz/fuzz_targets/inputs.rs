#![no_main]
//! Arbitrary bytes as graphs, options, starts, covers and audit queries:
//! every call returns a value or an error, never panics.
libfuzzer_sys::fuzz_target!(|data: &[u8]| hedonic_core_rs_fuzz::inputs(data));
