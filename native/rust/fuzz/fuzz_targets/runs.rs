#![no_main]
//! Small valid games: every run ends at a certified equilibrium.
libfuzzer_sys::fuzz_target!(|data: &[u8]| hedonic_core_rs_fuzz::runs(data));
