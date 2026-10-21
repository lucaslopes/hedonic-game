//! A hint to bring the cache line of `slice[index]` in: no effect on any
//! result, and no access is made (an index out of bounds is ignored).

#[inline(always)]
pub fn prefetch<T>(slice: &[T], index: usize) {
    if index < slice.len() {
        let p = slice.as_ptr().wrapping_add(index);
        #[cfg(target_arch = "aarch64")]
        // SAFETY: prfm only hints the cache; it never faults, reads or writes
        // (so `nomem`: pending stores need not be flushed before it), and p
        // points into `slice`.
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, nomem, preserves_flags));
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above (SSE is in the x86_64 baseline).
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p as *const i8);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = p;
    }
}

use std::sync::atomic::{AtomicU8, Ordering};

/// Working sets above this many bytes get prefetches ahead of their visits;
/// below, the caches hold them and the hints only cost instructions.
pub const WORKING_SET_THRESHOLD: usize = 12 << 20;

static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// Benchmarks only: Some(on) forces prefetching on or off, None restores
/// the working-set rule. Results never depend on it.
#[doc(hidden)]
pub fn set_override(on: Option<bool>) {
    OVERRIDE.store(match on { None => 0, Some(false) => 1, Some(true) => 2 }, Ordering::Relaxed);
}

/// Whether a run whose state takes about `bytes` prefetches.
pub fn wanted(bytes: usize) -> bool {
    match OVERRIDE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => bytes > WORKING_SET_THRESHOLD,
    }
}
