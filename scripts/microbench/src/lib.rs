//! Shared helpers for the microbench bins.
//!
//! Since the workspace restructure the bins link `doubly-even-core`
//! (the pure-Rust algorithm crate; pyo3 lives only in the wrapper
//! crate) directly, so every "production arm" in the bins IS
//! production code — the historical hand-copied kernel clones are
//! retired. Only experimental variants, synthetic-input generators and
//! brute-force oracles remain local to the bins.
//!
//! `timing` mirrors `doubly_even_core::cycles` but stays local: it
//! calibrates over 50 ms (vs the kernel's 5 ms — microbenches report
//! absolute ns, so the error budget is tighter) and must exist without
//! the `phase_timers` feature gate. Portable across x86_64 (`rdtsc`)
//! and aarch64 (`cntvct_el0`) so the post-D15 cache-cliff sweeps
//! (`wht_sweep`, `phi_replay`, `singular_walk`) rerun unmodified on
//! GCP Axion. The two pre-D15 bins (`popcount_probe`, `nauty_decomp`)
//! remain x86-only.

pub mod timing {
    use std::sync::OnceLock;

    /// Raw monotonic cycle/tick counter (deltas on one pinned thread).
    #[inline(always)]
    pub fn mono_cycles() -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            return unsafe { core::arch::x86_64::_rdtsc() };
        }
        #[cfg(target_arch = "aarch64")]
        {
            let v: u64;
            unsafe {
                core::arch::asm!("mrs {v}, cntvct_el0", v = out(reg) v, options(nomem, nostack));
            }
            return v;
        }
        #[allow(unreachable_code)]
        {
            static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
            EPOCH
                .get_or_init(std::time::Instant::now)
                .elapsed()
                .as_nanos() as u64
        }
    }

    /// ns per tick, calibrated once against `Instant` over ~50 ms (longer
    /// than the kernel's 5 ms — microbenches report absolute ns, so the
    /// calibration error budget is tighter).
    pub fn ns_per_cycle() -> f64 {
        static NS_PER_CYCLE: OnceLock<f64> = OnceLock::new();
        *NS_PER_CYCLE.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let c0 = mono_cycles();
            while t0.elapsed().as_millis() < 50 {
                std::hint::spin_loop();
            }
            let cycles = mono_cycles().wrapping_sub(c0).max(1);
            t0.elapsed().as_nanos() as f64 / cycles as f64
        })
    }

    #[inline]
    pub fn cycles_to_ns(cycles: u64) -> f64 {
        cycles as f64 * ns_per_cycle()
    }
}

/// xorshift64* — deterministic synthetic inputs, same scheme as
/// `wl_refine`'s `synth_qd_codewords`.
pub struct XorShift64(pub u64);

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        XorShift64(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    #[inline]
    pub fn next(&mut self) -> u64 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.0 = s;
        s.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

/// Touch every cacheline of a scratch buffer (read-modify-write) to
/// evict the benchmark working set from L1/L2 — the "cold" mode proxy
/// for the ~210 KB CanonScratch traffic interleaved between φ/σ_Q calls
/// in production. 4 MB sweeps past Raptor Lake's 2 MB L2 and Neoverse
/// V2's 2 MB L2 while staying inside L3.
pub fn evict_l1_l2(junk: &mut [u8]) {
    debug_assert!(junk.len() >= 4 << 20);
    let mut i = 0;
    while i < junk.len() {
        junk[i] = junk[i].wrapping_add(1);
        i += 64;
    }
}
