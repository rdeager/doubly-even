//! Portable cycle counter for sampled sub-phase timing (`phase_timers`).
//!
//! `Instant::now()` costs ~20–25 ns per call via the vDSO — fine around a
//! ~70 µs σ_Q call, distorting inside a 1–4 µs φ cascade timed at five
//! marks. The raw counter is ~6–9 ns on x86_64 (`rdtsc`; invariant-TSC on
//! every host we run) and ~2 ns on aarch64 (`cntvct_el0`, the constant-rate
//! generic timer — 1 GHz mandated from ARMv8.6, so Neoverse V2 / Axion).
//! Calibrated once per process against `Instant` so consumers get ns.
//!
//! Same timing primitive as `scripts/microbench/` uses, so in-vivo sampled
//! splits and the standalone cache-cliff sweeps are directly comparable.
//! Lives at the crate root (not `experimental/`) because the hot-path
//! `parent_rule` imports it under `phase_timers` — the experimental
//! namespace is a one-way barrier hot-path code must not cross.

use std::sync::OnceLock;

/// Raw monotonic cycle/tick counter. Only meaningful as deltas on one
/// thread pinned to one core (true for the per-thread φ scratch path);
/// convert with [`cycles_to_ns`].
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
        // Portable fallback: correct, just pays the vDSO cost.
        static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
        EPOCH
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_nanos() as u64
    }
}

/// ns per counter tick, calibrated once per process against `Instant`
/// over a ~5 ms spin. Both rdtsc and cntvct_el0 are constant-rate on the
/// supported hardware, so one calibration is valid for the process
/// lifetime regardless of frequency scaling.
pub fn ns_per_cycle() -> f64 {
    static NS_PER_CYCLE: OnceLock<f64> = OnceLock::new();
    *NS_PER_CYCLE.get_or_init(|| {
        let t0 = std::time::Instant::now();
        let c0 = mono_cycles();
        while t0.elapsed().as_millis() < 5 {
            std::hint::spin_loop();
        }
        let cycles = mono_cycles().wrapping_sub(c0).max(1);
        t0.elapsed().as_nanos() as f64 / cycles as f64
    })
}

#[inline]
pub fn cycles_to_ns(cycles: u64) -> u64 {
    (cycles as f64 * ns_per_cycle()) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 10 ms `Instant` sleep must measure as 10 ms ± 30 % through the
    /// cycle counter — catches a broken calibration or a non-constant
    /// counter outright.
    #[test]
    fn calibration_roundtrip() {
        ns_per_cycle(); // force calibration outside the measured window
        let t0 = std::time::Instant::now();
        let c0 = mono_cycles();
        while t0.elapsed().as_millis() < 10 {
            std::hint::spin_loop();
        }
        let wall_ns = t0.elapsed().as_nanos() as f64;
        let cyc_ns = cycles_to_ns(mono_cycles().wrapping_sub(c0)) as f64;
        let ratio = cyc_ns / wall_ns;
        assert!(
            (0.7..1.3).contains(&ratio),
            "cycle-counter ns disagrees with Instant by {ratio:.2}×"
        );
    }
}
