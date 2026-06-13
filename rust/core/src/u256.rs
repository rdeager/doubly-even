//! Minimal unsigned 256-bit integer for the mass spine.
//!
//! σ(N, k) — the per-rank mass quota `Σ N!/|Aut|` — fits u128 only
//! through N = 29 (max σ(29, ·) ≈ 2^127.7; measured 2026-06-13). At
//! N = 30 the max is ≈ 2^136, so every accumulator that SUMS per-class
//! contributions (`mass_at_k`, `GlobalMassTracker`, quota) must be
//! wider. Individual contributions `N!/|Aut| ≤ N!` stay comfortably
//! inside u128 (30! ≈ 2^108), so the only operations needed are
//! add-u128-with-carry, compare, and decimal conversion at the Python
//! boundary. 2^256 ≈ 1.2e77 clears the N = 32 maximum (≈ 2^156) with
//! ~100 bits of headroom.

use std::fmt;

/// Unsigned 256-bit integer: `hi * 2^128 + lo`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Debug, Hash)]
pub struct U256 {
    /// Most-significant 128 bits. Field order matters: derived
    /// `Ord`/`PartialOrd` compare `hi` first, which is the numeric order.
    pub hi: u128,
    pub lo: u128,
}

impl U256 {
    pub const ZERO: U256 = U256 { hi: 0, lo: 0 };
    /// The "infinite quota" sentinel (parallel workers defer all
    /// mass-stop decisions to the shared tracker).
    pub const MAX: U256 = U256 {
        hi: u128::MAX,
        lo: u128::MAX,
    };

    /// Checked addition; panics on (cosmologically impossible) overflow,
    /// mirroring the `checked_add().expect()` discipline of the old
    /// u128 spine.
    pub fn checked_add(self, rhs: U256) -> U256 {
        let (lo, carry) = self.lo.overflowing_add(rhs.lo);
        let hi = self
            .hi
            .checked_add(rhs.hi)
            .and_then(|h| h.checked_add(carry as u128))
            .expect("U256 overflow");
        U256 { hi, lo }
    }

    pub fn add_u128(self, rhs: u128) -> U256 {
        self.checked_add(U256::from(rhs))
    }

    /// Multiply by a small scalar (decimal parsing). Panics on overflow.
    fn mul_small(self, m: u64) -> U256 {
        let m = m as u128;
        // Split lo into two 64-bit halves to keep partial products in u128.
        let lo_lo = (self.lo & u64::MAX as u128) * m;
        let lo_hi = (self.lo >> 64) * m;
        let carry_into_hi = lo_hi >> 64;
        let (lo, c) = ((lo_hi << 64) as u128).overflowing_add(lo_lo);
        let hi = self
            .hi
            .checked_mul(m)
            .and_then(|h| h.checked_add(carry_into_hi))
            .and_then(|h| h.checked_add(c as u128))
            .expect("U256 mul overflow");
        U256 { hi, lo }
    }

    /// Divide by a small scalar, returning (quotient, remainder).
    /// Long division over the four 64-bit limbs.
    fn divmod_small(self, d: u64) -> (U256, u64) {
        debug_assert!(d != 0);
        let d = d as u128;
        let limbs = [
            (self.hi >> 64) as u64,
            self.hi as u64,
            (self.lo >> 64) as u64,
            self.lo as u64,
        ];
        let mut q = [0u64; 4];
        let mut rem: u128 = 0;
        for (i, &limb) in limbs.iter().enumerate() {
            let cur = (rem << 64) | limb as u128;
            q[i] = (cur / d) as u64;
            rem = cur % d;
        }
        (
            U256 {
                hi: ((q[0] as u128) << 64) | q[1] as u128,
                lo: ((q[2] as u128) << 64) | q[3] as u128,
            },
            rem as u64,
        )
    }

    /// Parse a decimal string (the Python-boundary inbound format).
    pub fn from_decimal(s: &str) -> Result<U256, String> {
        let s = s.trim();
        if s.is_empty() {
            return Err("empty string".into());
        }
        let mut acc = U256::ZERO;
        for c in s.chars() {
            let d = c
                .to_digit(10)
                .ok_or_else(|| format!("invalid decimal digit {c:?} in {s:?}"))?;
            acc = acc.mul_small(10).add_u128(d as u128);
        }
        Ok(acc)
    }

    /// The value as u128 if it fits (legacy N ≤ 29 paths).
    pub fn to_u128(self) -> Option<u128> {
        (self.hi == 0).then_some(self.lo)
    }

    /// Wrapping-free subtraction; panics if `rhs > self` (diff
    /// formatting in the mass gate only subtracts the smaller side).
    pub fn checked_sub(self, rhs: U256) -> U256 {
        assert!(self >= rhs, "U256 subtraction underflow");
        let (lo, borrow) = self.lo.overflowing_sub(rhs.lo);
        U256 {
            hi: self.hi - rhs.hi - borrow as u128,
            lo,
        }
    }
}

impl From<u128> for U256 {
    fn from(v: u128) -> Self {
        U256 { hi: 0, lo: v }
    }
}

impl fmt::Display for U256 {
    /// Decimal (the Python-boundary outbound format).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.hi == 0 {
            return write!(f, "{}", self.lo);
        }
        let mut digits = Vec::with_capacity(78);
        let mut cur = *self;
        while cur != U256::ZERO {
            let (q, r) = cur.divmod_small(10);
            digits.push(b'0' + r as u8);
            cur = q;
        }
        digits.reverse();
        f.write_str(std::str::from_utf8(&digits).expect("ascii digits"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u128_round_trip() {
        for v in [0u128, 1, 9, 10, u64::MAX as u128, u128::MAX] {
            let x = U256::from(v);
            assert_eq!(x.to_string(), v.to_string());
            assert_eq!(U256::from_decimal(&v.to_string()).unwrap(), x);
            assert_eq!(x.to_u128(), Some(v));
        }
    }

    #[test]
    fn carrying_add_crosses_the_u128_boundary() {
        let x = U256::from(u128::MAX).add_u128(1);
        assert_eq!(x, U256 { hi: 1, lo: 0 });
        assert_eq!(x.to_string(), "340282366920938463463374607431768211456"); // 2^128
        assert_eq!(x.to_u128(), None);
        assert_eq!(
            U256::from_decimal("340282366920938463463374607431768211456").unwrap(),
            x
        );
    }

    #[test]
    fn sigma_30_scale_value_round_trips() {
        // Exact max σ(30, k) (gaborit_sigma; 2^136-scale): a value the
        // u128 spine cannot hold.
        let s = "150480503525118908010130894718780438761875";
        let v = U256::from_decimal(s).unwrap();
        assert_eq!(v.to_string(), s);
        assert!(v > U256::from(u128::MAX));
    }

    #[test]
    fn ordering_is_numeric() {
        let a = U256 { hi: 1, lo: 0 };
        let b = U256 { hi: 0, lo: u128::MAX };
        assert!(a > b);
        assert!(U256::from(7u128) < U256::from(8u128));
        let mut acc = U256::ZERO;
        for _ in 0..1000 {
            acc = acc.add_u128(u128::MAX / 2);
        }
        assert_eq!(acc, {
            let mut x = U256::ZERO;
            x = x.checked_add(U256::from(u128::MAX / 2).mul_small(1000));
            x
        });
    }

    #[test]
    fn accumulation_matches_python_oracle() {
        // Σ of 5 large u128 contributions, oracle computed with Python ints:
        // 5 * (2^127 - 1) = 850705917302346158658436518579420528635
        let mut acc = U256::ZERO;
        for _ in 0..5 {
            acc = acc.add_u128((1u128 << 127) - 1);
        }
        assert_eq!(acc.to_string(), "850705917302346158658436518579420528635");
    }
}
