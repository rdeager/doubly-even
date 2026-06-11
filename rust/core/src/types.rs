//! Type aliases for the doubly-even kernel.
//!
//! ### Bit width
//!
//! Every binary vector — codeword, dual element, Q-coordinate, σ_Q column —
//! is a single `u64`. The published target is N=32 (Bouyukliev/DFGHILM
//! validation tables); N=64 is generous headroom. We assert this at the
//! FFI boundary so a bad call fails loudly rather than silently truncating.
//!
//! Going to `u128` is a single type-alias change here; we'll do it when
//! N=64 stops being headroom.

/// Maximum supported code length. The kernel rejects FFI inputs with `n > MAX_N`.
pub const MAX_N: u32 = 64;

/// Binary vector in `F_2^N`, bit `i` is component `i`. XOR is addition.
pub type BinVec = u64;

/// A column permutation: `sigma[i] = j` means "old column `i` becomes new column `j`".
/// Same convention as `doubly_even.spec.vectors.apply_permutation`.
pub type ColPerm = Vec<u32>;

/// A `GL(L, F_2)` matrix in column form: `M[j]` is column `j` as a packed
/// `u64`. Bit `i` of `M[j]` is the entry at row `i`, column `j`.
///
/// The action on a column vector `v ∈ F_2^L` is
/// `mat_apply(M, v) = XOR_{i: v_i = 1} M[i]`. See
/// `doubly_even.canon.matrix_group` for the Python reference.
pub type Mat = Vec<BinVec>;
