"""Differential test for ``paired_iso.reconstruct_canon_info``.

Phase A of the cheap-equivalence-verifier plan
(``/home/dev/.claude/plans/let-s-implement-the-previous-memoized-simon.md``):
validate the algebra of reconstructing ``CanonInfo`` for D from cached
``CanonInfo`` for cf + a witness permutation π, before porting to Rust.

Strategy: for each small ``(N, k)`` cell, enumerate every equivalence
class. For each class, take the canonical-form RREF as ``cf``; build K
permuted copies ``d`` by applying random column perms to cf and
re-RREFing; ask ``paired_iso_with_witness(d, cf)`` for a witness π; run
``reconstruct_canon_info(canon_info(C_cf), π, N)`` and compare against
``canon_info(C_d)`` (the nauty oracle).

The reconstructed ``canonical_column_order`` need not equal the oracle's
literally — many σ produce the same canonical form. We check the
*observable* property: ``canonical_form(d, σ_reconstructed) ==
canonical_form(cf, σ_cf)``. The aut group, aut order, and column-orbit
partition are checked directly.
"""

from __future__ import annotations

import random

import pytest

from doubly_even.canon.nauty import CanonInfo, canon_info
from doubly_even.canon.paired_iso import (
    paired_iso_with_witness,
    reconstruct_canon_info,
)
from doubly_even.canon.permutations import group_order
from doubly_even.enumerate.augment import enumerate_doubly_even_at
from doubly_even.spec.codes import Code
from doubly_even.spec.vectors import apply_permutation


def _canonical_rref(rref: tuple[int, ...], info: CanonInfo, n: int) -> tuple[int, ...]:
    """Apply ``info.canonical_column_order`` to rows of ``rref`` and re-RREF."""
    permuted = tuple(
        apply_permutation(b, list(info.canonical_column_order)) for b in rref
    )
    return Code(n, permuted).rref_basis()[0]


def _random_permute(rref: tuple[int, ...], n: int, rng: random.Random) -> tuple[int, ...]:
    perm = list(range(n))
    rng.shuffle(perm)
    permuted = tuple(apply_permutation(b, perm) for b in rref)
    return Code(n, permuted).rref_basis()[0]


# (N, k) cells. Coverage: every small case where >= 1 class exists.
_CELLS = [
    (8, 1), (8, 2), (8, 3), (8, 4),
    (10, 1), (10, 2), (10, 3),
    (12, 1), (12, 2), (12, 3), (12, 4),
    (14, 2), (14, 3), (14, 4),
]


@pytest.mark.parametrize(("N", "k"), _CELLS)
def test_reconstruct_matches_oracle(N: int, k: int) -> None:
    rng = random.Random(0xC0FFEE ^ (N << 8) ^ k)

    for ec in enumerate_doubly_even_at(N, k):
        # cf is the class's canonical-form RREF; canon_info(cf) is the
        # cached info we'd be reusing.
        cf_basis_code = Code(N, ec.code.basis)
        cf_info_raw = canon_info(cf_basis_code)
        cf_rref = _canonical_rref(ec.code.basis, cf_info_raw, N)
        # Re-canonicalise from the canonical-form basis — this is what
        # the verifier dispatch will have cached.
        cf_info = canon_info(Code(N, cf_rref))
        # The canonical column order from cf_info applied to cf_rref must
        # be the identity (cf is already canonical); used as the
        # reference canonical form below.
        canonical_form_ref = _canonical_rref(cf_rref, cf_info, N)

        # Generate 4 random permuted copies of cf and test each.
        for _ in range(4):
            d_rref = _random_permute(cf_rref, N, rng)
            pi = paired_iso_with_witness(d_rref, cf_rref, N)
            assert pi is not None, (
                f"paired_iso failed to find a witness at N={N}, k={k}"
            )

            # Verify π is actually a witness: applying π to D rows and
            # re-RREFing yields cf's row-span (== cf_rref since cf is RREF).
            permuted = tuple(apply_permutation(b, list(pi)) for b in d_rref)
            assert Code(N, permuted).rref_basis()[0] == cf_rref, (
                f"witness π is not a valid iso at N={N}, k={k}"
            )

            # Reconstruct and compare to the oracle.
            d_info_oracle = canon_info(Code(N, d_rref))
            d_info_recon = reconstruct_canon_info(cf_info, pi, N)

            # aut_order: invariant. Must match exactly.
            assert d_info_recon.aut_order == d_info_oracle.aut_order, (
                f"aut_order mismatch at N={N}, k={k}: "
                f"recon={d_info_recon.aut_order}, "
                f"oracle={d_info_oracle.aut_order}"
            )

            # canonical_column_order: check the observable — applying the
            # reconstructed σ to D and re-RREFing must yield the same
            # canonical form as cf's own canonicalisation.
            d_canon = _canonical_rref(d_rref, d_info_recon, N)
            assert d_canon == canonical_form_ref, (
                f"reconstructed σ does not produce the right canonical "
                f"form at N={N}, k={k}: got {d_canon}, "
                f"want {canonical_form_ref}"
            )

            # aut_generators: each reconstructed generator must fix D as a
            # subspace, and the subgroup generated must equal Aut(D) by
            # order. (We don't compare gen lists element-wise — different
            # generating sets for the same group are both correct.)
            d_basis = list(d_rref)
            for g in d_info_recon.aut_generators:
                permuted_basis = [apply_permutation(b, list(g)) for b in d_basis]
                # The image must span the same subspace (i.e. its RREF
                # equals d_rref).
                assert Code(N, tuple(permuted_basis)).rref_basis()[0] == d_rref, (
                    f"reconstructed aut generator does not fix D at "
                    f"N={N}, k={k}"
                )
            # Subgroup-equality: <recon_gens> alone must have order |Aut(D)|.
            # (The combined-with-oracle version is vacuous because the oracle
            # already generates Aut(D); we need to test the reconstruction
            # is sufficient on its own.)
            assert (
                group_order(d_info_recon.aut_generators, N)
                == d_info_oracle.aut_order
            ), (
                f"reconstructed gens generate a proper subgroup of Aut(D) "
                f"at N={N}, k={k}: "
                f"order={group_order(d_info_recon.aut_generators, N)}, "
                f"want={d_info_oracle.aut_order}"
            )

            # column_orbits: partition equality (as set of frozensets).
            def _orbits_to_partition(orbits: tuple[int, ...]) -> frozenset[frozenset[int]]:
                buckets: dict[int, list[int]] = {}
                for i, oid in enumerate(orbits):
                    buckets.setdefault(oid, []).append(i)
                return frozenset(frozenset(v) for v in buckets.values())

            assert _orbits_to_partition(d_info_recon.column_orbits) == _orbits_to_partition(
                d_info_oracle.column_orbits
            ), (
                f"column_orbits partition mismatch at N={N}, k={k}: "
                f"recon={d_info_recon.column_orbits}, "
                f"oracle={d_info_oracle.column_orbits}"
            )
