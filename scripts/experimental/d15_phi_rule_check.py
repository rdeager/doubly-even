"""D15 Phase 0: prototype the coset-spectrum parent rule on the clean/ enumerator.

The rule under test (internal label D15, public name "coset-spectrum parent
rule"): the canonical parent of a rank-(k+1) code D is selected among D's
2^(k+1)-1 hyperplanes (index-2 subcodes H_u, u a nonzero functional on D) by
lex-minimising the complement-coset weight spectrum

    phi(u) = (phi_4(u), phi_8(u), phi_12(u), ...),
    phi_w(u) = #{x in D : wt(x) = w, u(x) = 1},

over all weight strata of D ascending. If the argmin is a single functional
it names the parent orbit outright; otherwise the tie is broken exactly like
the legacy rule, restricted to the argmin set: the Aut(D)-orbit of the argmin
hyperplane whose sigma_D-permuted RREF is lex-least.

McKay soundness: phi is codeword-weight data only, so its argmin set is
Aut(D)-invariant and transported by any isomorphism; the sigma-key tie-break
picks exactly one orbit (same well-definedness argument the legacy rule
relies on). Hence m(D) is an iso-invariant single-orbit parent function and
the enumeration stays isomorph-free.

Two validation modes per N:

  A. audit-on-legacy-tree - run the unchanged legacy traversal; per candidate
     evaluate BOTH tests; assert per rank that
     phi_accept_unique + phi_tie_accept == legacy accepts EXACTLY.
     (Both rules provably accept exactly one (parent-class, coset-orbit)
     pair per class, and the legacy tree tests every pair. This identity is
     the Phase 1 hard gate; we rehearse it here.)
  B. phi-driven traversal - enumerate with the phi rule end to end; per-rank
     class counts and sorted |Aut| multisets must match legacy, and
     Sum N!/|Aut| must equal gaborit_sigma(N, k) at every rank.

Also tallies phi outcomes (reject / accept-unique / tie) per parent rank.
NOTE: small-N tie rates are PESSIMISTIC - high-symmetry codes dominate at
small N. The decisive volume-weighted kappa measurement is the Phase 1 Rust
audit; this script only debugs the rule definition.

Usage:
    uv run python scripts/experimental/d15_phi_rule_check.py [N ...]
    (default: 8 10 12 13 14)
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict

from doubly_even.clean._augment import (
    EnumeratedCode,
    _subspace_in_orbit,
    is_canonical_augmentation,
)
from doubly_even.clean._canon import CanonInfo, canon_info
from doubly_even.clean._mass import gaborit_sigma
from doubly_even.clean._qc import qc_candidates
from doubly_even.clean._spec import Code, apply_perm, rref_gf2

# ---------------------------------------------------------------------------
# The phi cascade
# ---------------------------------------------------------------------------


def _frame_strata(c_rref_rows: list[int], v: int) -> tuple[list[int], dict[int, list[int]]]:
    """Codewords of D = <C, v> in the fixed frame basis [C's RREF rows, v].

    Returns (frame_rows, by_weight) where by_weight[w] is the list of
    COORDINATE VECTORS (ints over k+1 bits; bit j = coefficient of
    frame_rows[j]) of the weight-w codewords. The frame rows are linearly
    independent (v is a nonzero coset rep, so v not in C), hence coordinate
    vector 0 <-> codeword 0 only and every bucket holds weight >= 4.
    """
    rows = list(c_rref_rows) + [v]
    size = 1 << len(rows)
    words = [0] * size
    by_weight: dict[int, list[int]] = defaultdict(list)
    for i in range(1, size):
        low = (i & -i).bit_length() - 1
        w = words[i ^ (1 << low)] ^ rows[low]
        words[i] = w
        by_weight[w.bit_count()].append(i)
    return rows, by_weight


def phi_outcome(c_rref_rows: list[int], v: int) -> tuple[str, set[int] | None]:
    """Lazy lex cascade over weight strata.

    Returns ('reject', None) | ('accept_unique', None) | ('tie', M) with M
    the surviving argmin set of nonzero functionals (u = 0 is excluded from
    the start: phi(0) == 0 would always win and reject everything).
    """
    _, by_weight = _frame_strata(c_rref_rows, v)
    kp1 = len(c_rref_rows) + 1
    u_c = 1 << (kp1 - 1)
    m_set = set(range(1, 1 << kp1))
    for w in sorted(by_weight):
        stratum = by_weight[w]
        counts: dict[int, int] = {}
        best: int | None = None
        for u in m_set:
            c = 0
            for x in stratum:
                c += (u & x).bit_count() & 1
            counts[u] = c
            if best is None or c < best:
                best = c
        m_set = {u for u in m_set if counts[u] == best}
        if u_c not in m_set:
            return "reject", None
        if len(m_set) == 1:
            return "accept_unique", None
    return "tie", m_set


def _hyperplane_basis(frame_rows: list[int], u: int) -> list[int]:
    """Codeword basis of H_u = {x : u(x) = 0}, the kernel of functional u."""
    kp1 = len(frame_rows)
    j0 = (u & -u).bit_length() - 1
    basis: list[int] = []
    for j in range(kp1):
        if j == j0:
            continue
        coord = (1 << j) | ((((u >> j) & 1)) << j0)
        word = 0
        x = coord
        while x:
            jj = (x & -x).bit_length() - 1
            word ^= frame_rows[jj]
            x &= x - 1
        basis.append(word)
    return basis


def tie_break_accept(
    C: Code, D: Code, info_D: CanonInfo, m_set: set[int], frame_rows: list[int]
) -> bool:
    """Legacy-style tie-break restricted to the argmin set.

    m(D) = Aut(D)-orbit of the argmin hyperplane whose sigma_D-permuted RREF
    is lex-least; accept iff C lies in that orbit.
    """
    sigma = list(info_D.canonical_column_order)
    best_key: tuple[int, ...] | None = None
    best_basis: list[int] | None = None
    for u in sorted(m_set):
        h_basis = _hyperplane_basis(frame_rows, u)
        permuted = [apply_perm(b, sigma) for b in h_basis]
        key = tuple(rref_gf2(permuted, D.n)[0])
        if best_key is None or key < best_key:
            best_key = key
            best_basis = h_basis
    assert best_basis is not None
    target = Code(D.n, tuple(best_basis))
    return _subspace_in_orbit(C, target, info_D.aut_generators, D.n)


# ---------------------------------------------------------------------------
# Mode A: audit on the legacy tree (rehearses the Phase 1 hard gate)
# ---------------------------------------------------------------------------


def audit_on_legacy_tree(N: int, max_k: int) -> dict[int, Counter]:
    """Legacy traversal; both tests evaluated per candidate; per-rank tallies."""
    tallies: dict[int, Counter] = defaultdict(Counter)

    def walk(C: Code, info_C: CanonInfo) -> None:
        k = C.rank
        if k >= max_k:
            return
        c_rref_rows = list(C.rref()[0])
        for v in qc_candidates(C, info_C.aut_generators):
            D = C.extend(v)
            info_D = canon_info(D)
            legacy = is_canonical_augmentation(C, D, info_D)
            outcome, m_set = phi_outcome(c_rref_rows, v)
            if outcome == "tie":
                frame_rows = c_rref_rows + [v]
                if tie_break_accept(C, D, info_D, m_set, frame_rows):
                    outcome = "tie_accept"
                else:
                    outcome = "tie_reject"
            tallies[k][outcome] += 1
            tallies[k]["legacy_accept" if legacy else "legacy_reject"] += 1
            if legacy:
                walk(D, info_D)

    root = Code.zero(N)
    walk(root, canon_info(root))
    return tallies


# ---------------------------------------------------------------------------
# Mode B: phi-driven traversal
# ---------------------------------------------------------------------------


def traverse_phi(C: Code, max_k: int, info_C: CanonInfo | None = None):
    if info_C is None:
        info_C = canon_info(C)
    yield EnumeratedCode(code=C, info=info_C)
    if C.rank >= max_k:
        return
    c_rref_rows = list(C.rref()[0])
    for v in qc_candidates(C, info_C.aut_generators):
        outcome, m_set = phi_outcome(c_rref_rows, v)
        if outcome == "reject":
            continue
        D = C.extend(v)
        info_D = canon_info(D)
        if outcome == "tie":
            frame_rows = c_rref_rows + [v]
            if not tie_break_accept(C, D, info_D, m_set, frame_rows):
                continue
        yield from traverse_phi(D, max_k, info_D)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def per_rank_profile(enumerated) -> tuple[Counter, dict[int, list[int]]]:
    counts: Counter = Counter()
    auts: dict[int, list[int]] = defaultdict(list)
    for ec in enumerated:
        k = ec.code.rank
        counts[k] += 1
        auts[k].append(ec.aut_order)
    for k in auts:
        auts[k].sort()
    return counts, auts


def check_mass(N: int, counts: Counter, auts: dict[int, list[int]]) -> bool:
    fact = math.factorial(N)
    ok = True
    for k in sorted(counts):
        mass = sum(fact // a for a in auts[k])
        sigma = gaborit_sigma(N, k)
        if mass != sigma:
            print(f"    MASS MISMATCH at k={k}: {mass} != sigma={sigma}")
            ok = False
    return ok


def main(ns: list[int]) -> int:
    failures = 0
    for N in ns:
        max_k = N // 2
        print(f"\nN = {N}")

        legacy_counts, legacy_auts = per_rank_profile(_legacy_enum(N, max_k))
        phi_counts, phi_auts = per_rank_profile(traverse_phi(Code.zero(N), max_k))

        tallies = audit_on_legacy_tree(N, max_k)

        print("  k | legacy phi | phi-outcome on legacy tree (rej/acc_uniq/tie_acc/tie_rej) | gate")
        all_ranks = sorted(set(legacy_counts) | set(phi_counts) | set(tallies))
        for k in all_ranks:
            t = tallies.get(k, Counter())
            phi_acc = t["accept_unique"] + t["tie_accept"]
            gate_ok = phi_acc == t["legacy_accept"]
            counts_ok = legacy_counts.get(k, 0) == phi_counts.get(k, 0)
            auts_ok = legacy_auts.get(k, []) == phi_auts.get(k, [])
            status = "OK" if (gate_ok and counts_ok and auts_ok) else "FAIL"
            if status == "FAIL":
                failures += 1
            print(
                f"  {k:2d} | {legacy_counts.get(k, 0):5d} {phi_counts.get(k, 0):4d}"
                f" | {t['reject']:5d} {t['accept_unique']:5d} {t['tie_accept']:4d} {t['tie_reject']:4d}"
                f" | phi_acc={phi_acc} legacy_acc={t['legacy_accept']} {status}"
            )
        mass_ok = check_mass(N, phi_counts, phi_auts)
        if not mass_ok:
            failures += 1
        total_cand = sum(sum(t[o] for o in ("reject", "accept_unique", "tie_accept", "tie_reject"))
                         for t in tallies.values())
        total_tie = sum(t["tie_accept"] + t["tie_reject"] for t in tallies.values())
        print(f"  candidates={total_cand} ties={total_tie}"
              f" ({100.0 * total_tie / total_cand if total_cand else 0.0:.1f}% -"
              f" pessimistic at small N) mass={'OK' if mass_ok else 'FAIL'}")
    print(f"\n{'ALL OK' if failures == 0 else f'{failures} FAILURES'}")
    return 1 if failures else 0


def _legacy_enum(N: int, max_k: int):
    from doubly_even.clean._augment import enumerate_doubly_even_sequential

    return enumerate_doubly_even_sequential(N, max_k)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]] or [8, 10, 12, 13, 14]
    raise SystemExit(main(args))
