"""Witt-orbit phase-(b) scaffolding for the Q_C-coordinate pipeline.

EXPERIMENTAL / dormant. The active dispatcher in
:func:`doubly_even.enumerate.quotient.doubly_even_candidates_Q` runs
phase (a) — the σ_Q lookup-table orbit-min path — at every measured
``N``. Phase (b) (this module) is kept for pedagogical reference and
in case a Rust port of the Schreier-Sims-on-``GL(L, F_2)`` machinery
flips the per-step cost ratio enough to re-enable the structural win.

See ``architecture/04-optimisations.md`` §D7 for the empirical
breakdown, and the memory bullet ``phase_b_empirical_finding.md`` for
the parametric reasoning.
"""

from __future__ import annotations

from collections.abc import Iterable

from ...canon._linalg_f2 import Mat, mat_identity


def aut_orbit_minima_Q_witt(
    sigma_Qs: list[Mat],
    singular_set: Iterable[int],
    L: int,
) -> list[int]:
    """One Q-rep per ``⟨sigma_Qs⟩``-orbit, applying σ_Q directly.

    Drop-in alternative to
    :func:`doubly_even.enumerate.quotient.aut_orbit_minima_Q`
    (Milestone 4 phase (b)): same global orbit-decomposition BFS, but
    skips the ``2^L``-per-generator ``_sigma_Q_table`` build. Instead
    applies each generator via :func:`mat_apply` — one bit-walk of
    length ``popcount(current)`` per BFS step.

    The σ_Q table build dominates the post-(a) profile (~59% wall at
    ``N = 18`` per the architecture doc D6); for the L = 10..14 regime
    that matters at ``N = 18..22``, the per-step bit-walk is *in
    principle* cheaper than the amortised table build. In pure-Python
    practice the table-based pipeline (a) wins at every ``L`` — phase
    (b) is dormant scaffolding for a possible native-code session.

    Precondition for correctness of the orbit-min decomposition:
    ``singular_set`` is closed under ``⟨sigma_Qs⟩`` (so each orbit is
    fully inside it). The caller ensures this by passing the
    ``wt mod 4 = 0`` Q-coords, which ``Aut(C)`` preserves on doubly
    even cosets.

    With an empty generator set every element is its own orbit — return
    a sorted list of the input.
    """
    if not sigma_Qs:
        return sorted(singular_set)
    identity = mat_identity(L)
    gens = [g for g in sigma_Qs if g != identity]
    if not gens:
        return sorted(singular_set)
    reps_sorted = sorted(singular_set)
    seen: set[int] = set()
    minima: list[int] = []
    for v in reps_sorted:
        if v in seen:
            continue
        minima.append(v)
        seen.add(v)
        queue: list[int] = [v]
        while queue:
            next_queue: list[int] = []
            for current in queue:
                for g in gens:
                    # Inlined mat_apply: walk set bits of `current`.
                    out = 0
                    u = current
                    while u:
                        lsb = u & -u
                        out ^= g[lsb.bit_length() - 1]
                        u ^= lsb
                    if out not in seen:
                        seen.add(out)
                        next_queue.append(out)
            queue = next_queue
    return minima


def use_witt_path(sigma_Qs: list[Mat], L: int) -> bool:
    """Cheap predicate that would dispatch between phase (a) and (b).

    Hard-wired to ``False`` because phase (a) wins at every measured
    ``L`` in pure Python (see module docstring). Kept here as a single
    decision point a future native-code port could re-tune.
    """
    del sigma_Qs, L
    return False
