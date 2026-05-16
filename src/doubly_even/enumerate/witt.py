"""Witt-type closed-form counts for the quotient quadratic form.

For a doubly even code ``C`` of length ``N`` and rank ``k``, the
quotient ``Q_C := C⊥/C`` carries a quadratic form ``q(v + C) := wt(v)/2
mod 2`` defined on cosets with even-weight representatives. The number
of nonzero singular vectors of a non-degenerate quadratic space
``(V, q)`` over ``F_2`` has a known closed form, indexed by Witt type:

* ``Ω_+`` at dim ``2m``: ``(2^m − 1)(2^{m−1} + 1)``.
* ``Ω_-`` at dim ``2m``: ``(2^m + 1)(2^{m−1} − 1)``.
* Parabolic at dim ``2m + 1``: ``2^{2m} − 1`` (every odd-dim non-
  degenerate form has the same count).

Phase (b)'s structural win is in orbit enumeration, not in singular-
vector enumeration; so this module's :func:`singular_vectors` is an
alias for :func:`doubly_even.enumerate.quotient.singular_reps_Q` and
:func:`count_singular` is here purely as informational closed-form
documentation.

The form ``(Q_C, q)`` is **not** always non-degenerate: when ``1...1
∉ C`` the inner product on ``C⊥`` has a non-trivial radical (spanned by
``1...1``), and the closed forms above describe a non-degenerate
reduction modulo that radical rather than ``(Q_C, q)`` itself. A full
Witt classifier handling the radical is parked as future work — the
orbit-enumeration code does not need it.
"""

from __future__ import annotations

from .quotient import singular_reps_Q


def count_singular(m: int, eps: str) -> int:
    """Closed-form nonzero singular count for a non-degenerate Witt type.

    Returns 0 when ``m <= 0``. Recognised signs:

    * ``"+"`` — ``Ω_+`` (hyperbolic), dim ``2m``.
    * ``"-"`` — ``Ω_-``, dim ``2m``.
    * ``"0"`` — parabolic, dim ``2m + 1``.
    """
    if m <= 0:
        return 0
    if eps == "+":
        return (2**m - 1) * (2**(m - 1) + 1)
    if eps == "-":
        return (2**m + 1) * (2**(m - 1) - 1)
    if eps == "0":
        return (1 << (2 * m)) - 1
    raise ValueError(f"unknown Witt sign {eps!r}; expected '+', '-', or '0'")


def singular_vectors(V_basis: tuple[int, ...]) -> list[int]:
    """Nonzero singular Q-coords of ``(Q_C, q)`` for a doubly even parent.

    Thin alias for :func:`doubly_even.enumerate.quotient.singular_reps_Q`.
    The combinatorial enumerator implied by the Witt decomposition is a
    future optimisation; phase (b)'s structural win is in orbit
    enumeration, not here.
    """
    return singular_reps_Q(V_basis)
