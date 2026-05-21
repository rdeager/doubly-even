"""Gaborit's closed-form ``σ(N, k)`` — # of labelled doubly-even ``[N, k]`` codes.

Used by verification scripts to check ``Σ N!/|Aut(C_i)| == σ(N, k)``. NOT used
inside the recursion as a mass-stop (4–11 % perf lever, intentionally out of
scope — see production kernel). Copied verbatim from ``doubly_even/spec/mass.py``.
"""

from __future__ import annotations
from fractions import Fraction
from functools import lru_cache


def _omega(m: int, t: int, eps: int) -> int:
    if t < 0 or t > m:
        return 0
    if t == 0:
        return 1
    r = Fraction(1)
    for i in range(t):
        numer = (1 << (2 * m - 2 * i - 1)) + eps * (1 << (m - i - 1)) - 1
        denom = (1 << (i + 1)) - 1
        r *= Fraction(numer, denom)
    if r.denominator != 1:
        raise RuntimeError(f"Ω did not simplify to integer: {r}")
    return int(r)


@lru_cache(maxsize=None)
def gaborit_sigma(N: int, k: int) -> int:
    if k < 0 or N < 0:
        raise ValueError(f"need N, k ≥ 0, got N={N}, k={k}")
    if k == 0:
        return 1
    if 2 * k > N:
        return 0
    r = N % 8
    if r in (1, 7):
        return _omega((N - 1) // 2, k, +1)
    if r in (3, 5):
        return _omega((N - 1) // 2, k, -1)
    if r in (2, 6):
        result = Fraction(1)
        for i in range(k):
            result *= Fraction((1 << (N - 2 * i - 2)) - 1, (1 << (i + 1)) - 1)
        if result.denominator != 1:
            raise RuntimeError(f"σ did not simplify to integer: {result}")
        return int(result)
    m = (N - 2) // 2
    eps = +1 if r == 0 else -1
    return _omega(m, k - 1, eps) + (1 << k) * _omega(m, k, eps)
