"""Fourier-domain enumeration of doubly-even [N, k] column multiplicity vectors.

Variables: j_u = w_u / 4 for u in F_2^k \\ {0}, equivalently n̂(u) = N - 8 j_u.

For each u order assignment, maintain

    partial[v] := sum over u processed of  n̂(u) · (-1)^{u·v}
                 (initialised to N for all v — the n̂(0) = N contribution)

At the leaf (all u != 0 processed):

    M(v)  =  partial[v]  /  2^k    must be integer >= 0 for all v in F_2^k

The full-rank-k constraint is post-checked (cheap).

Pruning: max future increment to partial[v] is + N per remaining level. So

    partial[v] + N · (remaining levels)   <   0      =>   dead branch
"""

from __future__ import annotations

import argparse
import time


def popcount(x: int) -> int:
    return bin(x).count("1")


def enumerate_de_fourier(N: int, k: int) -> list[tuple[int, ...]]:
    nv = 1 << k
    max_j = N // 4

    # DFS over u != 0 in reverse-Hamming-weight order (heavy first, locks
    # many per-v constraints early).
    u_order = sorted(range(1, nv), key=lambda u: (-popcount(u), u))
    n_lvls = nv - 1

    # sign[lvl][v] = (-1)^{u_order[lvl] · v} as +1 / -1
    sign = [
        [1 if popcount(u_order[lvl] & v) & 1 == 0 else -1 for v in range(nv)]
        for lvl in range(n_lvls)
    ]

    partial = [N] * nv  # n̂(0) = N contribution to every v
    results: list[tuple[int, ...]] = []

    def recurse(lvl: int) -> None:
        if lvl == n_lvls:
            # All n̂(u != 0) assigned. M(v) = partial[v] / 2^k must be integer >= 0.
            M = [0] * nv
            for v in range(nv):
                pv = partial[v]
                if pv < 0:
                    return
                q, r = divmod(pv, nv)
                if r:
                    return
                M[v] = q
            results.append(tuple(M))
            return

        remaining = n_lvls - lvl - 1  # remaining levels AFTER this one
        sign_lvl = sign[lvl]
        for j in range(max_j + 1):
            nhat = N - 8 * j  # in [-N, N]
            ok = True
            # Update partial[v] for each v
            for v in range(nv):
                partial[v] += nhat * sign_lvl[v]
                # Prune: max possible future contribution is +N per remaining level.
                if partial[v] + N * remaining < 0:
                    ok = False
            if ok:
                recurse(lvl + 1)
            for v in range(nv):
                partial[v] -= nhat * sign_lvl[v]

    recurse(0)
    return results


def spans_full(M: tuple[int, ...], k: int) -> bool:
    nv = 1 << k
    rows: list[int] = []
    rank = 0
    for v in range(nv):
        if M[v] == 0:
            continue
        x = v
        for r in rows:
            lb = r & -r
            if x & lb:
                x ^= r
        if x:
            rows.append(x)
            rank += 1
            if rank == k:
                return True
    return False


def gl_generators(k: int):
    if k <= 1:
        return []
    nv = 1 << k
    mask = nv - 1

    def transvect(v: int) -> int:
        if (v >> 1) & 1:
            return v ^ 1
        return v

    def cyclic(v: int) -> int:
        return ((v << 1) | (v >> (k - 1))) & mask

    def swap01(v: int) -> int:
        b0 = v & 1
        b1 = (v >> 1) & 1
        if b0 == b1:
            return v
        return v ^ 0b11

    return [transvect, cyclic, swap01]


def count_orbits(multisets: list[tuple[int, ...]], k: int) -> int:
    nv = 1 << k
    gens = gl_generators(k)
    mset = set(multisets)
    visited: set[tuple[int, ...]] = set()
    orbits = 0
    for M in multisets:
        if M in visited:
            continue
        orbits += 1
        stack = [M]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for g in gens:
                neigh = tuple(cur[g(v)] for v in range(nv))
                if neigh in mset and neigh not in visited:
                    stack.append(neigh)
    return orbits


def count_classes(N: int, k: int) -> tuple[int, int, int, float, float]:
    t0 = time.time()
    multisets = enumerate_de_fourier(N, k)
    t_enum = time.time() - t0
    full = [M for M in multisets if spans_full(M, k)]
    t0 = time.time()
    n_orbits = count_orbits(full, k)
    t_orbit = time.time() - t0
    return n_orbits, len(full), len(multisets), t_enum, t_orbit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=str, default="8,12,16")
    parser.add_argument("--k", type=str, default="1,2,3,4")
    args = parser.parse_args()
    Ns = [int(n) for n in args.N.split(",")]
    Ks = [int(k) for k in args.k.split(",")]
    print(
        f"\n{'(N, k)':>10} {'classes':>10} {'full_rk':>10} {'all_msets':>12} "
        f"{'enum(s)':>10} {'orbit(s)':>10}",
        flush=True,
    )
    for N in Ns:
        for k in Ks:
            if k > N:
                continue
            n_classes, n_full, n_all, t_e, t_o = count_classes(N, k)
            print(
                f"{(N, k)!s:>10} {n_classes:>10} {n_full:>10} {n_all:>12} "
                f"{t_e:>10.3f} {t_o:>10.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
