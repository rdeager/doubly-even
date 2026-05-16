"""Long-lived Sage subprocess serving canon_info requests for end-to-end bench.

Protocol (one JSON line per request, one JSON line per response):

  Request:  {"req":"canon", "rref":[int...], "n":int}
  Response: {"col_order":[int...], "aut_gens":[[int...], ...],
             "aut_order":int, "column_orbits":[int...]}

  Request:  {"req":"quit"}
  Response: {"ok": true}

Uses sage.groups.perm_gps.partn_ref.refinement_binary.LinearBinaryCodeStruct
(the same Cython class Sage's permutation_automorphism_group('partition')
uses internally) — the fastest Sage canonicaliser for binary codes.

This is bench-only. The IPC overhead per request is ~50-100 µs, which we
report alongside the inner kernel cost.
"""
import json
import sys
import time

from sage.all import GF, Matrix
from sage.groups.perm_gps.partn_ref.refinement_binary import LinearBinaryCodeStruct


def rref_to_matrix(rref, n):
    rows = [[(word >> i) & 1 for i in range(n)] for word in rref]
    return Matrix(GF(2), rows)


def column_orbits_from_gens(gens, n):
    """Compute orbit identifiers under <gens> via union-find."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for g in gens:
        for i in range(n):
            j = g[i]
            if j != i:
                union(i, j)

    # Re-canonicalise so labels are 0..numorbits-1 in order of appearance
    seen: dict = {}
    out = []
    for i in range(n):
        r = find(i)
        if r not in seen:
            seen[r] = len(seen)
        out.append(seen[r])
    return out


def handle_canon(req):
    rref = req["rref"]
    n = req["n"]
    if not rref:
        # Zero code at length n: every column permutation fixes it,
        # so Aut = S_n and |Aut| = n!. Sage's LinearBinaryCodeStruct
        # doesn't accept zero-row matrices, so we handle this analytically.
        import math
        nfac = math.factorial(n)
        # Generators of S_n: a transposition + an n-cycle (Coxeter).
        if n == 0:
            gens, col_orbits = [], []
        elif n == 1:
            gens, col_orbits = [], [0]
        else:
            swap = list(range(n)); swap[0], swap[1] = 1, 0
            cycle = list(range(1, n)) + [0]
            gens = [swap, cycle]
            col_orbits = [0] * n
        return {
            "col_order": list(range(n)),
            "aut_gens": gens,
            "aut_order": nfac,
            "column_orbits": col_orbits,
        }
    G = rref_to_matrix(rref, n)
    S = LinearBinaryCodeStruct(G)
    gens, order, _base = S.automorphism_group()
    relabel = S.canonical_relabeling()  # 0-indexed permutation list

    # Sage's gens are 0-indexed permutations as plain Python lists.
    # canonical_relabeling() returns a list mapping old_index -> new_index.
    # Our convention matches: π[i] = j means column i becomes j.
    col_orbits = column_orbits_from_gens(gens, n)

    return {
        "col_order": list(relabel),
        "aut_gens": [list(g) for g in gens],
        "aut_order": int(order),
        "column_orbits": col_orbits,
    }


def main():
    # Warmup: trigger module-level Cython lazy imports
    G = Matrix(GF(2), [[1, 0, 1, 0], [0, 1, 1, 0]])
    LinearBinaryCodeStruct(G).automorphism_group()

    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        req = json.loads(line)
        cmd = req.get("req")
        if cmd == "quit":
            sys.stdout.write(json.dumps({"ok": True}) + "\n")
            sys.stdout.flush()
            break
        elif cmd == "canon":
            resp = handle_canon(req)
        else:
            resp = {"error": f"unknown req {cmd}"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


main()
