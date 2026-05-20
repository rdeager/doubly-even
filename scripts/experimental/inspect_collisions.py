"""Inspect t11_stacked collision pairs at N=20, 22 — identify by Aut and WE."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from doubly_even.enumerate.augment import enumerate_doubly_even

from pair_gram_experiment import (
    codeword_list,
    column_bitmasks_subset,
    pair_gram_per_col_sig,
    stacked_span_aware,
)

from pair_gram_class_audit import (
    codewords_by_weight,
    t11_per_col_tuple,
)


def t11_stacked_sig(code, N: int):
    cws = codeword_list(code)
    masks = column_bitmasks_subset(cws, N)
    cwbw = codewords_by_weight(cws)
    stk = stacked_span_aware(cwbw, code.rank, N)
    if stk is None:
        sig_stacked = ("BAILED",)
    else:
        masks_stk = column_bitmasks_subset(stk, N)
        sig_stacked = pair_gram_per_col_sig(masks_stk)
    return (t11_per_col_tuple(cws, N), sig_stacked)


def weight_enum(cws: list[int]) -> dict[int, int]:
    out: dict[int, int] = defaultdict(int)
    for c in cws:
        out[c.bit_count()] += 1
    return dict(out)


def main():
    for N in [16, 18, 20, 22]:
        print(f"=== N={N} ===")
        sigs_to_codes: dict[tuple, list] = defaultdict(list)
        for idx, ec in enumerate(enumerate_doubly_even(N)):
            sig = t11_stacked_sig(ec.code, N)
            sigs_to_codes[sig].append((idx, ec))
        n_buckets = len(sigs_to_codes)
        coll_buckets = [v for v in sigs_to_codes.values() if len(v) > 1]
        print(f"  total classes: {sum(len(v) for v in sigs_to_codes.values())}")
        print(f"  distinct t11_stacked sigs: {n_buckets}")
        print(f"  collision buckets: {len(coll_buckets)}")
        for bi, bucket in enumerate(coll_buckets):
            print(f"  Bucket {bi}: size={len(bucket)}")
            for idx, ec in bucket:
                cws = codeword_list(ec.code)
                we = weight_enum(cws)
                we_str = ",".join(f"{w}:{c}" for w, c in sorted(we.items()))
                print(
                    f"    idx={idx} k={ec.code.rank} |Aut|={ec.aut_order}\n"
                    f"      basis={list(ec.code.basis)}\n"
                    f"      WE={we_str}"
                )


if __name__ == "__main__":
    main()
