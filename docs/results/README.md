# Published results

Citable, machine-readable artefacts for the enumerations this package
has produced. The `N = 29` result is the final state of one
`scripts/run_streaming.py` run plus its `scripts/merge_stream.py`
post-pass; the `N = 30` result is from `scripts/run_counts.py`
(counts-only mode, the `N ≥ 30`-capable entry — no per-class binaries
are emitted).

## What's in this directory

| file | what it is |
|------|------------|
| [`n29.json`](n29.json) | The `N = 29` result: per-`k` class counts, per-`k` mass `Σ N!/|Aut(C)|`, per-`k` `gaborit_sigma(29, k)`, total class count (239,465,540), wall time (12.32 hr), platform, and the git SHA the kernel was built from. |
| [`n30.json`](n30.json) | The `N = 30` result: same schema, ranks `k = 0..14`, total class count (3,786,528,214), wall time (4.70 hr) on a 96-core `c4a-highmem-96-metal` (Axion) at `frontier_depth = 3` / `δ = 5`, and the git SHA. Counts-only run, so no companion binary stream exists. |

## Schema

`n29.json` mirrors the structure that `scripts/merge_stream.py` emits
when finalising a streaming run, with four scalar fields from the
companion `stats.json` (wall, threads, frontier depth, canon cache
cap) folded into a `run_metadata` block for self-containment.

The per-`k` block has three integer columns:

- `classes` — equivalence classes emitted at rank `k`.
- `mass` — running `Σ_{C ∈ rank-k classes} N! / |Aut(C)|`.
- `gaborit_sigma` — closed-form `σ(N, k)` from
  [`doubly_even.spec.mass.gaborit_sigma`](../../src/doubly_even/spec/mass.py).

Equality `mass == gaborit_sigma` at every `k` is the
**Gaborit mass-formula certificate** (DFGHILM B.1) — the load-bearing
correctness oracle for `N = 29`, since DFGHILM Table 3 publishes no
cells at this length.

## Auditing the certificate without rerunning the enumeration

Each row of `n29.json` is independently checkable. Total round-trip
audit on a laptop is one second:

```sh
uv run python -c '
import json
from doubly_even.spec.mass import gaborit_sigma
d = json.load(open("docs/results/n29.json"))
for k, row in d["per_k"].items():
    sigma = gaborit_sigma(29, int(k))
    assert int(row["mass"]) == int(row["gaborit_sigma"]) == sigma, (k, sigma)
total = sum(row["classes"] for row in d["per_k"].values())
assert total == d["total_classes"] == 239_465_540
print(f"OK — mass-formula certificate verified at every k=0..13; "
      f"total {total:,} classes")
'
```

This proves the JSON is **internally consistent and correctly
labelled against the closed-form `σ`**. It does not re-verify the
underlying enumeration; for that you would need the per-class
binaries (see below).

The same audit applies to `n30.json` — substitute `30`, the rank
range `k = 0..14`, and the total `3_786_528_214`. (`n30.json` is a
counts-only run, so there are no per-class binaries to fall back on;
the mass-formula certificate is the sole correctness oracle, as for
`N = 29`.)

## On request

The following are kept off-repository because of size:

- **N = 29 per-class binary tarball** — 5.2 GB compressed, ~21 GB
  uncompressed; one canonical generator basis + `|Aut(C)|` per
  emitted class, in `streaming.rs` binary layout. Sufficient to
  re-run `scripts/merge_stream.py` from scratch and reproduce
  `n29.json` byte-for-byte (modulo the `run_metadata` block, which
  is run-specific).
- **Per-`k` JSONL extracts** (e.g. `final_k12.jsonl`, `final_k13.jsonl`):
  one line per emitted class with the basis row vectors + Aut order,
  for the heaviest two ranks of `N = 29`. Useful if you want the
  canonical reps but not the full binary stream.
- **Raw kernel statistics** from the cloud `stats.json` — nauty
  internal counters (numnodes, maxlevel, generators per call),
  bucket-size histograms, BFS reject counts. The fields in
  `n29.json`'s `run_metadata.canon_calls` and `nauty_seconds` are
  derived from these.

Open a [GitHub issue](https://github.com/rdeager/doubly-even/issues)
or email the maintainer for any of these — the binaries are
addressable by run timestamp and N.

## Reproducing the result from scratch

End-to-end recipe (clean checkout → N = 29 result) is in
[`docs/reproducing.md`](../reproducing.md). Plan ~12 hr of c4a-72
wall time and ~$35 of GCP on-demand compute for an exact rerun, or
substitute any 72+ physical core ARM box with ≥ 200 GB RAM. See the
[cloud cost ladder](../cluster-deployment.md#hardware-cost-ladder-for-n--29-forward-looking)
for other platforms.
