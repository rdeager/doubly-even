# Cluster deployment notes

> **Untested at scale.** The kernel has been validated on a single
> machine with up to 72 physical cores in one NUMA domain
> (`c4a-standard-72`, [N = 28 in 61 min](performance.md#the-n--28-cloud-run-first-reproducible-result)).
> This document describes what would have to change for multi-node or
> >256-core deployment, derived from the kernel's actual shape — not
> from a working cluster implementation.
>
> If you splice this and it works, please file an issue or PR; we'd
> love to validate against your numbers.

## What's already cluster-ready

The kernel's parallelism is structurally embarrassing above the
seeder frontier depth. At `DOUBLY_EVEN_FRONTIER_DEPTH = 5`, the
`N = 26` run produces ~7,000 seeds at depth 5; for `N = 28` it's
on the order of 30,000–50,000. Each seed defines a disjoint subtree
that never communicates with any other seed during enumeration — no
inter-seed reads, no inter-seed writes. The only cross-seed state is:

1. **The per-rank mass accumulator** (`Σ N!/|Aut(C_i)|`), checked
   against `σ(N, k+1)` to allow subtree-skipping when the rank-`k+1`
   quota is full. A single `Arc<Mutex<Vec<u128>>>` in the in-process
   kernel; contention measured at < 1 % wall on a 72-thread c4a-72
   run.
2. **The final per-`k` count tally**, sent back to the driver after
   all seeds finish.

Neither of these requires fine-grained communication. The kernel is
already producer-consumer at the seeder/worker boundary (see
[`algorithm.md` §4](algorithm.md#4-outer-dfs-worker-parallelism-with-pipelined-seeder)),
and a cluster deployment is the natural extension: the seeder
generates work for *remote* workers instead of local threads.

## What ships today

These pieces are in tree, work today, and would be reused unchanged
or near-unchanged by a cluster deployment:

| piece                              | file                                          | what it does |
|------------------------------------|-----------------------------------------------|--------------|
| Streaming output                   | `rust/src/streaming.rs`, `scripts/run_streaming.py` | per-worker binary files written to a local directory; no in-memory `Vec` of the result set. Avoids the OOM that kills any in-memory N ≥ 29 attempt. |
| Sidecar progress reader            | `scripts/stream_progress.py`                  | tails the per-worker `.bin` files at a configurable interval, prints per-`k` progress + mass-vs-σ table. Works against any local directory — equally useful for a 30-minute local N=24 run as for a multi-hour cloud N=28 run. |
| Post-run aggregator + cross-check  | `scripts/merge_stream.py`                     | re-walks the per-worker files, accumulates per-`k` mass, cross-checks DFGHILM Table 3, writes a single `stats.json` |
| In-Rust mass-formula gate          | `rust/src/enumerate.rs` (`enumerate_doubly_even_streaming`) | asserts `Σ N!/|Aut| == σ(N, k)` at completion; mismatch is a fatal panic |
| Parallel seeder + worker pool      | `rust/src/enumerate.rs::enumerate_doubly_even_parallel` | bounded-channel producer-consumer between the depth-`d` seeder and `t` workers |

The streaming output path is the one critical refactor — without it,
the Python list materialisation alone peaks at ~2 KB × N classes,
which is ~50 GB at `N = 28` and ~300 GB at `N = 29`. It is shipped on
`main` as `scripts/run_streaming.py` + the `enumerate_doubly_even_streaming`
kernel entry point.

## What would have to change for a cluster

The cluster boundary is the depth-`d` frontier. Each seed at depth `d`
becomes a unit of work; a coordinator hands it to a remote worker;
each worker streams its results to its node-local disk; the
coordinator collects per-`k` mass at the end.

Three concrete files would need work, listed in order of difficulty:

### 1. `rust/src/enumerate.rs::enumerate_doubly_even_parallel` — split the seeder

Today the seeder runs on the main thread and pushes into a
`crossbeam_channel`. A cluster coordinator would:

- Run the seeder unchanged (it's pure CPU; ~30 K seeds for N=28).
- Replace the channel with a *remote queue* — the simplest target is
  shared filesystem (NFS/GCS Fuse) writing one `seed-<n>.bin` file
  per seed, with workers claiming via atomic-rename. Redis or NATS
  would also work.
- Drop the per-worker spawn-locally code; instead the coordinator
  binary just produces seeds and exits. Workers are independent
  processes started by the cluster orchestrator.

The seed file format already exists (the `BinaryWriter` in
`rust/src/streaming.rs` knows how to dump a `Code` + its current
DFS state). Adding a "dump seed" entrypoint that emits the
augmentation state at depth `d` is small (~50 LOC).

### 2. `GlobalMassTracker` — choose: cross-node reconciliation vs deferred

The shared mass tracker (`Arc<Mutex<Vec<u128>>>`) doesn't cross
machine boundaries. Two paths, ranked simplest-first:

- **Deferred mass-stop (v0, recommended).** Each worker enumerates
  its assigned seeds *fully*; the coordinator aggregates per-`k` mass
  at the end and asserts `Σ == σ(N, k)`. Loses the 4–11 % wall
  saving from per-rank early termination (see
  [`algorithm.md` §5](algorithm.md#5-mass-formula-early-stop)), but
  simple. A worker can still apply its own *local* mass-stop within
  the seed (the kernel does this already), so the saving is only lost
  across seed boundaries.
- **Periodic reconciliation (v1).** Workers checkpoint per-`k` mass
  to a shared file every M seconds; each worker's per-rank quota
  check consults the latest aggregated mass. Recovers the 4–11 %
  but requires consistent-enough updates. Likely worth it only at
  N ≥ 30 where per-rank mass-stop deltas multiply across many seeds.

### 3. `scripts/merge_stream.py` — promote to a per-node + cross-node aggregator

Today `merge_stream.py` walks one node's per-worker `.bin` files and
emits a `stats.json`. A cluster deployment would run it once per node
(producing a per-node `stats.json`) and then run a second pass that
sums per-node mass and cross-checks the DFGHILM cells. The current
script already accumulates per-`k` and per-class; promoting it to take
a directory of per-node JSON files is small (~100 LOC).

## Coordinator protocol sketch

This is not implemented. Listed for someone who is.

- **Work item:** a depth-`d` seed `(code, k, parent_canon_state)`.
  Emitted as one file (`seed-<n>.bin`) in a shared directory.
- **Claim semantics:** worker atomic-renames `seed-<n>.bin` →
  `seed-<n>.bin.claimed-<host>-<pid>`. Linux's rename-into-existing
  is atomic on the same FS, so no double-claim. Filesystem-level
  primitives suffice; Redis/NATS only buy faster polling.
- **Idempotent emission:** the kernel is deterministic per seed (the
  same seed always produces the same canonical class set). A retry
  after worker death is safe; the post-merge dedup pass on per-node
  `.bin` files handles the rare double-completion.
- **Per-node output:** each worker writes `out.w<wid>.bin` to its
  *local* NVMe. After all seeds on a node complete, the node-local
  `merge_stream.py` produces a per-node `stats.json` and ships it
  (plus the binaries, if you want to materialise classes) to the
  coordinator.
- **Final aggregation:** the coordinator sums per-node `stats.json`
  and asserts `Σ N!/|Aut| == σ(N, k)` for each `k`. Mismatch is a
  bug, not a partial result.

## NUMA on a single big box

The `GlobalMassTracker` mutex is NUMA-oblivious. On a single-NUMA
72-core c4a-72, contention was negligible (< 1 % wall — see the
[N=28 run notes](../README.md#the-n--28-cloud-run-first-reproducible-result)).
On a dual-socket EPYC 9754 or 4-socket Granite Rapids host this
becomes a cross-socket cache-line contention point; worker checks
of `is_full(k+1)` would bounce the cache line every few microseconds.

Two mitigations, neither implemented:

- **Per-socket sharded mass tracker.** Each socket has its own
  `Arc<Mutex<Vec<u128>>>`; a low-frequency "reconcile" thread
  periodically sums them. Same model as the cross-node protocol
  above, but with `numactl --membind` placing the shard per socket.
- **`pin_workers`.** Pin each worker thread to a specific logical
  core (Rust's `core_affinity` crate); pair shards with cores. We
  measured this on the 13700K hybrid topology and the win was
  ~3 % — below the engineering threshold; it was not shipped. On
  symmetric NUMA hardware the win is likely larger.

## Hardware cost ladder for `N = 29`

The `c4a-standard-72` row is now measured (2026-05-23). Other rows
remain forecasts based on the measured `N = 22 → 29` per-call cost +
class-count trends. **Per-call cost continues to drop with `N`**
(see [`performance.md`](performance.md#the-scaling-story-per-call-cost-drops-calls-per-class-explodes))
— at `N = 29` it landed at 29.5 µs/call. The dominant scaling
factor is *calls per emitted class*: 249 at `N = 28`, 364 at `N = 29`,
growing ~2.6× per 2-step.

| platform                                    | phys cores | RAM     | $/hr (≈) | N=29 wall                  | N=29 cost            |
|---------------------------------------------|-----------:|--------:|---------:|----------------------------|----------------------|
| GCP `c4a-standard-72` (Axion ARM)           |         72 |  288 GB |   $2.81  | **12.3 hr (measured)**     | **~$35 (measured)**  |
| AWS `c8g.metal-24xl` (Graviton4)            |         96 |  192 GB |   $3.83  |       ~9 hr (forecast)     |     ~$35 (forecast)  |
| GCP `c4a-highmem-96-metal` (Preview)        |         96 |  768 GB |   ~$4    |       ~9 hr (forecast)     |     ~$36 (forecast)  |
| GCP `c4-standard-192` (Granite Rapids)      |   96 + SMT |  720 GB |   ~$8    |       ~8 hr (forecast)     |     ~$64 (forecast)  |
| GCP `c4-standard-288-metal`                 |  192 + SMT | 2.16 TB |  $14.23  |       ~5.5 hr (forecast)   |     ~$78 (forecast)  |
| AWS `r8g.metal-48xl` (Graviton4)            |        192 |  1.5 TB |   ~$11   |       ~5.5 hr (forecast)   |     ~$60 (forecast)  |

The measured 12.3 hr on `c4a-standard-72` is heavier than the
pre-run forecast (~7.5 hr) — the slack came from two effects we did
not fully account for: (a) the `DOUBLY_EVEN_CANON_CACHE_CAP` had to
drop from 500 K to 200 K to fit the 288 GB ceiling at 72 workers,
which likely costs ~5–10 % wall to canon-cache thrash, and (b) the
calls-per-class growth from `N = 26` (91) to `N = 29` (364) was
faster than the 2026-05-21 forecast accounted for. The other rows
in the table have been revised proportionally; treat them as
order-of-magnitude estimates, not promises.

The streaming output path is **required** on every option — N=29's
per-class output × class count would exceed 192 GB on the
smaller-RAM configurations. The cost-cheapest path remains
`c4a-standard-72` (the platform that delivered both our N=28 and
N=29 results); the wall-cheapest is `c4-standard-288-metal` or AWS
`r8g.metal-48xl`.

`N = 30` is past single-VM viability for output (estimated 2–6 TB
written). Two paths to consider, neither implemented:

- **Single big VM with streamed output to fast NVMe.**
  `c4-standard-288-metal` + ~3 TB pd-balanced + ~50–200 hr wall.
  ~$5,000 at on-demand pricing, plus disk.
- **Distributed cluster.** ~10 `c4a-standard-72` nodes claiming seeds
  from a shared coordinator. ~5–20 hr wall depending on seed balance,
  ~$300–700 total. **Significantly cheaper but needs the coordinator
  protocol above.**

`N = 32` is beyond canonical-augmentation enumeration alone — at the
projected ~3–5 T canon calls it is past single-machine viability at
any reasonable budget. The route forward is either a column-
multiset / Fourier-domain engine targeting specifically the small-`k`
slice (a Rust port of the Python prototype in
`scripts/experimental/multiset_*.py`), or a GPU canonicaliser
(PEACE-style, multi-week effort, uncertain payoff at our graph
shape).

## Streaming output is not just for cloud

`scripts/run_streaming.py` and `scripts/stream_progress.py` work
unchanged against any local directory. For long local runs the same
recipe applies:

```sh
# Local long-running N=26 (5 min on a 13700K, 30 min on a smaller box):
DOUBLY_EVEN_THREADS=24 DOUBLY_EVEN_FRONTIER_DEPTH=5 \
    DOUBLY_EVEN_CANON_CACHE_CAP=500000 \
    uv run python scripts/run_streaming.py --N 26 \
    --output-dir /tmp/n26-local

# In another terminal — same script as cloud, just point at the local dir:
uv run python scripts/stream_progress.py --N 26 \
    --output-dir /tmp/n26-local --interval 30
```

The sidecar prints a per-`k` mass-vs-σ table and exits when the kernel
finishes; it is roughly as useful for a 30-minute local enumeration
as for a 4-hour cloud run.
