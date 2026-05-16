# doubly-even

Enumerate doubly even binary linear codes `[N, k]` up to permutation
equivalence.

A binary linear code is **doubly even** if every codeword's Hamming weight is
divisible by 4. The classification of such codes underlies the Adinkra
chromotopology problem in supersymmetric representation theory.

## What this package does

Implements the McKay 1998 canonical augmentation algorithm specialised for
doubly even codes, following Appendix B of *Doran–Faux–Gates–Hübsch–Iga–
Landweber–Miller* (DFGHILM). The enumerator emits one canonical
representative per equivalence class plus the order of its automorphism
group, and self-verifies via Gaborit's mass formula
`Σ N!/|Aut(C_i)| = σ(N, k)`.

Layering (see `markdown/architecture/01-layering.md`):

- `doubly_even.spec` — executable specification (math, slow, readable).
- `doubly_even.canon` — wrapper around an external canonical labeller (nauty).
- `doubly_even.enumerate` — the search loop and pre-canonical filters.
- `doubly_even.cli` — `dec enumerate --N --k` entry point.

## Status

Phase 0 (project scaffolding) complete. Phase 1 (`spec/` module) up next.

## Project documentation

Living outside this repository at [`/workspace/markdown/`](../markdown/):

- `algorithm/` — what the enumerator does, in math.
- `architecture/` — engineering decisions.
- `notes/` — paper summaries and the original architectural framing.
- `references/` — bibliography + paperctl request log.

## Usage (planned)

```sh
uv sync --dev
uv run pytest
uv run dec enumerate --N 16 --k 4 --output codes.jsonl
```

`dec` is not implemented yet; the CLI shim is part of Phase 4.

## License

TBD.
