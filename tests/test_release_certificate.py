"""Release-artifact certificate checks for the published ``N = 29`` and
``N = 30`` enumerations.

These results are the package's headline scientific claim, and at these
lengths there is **no external table** to cross-check against (DFGHILM
Table 3 stops at ``N = 28``). The load-bearing oracle is therefore
Gaborit's mass formula: at every rank ``k`` the sum ``Σ N!/|Aut(C)|``
over the emitted classes must equal the closed-form labelled count
``σ(N, k)``.

Re-running the enumeration to re-derive those sums costs hours of cloud
compute. This module instead audits the *checked-in certificates*
(``docs/results/n29.json``, ``docs/results/n30.json``) in milliseconds:

1. the schema is well-formed,
2. ``total_classes`` equals the sum of the per-rank class counts,
3. each stored per-rank ``mass`` equals the stored ``gaborit_sigma``, and
4. that stored ``gaborit_sigma`` is **independently recomputed** here from
   :func:`doubly_even.spec.mass.gaborit_sigma` — so the artifact is not
   merely internally consistent but agrees with a fresh evaluation of the
   closed form.

This needs neither the Rust kernel nor an enumeration run, so the
advertised result is verifiable from a clean checkout (``uv run pytest``)
even on a laptop. Step 4 is the auditable core: it would catch a
transcription error or a tampered certificate, not just a self-consistent
one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doubly_even.spec.mass import gaborit_sigma

RESULTS_DIR = Path(__file__).resolve().parents[1] / "docs" / "results"

# (N, total_classes) — the published headline counts. Pinned here so a
# silently corrupted or truncated certificate fails loudly rather than
# being trusted from its own (possibly wrong) ``total_classes`` field.
PUBLISHED = {
    29: 239_465_540,
    30: 3_786_528_214,
}


def _load(n: int) -> dict:
    path = RESULTS_DIR / f"n{n}.json"
    assert path.is_file(), f"missing release certificate: {path}"
    return json.loads(path.read_text())


@pytest.mark.parametrize("n", sorted(PUBLISHED))
def test_certificate_schema(n: int) -> None:
    """Top-level and per-rank schema is well-formed and self-describing."""
    doc = _load(n)
    for key in ("N", "total_classes", "mass_formula_ok", "per_k", "run_metadata"):
        assert key in doc, f"n{n}.json missing top-level key {key!r}"
    assert doc["N"] == n
    assert doc["mass_formula_ok"] is True

    per_k = doc["per_k"]
    assert per_k, "per_k is empty"
    ranks = sorted(int(k) for k in per_k)
    # Ranks must be contiguous from 0 (no gaps in the rank sweep).
    assert ranks == list(range(len(ranks))), f"non-contiguous ranks: {ranks}"
    for k, cell in per_k.items():
        for field in ("classes", "mass", "gaborit_sigma"):
            assert field in cell, f"n{n}.json k={k} missing {field!r}"
            assert isinstance(cell[field], int), f"n{n}.json k={k} {field} not int"

    meta = doc["run_metadata"]
    assert meta.get("git_sha"), "run_metadata.git_sha missing"


@pytest.mark.parametrize("n", sorted(PUBLISHED))
def test_certificate_total_matches_published(n: int) -> None:
    """total_classes equals both the published headline and the per-rank sum."""
    doc = _load(n)
    per_rank_sum = sum(cell["classes"] for cell in doc["per_k"].values())
    assert doc["total_classes"] == PUBLISHED[n], (
        f"n{n}.json total_classes={doc['total_classes']} "
        f"!= published {PUBLISHED[n]}"
    )
    assert per_rank_sum == PUBLISHED[n], (
        f"n{n}.json per-rank class sum {per_rank_sum} != total {PUBLISHED[n]}"
    )


@pytest.mark.parametrize("n", sorted(PUBLISHED))
def test_certificate_mass_formula(n: int) -> None:
    """The Gaborit mass-formula certificate, re-derived from the closed form.

    For every rank the stored ``mass`` must equal the stored
    ``gaborit_sigma``, and that value must equal a fresh evaluation of
    :func:`gaborit_sigma` here. This is what makes the published result
    auditable without rerunning the multi-hour enumeration.
    """
    doc = _load(n)
    for k_str, cell in doc["per_k"].items():
        k = int(k_str)
        expected = gaborit_sigma(n, k)
        assert cell["gaborit_sigma"] == expected, (
            f"n{n}.json k={k}: stored gaborit_sigma={cell['gaborit_sigma']} "
            f"!= recomputed {expected}"
        )
        assert cell["mass"] == expected, (
            f"n{n}.json k={k}: enumerated mass={cell['mass']} "
            f"!= gaborit_sigma {expected} (mass-formula certificate FAILED)"
        )
