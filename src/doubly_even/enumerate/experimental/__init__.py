"""Experimental / dormant enumeration scaffolding.

Holds two parked directions:

- **Witt phase-(b) machinery** (closed-form singular counts and
  the unreachable ``aut_orbit_minima_Q_witt`` orbit-min path). Phase (a)
  won at every measured ``N`` so phase (b) never became the active
  dispatch. Kept for pedagogical reference and in case the trade-off
  flips at some future ``N``.

- **BHM2012 direct-sum mass-seeding scaffolding** (``seeds.py``) —
  Phase 1 of the gluing-based mass-seeding plan: pre-credit the
  ``mass_at_k`` accumulator with codes reachable as ``C1 ⊕ C2``
  before the canonical-augmentation BFS starts, so the Gaborit
  mass-stop fires sooner. Not yet wired into the active recursion.

See ``/workspace/src/EXPERIMENTAL.md`` and the memory bullets
``phase_b_empirical_finding.md`` and ``project_paper_2012_audit.md``.
"""
