"""Enforce the canon/experimental and enumerate/experimental boundary.

The refactor that landed in commits b112ad4..6caa577 (Step 4a–4e + 8)
moved Feulner, paired-iso, Sage proxy, Schreier–Sims-on-GL(L, F_2),
and the Witt phase-(b) scaffolding into ``experimental/`` subpackages
so the spine reads cleanly. This test pins that boundary: a future
refactor that re-introduces ``doubly_even.canon.feulner`` (etc.)
under the spine should fail loudly.

Also verifies that the production kernel surface
(``canon_info_native``, ``doubly_even_candidates_q``) is present in
the default build.
"""

from __future__ import annotations

import importlib

import pytest


# --------- experimental subpackages import correctly under their new path


def test_experimental_modules_import():
    importlib.import_module("doubly_even.canon.experimental.feulner")
    importlib.import_module("doubly_even.canon.experimental.paired_iso")
    importlib.import_module("doubly_even.canon.experimental.sage_proxy")
    importlib.import_module(
        "doubly_even.canon.experimental.schreier_sims_gl"
    )
    importlib.import_module("doubly_even.enumerate.experimental.witt")
    importlib.import_module(
        "doubly_even.enumerate.experimental.quotient_witt"
    )


# --------- boundary: nothing experimental leaks back into the spine names


@pytest.mark.parametrize(
    "module_name",
    [
        "doubly_even.canon.feulner",
        "doubly_even.canon.paired_iso",
        "doubly_even.canon.sage_proxy",
        "doubly_even.canon.matrix_group",
        "doubly_even.enumerate.witt",
    ],
)
def test_spine_no_longer_owns_experimental_name(module_name: str):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_quotient_lost_witt_function():
    """``aut_orbit_minima_Q_witt`` moved out of ``enumerate.quotient``."""
    quotient = importlib.import_module("doubly_even.enumerate.quotient")
    assert not hasattr(quotient, "aut_orbit_minima_Q_witt"), (
        "aut_orbit_minima_Q_witt should now live under "
        "doubly_even.enumerate.experimental.quotient_witt"
    )


# --------- spine still exposes the production kernel API


def test_kernel_production_symbols_present():
    kernel = pytest.importorskip("doubly_even_kernel")
    assert hasattr(kernel, "canon_info_native"), (
        "kernel canon_info_native missing — spine dispatcher will break"
    )
    assert hasattr(kernel, "doubly_even_candidates_q"), (
        "kernel doubly_even_candidates_q missing — hot path will break"
    )
    assert hasattr(kernel, "enumerate_doubly_even"), (
        "kernel enumerate_doubly_even missing — D11 native recursion broken"
    )
