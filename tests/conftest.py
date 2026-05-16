"""Pytest configuration: slow-test skipping.

Slow tests are marked with ``@pytest.mark.slow`` and are skipped by default.
Run them with ``pytest --run-slow`` (or ``uv run pytest --run-slow``).
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.slow (default: skipped)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: test takes more than a couple seconds; skipped unless --run-slow",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow; pass --run-slow to enable")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
