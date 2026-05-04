"""Shared fixtures for vcif-hlo tests."""
from __future__ import annotations

import pytest

from vcif_hlo import IdDictionary, ReferentUniverseTensor


@pytest.fixture
def small_dict() -> IdDictionary:
    d = IdDictionary()
    for s in ('alpha', 'beta', 'gamma', 'delta'):
        d.intern(s)
    return d


@pytest.fixture
def small_universe(small_dict) -> ReferentUniverseTensor:
    """K=1 universe over {alpha, beta, gamma, delta}."""
    ids = [small_dict.intern(s) for s in ('alpha', 'beta', 'gamma', 'delta')]
    return ReferentUniverseTensor.from_ids(ids)
