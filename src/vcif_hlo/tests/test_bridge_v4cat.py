"""Bridge tests — load tensors from a v4cat catalogue."""
from __future__ import annotations

import pytest

from vcif_hlo import IdDictionary
from vcif_hlo.bridge_v4cat import load_node_universe


def test_load_node_universe_from_in_memory_catalogue():
    """Open a fresh v4cat in-memory catalogue, introduce two specs, and
    confirm load_node_universe reads them back as a K=1 universe."""
    pytest.importorskip('v4cat')
    from v4cat import SymmetryCatalogue

    with SymmetryCatalogue(':memory:') as cat:
        # Use the legacy introduce_object surface for fixture simplicity.
        cat.introduce_object('alpha', 'Alpha', year=1980)
        cat.introduce_object('beta', 'Beta', year=1985,
                             lineage=[('alpha', 'descended-from')])

        d = IdDictionary()
        U = load_node_universe(cat, d)

    # Note: the framework_seed catalogues many internal specs (kinds, etc.)
    # too. We just check our two new ones are present.
    assert 'alpha' in d
    assert 'beta' in d
    assert U.contains([d.intern('alpha')]).any()
    assert U.contains([d.intern('beta')]).any()
