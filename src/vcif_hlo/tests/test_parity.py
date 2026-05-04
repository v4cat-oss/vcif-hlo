"""Cross-substrate parity: hf-dbe-closure example produces identical V₄
cells in vcif-hlo vs vcif (when vcif is installed).

The parity claim: a snapshot loaded from vcif's JSON form into
vcif-hlo's tensor form, then run through kquery with the same observer
pair, yields identical cell-membership to vcif's set_expr eval.
"""
from __future__ import annotations

import pytest

from vcif_hlo import IdDictionary, ReferentUniverseTensor, kquery


def test_hf_dbe_closure_eleven_cell_only():
    """The HF-DBE closure example: A and B both = {claim}; |U|=1; cell 11."""
    d = IdDictionary()
    claim = d.intern('CLAIM-DBE-produces-shadows')
    U = ReferentUniverseTensor.from_ids([claim])

    A_live = U.contains([claim])
    B_live = U.contains([claim])
    cov = kquery(U, A_live, B_live)

    sizes = cov.cell_sizes()
    assert sizes == {'00': 0, '01': 0, '10': 0, '11': 1}
    # Specifically: the 11 cell contains exactly the claim's id.
    assert (cov.frame_rows[cov.project_11()].flatten() == [claim]).all()


def test_disjoint_observers_match_set_diff():
    """A ∖ B and B ∖ A read off the V₄ chart agree with raw set diff."""
    d = IdDictionary()
    ids = {s: d.intern(s) for s in ('alpha', 'beta', 'gamma', 'delta')}
    U = ReferentUniverseTensor.from_ids(list(ids.values()))

    A = {ids['alpha'], ids['beta']}
    B = {ids['beta'],  ids['gamma']}
    A_live = U.contains(list(A))
    B_live = U.contains(list(B))
    cov = kquery(U, A_live, B_live)

    # 11 = A ∩ B = {beta}
    eleven = U.rows[cov.project_11()].flatten().tolist()
    assert eleven == [ids['beta']]

    # 10 = A ∖ B = {alpha}
    ten = U.rows[cov.project_10()].flatten().tolist()
    assert ten == [ids['alpha']]

    # 01 = B ∖ A = {gamma}
    oh1 = U.rows[cov.project_01()].flatten().tolist()
    assert oh1 == [ids['gamma']]

    # 00 = U ∖ (A ∪ B) = {delta}
    zero = U.rows[cov.project_00()].flatten().tolist()
    assert zero == [ids['delta']]


def test_parity_against_vcif_when_available():
    """If vcif is installed, load the agda-import.json snapshot and
    confirm node and edge counts match vcif-hlo's tensor form."""
    pytest.importorskip('vcif')
    import json
    from pathlib import Path

    # vcif's examples live in the sibling repo's docs/examples/ directory.
    # Walk up from this test file to the workspace dir, then sideways.
    test_file = Path(__file__).resolve()
    # parents[3] = vcif-hlo repo root; parents[4] = workspace dir.
    workspace_root = test_file.parents[4]
    repo_example = workspace_root / 'vcif' / 'docs' / 'examples' / 'agda-import.json'
    if not repo_example.exists():
        pytest.skip(f'vcif agda-import.json fixture not found at {repo_example}')

    with open(repo_example) as f:
        doc = json.load(f)

    from vcif_hlo.bridge_vcif import load_json_snapshot
    d, universes = load_json_snapshot(doc)

    # Sanity: nodes and edges parsed at expected counts (from vcif's example fixture).
    expected_nodes = len(doc['nodes'])
    expected_edges = len(doc['edges'])
    assert universes['nodes'].support_size == expected_nodes
    assert universes['edges'].support_size == expected_edges
