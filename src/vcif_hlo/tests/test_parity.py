"""Cross-substrate parity tests.

Two layers:

  * **Internal V₄-chart parity** (the original tests): vcif-hlo's
    kquery agrees with raw set-difference algebra and with the
    HF-DBE closure expectation.

  * **Cross-substrate parity** (the cross-substrate group below):
    the same input data, classified by kquery in vcif's set_expr
    eval / vcif-rdf's SPARQL / vcif-hlo's tensor algebra, yields
    *identical V₄ cell membership* (modulo identifier renaming).

The cross-substrate group closes G1 from
`v4cat/cotype/audit_workspace_2026_05_04.md`.

The shadows produced by the DBE pass on G1 (from the conversation
that landed this file):

  * `parity_canonical_form`: the cells dict as
    `{'00': sorted[str], '01': sorted[str], '10': sorted[str],
      '11': sorted[str]}`. Identifier strings, post-renaming.
    Each substrate-specific extractor returns this canonical form.

  * `parity_check_function`: pairwise comparison of two canonical
    cells dicts; assertion failure carries the disagreeing cell
    code and per-cell members.
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


# =============================================================================
# Cross-substrate parity (closes G1 from the workspace audit).
# =============================================================================

# A 4-cell-coverage fixture: one identifier in each V₄ cell.
PARITY_SYNTHETIC = {
    'universe': ['alpha', 'beta', 'gamma', 'delta'],
    'A':        ['alpha', 'beta'],            # contains 11- and 10-members
    'B':        ['beta',  'gamma'],           # contains 11- and 01-members
    'expected': {
        '11': ['beta'],
        '10': ['alpha'],
        '01': ['gamma'],
        '00': ['delta'],
    },
}

# A boundary fixture: only cell 11 has a member (the HF-DBE closure shape).
PARITY_HF_DBE = {
    'universe': ['CLAIM-DBE-produces-shadows'],
    'A':        ['CLAIM-DBE-produces-shadows'],
    'B':        ['CLAIM-DBE-produces-shadows'],
    'expected': {
        '11': ['CLAIM-DBE-produces-shadows'],
        '10': [],
        '01': [],
        '00': [],
    },
}


# -----------------------------------------------------------------------------
# Substrate-specific cell extractors. Each takes a fixture dict (with
# 'universe', 'A', 'B' as lists of identifier strings) and returns the
# parity_canonical_form: dict with keys '00', '01', '10', '11', each a
# sorted list of identifier strings.
# -----------------------------------------------------------------------------


def cells_via_vcif(fixture: dict) -> dict[str, list[str]]:
    """vcif (JSON Schema substrate): literal-only set_expr cover."""
    pytest.importorskip('vcif')
    from vcif.importer import evaluate_cover_cells
    cover = {
        'id': 'cover:parity-test',
        'universe': {'op': 'literal', 'items': list(fixture['universe'])},
        'left':     {'op': 'literal', 'items': list(fixture['A'])},
        'right':    {'op': 'literal', 'items': list(fixture['B'])},
    }
    cells = evaluate_cover_cells(cover, catalogue=None, env={})
    return {code: sorted(cells[code]) for code in ('00', '01', '10', '11')}


def cells_via_vcif_rdf(fixture: dict) -> dict[str, list[str]]:
    """vcif-rdf (RDF/SHACL/SPARQL substrate): build a small carrier graph,
    run vcif-rdf's SPARQL kquery, resolve URIRefs back to identifier
    strings."""
    pytest.importorskip('vcif_rdf')
    pytest.importorskip('rdflib')
    from rdflib import Graph, URIRef, Literal
    from rdflib.namespace import RDF
    from vcif_rdf.kquery import cells as rdf_cells

    EX = 'https://example.org/parity#'
    VC = 'https://v4cat-oss.github.io/vcif-rdf/carrier#'
    NodeAssertion  = URIRef(VC + 'NodeAssertion')
    EdgeAssertion  = URIRef(VC + 'EdgeAssertion')
    CoverAssertion = URIRef(VC + 'CoverAssertion')
    P_identifier   = URIRef(VC + 'identifier')
    P_source       = URIRef(VC + 'source')
    P_edgeKind     = URIRef(VC + 'edgeKind')
    P_target       = URIRef(VC + 'target')
    P_universe     = URIRef(VC + 'universe')
    P_left         = URIRef(VC + 'leftObserver')
    P_right        = URIRef(VC + 'rightObserver')

    g = Graph()

    def add_node(local_uri_part: str, identifier: str) -> URIRef:
        uri = URIRef(EX + local_uri_part)
        g.add((uri, RDF.type, NodeAssertion))
        g.add((uri, P_identifier, Literal(identifier)))
        return uri

    # Object nodes (one per universe member)
    objects = {
        ident: add_node(f'node:{i}', ident)
        for i, ident in enumerate(fixture['universe'])
    }

    # Set markers + the in-set edge-kind (all NodeAssertions per the
    # carrier-vs-object discipline)
    set_U = add_node('set:U', 'set:U')
    set_A = add_node('set:A', 'set:A')
    set_B = add_node('set:B', 'set:B')
    in_set = add_node('inSet', 'in-set')

    # Membership edges
    edge_counter = 0

    def add_membership(member: URIRef, set_node: URIRef) -> None:
        nonlocal edge_counter
        e = URIRef(EX + f'edge:{edge_counter}')
        edge_counter += 1
        g.add((e, RDF.type, EdgeAssertion))
        g.add((e, P_source, member))
        g.add((e, P_edgeKind, in_set))
        g.add((e, P_target, set_node))

    for ident in fixture['universe']:
        add_membership(objects[ident], set_U)
    for ident in fixture['A']:
        add_membership(objects[ident], set_A)
    for ident in fixture['B']:
        add_membership(objects[ident], set_B)

    # Cover assertion
    cover_uri = URIRef(EX + 'cover')
    g.add((cover_uri, RDF.type, CoverAssertion))
    g.add((cover_uri, P_identifier, Literal('cover:parity-test')))
    g.add((cover_uri, P_universe, set_U))
    g.add((cover_uri, P_left, set_A))
    g.add((cover_uri, P_right, set_B))

    # Run SPARQL kquery
    raw = rdf_cells(g, cover_uri, in_set)

    # Resolve URIRefs back to identifier strings (the parity_canonical_form)
    out = {}
    for code in ('00', '01', '10', '11'):
        names = []
        for uri in raw.get(code, []):
            ident = g.value(uri, P_identifier)
            if ident is not None:
                names.append(str(ident))
        out[code] = sorted(names)
    return out


def cells_via_vcif_hlo(fixture: dict) -> dict[str, list[str]]:
    """vcif-hlo (tensor substrate): build a K=1 universe, compute kquery,
    resolve Ids back through the dictionary."""
    d = IdDictionary()
    u_ids = [d.intern(s) for s in fixture['universe']]
    a_ids = [d.intern(s) for s in fixture['A']]
    b_ids = [d.intern(s) for s in fixture['B']]

    U = ReferentUniverseTensor.from_ids(u_ids)
    A_live = U.contains(a_ids)
    B_live = U.contains(b_ids)

    cover = kquery(U, A_live, B_live)

    out = {}
    for code_str in ('00', '01', '10', '11'):
        code = int(code_str, 2)
        members = cover.members(code)
        # K=1: each row is [id]
        out[code_str] = sorted(d.name(int(row[0])) for row in members)
    return out


# -----------------------------------------------------------------------------
# Parametrized cross-substrate parity tests. 3 pairings × 2 fixtures = 6.
# Plus an all-three convergence test per fixture = 2 more. 8 total.
# -----------------------------------------------------------------------------


_FIXTURES = [
    pytest.param(PARITY_SYNTHETIC, id='synthetic-4cells'),
    pytest.param(PARITY_HF_DBE,    id='hf-dbe-boundary'),
]

_PAIRINGS = [
    pytest.param('vcif', 'vcif-rdf', cells_via_vcif,     cells_via_vcif_rdf,
                 id='vcif↔vcif-rdf'),
    pytest.param('vcif', 'vcif-hlo', cells_via_vcif,     cells_via_vcif_hlo,
                 id='vcif↔vcif-hlo'),
    pytest.param('vcif-rdf', 'vcif-hlo', cells_via_vcif_rdf, cells_via_vcif_hlo,
                 id='vcif-rdf↔vcif-hlo'),
]


@pytest.mark.parametrize('fixture', _FIXTURES)
@pytest.mark.parametrize('left_name,right_name,left_fn,right_fn', _PAIRINGS)
def test_parity_cross_substrate_pairing(
    left_name, right_name, left_fn, right_fn, fixture,
):
    """Cross-substrate parity: two substrates produce identical V₄ cells
    on the same fixture, modulo identifier renaming.

    Closes G1 (cross-substrate parity) from the workspace audit at the
    pairwise level: every pair of substrates agrees, cell-by-cell.
    """
    cells_left = left_fn(fixture)
    cells_right = right_fn(fixture)
    for code in ('00', '01', '10', '11'):
        assert cells_left[code] == cells_right[code], (
            f'{left_name} ↔ {right_name}: cell {code!r} disagrees:\n'
            f'  {left_name}: {cells_left[code]}\n'
            f'  {right_name}: {cells_right[code]}'
        )


@pytest.mark.parametrize('fixture', _FIXTURES)
def test_parity_all_three_match_expected(fixture):
    """Triple-convergence: all three substrates AND the fixture's
    declared `expected` cells coincide.

    Closes G1 at the all-three level: not just pairwise agreement, but
    convergence on the expected V₄ classification."""
    expected = {k: sorted(v) for k, v in fixture['expected'].items()}
    cells_v     = cells_via_vcif(fixture)
    cells_v_rdf = cells_via_vcif_rdf(fixture)
    cells_v_hlo = cells_via_vcif_hlo(fixture)
    for code in ('00', '01', '10', '11'):
        assert (
            cells_v[code]
            == cells_v_rdf[code]
            == cells_v_hlo[code]
            == expected[code]
        ), (
            f'cell {code!r} divergence:\n'
            f'  vcif:      {cells_v[code]}\n'
            f'  vcif-rdf:  {cells_v_rdf[code]}\n'
            f'  vcif-hlo:  {cells_v_hlo[code]}\n'
            f'  expected:  {expected[code]}'
        )
