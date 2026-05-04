"""Recognizer tests — ResolveReferences, RecursiveDefinition."""
from __future__ import annotations

import numpy as np
import pytest

from vcif_hlo import (
    IdDictionary,
    ReferentUniverseTensor,
    cartesian_product,
    project_recursive_evidence,
    recursive_definition_dag,
    resolve_references_dag,
    resolve_references_pair_dag,
)


def test_resolve_references_stage_a_picks_only_eleven_cell():
    """RefTerms = terms with both has-kind=Def AND has-head-name=N."""
    d = IdDictionary()
    term_ids = [d.intern(s) for s in ('term:A', 'term:B', 'term:C', 'term:D')]
    Omega_terms = ReferentUniverseTensor.from_ids(term_ids)

    # term:A has kind=Def AND head-name=N → 11
    # term:B has kind=Def only             → 10
    # term:C has head-name=N only          → 01
    # term:D has neither                   → 00
    A_kind = Omega_terms.contains([d.intern('term:A'), d.intern('term:B')])
    A_name = Omega_terms.contains([d.intern('term:A'), d.intern('term:C')])

    dag = resolve_references_dag()
    out = dag.evaluate({
        'Omega_terms':       Omega_terms,
        'terms_kind_def':    A_kind,
        'terms_head_name_N': A_name,
        'Omega_defs':        ReferentUniverseTensor.from_ids([]),
        'defs_kind_AgdaDef': np.zeros(0, dtype=np.bool_),
        'defs_define_N':     np.zeros(0, dtype=np.bool_),
    })
    # mask_RefTerms = 11 cell. Only term:A.
    expected = Omega_terms.contains([d.intern('term:A')])
    assert (out['mask_RefTerms'] == expected).all()


def test_resolve_references_pair_stage_yields_three_named_cells():
    """RefPair stage classifies (term, def) pairs into derive/stable/unsupported/blind."""
    d = IdDictionary()
    pair_universe = ReferentUniverseTensor.from_tuples(
        [
            (d.intern('term:1'), d.intern('def:X')),
            (d.intern('term:1'), d.intern('def:Y')),
            (d.intern('term:2'), d.intern('def:X')),
        ],
        arity=2,
    )
    # Existing references-def edges already cover (term:1, def:X) only.
    # That's index 0 of the pair universe.
    pair_existing_ref = np.array([True, False, False], dtype=np.bool_)

    dag = resolve_references_pair_dag()
    out = dag.evaluate({
        'Omega_pairs':        pair_universe,
        'pair_existing_ref':  pair_existing_ref,
    })
    # All three pairs are in A (the universe itself); existing edges in B.
    # → 11 = pair already represented (term:1, def:X)
    # → 10 = pair to derive ((term:1, def:Y), (term:2, def:X))
    # → 01 = empty (no edges outside the pair universe in this fixture)
    # → 00 = empty (nothing blind to both)
    assert (out['stable_live']      == np.array([True, False, False])).all()
    assert (out['derive_live']      == np.array([False, True, True])).all()
    assert (out['unsupported_live'] == np.array([False, False, False])).all()
    assert (out['blind_live']       == np.array([False, False, False])).all()


def test_cartesian_product_lifts_universes():
    """A × B over two K=1 universes is a K=2 universe of |A|·|B| pairs."""
    A = ReferentUniverseTensor.from_ids([1, 2])
    B = ReferentUniverseTensor.from_ids([10, 20, 30])
    AB = cartesian_product(A, B)
    assert AB.arity == 2
    assert AB.support_size == 6
    # Order: (1,10), (1,20), (1,30), (2,10), (2,20), (2,30)
    assert tuple(AB.rows[0]) == (1, 10)
    assert tuple(AB.rows[5]) == (2, 30)


def test_cartesian_product_empty_left():
    A = ReferentUniverseTensor.from_ids([])
    B = ReferentUniverseTensor.from_ids([10, 20])
    AB = cartesian_product(A, B)
    assert AB.arity == 2
    assert AB.support_size == 0


def test_recursive_definition_promotes_evidence_to_marker():
    """Definitions with self-reference evidence but no recursive marker land in 10."""
    d = IdDictionary()
    def_ids = [d.intern(s) for s in ('def:F', 'def:G', 'def:H')]
    Omega_defs = ReferentUniverseTensor.from_ids(def_ids)

    # Pair universe: (def, term)
    pair_rows = [
        (d.intern('def:F'), d.intern('term:F.body.1')),
        (d.intern('def:G'), d.intern('term:G.body.1')),
    ]
    Omega_pairs = ReferentUniverseTensor.from_tuples(pair_rows, arity=2)

    # def:F has term in body AND term references def:F → self-reference 11
    # def:G has term in body but term doesn't reference def:G → 10 of pair-cover
    pair_term_in_def_body     = np.array([True, True], dtype=np.bool_)
    pair_term_references_def  = np.array([True, False], dtype=np.bool_)

    # def:H is already marked recursive (legacy marker); evidence we'll project
    # below. Suppose def:F is also already marked → cell 11.
    # def:G has no marker, no evidence → cell 00.
    defs_already_recursive = Omega_defs.contains([d.intern('def:F'), d.intern('def:H')])

    dag = recursive_definition_dag()

    # Stage 1: compute self-reference pairs (cell 11 of pair-cover).
    stage1 = dag.evaluate({
        'Omega_pairs':              Omega_pairs,
        'pair_term_in_def_body':    pair_term_in_def_body,
        'pair_term_references_def': pair_term_references_def,
        'Omega_defs':               Omega_defs,
        'defs_recursive_evidence':  np.zeros(Omega_defs.live.shape, dtype=np.bool_),  # provisional
        'defs_already_recursive':   defs_already_recursive,
    })

    # The self-reference pair mask is from cover_self_ref_pairs's 11.
    # Project it to def-ids.
    self_ref_mask = stage1['mask_self_reference_pairs']
    assert (self_ref_mask == np.array([True, False])).all()

    evidence = project_recursive_evidence(Omega_defs, Omega_pairs, self_ref_mask)
    # def:F has evidence; def:G does not; def:H does not.
    expected_evidence = Omega_defs.contains([d.intern('def:F')])
    assert (evidence == expected_evidence).all()

    # Stage 2: re-evaluate with the real evidence mask.
    stage2 = dag.evaluate({
        'Omega_pairs':              Omega_pairs,
        'pair_term_in_def_body':    pair_term_in_def_body,
        'pair_term_references_def': pair_term_references_def,
        'Omega_defs':                Omega_defs,
        'defs_recursive_evidence':   evidence,
        'defs_already_recursive':    defs_already_recursive,
    })

    # def:F: evidence + marker → 11 (stable)
    # def:G: neither            → 00 (blind)
    # def:H: marker only        → 01 (unsupported_marker)
    assert (stage2['mask_stable']              ==
            Omega_defs.contains([d.intern('def:F')])).all()
    assert (stage2['mask_promote']             ==
            np.array([False, False, False])).all()  # nothing to promote
    assert (stage2['mask_unsupported_marker']  ==
            Omega_defs.contains([d.intern('def:H')])).all()
    assert (stage2['mask_blind']               ==
            Omega_defs.contains([d.intern('def:G')])).all()


def test_recursive_definition_promote_when_evidence_without_marker():
    """The 10 cell — recursive evidence exists but marker absent."""
    d = IdDictionary()
    def_ids = [d.intern(s) for s in ('def:F',)]
    Omega_defs = ReferentUniverseTensor.from_ids(def_ids)

    # No marker yet, but evidence exists.
    evidence = np.array([True], dtype=np.bool_)
    no_marker = np.array([False], dtype=np.bool_)

    dag = recursive_definition_dag()
    out = dag.evaluate({
        'Omega_pairs':              ReferentUniverseTensor.from_tuples([], arity=2),
        'pair_term_in_def_body':    np.zeros(0, dtype=np.bool_),
        'pair_term_references_def': np.zeros(0, dtype=np.bool_),
        'Omega_defs':                Omega_defs,
        'defs_recursive_evidence':   evidence,
        'defs_already_recursive':    no_marker,
    })
    assert out['mask_promote'].tolist() == [True]
    assert out['mask_stable'].tolist() == [False]
