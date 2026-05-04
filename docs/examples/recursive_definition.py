"""
Worked example: recursive-definition recognizer (writeup § 7).

Synthetic universe of definitions and definition-term pairs;
recognises self-referencing definitions and reports per-cell action.

Run from the repo root:

    python docs/examples/recursive_definition.py
"""
from __future__ import annotations

import numpy as np

from vcif_hlo import (
    IdDictionary,
    ReferentUniverseTensor,
    project_recursive_evidence,
    recursive_definition_dag,
)


def main():
    d = IdDictionary()

    Omega_defs = ReferentUniverseTensor.from_ids([
        d.intern('def:fact'),       # recursive (self-references)
        d.intern('def:double'),     # not recursive
        d.intern('def:legacy'),     # marked recursive but no evidence
    ])

    Omega_pairs = ReferentUniverseTensor.from_tuples([
        (d.intern('def:fact'),   d.intern('term:fact/body/call#1')),
        (d.intern('def:double'), d.intern('term:double/body/x*2#1')),
    ], arity=2)

    # def:fact's body has a term that references def:fact → 11 of pair-cover
    # def:double's body has a term that does NOT reference def:double → 10
    pair_term_in_def_body    = np.array([True,  True],  dtype=np.bool_)
    pair_term_references_def = np.array([True,  False], dtype=np.bool_)

    defs_already_recursive = Omega_defs.contains([d.intern('def:legacy')])

    dag = recursive_definition_dag()

    # Stage 1: compute self-reference pair mask, then project to def-ids.
    stage1 = dag.evaluate({
        'Omega_pairs':              Omega_pairs,
        'pair_term_in_def_body':    pair_term_in_def_body,
        'pair_term_references_def': pair_term_references_def,
        'Omega_defs':                Omega_defs,
        'defs_recursive_evidence':  np.zeros(Omega_defs.live.shape, dtype=np.bool_),
        'defs_already_recursive':    defs_already_recursive,
    })
    self_ref_mask = stage1['mask_self_reference_pairs']
    evidence = project_recursive_evidence(Omega_defs, Omega_pairs, self_ref_mask)

    # Stage 2: evidence vs marker.
    stage2 = dag.evaluate({
        'Omega_pairs':              Omega_pairs,
        'pair_term_in_def_body':    pair_term_in_def_body,
        'pair_term_references_def': pair_term_references_def,
        'Omega_defs':                Omega_defs,
        'defs_recursive_evidence':   evidence,
        'defs_already_recursive':    defs_already_recursive,
    })

    def names(mask):
        return [d.name(int(i)) for i in Omega_defs.rows[mask].flatten()]

    print(f'mask_promote (10):              {names(stage2["mask_promote"])}')
    print(f'mask_unsupported_marker (01):  {names(stage2["mask_unsupported_marker"])}')
    print(f'mask_stable (11):              {names(stage2["mask_stable"])}')
    print(f'mask_blind  (00):              {names(stage2["mask_blind"])}')


if __name__ == '__main__':
    main()
