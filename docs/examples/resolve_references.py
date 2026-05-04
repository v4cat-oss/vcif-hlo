"""
Worked example: ResolveReferencesForName (writeup § 6).

Builds a small synthetic universe of terms and definitions, runs the
resolve-references DAG, and prints the four V₄ cells of the pair-cover.

Run from the repo root:

    python docs/examples/resolve_references.py
"""
from __future__ import annotations

import numpy as np

from vcif_hlo import (
    IdDictionary,
    ReferentUniverseTensor,
    cartesian_product,
    resolve_references_dag,
    resolve_references_pair_dag,
)


def main():
    d = IdDictionary()

    # Synthetic term universe.
    term_ids = [d.intern(s) for s in (
        'term:Foo.f/body/Def#1',     # has-kind=Def, head-name=map → 11 (RefTerm)
        'term:Foo.f/body/Def#2',     # has-kind=Def, head-name=other → 10
        'term:Bar.g/body/App#1',     # has-kind=App                  → 00 / 01
    )]
    Omega_terms = ReferentUniverseTensor.from_ids(term_ids)

    terms_kind_def    = Omega_terms.contains([d.intern('term:Foo.f/body/Def#1'),
                                              d.intern('term:Foo.f/body/Def#2')])
    terms_head_name_N = Omega_terms.contains([d.intern('term:Foo.f/body/Def#1')])

    # Synthetic definition universe.
    def_ids = [d.intern(s) for s in (
        'def:Data.List.map',         # kind=AgdaDefinition, defines=map → 11 (TargetDef)
        'def:Data.List.filter',      # kind=AgdaDefinition, defines=filter → 10
    )]
    Omega_defs = ReferentUniverseTensor.from_ids(def_ids)

    defs_kind_AgdaDef = Omega_defs.contains([d.intern('def:Data.List.map'),
                                             d.intern('def:Data.List.filter')])
    defs_define_N     = Omega_defs.contains([d.intern('def:Data.List.map')])

    # Stage A: kquery on terms and on defs.
    dag_a = resolve_references_dag()
    out_a = dag_a.evaluate({
        'Omega_terms':       Omega_terms,
        'terms_kind_def':    terms_kind_def,
        'terms_head_name_N': terms_head_name_N,
        'Omega_defs':        Omega_defs,
        'defs_kind_AgdaDef': defs_kind_AgdaDef,
        'defs_define_N':     defs_define_N,
    })

    ref_terms_mask = out_a['mask_RefTerms']
    target_defs_mask = out_a['mask_TargetDefs']

    print('Stage A:')
    print(f'  RefTerms  ({int(ref_terms_mask.sum())}): '
          f'{[d.name(int(i)) for i in Omega_terms.rows[ref_terms_mask].flatten()]}')
    print(f'  TargetDefs ({int(target_defs_mask.sum())}): '
          f'{[d.name(int(i)) for i in Omega_defs.rows[target_defs_mask].flatten()]}')

    # Lift to pair universe = RefTerms × TargetDefs.
    RefTerms_U = ReferentUniverseTensor(
        rows=Omega_terms.rows[ref_terms_mask], live=ref_terms_mask[ref_terms_mask], arity=1,
    )
    TargetDefs_U = ReferentUniverseTensor(
        rows=Omega_defs.rows[target_defs_mask], live=target_defs_mask[target_defs_mask], arity=1,
    )
    Omega_pairs = cartesian_product(RefTerms_U, TargetDefs_U)

    # Suppose no references-def edges exist yet.
    pair_existing_ref = np.zeros(Omega_pairs.live.shape, dtype=np.bool_)

    # Stage B: pair kquery.
    dag_b = resolve_references_pair_dag()
    out_b = dag_b.evaluate({
        'Omega_pairs':       Omega_pairs,
        'pair_existing_ref': pair_existing_ref,
    })

    print('Stage B:')
    print(f'  derive_live      ({int(out_b["derive_live"].sum())}): '
          f'{[(d.name(int(r[0])), d.name(int(r[1]))) for r in Omega_pairs.rows[out_b["derive_live"]]]}')
    print(f'  unsupported_live ({int(out_b["unsupported_live"].sum())})')
    print(f'  stable_live      ({int(out_b["stable_live"].sum())})')
    print(f'  blind_live       ({int(out_b["blind_live"].sum())})')


if __name__ == '__main__':
    main()
