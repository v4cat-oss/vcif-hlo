"""
vcif_hlo.recognizers — worked CISC recognizer DAGs from the writeup.

Two recognizers:

  * resolve_references — writeup § 6: given a name anchor, find term
    nodes that reference a definition with that name; if the
    references-def edge is missing, mark the cell-10 derive set.
  * recursive_definition — writeup § 7: detect definitions whose body
    references the definition itself; emit cell-10 promote-to-recursive
    derive set.

Both build a QueryDAG and return it; callers compose env (input
universes + masks) and call `dag.evaluate(env)`. The recognizers
demonstrate the writeup's claim that CISC = typed DAG of
universe-shaped tensor nodes.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .dag import QueryDAG
from .tensors import ReferentUniverseTensor


# -----------------------------------------------------------------------------
# resolve_references — writeup § 6
# -----------------------------------------------------------------------------

def resolve_references_dag() -> QueryDAG:
    """Build the ResolveReferencesForName DAG.

    Required env entries (for `dag.evaluate`):

      * Ω_terms             : universe of term nodes (K=1)
      * terms_kind_def      : mask over Ω_terms — terms with has-kind kind:Def
      * terms_head_name_N   : mask over Ω_terms — terms with has-head-name N
      * Ω_defs              : universe of definition nodes (K=1)
      * defs_kind_AgdaDef   : mask over Ω_defs
      * defs_define_N       : mask over Ω_defs
      * Ω_pairs             : universe of (term, def) pair tuples (K=2);
                              caller constructs the cartesian product of
                              RefTerms × TargetDefs after stage 1+2 evaluate.
      * pair_existing_ref   : mask over Ω_pairs — pair (t, d) for which
                              t -references-def-> d already exists.

    The DAG is split into two stages with a Python-level join between
    them (see `recognizers_evaluate_resolve_references` for the
    composed evaluator). Stage A runs over Ω_terms and Ω_defs; Stage B
    runs over the pair universe constructed from Stage A's outputs.
    """
    dag = QueryDAG()

    # Stage A: per-term and per-def kquery gates.
    dag.add_universe('Omega_terms', lambda env: env['Omega_terms'])
    dag.add_mask('terms_kind_def',
                 'Omega_terms', lambda env: env['terms_kind_def'])
    dag.add_mask('terms_head_name_N',
                 'Omega_terms', lambda env: env['terms_head_name_N'])
    dag.add_kquery(
        'cover_RefTerms',
        universe='Omega_terms',
        A_mask='terms_kind_def',
        B_mask='terms_head_name_N',
    )
    dag.add_projection('mask_RefTerms', cover='cover_RefTerms', kind='11')

    dag.add_universe('Omega_defs', lambda env: env['Omega_defs'])
    dag.add_mask('defs_kind_AgdaDef',
                 'Omega_defs', lambda env: env['defs_kind_AgdaDef'])
    dag.add_mask('defs_define_N',
                 'Omega_defs', lambda env: env['defs_define_N'])
    dag.add_kquery(
        'cover_TargetDefs',
        universe='Omega_defs',
        A_mask='defs_kind_AgdaDef',
        B_mask='defs_define_N',
    )
    dag.add_projection('mask_TargetDefs', cover='cover_TargetDefs', kind='11')

    return dag


def resolve_references_pair_dag() -> QueryDAG:
    """Stage B: pair-universe DAG. Caller supplies Ω_pairs (cartesian
    product of stage-A live RefTerms × TargetDefs) and pair_existing_ref.
    """
    dag = QueryDAG()
    dag.add_universe('Omega_pairs', lambda env: env['Omega_pairs'])
    # Left observer: full pair universe. Right observer: existing references-def edges.
    dag.add_mask('pair_universe',
                 'Omega_pairs', lambda env: env['Omega_pairs'].live)
    dag.add_mask('pair_existing_ref',
                 'Omega_pairs', lambda env: env['pair_existing_ref'])
    dag.add_kquery(
        'cover_RefPair',
        universe='Omega_pairs',
        A_mask='pair_universe',
        B_mask='pair_existing_ref',
    )
    dag.add_projection('derive_live',      cover='cover_RefPair', kind='10')
    dag.add_projection('unsupported_live', cover='cover_RefPair', kind='01')
    dag.add_projection('stable_live',      cover='cover_RefPair', kind='11')
    dag.add_projection('blind_live',       cover='cover_RefPair', kind='00')
    return dag


def cartesian_product(
    A: ReferentUniverseTensor, B: ReferentUniverseTensor,
) -> ReferentUniverseTensor:
    """Build A × B as a K=arity_A+arity_B universe.

    Used by the ResolveReferences recognizer (writeup § 6, stage 2 lift).
    Live rows of A times live rows of B; padding rows are dropped.
    """
    a_rows = A.support()
    b_rows = B.support()
    if a_rows.shape[0] == 0 or b_rows.shape[0] == 0:
        # Empty product
        empty = np.empty((0, A.arity + B.arity), dtype=np.int64)
        return ReferentUniverseTensor(
            rows=empty,
            live=np.zeros(0, dtype=np.bool_),
            arity=A.arity + B.arity,
        )
    # Cross product via broadcasting.
    a_repeat = np.repeat(a_rows, b_rows.shape[0], axis=0)
    b_tile = np.tile(b_rows, (a_rows.shape[0], 1))
    rows = np.concatenate([a_repeat, b_tile], axis=1)
    live = np.ones(rows.shape[0], dtype=np.bool_)
    return ReferentUniverseTensor(rows=rows, live=live, arity=A.arity + B.arity)


# -----------------------------------------------------------------------------
# recursive_definition — writeup § 7
# -----------------------------------------------------------------------------

def recursive_definition_dag() -> QueryDAG:
    """Build the recursive-definition recognizer DAG.

    Required env:

      * Omega_defs                  : universe of definitions (K=1)
      * Omega_pairs                 : universe of (def, term) pairs (K=2)
      * pair_term_in_def_body       : mask over Omega_pairs
                                       True where term ∈ body/descendant
                                       closure of definition.
      * pair_term_references_def    : mask over Omega_pairs
                                       True where term has a references-def
                                       edge to definition.
      * defs_already_recursive      : mask over Omega_defs
                                       True where definition is already
                                       marked recursive.

    Outputs (after evaluate):

      * cover_self_ref_pairs.cells  : (00, 01, 10, 11) over Omega_pairs.
      * mask_self_reference_pairs   : the 11 cell — body-mention coincides
                                       with a references-def edge.
      * (caller projects self-reference pairs onto Omega_defs to produce
        recursive_evidence — see `project_recursive_evidence` below.)
      * cover_recursive             : (00, 01, 10, 11) over Omega_defs.
      * mask_promote                : 10 — recursive evidence exists but
                                       marker absent (promote to recursive).
      * mask_unsupported_marker     : 01 — marker exists without evidence.
      * mask_stable                 : 11 — marker and evidence agree.
      * mask_blind                  : 00 — neither evidence nor marker.
    """
    dag = QueryDAG()

    dag.add_universe('Omega_pairs', lambda env: env['Omega_pairs'])
    dag.add_mask('pair_term_in_def_body',
                 'Omega_pairs', lambda env: env['pair_term_in_def_body'])
    dag.add_mask('pair_term_references_def',
                 'Omega_pairs', lambda env: env['pair_term_references_def'])
    dag.add_kquery(
        'cover_self_ref_pairs',
        universe='Omega_pairs',
        A_mask='pair_term_in_def_body',
        B_mask='pair_term_references_def',
    )
    dag.add_projection('mask_self_reference_pairs',
                       cover='cover_self_ref_pairs', kind='11')

    dag.add_universe('Omega_defs', lambda env: env['Omega_defs'])
    # The caller fills `defs_recursive_evidence` by projecting
    # self_reference_pairs to its first coordinate (the def-id).
    dag.add_mask('defs_recursive_evidence',
                 'Omega_defs',
                 lambda env: env['defs_recursive_evidence'])
    dag.add_mask('defs_already_recursive',
                 'Omega_defs', lambda env: env['defs_already_recursive'])
    dag.add_kquery(
        'cover_recursive',
        universe='Omega_defs',
        A_mask='defs_recursive_evidence',
        B_mask='defs_already_recursive',
    )
    dag.add_projection('mask_promote',
                       cover='cover_recursive', kind='10')
    dag.add_projection('mask_unsupported_marker',
                       cover='cover_recursive', kind='01')
    dag.add_projection('mask_stable',
                       cover='cover_recursive', kind='11')
    dag.add_projection('mask_blind',
                       cover='cover_recursive', kind='00')
    return dag


def project_recursive_evidence(
    Omega_defs: ReferentUniverseTensor,
    Omega_pairs: ReferentUniverseTensor,
    self_reference_pair_mask: np.ndarray,
) -> np.ndarray:
    """Project self-reference pair-mask onto Omega_defs by its first
    coordinate (def-id).

    Returns a Boolean mask over Omega_defs's rows: True where some
    pair (def_id, term_id) has self_reference_pair_mask = True.
    """
    if Omega_defs.arity != 1:
        raise ValueError('Omega_defs must be K=1')
    if Omega_pairs.arity != 2:
        raise ValueError('Omega_pairs must be K=2')
    # def-ids that survived the pair-cover's 11 projection
    surviving_pairs = Omega_pairs.rows[self_reference_pair_mask]
    if surviving_pairs.size == 0:
        return np.zeros(Omega_defs.live.shape, dtype=np.bool_)
    surviving_def_ids = surviving_pairs[:, 0]
    # mask over Omega_defs: True for rows whose id ∈ surviving_def_ids
    return np.isin(Omega_defs.rows[:, 0], surviving_def_ids) & Omega_defs.live
