"""
vcif_hlo.bridge_v4cat — load tensors from a v4cat catalogue; apply
derivation tensors back via v4cat's public ISA.

The bridge runs strictly OUTSIDE fused tensor regions. Strings cross
the boundary here (going in: catalogue ids → IdDictionary → Id
tensors; going out: derive_mask → IdDictionary.name → cat.edge calls).

Idempotent on the apply side via sqlite3.IntegrityError suppression
— the same pattern vcif and vcif-rdf use.
"""
from __future__ import annotations

import sqlite3
from typing import Iterable

import numpy as np

from .dictionary import IdDictionary
from .tensors import ReferentUniverseTensor


# -----------------------------------------------------------------------------
# Loading: catalogue → tensors
# -----------------------------------------------------------------------------

def load_node_universe(
    catalogue,
    dictionary: IdDictionary,
    *,
    spec_filter: str | None = None,
) -> ReferentUniverseTensor:
    """Read all `specs` rows from a v4cat catalogue into a K=1 universe.

    `spec_filter` is an optional SQL WHERE clause body (without the
    'WHERE' keyword) — e.g., `"id LIKE 'def:%'"` to filter to
    definition specs only. Strings are interned through `dictionary`.
    """
    sql = 'SELECT id FROM specs'
    if spec_filter:
        sql += f' WHERE {spec_filter}'
    rows = catalogue.conn.execute(sql).fetchall()
    ids = [dictionary.intern(r[0]) for r in rows]
    return ReferentUniverseTensor.from_ids(ids)


def load_edge_universe(
    catalogue,
    dictionary: IdDictionary,
    *,
    kinds: Iterable[str] | None = None,
) -> ReferentUniverseTensor:
    """Read all edges (witnesses + lineages) into a K=3 universe.

    Each row is `(source_id, kind_id, target_id)`. Strings are
    interned through `dictionary`. Optional `kinds` filter restricts to
    a given set of edge-kind labels.
    """
    out_rows: list[tuple[int, int, int]] = []
    kinds_list = None if kinds is None else list(kinds)
    placeholders = (
        ','.join('?' * len(kinds_list)) if kinds_list else None
    )

    # witnesses table: (spec_id, break_number, kind, ...)
    sql_w = 'SELECT spec_id, kind, break_number FROM witnesses'
    params_w: tuple = ()
    if kinds_list:
        sql_w += f' WHERE kind IN ({placeholders})'
        params_w = tuple(kinds_list)
    try:
        for s, k, t in catalogue.conn.execute(sql_w, params_w).fetchall():
            out_rows.append((
                dictionary.intern(s),
                dictionary.intern(k),
                dictionary.intern(t),
            ))
    except sqlite3.OperationalError:
        pass  # table absent — allow tests with stub catalogues

    # lineages table: (descendant_id, ancestor_id, kind, ...)
    sql_l = 'SELECT descendant_id, kind, ancestor_id FROM lineages'
    params_l: tuple = ()
    if kinds_list:
        sql_l += f' WHERE kind IN ({placeholders})'
        params_l = tuple(kinds_list)
    try:
        for s, k, t in catalogue.conn.execute(sql_l, params_l).fetchall():
            out_rows.append((
                dictionary.intern(s),
                dictionary.intern(k),
                dictionary.intern(t),
            ))
    except sqlite3.OperationalError:
        pass

    return ReferentUniverseTensor.from_tuples(out_rows, arity=3)


# -----------------------------------------------------------------------------
# Apply: tensors → catalogue
# -----------------------------------------------------------------------------

def apply_derive_mask(
    catalogue,
    dictionary: IdDictionary,
    universe: ReferentUniverseTensor,
    derive_mask: np.ndarray,
    *,
    edge_kind: str,
) -> dict:
    """For each (true) row of `derive_mask` in a K=3 edge universe,
    call `catalogue.edge(source, edge_kind, target)`. Idempotent via
    IntegrityError suppression.

    Returns a counts report.
    """
    if universe.arity != 3:
        raise ValueError(f'apply_derive_mask requires K=3 universe; got K={universe.arity}')
    derive_mask = np.asarray(derive_mask, dtype=np.bool_)
    if derive_mask.shape != universe.live.shape:
        raise ValueError(
            f'derive_mask shape {derive_mask.shape} != universe shape {universe.live.shape}'
        )

    report = {'edges_added': 0, 'edges_skipped': 0}
    rows = universe.rows[derive_mask]
    for row in rows:
        source_id, _kind_in_row, target_id = (int(x) for x in row)
        source = dictionary.name(source_id)
        target = dictionary.name(target_id)
        try:
            catalogue.edge(source, target, edge_kind)
            report['edges_added'] += 1
        except sqlite3.IntegrityError:
            report['edges_skipped'] += 1
        except (ValueError, RuntimeError):
            report['edges_skipped'] += 1
    return report


def apply_derive_pair_mask(
    catalogue,
    dictionary: IdDictionary,
    pair_universe: ReferentUniverseTensor,
    derive_mask: np.ndarray,
    *,
    edge_kind: str,
) -> dict:
    """For each (true) row of `derive_mask` in a K=2 (source, target)
    pair universe, call `catalogue.edge(source, edge_kind, target)`.
    Used by recognizers like ResolveReferences whose derive cell holds
    pairs, not full triples.
    """
    if pair_universe.arity != 2:
        raise ValueError(
            f'apply_derive_pair_mask requires K=2 universe; got K={pair_universe.arity}'
        )
    derive_mask = np.asarray(derive_mask, dtype=np.bool_)
    if derive_mask.shape != pair_universe.live.shape:
        raise ValueError(
            f'derive_mask shape {derive_mask.shape} != pair_universe shape {pair_universe.live.shape}'
        )
    report = {'edges_added': 0, 'edges_skipped': 0}
    rows = pair_universe.rows[derive_mask]
    for row in rows:
        source_id, target_id = (int(x) for x in row)
        source = dictionary.name(source_id)
        target = dictionary.name(target_id)
        try:
            catalogue.edge(source, target, edge_kind)
            report['edges_added'] += 1
        except sqlite3.IntegrityError:
            report['edges_skipped'] += 1
        except (ValueError, RuntimeError):
            report['edges_skipped'] += 1
    return report
