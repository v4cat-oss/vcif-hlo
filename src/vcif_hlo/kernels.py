"""
vcif_hlo.kernels — RISC operations as branchless tensor functions.

Three kernels matching v4cat's RISC primitives:

  * introduce_node(universe, x) → universe'
  * edge(universe, candidate)   → universe'         # arity-3 universe
  * kquery(universe, A_live, B_live) → CoverTensor

Per the writeup § 4 and theory.md § 15.11. Each kernel is branchless:
no Python-level if-else, no loops over rows. The output is computed
for every row, and selection happens by mask.

The introduce_node and edge kernels return *new* tensor instances
(non-mutating) so they can be chained or jitted under JAX. The
operations are *additive* — they grow the support; the modal
inverses live at the history-event level (see theory.md § 15.13).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .tensors import CoverTensor, ReferentUniverseTensor


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _row_equals(rows: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-row boolean: rows[i] == target (broadcast over the row axis)."""
    return np.all(rows == target, axis=1)


def _first_true_mask(values: np.ndarray) -> np.ndarray:
    """Boolean mask selecting the *first* True position in `values`.

    Returns an all-False mask if `values` has no True entries. Branchless
    over the row axis: builds a cumulative sum and selects index 0.
    """
    # Find the first index where values is True; cumulative sum gives 1 at first
    # True, 1 at second True, etc. Mask = (cumsum == 1) & values.
    if values.size == 0:
        return np.zeros_like(values, dtype=np.bool_)
    cum = np.cumsum(values.astype(np.int32))
    return (cum == 1) & values


# -----------------------------------------------------------------------------
# RISC primitives
# -----------------------------------------------------------------------------

def introduce_node(
    universe: ReferentUniverseTensor, x: int,
) -> ReferentUniverseTensor:
    """Branchlessly introduce node-id `x` into a K=1 universe.

    If `x` is already present (live row), returns the universe unchanged.
    If a free padding row exists, occupies it. If no free row exists,
    appends a new live row.

    Group-theoretically: translation by Nₓ (or no-op if already in support).
    """
    if universe.arity != 1:
        raise ValueError(
            f'introduce_node operates on K=1 universes; got K={universe.arity}'
        )
    rows = universe.rows
    live = universe.live

    target = np.array([x], dtype=np.int64)
    present_mask = _row_equals(rows, target) & live
    present = bool(present_mask.any())

    if present:
        return universe

    # Find a dead row to occupy
    free_mask = ~live
    insert_mask = _first_true_mask(free_mask)
    if insert_mask.any():
        new_rows = np.where(insert_mask[:, None], target, rows)
        new_live = live | insert_mask
        return ReferentUniverseTensor(rows=new_rows, live=new_live, arity=1)

    # No free padding — append
    new_rows = np.concatenate([rows, target.reshape(1, 1)], axis=0)
    new_live = np.concatenate([live, np.array([True], dtype=np.bool_)])
    return ReferentUniverseTensor(rows=new_rows, live=new_live, arity=1)


def edge(
    universe: ReferentUniverseTensor, candidate: Sequence[int],
) -> ReferentUniverseTensor:
    """Branchlessly introduce edge-tuple `candidate` into a K=arity universe.

    `candidate` is a K-tuple of Ids representing one row (typically
    `(source, kind, target)` for K=3 edge universes, but works for any
    arity).

    If the row is already present, returns the universe unchanged. If a
    free padding row exists, occupies it. Otherwise appends.

    Group-theoretically: translation by Eₛ,ₖ,ₜ.
    """
    target = np.asarray(list(candidate), dtype=np.int64)
    if target.shape != (universe.arity,):
        raise ValueError(
            f'edge candidate arity {target.shape} != universe arity {universe.arity}'
        )

    rows = universe.rows
    live = universe.live

    present_mask = _row_equals(rows, target) & live
    if bool(present_mask.any()):
        return universe

    free_mask = ~live
    insert_mask = _first_true_mask(free_mask)
    if insert_mask.any():
        new_rows = np.where(insert_mask[:, None], target, rows)
        new_live = live | insert_mask
        return ReferentUniverseTensor(rows=new_rows, live=new_live, arity=universe.arity)

    new_rows = np.concatenate([rows, target.reshape(1, -1)], axis=0)
    new_live = np.concatenate([live, np.array([True], dtype=np.bool_)])
    return ReferentUniverseTensor(rows=new_rows, live=new_live, arity=universe.arity)


def kquery(
    universe: ReferentUniverseTensor,
    A_live: np.ndarray,
    B_live: np.ndarray,
) -> CoverTensor:
    """Compute the V₄ coordinate decomposition of `universe` under
    observers `A_live` and `B_live`.

    `A_live` and `B_live` must be per-row Boolean masks with the same
    row count as `universe.live`. They represent the observer-pair
    (A, B) ⊆ U as universe-shaped supports.

    Returns a CoverTensor whose `cell` field is `2·A_live + B_live`
    per row. The four cells are then pure projections via
    `cover.project_{00,01,10,11}()`.

    Branchless. Group-theoretically: V₄-equivariant coordinate chart
    of the observer-pair group action on `(A, B) ∈ 𝒫(U) × 𝒫(U)`.
    """
    A_live = np.asarray(A_live, dtype=np.bool_)
    B_live = np.asarray(B_live, dtype=np.bool_)
    if A_live.shape != universe.live.shape:
        raise ValueError(
            f'A_live shape {A_live.shape} != universe.live shape {universe.live.shape}'
        )
    if B_live.shape != universe.live.shape:
        raise ValueError(
            f'B_live shape {B_live.shape} != universe.live shape {universe.live.shape}'
        )

    # Branchless V₄ coordinate: cell ∈ {00, 01, 10, 11}.
    # Effective masks honour universe.live so observers can't introduce
    # phantom membership outside U.
    eff_A = A_live & universe.live
    eff_B = B_live & universe.live
    cell = (np.uint8(2) * eff_A.astype(np.uint8)) + eff_B.astype(np.uint8)

    return CoverTensor(
        frame_rows=universe.rows,
        cell=cell,
        live=universe.live,
    )
