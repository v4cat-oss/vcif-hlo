"""
vcif_hlo.geometric_currying — role-closure tensors over edge universes.

Per cotype/shadow_geometric_currying_vcif_hlo_carrier.md, the geometric-
currying substrate (cotype/shadow_geometric_currying.md) reframes edges
as projections of closed event-cells with three role obligations
(source, kind, target). In the tensor substrate, closure becomes
sparse-matrix algebra:

    role_source_closed : tensor<P × Bool>
    role_kind_closed   : tensor<P × Bool>
    role_target_closed : tensor<P × Bool>

    edge_closed = role_source_closed & role_kind_closed & role_target_closed

For a vcif-hlo arity-3 edge universe (rows of (source_idx, kind_idx,
target_idx)), the role-closed tensor is "live AND non-padding at the
role's column." This is fusion-friendly: a single broadcast-elementwise
expression compiles cleanly to OpenHLO / StableHLO ops.

Naming caveat: this module is named ``geometric_currying`` (not
``cells``) for symmetry with v4cat's ``event_cells.py`` -- avoiding the
overload between (1) the kquery 4-cell partition, (2) v4cat's existing
8-way kind enum, and (3) the new geometric event-cell.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .tensors import ReferentUniverseTensor


# ---------------------------------------------------------------------------
# Padding sentinel
# ---------------------------------------------------------------------------

PADDING_ID: int = 0  # vcif_hlo.dictionary.IdDictionary reserves 0 for padding


# ---------------------------------------------------------------------------
# Role-closure tensors
# ---------------------------------------------------------------------------

def role_source_closed(universe: ReferentUniverseTensor) -> np.ndarray:
    """Per-row Bool: row is live AND the source position is non-padding."""
    return _role_closed(universe, role_index=0)


def role_kind_closed(universe: ReferentUniverseTensor) -> np.ndarray:
    """Per-row Bool: row is live AND the kind position is non-padding."""
    return _role_closed(universe, role_index=1)


def role_target_closed(universe: ReferentUniverseTensor) -> np.ndarray:
    """Per-row Bool: row is live AND the target position is non-padding."""
    return _role_closed(universe, role_index=2)


def _role_closed(
    universe: ReferentUniverseTensor, *, role_index: int,
) -> np.ndarray:
    """Generic role-closure tensor for a chosen position."""
    if universe.arity < role_index + 1:
        # Arity too low for this role — every row's role is "vacuously closed."
        return universe.live.copy()
    return universe.live & (universe.rows[:, role_index] != PADDING_ID)


# ---------------------------------------------------------------------------
# Edge-closure (the saturating-mode invariant)
# ---------------------------------------------------------------------------

def edge_closed(universe: ReferentUniverseTensor) -> np.ndarray:
    """The saturating-mode edge-closure mask:

        edge_closed = role_source_closed & role_kind_closed & role_target_closed

    For an arity-3 edge universe, this is True iff the row is live AND
    all three role positions carry non-padding identities. The
    geometric-currying invariant operationalised at the tensor layer.
    """
    return (
        role_source_closed(universe)
        & role_kind_closed(universe)
        & role_target_closed(universe)
    )


# ---------------------------------------------------------------------------
# Path advancement
# ---------------------------------------------------------------------------

def advance_mask(
    universe: ReferentUniverseTensor,
    scheduled_mask: np.ndarray,
) -> np.ndarray:
    """The path-advancement mask:

        advance_mask = scheduled_mask & edge_closed

    Per cotype/shadow_geometric_currying.md, a path may advance through
    a cell only when the cell is closed. ``scheduled_mask`` carries
    which rows are scheduled for the path under consideration; the
    AND with ``edge_closed`` yields the rows where advancement is
    licensed.
    """
    if scheduled_mask.shape != universe.live.shape:
        raise ValueError(
            f"scheduled_mask shape {scheduled_mask.shape} != "
            f"universe live shape {universe.live.shape}"
        )
    return scheduled_mask & edge_closed(universe)
