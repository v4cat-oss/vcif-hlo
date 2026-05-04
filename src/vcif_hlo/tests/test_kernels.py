"""RISC kernel tests — introduce_node, edge, kquery (branchless)."""
from __future__ import annotations

import numpy as np
import pytest

from vcif_hlo import (
    ReferentUniverseTensor,
    edge,
    introduce_node,
    kquery,
)


# -----------------------------------------------------------------------------
# introduce_node
# -----------------------------------------------------------------------------

def test_introduce_node_appends_when_no_padding(small_universe):
    U = small_universe
    n_before = U.support_size
    U2 = introduce_node(U, x=999)
    assert U2.support_size == n_before + 1
    # The new id is in U2
    assert U2.contains([999]).any()


def test_introduce_node_idempotent_when_present(small_universe):
    U = small_universe
    # 'alpha' is at id 0
    U2 = introduce_node(U, x=0)
    # Same universe (or at least same support)
    assert U2.support_size == U.support_size
    assert (U2.live == U.live).all()


def test_introduce_node_fills_padding():
    """Non-live row at index 1 gets occupied without growth."""
    rows = np.array([[10], [99], [30]])
    live = np.array([True, False, True])
    U = ReferentUniverseTensor(rows=rows, live=live, arity=1)
    U2 = introduce_node(U, x=20)
    # Row 1 should be (20) and live now
    assert U2.P == 3
    assert U2.live[1]
    assert U2.rows[1, 0] == 20


def test_introduce_node_rejects_higher_arity():
    U = ReferentUniverseTensor.from_tuples([(1, 2)], arity=2)
    with pytest.raises(ValueError):
        introduce_node(U, x=5)


# -----------------------------------------------------------------------------
# edge
# -----------------------------------------------------------------------------

def test_edge_adds_to_K3_universe():
    U = ReferentUniverseTensor.from_tuples(
        [(1, 2, 3), (4, 5, 6)], arity=3,
    )
    U2 = edge(U, candidate=(7, 8, 9))
    assert U2.support_size == 3


def test_edge_idempotent():
    U = ReferentUniverseTensor.from_tuples(
        [(1, 2, 3), (4, 5, 6)], arity=3,
    )
    U2 = edge(U, candidate=(1, 2, 3))
    assert U2.support_size == 2


def test_edge_arity_mismatch():
    U = ReferentUniverseTensor.from_tuples([(1, 2, 3)], arity=3)
    with pytest.raises(ValueError):
        edge(U, candidate=(1, 2))     # wrong arity


# -----------------------------------------------------------------------------
# kquery — the heart of the design
# -----------------------------------------------------------------------------

def test_kquery_disjoint_observers_yield_only_10_and_01():
    U = ReferentUniverseTensor.from_ids([10, 20])
    A_live = U.contains([10])
    B_live = U.contains([20])
    cov = kquery(U, A_live, B_live)
    assert cov.cell_sizes() == {'00': 0, '01': 1, '10': 1, '11': 0}


def test_kquery_identical_observers_yield_only_11():
    U = ReferentUniverseTensor.from_ids([10, 20, 30])
    A_live = U.contains([10, 20])
    B_live = U.contains([10, 20])
    cov = kquery(U, A_live, B_live)
    assert cov.cell_sizes() == {'00': 1, '01': 0, '10': 0, '11': 2}


def test_kquery_blind_member_in_00():
    U = ReferentUniverseTensor.from_ids([10, 20, 30, 40])
    A_live = U.contains([10])
    B_live = U.contains([20])
    cov = kquery(U, A_live, B_live)
    sizes = cov.cell_sizes()
    assert sizes['10'] == 1   # 10 in A only
    assert sizes['01'] == 1   # 20 in B only
    assert sizes['00'] == 2   # 30, 40 blind to both
    assert sizes['11'] == 0


def test_kquery_branchless_cell_code_formula():
    """cell_code = 2·A_live + B_live across all 4 combinations."""
    U = ReferentUniverseTensor.from_ids([1, 2, 3, 4])
    # u=1: ¬A ¬B → 0
    # u=2: ¬A  B → 1
    # u=3:  A ¬B → 2
    # u=4:  A  B → 3
    A_live = U.contains([3, 4])
    B_live = U.contains([2, 4])
    cov = kquery(U, A_live, B_live)
    assert cov.cell.tolist() == [0, 1, 2, 3]


def test_kquery_observer_outside_U_is_filtered():
    """If A_live claims a row that's not live in U, the cover honours U."""
    U = ReferentUniverseTensor.from_ids([10, 20])
    # Mark dead row in A_live (none here, but test the logic)
    rows = np.array([[10], [99], [20]])
    live = np.array([True, False, True])
    Upad = ReferentUniverseTensor(rows=rows, live=live, arity=1)
    A_live = np.array([False, True, False])  # claims dead row
    B_live = np.array([False, False, True])
    cov = kquery(Upad, A_live, B_live)
    # Dead row never appears in any projection
    assert not cov.project_10()[1]
    assert not cov.project_01()[1]
    assert not cov.project_11()[1]


def test_kquery_shape_mismatch_raises():
    U = ReferentUniverseTensor.from_ids([1, 2, 3])
    with pytest.raises(ValueError):
        kquery(U, np.array([True, True]), np.array([True, True, True]))
