"""ReferentUniverseTensor + CoverTensor — backend-agnostic correctness."""
from __future__ import annotations

import numpy as np
import pytest

from vcif_hlo import CoverTensor, ReferentUniverseTensor


def test_from_ids_creates_K1_universe():
    U = ReferentUniverseTensor.from_ids([10, 20, 30])
    assert U.arity == 1
    assert U.P == 3
    assert U.support_size == 3
    assert U.rows.shape == (3, 1)
    assert (U.live == np.array([True, True, True])).all()


def test_from_tuples_creates_K3_universe():
    U = ReferentUniverseTensor.from_tuples(
        [(1, 2, 3), (4, 5, 6)], arity=3,
    )
    assert U.arity == 3
    assert U.P == 2
    assert U.rows.shape == (2, 3)
    assert U.support_size == 2


def test_contains_only_K1():
    U = ReferentUniverseTensor.from_tuples([(1, 2)], arity=2)
    with pytest.raises(ValueError):
        U.contains([1])


def test_contains_returns_per_row_mask():
    U = ReferentUniverseTensor.from_ids([10, 20, 30, 40])
    mask = U.contains([20, 40, 99])  # 99 not in U
    assert (mask == np.array([False, True, False, True])).all()


def test_contains_with_empty_target():
    U = ReferentUniverseTensor.from_ids([10, 20])
    mask = U.contains([])
    assert (mask == np.array([False, False])).all()


def test_contains_tuple():
    U = ReferentUniverseTensor.from_tuples(
        [(1, 2, 3), (4, 5, 6), (1, 2, 99)], arity=3,
    )
    mask = U.contains_tuple((1, 2, 3))
    assert (mask == np.array([True, False, False])).all()


def test_cover_tensor_cell_codes():
    """Cover constructed from cell array of {0,1,2,3} projects each cell."""
    rows = np.array([[10], [20], [30], [40]])
    live = np.array([True, True, True, True])
    cell = np.array([0, 1, 2, 3], dtype=np.uint8)
    cov = CoverTensor(frame_rows=rows, cell=cell, live=live)

    assert (cov.project_00() == np.array([True, False, False, False])).all()
    assert (cov.project_01() == np.array([False, True, False, False])).all()
    assert (cov.project_10() == np.array([False, False, True, False])).all()
    assert (cov.project_11() == np.array([False, False, False, True])).all()


def test_cover_diff_left_right_union():
    rows = np.array([[10], [20], [30], [40]])
    live = np.array([True, True, True, True])
    cell = np.array([0, 1, 2, 3], dtype=np.uint8)
    cov = CoverTensor(frame_rows=rows, cell=cell, live=live)

    # diff = 10 ∪ 01
    assert (cov.project_diff() == np.array([False, True, True, False])).all()
    # left = A members (cell bit 1 set: codes 2, 3)
    assert (cov.project_left() == np.array([False, False, True, True])).all()
    # right = B members (cell bit 0 set: codes 1, 3)
    assert (cov.project_right() == np.array([False, True, False, True])).all()
    # union = A ∪ B (cell != 0)
    assert (cov.project_union() == np.array([False, True, True, True])).all()


def test_cover_dead_rows_never_in_any_cell():
    rows = np.array([[10], [20], [30]])
    live = np.array([True, False, True])    # row 1 is padding
    cell = np.array([3, 3, 3], dtype=np.uint8)
    cov = CoverTensor(frame_rows=rows, cell=cell, live=live)
    # row 1 has cell=11 BUT live=False, so it's not selected
    assert (cov.project_11() == np.array([True, False, True])).all()


def test_cover_sizes():
    rows = np.array([[10], [20], [30], [40], [50]])
    live = np.array([True, True, True, True, True])
    cell = np.array([0, 1, 2, 3, 3], dtype=np.uint8)
    cov = CoverTensor(frame_rows=rows, cell=cell, live=live)
    assert cov.cell_sizes() == {'00': 1, '01': 1, '10': 1, '11': 2}


def test_cover_tensor_validates_shape_mismatch():
    with pytest.raises(ValueError):
        CoverTensor(
            frame_rows=np.array([[10], [20]]),
            cell=np.array([0, 1, 2], dtype=np.uint8),  # wrong length
            live=np.array([True, True]),
        )
