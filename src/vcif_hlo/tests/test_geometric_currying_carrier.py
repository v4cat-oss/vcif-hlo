"""
Tests for the gc-vcif-hlo-carrier sub-fire.

Closes vcif-hlo#4. Verifies the role-closure tensors + edge_closed
+ advance_mask compute correctly from a vcif-hlo edge universe.
"""
from __future__ import annotations

import numpy as np
import pytest

from vcif_hlo.geometric_currying import (
    advance_mask,
    edge_closed,
    role_kind_closed,
    role_source_closed,
    role_target_closed,
    PADDING_ID,
)
from vcif_hlo.tensors import ReferentUniverseTensor


def _u3(rows, live=None):
    """Build an arity-3 (edge) ReferentUniverseTensor."""
    rows = np.asarray(rows, dtype=np.int64)
    if live is None:
        live = np.ones(rows.shape[0], dtype=np.bool_)
    else:
        live = np.asarray(live, dtype=np.bool_)
    return ReferentUniverseTensor(rows=rows, live=live, arity=3)


def test_role_source_closed_simple():
    u = _u3([[1, 2, 3], [4, 5, 6]])
    assert role_source_closed(u).tolist() == [True, True]


def test_role_source_closed_filters_padding():
    u = _u3([[PADDING_ID, 2, 3], [4, 5, 6]])
    assert role_source_closed(u).tolist() == [False, True]


def test_role_kind_and_target_closed_filter_padding():
    u = _u3([[1, PADDING_ID, 3], [1, 2, PADDING_ID]])
    assert role_kind_closed(u).tolist() == [False, True]
    assert role_target_closed(u).tolist() == [True, False]


def test_role_closed_respects_live_mask():
    """Dead rows are not closed even if their role positions are non-padding."""
    u = _u3([[1, 2, 3], [4, 5, 6]], live=[True, False])
    assert role_source_closed(u).tolist() == [True, False]
    assert role_kind_closed(u).tolist()   == [True, False]
    assert role_target_closed(u).tolist() == [True, False]


def test_edge_closed_is_elementwise_and():
    u = _u3([
        [1, 2, 3],            # all roles closed
        [PADDING_ID, 5, 6],   # source open
        [7, PADDING_ID, 9],   # kind open
        [10, 11, PADDING_ID], # target open
    ])
    expected = [True, False, False, False]
    assert edge_closed(u).tolist() == expected


def test_advance_mask_fuses_scheduling_and_closure():
    u = _u3([[1, 2, 3], [4, 5, 6], [7, PADDING_ID, 9]])
    scheduled = np.array([True, False, True], dtype=np.bool_)
    # row 0: scheduled & closed -> True
    # row 1: not scheduled -> False
    # row 2: scheduled but kind open -> False
    assert advance_mask(u, scheduled).tolist() == [True, False, False]


def test_advance_mask_shape_mismatch_raises():
    u = _u3([[1, 2, 3]])
    bad = np.array([True, False], dtype=np.bool_)
    with pytest.raises(ValueError, match="scheduled_mask shape"):
        advance_mask(u, bad)
