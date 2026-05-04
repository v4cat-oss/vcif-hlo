"""
vcif_hlo.tensors — universe-shaped carrier types.

Two core types:

  * ReferentUniverseTensor — a universe-shaped carrier of identity
    distinctions. K=1 → node universe; K=2 → binary relation; K=3 →
    edge triple; K=n → n-ary referent.

  * CoverTensor — V₄-fiber decomposition over a universe. Carries the
    unquotiented `cell ∈ {00, 01, 10, 11}` per row so any later
    projection is a pure mask.

Backend-agnostic: the implementations only use the array-API surface
that NumPy and JAX both expose. Tests run under NumPy by default;
JAX is optional and goes through the same calls.

Per the writeup §§ 1, 5 and v4cat theory.md § 15.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def _as_int_array(values: Sequence[int]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int64)


def _as_bool_array(values: Sequence[bool]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.bool_)


@dataclass
class ReferentUniverseTensor:
    """Universe-shaped tensor of identity distinctions.

    Attributes
    ----------
    rows : array of shape (P, K), dtype int (Id values).
        Each row is a K-tuple of Ids.
    live : array of shape (P,), dtype bool.
        Per-row liveness mask. Inactive rows are padding.
    arity : int
        K — the row arity (1=node, 2=binary, 3=edge-triple, ...).

    The universe size P is the *padded* row count; the *live* size is
    `int(live.sum())`. Padded rows allow branchless tensor algebra.
    """

    rows: np.ndarray
    live: np.ndarray
    arity: int

    def __post_init__(self):
        if self.rows.ndim != 2:
            raise ValueError(f'rows must be 2-D (P × K); got shape {self.rows.shape}')
        if self.live.ndim != 1:
            raise ValueError(f'live must be 1-D (P); got shape {self.live.shape}')
        if self.rows.shape[0] != self.live.shape[0]:
            raise ValueError(
                f'rows and live row-counts disagree: {self.rows.shape[0]} vs {self.live.shape[0]}'
            )
        if self.rows.shape[1] != self.arity:
            raise ValueError(
                f'rows arity {self.rows.shape[1]} disagrees with declared arity {self.arity}'
            )

    @property
    def P(self) -> int:
        """Padded row count (allocated capacity, including dead rows)."""
        return self.rows.shape[0]

    @property
    def support_size(self) -> int:
        """Number of live rows."""
        return int(self.live.sum())

    @classmethod
    def from_ids(cls, ids: Iterable[int]) -> 'ReferentUniverseTensor':
        """Build a K=1 (node) universe from a flat iterable of Ids.

        Padding-free: P = number of input Ids, all live.
        """
        ids_arr = _as_int_array(ids).reshape(-1, 1)
        live_arr = np.ones(ids_arr.shape[0], dtype=np.bool_)
        return cls(rows=ids_arr, live=live_arr, arity=1)

    @classmethod
    def from_tuples(
        cls, tuples: Iterable[Sequence[int]], arity: int,
    ) -> 'ReferentUniverseTensor':
        """Build a K=arity universe from a flat iterable of arity-tuples."""
        rows = np.asarray(list(tuples), dtype=np.int64)
        if rows.size == 0:
            rows = rows.reshape(0, arity)
        if rows.ndim != 2 or rows.shape[1] != arity:
            raise ValueError(
                f'tuple shape mismatch: arity={arity}, got rows shape {rows.shape}'
            )
        live = np.ones(rows.shape[0], dtype=np.bool_)
        return cls(rows=rows, live=live, arity=arity)

    def support(self) -> np.ndarray:
        """Return only the live rows (shape: support_size × K)."""
        return self.rows[self.live]

    def contains(self, ids: Iterable[int]) -> np.ndarray:
        """Per-row mask: True where row[0] ∈ ids (K=1 universes)."""
        if self.arity != 1:
            raise ValueError(
                f'contains() is only defined for K=1 universes; this is K={self.arity}'
            )
        target = _as_int_array(ids)
        if target.size == 0:
            return np.zeros(self.P, dtype=np.bool_)
        return np.isin(self.rows[:, 0], target) & self.live

    def contains_tuple(self, tup: Sequence[int]) -> np.ndarray:
        """Per-row mask: True where row equals `tup` exactly."""
        target = _as_int_array(tup)
        if target.shape != (self.arity,):
            raise ValueError(f'tuple arity {target.shape} != universe arity {self.arity}')
        match = np.all(self.rows == target, axis=1)
        return match & self.live


@dataclass
class CoverTensor:
    """V₄-fiber decomposition over a universe.

    Per the writeup § 5 and theory.md § 15.5: kquery materializes
    `cell_code = 2·A_live + B_live ∈ {0, 1, 2, 3}` per row. The four
    cells are then pure masks over `cell`.

    Attributes
    ----------
    frame_rows : array (P × K), dtype int.
        Identical to the source universe's rows.
    cell : array (P,), dtype uint8 with values in {0, 1, 2, 3}.
    live : array (P,), dtype bool. Same as source universe's live.
    """

    frame_rows: np.ndarray
    cell: np.ndarray
    live: np.ndarray

    def __post_init__(self):
        if self.cell.dtype != np.uint8:
            self.cell = self.cell.astype(np.uint8)
        if self.frame_rows.shape[0] != self.cell.shape[0]:
            raise ValueError(
                f'frame_rows and cell row-counts disagree: '
                f'{self.frame_rows.shape[0]} vs {self.cell.shape[0]}'
            )

    @property
    def P(self) -> int:
        return self.cell.shape[0]

    @property
    def arity(self) -> int:
        return self.frame_rows.shape[1]

    def cell_mask(self, code: int) -> np.ndarray:
        """Boolean mask: rows where cell == code AND live."""
        if not 0 <= code <= 3:
            raise ValueError(f'cell code must be 0..3; got {code}')
        return (self.cell == np.uint8(code)) & self.live

    def project_00(self) -> np.ndarray:
        return self.cell_mask(0)

    def project_01(self) -> np.ndarray:
        return self.cell_mask(1)

    def project_10(self) -> np.ndarray:
        return self.cell_mask(2)

    def project_11(self) -> np.ndarray:
        return self.cell_mask(3)

    def project_diff(self) -> np.ndarray:
        """10 ∪ 01 — the symmetric-difference projection."""
        return ((self.cell == np.uint8(1)) | (self.cell == np.uint8(2))) & self.live

    def project_left(self) -> np.ndarray:
        """Members of A — cell bit 1 set (codes 2, 3)."""
        return ((self.cell & np.uint8(2)) != 0) & self.live

    def project_right(self) -> np.ndarray:
        """Members of B — cell bit 0 set (codes 1, 3)."""
        return ((self.cell & np.uint8(1)) != 0) & self.live

    def project_union(self) -> np.ndarray:
        """A ∪ B — cell != 00 (and live)."""
        return (self.cell != np.uint8(0)) & self.live

    def cell_sizes(self) -> dict[str, int]:
        """Counts per V₄ cell."""
        return {
            '00': int(self.project_00().sum()),
            '01': int(self.project_01().sum()),
            '10': int(self.project_10().sum()),
            '11': int(self.project_11().sum()),
        }

    def members(self, code: int) -> np.ndarray:
        """Rows belonging to `cell_mask(code)` — shape (n × K)."""
        return self.frame_rows[self.cell_mask(code)]
