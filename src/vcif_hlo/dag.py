"""
vcif_hlo.dag — typed DAG of universe constructors, kquery gates, and projections.

Per the writeup §§ 5-7: a CISC recognizer is not a new primitive; it
is a *named* DAG of RISC-compatible tensor nodes. The wires are
universe-shaped tensors; the gates are kquery; the outputs are
covers, masks, and derived rows.

This module provides the DAG construction and topological-evaluation
machinery. Nodes are typed by frame-shape (K=arity); wires are
type-checked at evaluation time.

The DAG is *deliberately Python-level*: it builds a list of node
specs and runs them eagerly. A future fire could compile the DAG to
JAX `jax.jit` and export to StableHLO via `jax.export` (see
`shadow_stablehlo_export_gap.md` in v4cat's cotype). The shape of
the DAG is the same in both cases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .kernels import kquery
from .tensors import CoverTensor, ReferentUniverseTensor


@dataclass
class NodeRef:
    """Opaque reference to a DAG node by name."""
    name: str

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class _UniverseNode:
    """Builds a ReferentUniverseTensor."""
    name: str
    build_fn: Callable[[dict], ReferentUniverseTensor]
    deps: tuple[str, ...] = ()


@dataclass
class _MaskNode:
    """Computes a Boolean mask over a universe (the support of A or B)."""
    name: str
    universe: str
    build_fn: Callable[[dict], np.ndarray]
    deps: tuple[str, ...] = ()


@dataclass
class _KqueryNode:
    """A kquery gate combining (universe, A-mask, B-mask)."""
    name: str
    universe: str
    A_mask: str
    B_mask: str


@dataclass
class _ProjectionNode:
    """A projection of a CoverTensor — cell mask over the same universe."""
    name: str
    cover: str
    kind: str  # one of '00', '01', '10', '11', 'diff', 'left', 'right', 'union'


_PROJECTION_KINDS = ('00', '01', '10', '11', 'diff', 'left', 'right', 'union')


@dataclass
class QueryDAG:
    """Typed DAG of universe constructors, kquery gates, and projections.

    Nodes are named; references between nodes are by name. Build with
    add_*; evaluate runs the DAG in topological order against an env.
    """

    _nodes: dict[str, Any] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ build

    def add_universe(
        self,
        name: str,
        build_fn: Callable[[dict], ReferentUniverseTensor],
        *,
        deps: tuple[str, ...] = (),
    ) -> NodeRef:
        """Add a node that constructs a ReferentUniverseTensor.

        `build_fn` is called with the partial environment (already-evaluated
        node names → values). `deps` is the names of upstream nodes
        whose values are read by `build_fn`.
        """
        self._add(name, _UniverseNode(name, build_fn, tuple(deps)))
        return NodeRef(name)

    def add_mask(
        self,
        name: str,
        universe: str,
        build_fn: Callable[[dict], np.ndarray],
        *,
        deps: tuple[str, ...] = (),
    ) -> NodeRef:
        """Add a Boolean-mask node over an existing universe."""
        if universe not in self._nodes:
            raise KeyError(f'universe {universe!r} not in DAG')
        self._add(name, _MaskNode(name, universe, build_fn, tuple(deps)))
        return NodeRef(name)

    def add_kquery(
        self, name: str, universe: str, A_mask: str, B_mask: str,
    ) -> NodeRef:
        """Add a kquery gate."""
        for n in (universe, A_mask, B_mask):
            if n not in self._nodes:
                raise KeyError(f'{n!r} not in DAG')
        self._add(name, _KqueryNode(name, universe, A_mask, B_mask))
        return NodeRef(name)

    def add_projection(
        self, name: str, cover: str, kind: str,
    ) -> NodeRef:
        """Add a projection node over a CoverTensor."""
        if cover not in self._nodes:
            raise KeyError(f'cover {cover!r} not in DAG')
        if kind not in _PROJECTION_KINDS:
            raise ValueError(f'projection kind {kind!r} not in {_PROJECTION_KINDS}')
        self._add(name, _ProjectionNode(name, cover, kind))
        return NodeRef(name)

    # ----------------------------------------------------------------- evaluate

    def evaluate(self, env: dict | None = None) -> dict:
        """Run the DAG in insertion order; return all named outputs.

        `env` provides any external values referenced by build_fn deps.
        Insertion order is honoured directly (no topological sort needed
        since add_* enforces forward references).
        """
        out: dict = dict(env or {})
        for name in self._order:
            node = self._nodes[name]
            value = self._eval_node(node, out)
            out[name] = value
        return out

    # -------------------------------------------------------------- internals

    def _add(self, name: str, node: Any) -> None:
        if name in self._nodes:
            raise ValueError(f'duplicate node name {name!r}')
        self._nodes[name] = node
        self._order.append(name)

    def _eval_node(self, node: Any, env: dict) -> Any:
        if isinstance(node, _UniverseNode):
            return node.build_fn(env)
        if isinstance(node, _MaskNode):
            U = env[node.universe]
            mask = node.build_fn(env)
            mask = np.asarray(mask, dtype=np.bool_)
            if mask.shape != U.live.shape:
                raise ValueError(
                    f'mask {node.name!r} shape {mask.shape} != universe '
                    f'{node.universe!r} shape {U.live.shape}'
                )
            return mask
        if isinstance(node, _KqueryNode):
            U = env[node.universe]
            A_live = env[node.A_mask]
            B_live = env[node.B_mask]
            return kquery(U, A_live, B_live)
        if isinstance(node, _ProjectionNode):
            cover: CoverTensor = env[node.cover]
            return _project(cover, node.kind)
        raise TypeError(f'unknown node type {type(node).__name__}')


def _project(cover: CoverTensor, kind: str) -> np.ndarray:
    if kind == '00':
        return cover.project_00()
    if kind == '01':
        return cover.project_01()
    if kind == '10':
        return cover.project_10()
    if kind == '11':
        return cover.project_11()
    if kind == 'diff':
        return cover.project_diff()
    if kind == 'left':
        return cover.project_left()
    if kind == 'right':
        return cover.project_right()
    if kind == 'union':
        return cover.project_union()
    raise ValueError(kind)
