"""
vcif_hlo — tensor/OpenHLO carrier for v4cat catalogues.

Public API:

    IdDictionary               — string ↔ Id boundary
    ReferentUniverseTensor     — universe-shaped tensor of identity distinctions
    CoverTensor                — V₄-fiber decomposition over a universe
    introduce_node, edge       — branchless RISC kernels (translations)
    kquery                     — V₄-equivariant coordinate chart
    QueryDAG                   — typed DAG of universe constructors + kquery + projections
    cartesian_product          — universe-lifting helper for pair recognizers
    resolve_references_dag     — writeup § 6 worked recognizer
    recursive_definition_dag   — writeup § 7 worked recognizer

The catalogue's identity is unchanged by vcif-hlo. All writes go
through `v4cat.SymmetryCatalogue`'s public verbs (introduce_node,
edge); the bridge module enforces that boundary.

Algebraic basis: v4cat theory.md § 15. RISC writes are translations;
kquery is the V₄ coordinate chart. Operationalised here as
`cell_code = 2·A_live + B_live`.
"""
from .dictionary import IdDictionary
from .tensors import ReferentUniverseTensor, CoverTensor
from .kernels import introduce_node, edge, kquery
from .dag import QueryDAG, NodeRef
from .recognizers import (
    cartesian_product,
    project_recursive_evidence,
    recursive_definition_dag,
    resolve_references_dag,
    resolve_references_pair_dag,
)

__all__ = [
    # boundary
    'IdDictionary',
    # tensors
    'ReferentUniverseTensor', 'CoverTensor',
    # RISC kernels
    'introduce_node', 'edge', 'kquery',
    # CISC composition
    'QueryDAG', 'NodeRef',
    # recognizers
    'cartesian_product', 'project_recursive_evidence',
    'recursive_definition_dag',
    'resolve_references_dag', 'resolve_references_pair_dag',
]
