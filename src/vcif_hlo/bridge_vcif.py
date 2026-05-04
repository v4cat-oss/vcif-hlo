"""
vcif_hlo.bridge_vcif — load tensor universes from vcif/vcif-rdf snapshots.

Both substrates carry the same v4cat-native content (nodes + edges +
covers + ...). This bridge reads either a vcif JSON dict or a
vcif-rdf rdflib Graph and produces an `IdDictionary` plus tensor
universes ready for kquery.

Optional dependencies: `vcif` (for JSON path) and `rdflib` (for RDF
path). Each is gated by import-availability; missing dependency
raises ImportError only when the corresponding loader is called.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .dictionary import IdDictionary
from .tensors import ReferentUniverseTensor


def load_json_snapshot(doc: dict) -> tuple[IdDictionary, dict]:
    """Read a vcif JSON document into an IdDictionary + tensor universes.

    Returns (dictionary, universes) where universes is a dict with keys:

      * 'nodes' : K=1 ReferentUniverseTensor of all node ids.
      * 'edges' : K=3 ReferentUniverseTensor of (source, kind, target) triples.

    The dict input is whatever `json.load` produces from a vcif
    profile (snapshot, patch, etc.) — only the `nodes` and `edges`
    fields are read.
    """
    if not isinstance(doc, dict):
        raise TypeError(f'expected dict, got {type(doc).__name__}')

    d = IdDictionary()

    node_ids: list[int] = []
    for n in doc.get('nodes', []):
        node_ids.append(d.intern(n['id']))

    edge_triples: list[tuple[int, int, int]] = []
    for e in doc.get('edges', []):
        s = d.intern(e['source'])
        k = d.intern(e['kind'])
        t = d.intern(e['target'])
        edge_triples.append((s, k, t))

    nodes_U = ReferentUniverseTensor.from_ids(node_ids)
    edges_U = ReferentUniverseTensor.from_tuples(edge_triples, arity=3)
    return d, {'nodes': nodes_U, 'edges': edges_U}


def load_rdf_snapshot(graph: Any) -> tuple[IdDictionary, dict]:
    """Read a vcif-rdf rdflib Graph into an IdDictionary + tensor universes.

    Walks the graph's `vc:NodeAssertion` and `vc:EdgeAssertion`
    subjects (where the carrier slots `vc:source`, `vc:edgeKind`,
    `vc:target` are the only RDF predicates the carrier uses).

    Returns (dictionary, universes) with the same shape as
    `load_json_snapshot`.
    """
    try:
        from rdflib import URIRef
        from rdflib.namespace import RDF
    except ImportError as e:
        raise ImportError(
            'load_rdf_snapshot requires rdflib; install vcif-hlo[vcif-rdf] '
            'or rdflib directly'
        ) from e

    VC = 'https://v4cat-oss.github.io/vcif-rdf/carrier#'
    VC_NodeAssertion = URIRef(VC + 'NodeAssertion')
    VC_EdgeAssertion = URIRef(VC + 'EdgeAssertion')
    VC_identifier = URIRef(VC + 'identifier')
    VC_source = URIRef(VC + 'source')
    VC_edgeKind = URIRef(VC + 'edgeKind')
    VC_target = URIRef(VC + 'target')

    d = IdDictionary()

    node_ids: list[int] = []
    for node in graph.subjects(RDF.type, VC_NodeAssertion):
        ident = graph.value(node, VC_identifier)
        if ident is not None:
            node_ids.append(d.intern(str(ident)))

    edge_triples: list[tuple[int, int, int]] = []
    for edge in graph.subjects(RDF.type, VC_EdgeAssertion):
        s_node = graph.value(edge, VC_source)
        k_node = graph.value(edge, VC_edgeKind)
        t_node = graph.value(edge, VC_target)
        if s_node is None or k_node is None or t_node is None:
            continue
        s_id = graph.value(s_node, VC_identifier)
        k_id = graph.value(k_node, VC_identifier)
        t_id = graph.value(t_node, VC_identifier)
        if s_id is None or k_id is None or t_id is None:
            continue
        s = d.intern(str(s_id))
        k = d.intern(str(k_id))
        t = d.intern(str(t_id))
        edge_triples.append((s, k, t))

    nodes_U = ReferentUniverseTensor.from_ids(node_ids)
    edges_U = ReferentUniverseTensor.from_tuples(edge_triples, arity=3)
    return d, {'nodes': nodes_U, 'edges': edges_U}


def load_snapshot(source: Any, *, format: str | None = None) -> tuple[IdDictionary, dict]:
    """Convenience wrapper. `format` is 'json' or 'rdf'.

    For JSON: `source` should be a parsed dict (use `json.load(open(path))`
    upstream).

    For RDF: `source` should be an rdflib.Graph (parsed via
    `Graph().parse(...)` upstream).

    Auto-detection is intentionally NOT supported — explicit format is
    safer (per plan's open sub-decision F).
    """
    if format is None:
        raise ValueError("format= is required; pass 'json' or 'rdf'")
    if format == 'json':
        return load_json_snapshot(source)
    if format == 'rdf':
        return load_rdf_snapshot(source)
    raise ValueError(f"unknown format {format!r}; expected 'json' or 'rdf'")
