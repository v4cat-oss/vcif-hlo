# vcif-hlo — tensor/OpenHLO carrier for v4cat

> **Version**: 0.1
> **Status**: Alpha
> **Substrate-column siblings**: [vcif][vcif] (JSON), [vcif-rdf][vcif-rdf] (RDF/SHACL/SPARQL)
> **Algebraic basis**: [v4cat `theory.md` § 15][theory15]

[vcif]: https://github.com/v4cat-oss/vcif
[vcif-rdf]: https://github.com/v4cat-oss/vcif-rdf
[v4cat]: https://github.com/v4cat-oss/v4cat
[theory15]: https://github.com/v4cat-oss/v4cat/blob/main/src/v4cat/theory.md

vcif-hlo is the third substrate column in the v4cat carrier grid:
v4cat's RISC operations compile to **branchless tensor DAGs over
interned identity IDs**. Strings exist only at the dictionary
boundary; OpenHLO sees only `Id` tensors.

## 0. Slogan

> **RISC writes are translations; kquery is the V₄ coordinate chart.**

(From v4cat `theory.md` § 15.16.) vcif-hlo *operationalises* it as
branchless tensor algebra:

```text
cell_code = 2 · A_live + B_live ∈ {0, 1, 2, 3}      # the V₄ coordinate

00 = U_live ∧ ¬A_live ∧ ¬B_live    cell == 0
01 = U_live ∧ ¬A_live ∧  B_live    cell == 1
10 = U_live ∧  A_live ∧ ¬B_live    cell == 2
11 = U_live ∧  A_live ∧  B_live    cell == 3
```

A single broadcast + multiply + add yields the four cells. The
projections are pure masks.

## 1. The ambient assertion structure as tensors

Per [theory.md § 15.1][theory15], v4cat is a free-abelian
assertion-history group action `H = ℤ^𝔄`. The carrier here represents
that action over a **tensor of interned identifiers**:

```text
ReferentUniverseTensor[K]:
  rows : tensor<P × K × Id>
  live : tensor<P × Bool>
```

where:

| K | Shape |
|---|---|
| 1 | node universe |
| 2 | binary relation universe |
| 3 | edge-triple universe (the (source, kind, target) shape) |
| n | n-ary referent universe |

Padded rows allow branchless tensor algebra: `live` distinguishes
real rows from padding without conditional control flow.

## 2. The string-↔-Id boundary (`IdDictionary`)

The dictionary is intentionally OUTSIDE the semantic core. Strings
cross the boundary at exactly two points: import (`intern`) and
export (`name`). Inside the kernels, only `Id` tensors flow.

```python
d = IdDictionary()
d.intern('alpha')         # → 0
d.intern('beta')          # → 1
d.intern('alpha')         # → 0  (idempotent)
d.name(1)                 # → 'beta'
```

Stable: once interned, the Id is permanent for the dictionary's
lifetime. This is the contract that lets tensor kernels be
position-stable across imports.

## 3. The three RISC operations as tensor kernels

### `introduce_node` (writeup § 4)

Group-theoretically: translation by `Nₓ ∈ 𝔄_node`.

```python
def introduce_node(U: ReferentUniverseTensor, x: Id) -> ReferentUniverseTensor:
    # Branchless: present_mask | first_free_mask, then select.
```

If `x` is already in the live support, the kernel is the identity
(idempotence-as-quotient per theory.md § 15.4). Else: occupy a
padding row (no growth) or append.

### `edge` (writeup § 4)

Group-theoretically: translation by `Eₛ,ₖ,ₜ ∈ 𝔄_edge`. The kind k
is just the middle Id of a K=3 row — *not* a privileged predicate.

```python
def edge(U: ReferentUniverseTensor, candidate: tuple[Id, Id, Id]) -> ReferentUniverseTensor:
    # Same shape as introduce_node, K=3.
```

### `kquery` (writeup § 4, § 5; theory.md § 15.5)

The V₄-equivariant coordinate chart of the observer-pair group
`O_U = V₄^U`:

```python
def kquery(U: ReferentUniverseTensor, A_live, B_live) -> CoverTensor:
    cell = (2 * A_live.astype(uint8)) + B_live.astype(uint8)
    return CoverTensor(frame_rows=U.rows, cell=cell, live=U.live)
```

**No branching.** Single broadcast + multiply + add. The four cells
are `cover.project_{00,01,10,11}()` — pure boolean masks over `cell`.

## 4. CISC = typed DAG of universe constructors + kquery gates + projections

A CISC recognizer is *not* a new primitive. It is a named DAG whose
nodes are:

| Node kind | Function |
| --- | --- |
| `UniverseConstructor` | builds a ReferentUniverseTensor (from catalogue / dictionary / literal) |
| `MaskBuilder` | computes a Boolean mask over an existing universe |
| `KqueryGate` | `(A, B, U) → CoverTensor` |
| `Projection` | `(CoverTensor, kind) → mask` for kind ∈ {00, 01, 10, 11, diff, left, right, union} |

Edges are tensor dependencies. The DAG is *typed by frame shape*: K
must match across kquery wires. CISC composition is just chaining
these.

## 5. Worked recognizer: ResolveReferencesForName (writeup § 6)

Two-stage DAG.

**Stage A** — per-term and per-def kquery gates:

```text
RefTerms   = kquery(Ω_terms, terms_kind_def, terms_head_name_N).11
TargetDefs = kquery(Ω_defs,  defs_kind_AgdaDef, defs_define_N).11
```

**Stage B** — lift to pair universe `Ω_pairs = RefTerms × TargetDefs`,
then:

```text
RefPair = kquery(Ω_pairs, pair_universe, pair_existing_ref)

derive_live      = RefPair.10   (term → def edge to derive)
unsupported_live = RefPair.01   (existing references-def edge with no candidate)
stable_live      = RefPair.11   (already represented)
blind_live       = RefPair.00   (pair seen by neither)
```

The `cartesian_product` helper builds `Ω_pairs` from the live RefTerms
and TargetDefs.

## 6. Worked recognizer: recursive_definition (writeup § 7)

Two-stage DAG. Stage 1 over a (def, term) pair universe:

```text
SelfReferencePairs =
  kquery(Ω_pairs, pair_term_in_def_body, pair_term_references_def).11
```

Project to def-ids via `project_recursive_evidence`. Stage 2 over
`Ω_defs`:

```text
cover_recursive = kquery(Ω_defs, defs_recursive_evidence, defs_already_recursive)

mask_promote             = cover_recursive.10   # evidence without marker
mask_unsupported_marker  = cover_recursive.01   # marker without evidence
mask_stable              = cover_recursive.11   # both
mask_blind               = cover_recursive.00   # neither
```

This recognizer is the v4cat-native test for recursive definitions:
not a fixed predicate, a V₄-cover whose four cells each license a
specific lifecycle action.

## 7. Bridges to v4cat and to vcif/vcif-rdf

### v4cat side (`bridge_v4cat`)

```python
load_node_universe(catalogue, dictionary)        # → K=1 ReferentUniverseTensor
load_edge_universe(catalogue, dictionary)        # → K=3 over witnesses ∪ lineages
apply_derive_mask(catalogue, dictionary, U, mask, edge_kind)
                                                  # → calls cat.edge(s, k, t) per row
apply_derive_pair_mask(...)                       # K=2 derive masks for ResolveRef
```

Idempotent on apply via `sqlite3.IntegrityError` suppression — the
same pattern vcif and vcif-rdf use.

### vcif/vcif-rdf side (`bridge_vcif`)

```python
load_json_snapshot(doc)                          # vcif
load_rdf_snapshot(graph)                         # vcif-rdf
load_snapshot(source, format='json' | 'rdf')     # convenience
```

All three return `(IdDictionary, {'nodes': K=1, 'edges': K=3})`.
Auto-detection of format is intentionally NOT supported (explicit is safer).

## 8. Branchless / fusable kernel design (writeup § 8)

Inside fused regions:

```text
broadcast equality         row equality
boolean mask algebra       reduce_or / reduce_and
bitwise cell-code algebra  select
```

Avoid:

```text
ragged lists               Python loops
string checks              hash-table semantics
dynamic append             semantic if/else
```

Use padded tensors plus masks. Carry all four cells forward; project
later by named masks.

## 9. Fusion boundaries (writeup § 9)

The fusable interior:

```text
edge_select → intersection → product → membership →
kquery → cell mask → derived-row construction
```

Likely fusion barriers (where shape changes):

```text
canonical sort/unique    transitive closure    fixed-point closure
dynamic compaction       SCC computation       large relation saturation
```

Stage the architecture as fuse-block + barrier + fuse-block, with
v4cat-imports and `apply_derive_mask` calls living at the barriers.

## 10. Backend (NumPy / JAX / StableHLO)

vcif-hlo's kernels are written in **backend-agnostic NumPy-API style**.
Both `numpy` and `jax.numpy` expose the same surface
(`broadcast_to`, `where`, `logical_and`, etc.), so the same kernel
runs under either.

```python
import numpy as np
# import jax.numpy as np    # drop-in replacement
```

JAX is optional (`pip install vcif-hlo[jax]`). When installed, the
recognizers can be `jax.jit`-compiled and exported to StableHLO MLIR
via `jax.export.export(...).serialize()`. The export path is
registered as a promissory cell in v4cat's cotype
(`shadow_stablehlo_export_gap.md`).

## 11. Profiles (writeup § 10)

Four profiles, each a way to *shape* a tensor carrier document:

| Profile | Carrier content |
|---|---|
| **VCIF-HLO-Core** | NodeUniverse + EdgeUniverse |
| **VCIF-HLO-Cover** | + CoverTensor (V₄-fiber decomposition) |
| **VCIF-HLO-DAG** | + named QueryDAG |
| **VCIF-HLO-Self** | + carrier self-hosting cover |

VCIF-HLO-Self is the recursive bootstrap: the carrier's own
implemented-vs-catalogued claims classified by kquery against U =
all carrier claims. Cell 11 = honest commitments; 10 / 01 / 00 = the
three flavors of dishonesty per theory.md § 14.5.

## 12. Self-hosting cover (writeup § 11)

Carrier-claim universe, abbreviated:

```text
U_HLOCarrierClaims = {
  has-node-universe-tensor,
  has-edge-universe-tensor,
  has-cover-tensor,
  has-kquery-set-kernel,
  uses-id-dictionary-boundary,
  does-not-inspect-strings,
  represents-edge-kind-as-id-coordinate,
  carries-cell-codes-{00,01,10,11},
  derives-edges-only-from-cell-masks,
}

A = claims witnessed by exported kernels / API metadata
B = claims catalogued in VCIF/v4cat

CarrierClosure = kquery_U(A, B)

11 = implemented and catalogued (honest)
10 = implemented but not catalogued
01 = catalogued but not implemented
00 = accountable but absent from both
```

A future fire would add a `vcif_hlo.self_check()` that materializes
this cover on every package import. v0.1 documents the structure;
the running check is on the vcif-hlo roadmap.

## 13. Negative space — what vcif-hlo does NOT do

- Strings inside fused kernels: **never**. The dictionary is the
  boundary.
- RDF predicates as v4cat edge-kinds: **never** (per the
  carrier-vs-object discipline shared with vcif-rdf).
- Hidden recognizers (Python lambdas, `eval`, etc.): **never**.
  Recognizers are typed DAGs constructed via `QueryDAG.add_*`.
- Saturating closures (transitive closure, SCC): **fusion barrier**.
  Compute eagerly, cross the barrier, re-enter the next fuse block.
- Implicit projections: **never**. Every projection has a name and
  a kind from the eight allowed values.

## 14. Compact statement

> v4cat supplies the ontology: identities, edge triples, referent
> universes, V₄ covers.
>
> VCIF / VCIF-RDF supply the interchange: portable declarations of
> those universes, covers, DAGs, and residues.
>
> vcif-hlo supplies the carrier: branchless tensor execution over
> interned identity IDs.
>
> Shadow-architecture supplies the discipline: decompose into
> reusable shadows, regroup common structure, snap the result onto
> the catalogue grid, and self-audit the closure.

Tighter:

> Strings name the world outside.
> IDs carry it inside.
> kquery splits it into V₄ fibers.
> CISC is just a fused DAG of those splits.
> v4cat audits the whole carrier against itself.
