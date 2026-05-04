# vcif-hlo — tensor/OpenHLO carrier for v4cat catalogues

The **third substrate-column counterpart** to [vcif][vcif] (JSON Schema) and
[vcif-rdf][vcif-rdf] (RDF/SHACL/SPARQL). v4cat semantics; tensor substrate.

[vcif]: https://github.com/v4cat-oss/vcif
[vcif-rdf]: https://github.com/v4cat-oss/vcif-rdf
[v4cat]: https://github.com/v4cat-oss/v4cat

## Algebraic basis

Per [v4cat `theory.md` § 15][theory15]: v4cat is a free-abelian
assertion-history group action `H = ℤ^𝔄`, the visible catalogue is the
support quotient `π(H)`, and `kquery` is the V₄-equivariant coordinate
chart of the observer-pair group `V₄^U`. **RISC writes are
translations; kquery is the V₄ coordinate chart.**

[theory15]: https://github.com/v4cat-oss/v4cat/blob/main/src/v4cat/theory.md

vcif-hlo *operationalises* the slogan as branchless tensor algebra:

```text
cell_code = 2 · A_live + B_live ∈ {0, 1, 2, 3}    # the V₄ coordinate
00 = U_live ∧ ¬A_live ∧ ¬B_live    cell_code == 0
01 = U_live ∧ ¬A_live ∧  B_live    cell_code == 1
10 = U_live ∧  A_live ∧ ¬B_live    cell_code == 2
11 = U_live ∧  A_live ∧  B_live    cell_code == 3
```

A single broadcast + multiply + add yields the four cells. No
branching. CISC recognizers are typed DAGs of universe constructors,
kquery gates, and projections.

## What ships in v0.1

- `IdDictionary` — the string ↔ Id boundary; the only place strings
  exist. Inside the kernels, only Id tensors flow.
- `ReferentUniverseTensor` — universe-shaped tensor of identity
  distinctions; arity K parameterises node (K=1), binary
  relation (K=2), or edge triple (K=3) universes.
- `CoverTensor` — V₄-fiber decomposition over a universe; carries the
  unquotiented cell code so any later projection (11, 10∪01, 10, etc.)
  is a pure mask.
- RISC kernels — `introduce_node`, `edge`, `kquery`. Branchless,
  tensor-shaped.
- `QueryDAG` — typed DAG of universe constructors + kquery gates +
  projections; CISC recognizers compose as DAGs.
- Two worked recognizers: `resolve_references` (per writeup § 6) and
  `recursive_definition` (per writeup § 7).
- Bridges to v4cat (read universes from a SymmetryCatalogue; apply
  derived edges back), and to vcif/vcif-rdf snapshots.
- CLI: `vcif-hlo validate / inspect / kquery / run`.

## Backend

Backend-agnostic via the array-API surface NumPy and JAX both expose:
`broadcast_to`, `where`, `logical_and/or/not`, `sum`, `astype`,
`array`. NumPy is the default (light dependency, runs everywhere).
JAX is optional — install via `pip install vcif-hlo[jax]` to compile
recognizers via `jax.jit` and (in a future release) export to
StableHLO MLIR via `jax.export`.

The StableHLO export path is registered as a promissory cell in
v4cat's cotype (`shadow_stablehlo_export_gap.md`).

## Install

```sh
pip install vcif-hlo                # NumPy backend
pip install "vcif-hlo[jax]"         # + JAX backend for jit / StableHLO export
pip install "vcif-hlo[vcif,vcif-rdf]"  # + bridges to JSON / RDF snapshots
```

## CLI

```sh
vcif-hlo validate <doc>                              # tensor-shape conformance
vcif-hlo inspect <doc>                               # universe shapes + sizes
vcif-hlo kquery --U <universe> --A <set> --B <set>   # ad-hoc V₄ classification
vcif-hlo run <recognizer> --catalogue cat.db ...     # execute a named recognizer
```

## Python API

```python
from vcif_hlo import IdDictionary, ReferentUniverseTensor, kquery

d = IdDictionary()
ids = [d.intern(s) for s in ('alpha', 'beta', 'gamma')]
U = ReferentUniverseTensor.from_ids(ids)

A_live = U.contains([d.intern('alpha'), d.intern('beta')])
B_live = U.contains([d.intern('beta'),  d.intern('gamma')])

cover = kquery(U, A_live, B_live)
print(cover.project_11())   # → [False, True, False]    (beta in both)
print(cover.project_10())   # → [True,  False, False]   (alpha in A only)
print(cover.project_01())   # → [False, False, True]    (gamma in B only)
print(cover.project_00())   # → [False, False, False]   (nothing blind)
```

## Layout

```text
vcif-hlo/
├── pyproject.toml
├── LICENSE                            MIT
├── README.md                          this file
├── docs/
│   ├── spec.md                        full carrier spec (14 sections)
│   └── examples/
│       ├── resolve_references.py      writeup § 6
│       ├── recursive_definition.py    writeup § 7
│       └── hf_dbe_closure.py          parity-with-vcif demo
└── src/vcif_hlo/
    ├── __init__.py                    public API
    ├── __main__.py                    enables `python -m vcif_hlo`
    ├── cli.py                         console-script entry
    ├── dictionary.py                  IdDictionary — string ↔ Id boundary
    ├── tensors.py                     ReferentUniverseTensor, CoverTensor
    ├── kernels.py                     introduce_node, edge, kquery
    ├── dag.py                         QueryDAG composition + eval
    ├── recognizers.py                 ResolveReferences + RecursiveDefinition
    ├── bridge_v4cat.py                load/apply with SymmetryCatalogue
    ├── bridge_vcif.py                 load from vcif/vcif-rdf snapshots
    └── tests/
        └── ...
```

## Relationship to v4cat-oss siblings

| Distribution | Substrate | Validator | Query language |
|---|---|---|---|
| [v4cat][v4cat] | Python + SQLite | (none) | Python `kquery` |
| [v4cat-mcp](https://github.com/v4cat-oss/v4cat-mcp) | MCP-over-stdio (RPC) | (transport) | MCP tool args |
| [vcif][vcif] | JSON | JSON Schema 2020-12 | Python set_expr eval |
| [vcif-rdf][vcif-rdf] | RDF/Turtle | SHACL + SHACL-SPARQL | SPARQL 1.1 |
| **[vcif-hlo](https://github.com/v4cat-oss/vcif-hlo)** (this repo) | Tensor (NumPy/JAX/StableHLO) | tensor-shape checks | branchless mask algebra |

Three substrate columns now filled. The catalogue's identity is
unchanged across all five distributions. v4cat sits at the kernel-cell
as the universal; every column is a co-projection of v4cat
parameterised by `(depth, substrate)` in the carrier grid.

## License

MIT.
