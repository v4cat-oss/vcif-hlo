"""
vcif_hlo.cli — console-script entry point.

Subcommands:

    vcif-hlo validate <doc>                    JSON snapshot (vcif-format)
                                                round-trips into tensor form
    vcif-hlo inspect <doc>                     universe shapes + sizes
    vcif-hlo kquery --U ids... --A ids... --B ids...
                                                ad-hoc V₄ classification
    vcif-hlo run <recognizer> ...              execute a named recognizer
"""
from __future__ import annotations

import argparse
import json
import sys

from .bridge_vcif import load_json_snapshot
from .kernels import kquery
from .tensors import ReferentUniverseTensor


def _cmd_validate(args: argparse.Namespace) -> int:
    with open(args.doc) as f:
        doc = json.load(f)
    d, universes = load_json_snapshot(doc)
    if universes['nodes'].arity != 1:
        print('nodes universe must be K=1', file=sys.stderr)
        return 2
    if universes['edges'].arity != 3:
        print('edges universe must be K=3', file=sys.stderr)
        return 2
    print(f'{args.doc}: OK '
          f'({universes["nodes"].support_size} nodes, '
          f'{universes["edges"].support_size} edges, '
          f'{len(d)} interned ids)')
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    with open(args.doc) as f:
        doc = json.load(f)
    d, universes = load_json_snapshot(doc)
    print(f'kind:          {doc.get("kind")}')
    print(f'id:            {doc.get("id")}')
    print(f'dictionary:    {len(d)} interned ids')
    print(f'nodes (K=1):   P={universes["nodes"].P}, support={universes["nodes"].support_size}')
    print(f'edges (K=3):   P={universes["edges"].P}, support={universes["edges"].support_size}')
    return 0


def _cmd_kquery(args: argparse.Namespace) -> int:
    # Build a small in-memory K=1 universe from the --U list of ids,
    # then mark per-id presence in A and B.
    u_ids = args.U
    a_set = set(args.A)
    b_set = set(args.B)
    # Numeric ids only here; for strings, the user should pre-intern.
    # (CLI ergonomics: small enough to pass on the command line.)
    try:
        u_int = [int(x) for x in u_ids]
        a_int = {int(x) for x in a_set}
        b_int = {int(x) for x in b_set}
    except ValueError:
        print('--U / --A / --B values must be integers (Ids)', file=sys.stderr)
        return 2
    U = ReferentUniverseTensor.from_ids(u_int)
    A_live = U.contains(list(a_int))
    B_live = U.contains(list(b_int))
    cover = kquery(U, A_live, B_live)
    sizes = cover.cell_sizes()
    print(f'|U|={U.support_size}  cells: 11={sizes["11"]} 10={sizes["10"]} 01={sizes["01"]} 00={sizes["00"]}')
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    print(f'recognizer {args.recognizer!r}: not yet implemented in CLI')
    print('use the Python API directly: from vcif_hlo import resolve_references_dag, ...')
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='vcif-hlo', description='vcif-hlo CLI')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_validate = sub.add_parser('validate', help='tensor-shape conformance over a JSON snapshot')
    p_validate.add_argument('doc')
    p_validate.set_defaults(fn=_cmd_validate)

    p_inspect = sub.add_parser('inspect', help='universe shapes + sizes')
    p_inspect.add_argument('doc')
    p_inspect.set_defaults(fn=_cmd_inspect)

    p_kquery = sub.add_parser('kquery', help='ad-hoc V₄ classification')
    p_kquery.add_argument('--U', nargs='*', required=True, help='universe ids (ints)')
    p_kquery.add_argument('--A', nargs='*', required=True, help='left observer ids (ints, must be subset of U)')
    p_kquery.add_argument('--B', nargs='*', required=True, help='right observer ids (ints, must be subset of U)')
    p_kquery.set_defaults(fn=_cmd_kquery)

    p_run = sub.add_parser('run', help='execute a named recognizer (stub)')
    p_run.add_argument('recognizer')
    p_run.set_defaults(fn=_cmd_run)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
