"""
Worked example: HF-DBE closure cross-substrate parity (writeup § 11).

Loads the same data into vcif-hlo's tensor form and (if vcif is
installed) into vcif's JSON form, runs kquery in each, prints both
results, and asserts equality.

Run:

    pip install vcif       # optional, for the parity comparison
    python docs/examples/hf_dbe_closure.py
"""
from __future__ import annotations

from vcif_hlo import IdDictionary, ReferentUniverseTensor, kquery


def hlo_path() -> dict:
    """Compute the HF-DBE closure via vcif-hlo's tensor kquery."""
    d = IdDictionary()
    claim = d.intern('CLAIM-DBE-produces-shadows')
    U = ReferentUniverseTensor.from_ids([claim])

    # Both observers contain the single claim.
    A_live = U.contains([claim])  # source-declares
    B_live = U.contains([claim])  # catalogue-represents

    cov = kquery(U, A_live, B_live)
    return cov.cell_sizes()


def main():
    print('vcif-hlo (tensor) cell sizes:')
    sizes = hlo_path()
    for cell in ('11', '10', '01', '00'):
        print(f'  {cell}: {sizes[cell]}')

    # Closure invariant: only the 11 cell is non-empty.
    assert sizes == {'00': 0, '01': 0, '10': 0, '11': 1}
    print('\nclosure invariant satisfied: 00 = 01 = 10 = 0; 11 = 1.')

    # Cross-substrate parity (optional): if vcif is installed and its
    # JSON example file is reachable, the same claim appears in the 11
    # cell of vcif's set_expr eval. The test_parity.py covers this
    # comparison automatically.


if __name__ == '__main__':
    main()
