"""QueryDAG composition tests."""
from __future__ import annotations

import numpy as np
import pytest

from vcif_hlo import QueryDAG, ReferentUniverseTensor


def test_minimal_dag_with_one_kquery():
    """Build a 4-node DAG: universe + two masks + kquery + 11 projection."""
    dag = QueryDAG()
    dag.add_universe('U', lambda env: env['U_in'])
    dag.add_mask('A', 'U', lambda env: env['A_in'])
    dag.add_mask('B', 'U', lambda env: env['B_in'])
    dag.add_kquery('cov', universe='U', A_mask='A', B_mask='B')
    dag.add_projection('mask_11', cover='cov', kind='11')

    U = ReferentUniverseTensor.from_ids([10, 20, 30])
    A_in = U.contains([10, 20])
    B_in = U.contains([20, 30])
    out = dag.evaluate({'U_in': U, 'A_in': A_in, 'B_in': B_in})

    # Only id=20 is in both
    assert (out['mask_11'] == np.array([False, True, False])).all()


def test_dag_chained_projections():
    """One cover, multiple projection nodes."""
    dag = QueryDAG()
    dag.add_universe('U', lambda env: env['U'])
    dag.add_mask('A', 'U', lambda env: env['A'])
    dag.add_mask('B', 'U', lambda env: env['B'])
    dag.add_kquery('cov', 'U', 'A', 'B')
    dag.add_projection('p11', cover='cov', kind='11')
    dag.add_projection('p10', cover='cov', kind='10')
    dag.add_projection('p01', cover='cov', kind='01')
    dag.add_projection('p00', cover='cov', kind='00')

    U = ReferentUniverseTensor.from_ids([1, 2, 3, 4])
    out = dag.evaluate({
        'U': U,
        'A': U.contains([1, 2]),
        'B': U.contains([2, 3]),
    })
    assert int(out['p11'].sum()) == 1   # {2}
    assert int(out['p10'].sum()) == 1   # {1}
    assert int(out['p01'].sum()) == 1   # {3}
    assert int(out['p00'].sum()) == 1   # {4}


def test_dag_rejects_missing_universe():
    dag = QueryDAG()
    with pytest.raises(KeyError):
        dag.add_mask('A', 'no_such_universe', lambda env: None)


def test_dag_rejects_missing_kquery_inputs():
    dag = QueryDAG()
    dag.add_universe('U', lambda env: env['U'])
    with pytest.raises(KeyError):
        dag.add_kquery('cov', 'U', 'no_A', 'no_B')


def test_dag_rejects_unknown_projection_kind():
    dag = QueryDAG()
    dag.add_universe('U', lambda env: env['U'])
    dag.add_mask('A', 'U', lambda env: env['A'])
    dag.add_mask('B', 'U', lambda env: env['B'])
    dag.add_kquery('cov', 'U', 'A', 'B')
    with pytest.raises(ValueError):
        dag.add_projection('bad', cover='cov', kind='42')


def test_dag_rejects_duplicate_node_name():
    dag = QueryDAG()
    dag.add_universe('U', lambda env: env['U'])
    with pytest.raises(ValueError):
        dag.add_universe('U', lambda env: env['U'])


def test_dag_mask_shape_mismatch_at_eval():
    """A mask whose build_fn returns wrong-shape array should error at eval."""
    dag = QueryDAG()
    dag.add_universe('U', lambda env: env['U'])
    dag.add_mask('bad_A', 'U', lambda env: np.array([True, True]))  # too short
    with pytest.raises(ValueError):
        U = ReferentUniverseTensor.from_ids([1, 2, 3])
        dag.evaluate({'U': U})
