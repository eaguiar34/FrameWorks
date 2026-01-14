import numpy as np
from solver.diagrams import diagrams_from_end_forces_and_loads
from solver.member_loads import LocalMemberLoad

def test_diagrams_match_end_shear_with_uniform_load():
    L = 3.0
    Vi = 10.0
    Mi = 0.0
    fL = np.array([0.0, Vi, Mi, 0.0, 0.0, 0.0], dtype=float)
    q = 2.0
    loads = [LocalMemberLoad(kind="UDL", direction="Y", value=q, a=0.0, b=None)]
    di = diagrams_from_end_forces_and_loads(L=L, end_forces_local=fL, loads_local=loads, n_points=200)
    assert np.isclose(di.V[0], Vi)
    assert np.isclose(di.V[-1], Vi - q*L, atol=1e-6)
