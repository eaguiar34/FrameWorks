import numpy as np
from core.model import Model2D, Node2D, Material, Section2D, Element2D, Support2D
from core.loads import LoadCase2D
from solver.frame2d import solve_frame_2d

def test_simply_supported_midspan_couple_reactions():
    # Simply supported beam: couple at midspan => R1 = +M/L, R2 = -M/L (sum Fy = 0)
    E = 29000.0
    A = 10.0
    I = 1000.0
    L = 10.0
    M0 = 12.0

    m = Model2D()
    m.add_material(Material("STEEL", E=E, gamma=None))
    m.add_section(Section2D("S", A=A, Izz=I))

    m.add_node(Node2D("N1", 0.0, 0.0))
    m.add_node(Node2D("N2", L, 0.0))
    m.add_element(Element2D("E1", "FRAME2D", "N1", "N2", "STEEL", "S"))

    # Pin at left (ux,uy fixed), roller at right (uy fixed). Rotations free.
    m.add_support(Support2D("N1", ux_fixed=True, uy_fixed=True, rz_fixed=False))
    m.add_support(Support2D("N2", ux_fixed=False, uy_fixed=True, rz_fixed=False))

    lc = LoadCase2D("LC1")
    lc.add_member_point_moment("E1", Mz=M0, a=L/2, coord="LOCAL")

    res = solve_frame_2d(m, lc)
    R1 = res.reactions[("N1","uy")]
    R2 = res.reactions[("N2","uy")]

    assert np.isclose(R1,  M0/L, rtol=1e-6, atol=1e-6)
    assert np.isclose(R2, -M0/L, rtol=1e-6, atol=1e-6)
    assert np.isclose(R1 + R2, 0.0, rtol=1e-6, atol=1e-6)
