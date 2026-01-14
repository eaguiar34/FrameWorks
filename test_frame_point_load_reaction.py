import numpy as np
from core.model import Model2D, Node2D, Material, Section2D, Element2D, Support2D
from core.loads import LoadCase2D
from solver.frame2d import solve_frame_2d

def test_cantilever_point_load_reaction_balance():
    E = 29000.0
    A = 10.0
    I = 1000.0
    L = 10.0
    P = -5.0

    m = Model2D()
    m.add_material(Material("STEEL", E=E, gamma=None))
    m.add_section(Section2D("S", A=A, Izz=I))

    m.add_node(Node2D("N1", 0.0, 0.0))
    m.add_node(Node2D("N2", L, 0.0))
    m.add_element(Element2D("E1", "FRAME2D", "N1", "N2", "STEEL", "S"))
    m.add_support(Support2D("N1", ux_fixed=True, uy_fixed=True, rz_fixed=True))

    lc = LoadCase2D("LC1")
    lc.add_member_point_local_y("E1", P=P, a=L*0.3)

    res = solve_frame_2d(m, lc)
    Ry = res.reactions[("N1","uy")]
    assert np.isclose(Ry, -P, rtol=1e-6, atol=1e-6)
