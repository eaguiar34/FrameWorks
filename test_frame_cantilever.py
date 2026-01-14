import numpy as np
from core.model import Model2D, Node2D, Material, Section2D, Element2D, Support2D
from core.loads import LoadCase2D
from solver.frame2d import solve_frame_2d

def test_cantilever_tip_deflection_matches_closed_form():
    E = 200000.0
    I = 8.0e6
    A = 1.0e4
    L = 3000.0
    P = -10000.0

    m = Model2D()
    m.add_material(Material("STEEL", E=E))
    m.add_section(Section2D("S", A=A, Izz=I))

    m.add_node(Node2D("N1", 0.0, 0.0))
    m.add_node(Node2D("N2", L, 0.0))

    m.add_element(Element2D("E1", "FRAME2D", "N1", "N2", "STEEL", "S"))
    m.add_support(Support2D("N1", ux_fixed=True, uy_fixed=True, rz_fixed=True))

    lc = LoadCase2D("LC1")
    lc.add_nodal_force("N2", Fy=P)

    res = solve_frame_2d(m, lc)
    dof = m.dof_map_frame()
    uy_tip = res.u[dof[("N2","uy")]]

    delta_expected = P * (L**3) / (3 * E * I)
    assert np.isclose(uy_tip, delta_expected, rtol=1e-3, atol=1e-6)
