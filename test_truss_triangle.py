import numpy as np
from core.model import Model2D, Node2D, Material, Section2D, Element2D, Support2D
from core.loads import LoadCase2D
from solver.truss2d import solve_truss_2d

def test_triangle_truss_symmetry_and_balance():
    m = Model2D()
    m.add_material(Material("STEEL", E=29000.0))
    m.add_section(Section2D("S", A=1.0, Izz=1.0, I_min=1.0))

    m.add_node(Node2D("N1", 0.0, 0.0))
    m.add_node(Node2D("N2", 10.0, 0.0))
    m.add_node(Node2D("N3", 5.0, 5.0))

    m.add_element(Element2D("E1", "TRUSS2D", "N1", "N3", "STEEL", "S"))
    m.add_element(Element2D("E2", "TRUSS2D", "N3", "N2", "STEEL", "S"))
    m.add_element(Element2D("E3", "TRUSS2D", "N1", "N2", "STEEL", "S"))

    m.add_support(Support2D("N1", ux_fixed=True, uy_fixed=True))
    m.add_support(Support2D("N2", ux_fixed=False, uy_fixed=True))

    lc = LoadCase2D("LC1")
    lc.add_nodal_force("N3", Fy=-10.0)

    res = solve_truss_2d(m, lc)

    assert np.isclose(res.elem_axial["E1"], res.elem_axial["E2"], rtol=1e-6, atol=1e-6)
    Ry = res.reactions[("N1","uy")] + res.reactions[("N2","uy")]
    assert np.isclose(Ry, 10.0, rtol=1e-6, atol=1e-6)
