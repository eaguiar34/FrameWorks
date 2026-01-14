from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass(frozen=True)
class LocalMemberLoad:
    kind: str            # "UDL" | "POINT" | "MOMENT"
    direction: str       # "X" | "Y" | "Z" in LOCAL coords
    value: float         # UDL: force/length; POINT: force; MOMENT: moment
    a: float = 0.0
    b: Optional[float] = None

def _hermite_shapes(x: float, L: float) -> np.ndarray:
    xi = x / L
    N1 = 1 - 3*xi**2 + 2*xi**3
    N2 = L*(xi - 2*xi**2 + xi**3)
    N3 = 3*xi**2 - 2*xi**3
    N4 = L*(-xi**2 + xi**3)
    return np.array([N1, N2, N3, N4], dtype=float)  # maps to [v1,th1,v2,th2]

def _hermite_dshapes_dx(x: float, L: float) -> np.ndarray:
    """dN/dx for the Hermite displacement shapes above."""
    xi = x / L
    dN1dx = (-6*xi + 6*xi**2) / L
    dN2dx = (1 - 4*xi + 3*xi**2)
    dN3dx = (6*xi - 6*xi**2) / L
    dN4dx = (-2*xi + 3*xi**2)
    return np.array([dN1dx, dN2dx, dN3dx, dN4dx], dtype=float)

def _axial_shapes(x: float, L: float) -> np.ndarray:
    xi = x / L
    return np.array([1 - xi, xi], dtype=float)  # maps to [u1,u2]

def consistent_nodal_loads_local(L: float, loads_local: List[LocalMemberLoad], n_gauss: int = 4) -> np.ndarray:
    """Return equivalent nodal load vector in LOCAL coords [Fx_i, Fy_i, M_i, Fx_j, Fy_j, M_j]."""
    fe = np.zeros(6, dtype=float)
    if not loads_local:
        return fe

    pts, wts = np.polynomial.legendre.leggauss(max(2, int(n_gauss)))

    for ld in loads_local:
        if ld.kind == "UDL":
            a = max(0.0, min(L, float(ld.a)))
            b = float(L) if ld.b is None else max(0.0, min(L, float(ld.b)))
            if b <= a:
                continue
            for t, wt in zip(pts, wts):
                x = 0.5*(b-a)*t + 0.5*(a+b)
                dx = 0.5*(b-a)
                if ld.direction == "Y":
                    Nv = _hermite_shapes(x, L)
                    fe[1] += Nv[0] * ld.value * dx * wt
                    fe[2] += Nv[1] * ld.value * dx * wt
                    fe[4] += Nv[2] * ld.value * dx * wt
                    fe[5] += Nv[3] * ld.value * dx * wt
                elif ld.direction == "X":
                    Nu = _axial_shapes(x, L)
                    fe[0] += Nu[0] * ld.value * dx * wt
                    fe[3] += Nu[1] * ld.value * dx * wt
                else:
                    raise ValueError("UDL direction must be X or Y")
        elif ld.kind == "POINT":
            x = max(0.0, min(L, float(ld.a)))
            if ld.direction == "Y":
                Nv = _hermite_shapes(x, L)
                fe[1] += Nv[0] * ld.value
                fe[2] += Nv[1] * ld.value
                fe[4] += Nv[2] * ld.value
                fe[5] += Nv[3] * ld.value
            elif ld.direction == "X":
                Nu = _axial_shapes(x, L)
                fe[0] += Nu[0] * ld.value
                fe[3] += Nu[1] * ld.value
            else:
                raise ValueError("POINT direction must be X or Y")
        elif ld.kind == "MOMENT":
            # Point couple about local Z; does work on rotation theta(x)=dv/dx
            x = max(0.0, min(L, float(ld.a)))
            if ld.direction != "Z":
                raise ValueError("MOMENT direction must be Z")
            dNdx = _hermite_dshapes_dx(x, L)  # maps to [v1,th1,v2,th2]
            fe[1] += dNdx[0] * ld.value
            fe[2] += dNdx[1] * ld.value
            fe[4] += dNdx[2] * ld.value
            fe[5] += dNdx[3] * ld.value
        else:
            raise ValueError("Unknown load kind")

    return fe

def to_local_from_global_vec(gx: float, gy: float, c: float, s: float) -> Tuple[float, float]:
    """Given a GLOBAL vector (gx,gy), return (local_x, local_y)."""
    # local = [[c, s],[-s, c]] * global
    return float(c*gx + s*gy), float(-s*gx + c*gy)
