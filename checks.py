from __future__ import annotations
from typing import Dict
import numpy as np
from core.model import Model2D

def euler_buckling_capacity(E: float, I: float, L: float, K: float = 1.0) -> float:
    if E <= 0 or I <= 0 or L <= 0 or K <= 0:
        return float("nan")
    return (np.pi**2) * E * I / ((K*L)**2)

def truss_buckling_utilization(model: Model2D, elem_axial: Dict[str, float]) -> Dict[str, float]:
    util: Dict[str, float] = {}
    for eid, N in elem_axial.items():
        e = model.elements[eid]
        if e.etype != "TRUSS2D":
            continue
        if N >= 0:
            util[eid] = 0.0
            continue
        mat = model.materials[e.material]
        sec = model.sections[e.section]
        if sec.I_min is None:
            util[eid] = float("nan")
            continue
        L = model.element_length(eid)
        Pcr = euler_buckling_capacity(mat.E, sec.I_min, L, e.K)
        util[eid] = float(abs(N)/Pcr) if np.isfinite(Pcr) and Pcr > 0 else float("nan")
    return util

def frame_basic_stress_demands(model: Model2D, end_forces_local: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for eid, fL in end_forces_local.items():
        e = model.elements[eid]
        if e.etype != "FRAME2D":
            continue
        sec = model.sections[e.section]
        Ni, Vi, Mi, Nj, Vj, Mj = [float(x) for x in fL.tolist()]
        Nmax = max(abs(Ni), abs(Nj))
        Vmax = max(abs(Vi), abs(Vj))
        Mmax = max(abs(Mi), abs(Mj))
        sigma_axial = Nmax / sec.A if sec.A > 0 else float("nan")
        M_over_I = Mmax / sec.Izz if sec.Izz > 0 else float("nan")
        out[eid] = dict(Nmax=Nmax, Vmax=Vmax, Mmax=Mmax, sigma_axial=sigma_axial, M_over_I=M_over_I)
    return out
