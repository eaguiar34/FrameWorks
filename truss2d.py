from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from core.model import Model2D

@dataclass
class TrussResult2D:
    u: np.ndarray
    reactions: Dict[Tuple[str, str], float]
    elem_axial: Dict[str, float]
    nodal_equilibrium_sum: Dict[str, tuple[float,float]]

def _ke_truss(E: float, A: float, xi: float, yi: float, xj: float, yj: float):
    dx, dy = xj-xi, yj-yi
    L = float(np.hypot(dx, dy))
    if L <= 0.0:
        raise ValueError("Zero-length truss element.")
    c, s = dx/L, dy/L
    ke = (E*A/L) * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s],
    ], dtype=float)
    return ke, L, c, s

def solve_truss_2d(model: Model2D, loadcase) -> TrussResult2D:
    dof = model.dof_map_truss()
    ndof = 2*len(model.nodes)
    K = np.zeros((ndof, ndof), dtype=float)

    for eid, e in model.elements.items():
        if e.etype != "TRUSS2D":
            continue
        mat = model.materials[e.material]
        sec = model.sections[e.section]
        ni = model.nodes[e.ni]; nj = model.nodes[e.nj]
        ke, _, _, _ = _ke_truss(mat.E, sec.A, ni.x, ni.y, nj.x, nj.y)
        edofs = [dof[(e.ni,"ux")], dof[(e.ni,"uy")], dof[(e.nj,"ux")], dof[(e.nj,"uy")]]
        for a in range(4):
            for b in range(4):
                K[edofs[a], edofs[b]] += ke[a,b]

    F = np.zeros(ndof, dtype=float)
    for nid, (Fx, Fy, _) in loadcase.nodal_forces.items():
        F[dof[(nid,"ux")]] += Fx
        F[dof[(nid,"uy")]] += Fy

    fixed: List[int] = []
    for s in model.supports:
        if s.ux_fixed: fixed.append(dof[(s.node_id,"ux")])
        if s.uy_fixed: fixed.append(dof[(s.node_id,"uy")])
    fixed = sorted(set(fixed))
    free = [i for i in range(ndof) if i not in fixed]
    if not free:
        raise ValueError("No free DOFs.")

    Kff = K[np.ix_(free, free)]
    Ff = F[free]
    cond = np.linalg.cond(Kff)
    if not np.isfinite(cond) or cond > 1e12:
        raise ValueError(f"Truss unstable/ill-conditioned (cond={cond:.2e}).")

    uf = np.linalg.solve(Kff, Ff)
    u = np.zeros(ndof, dtype=float)
    u[free] = uf

    R = K @ u - F
    inv = {v:k for k,v in dof.items()}
    reactions: Dict[Tuple[str,str], float] = {}
    for idx in fixed:
        nid, dofname = inv[idx]
        reactions[(nid, dofname)] = float(R[idx])

    elem_axial: Dict[str, float] = {}
    for eid, e in model.elements.items():
        if e.etype != "TRUSS2D":
            continue
        mat = model.materials[e.material]
        sec = model.sections[e.section]
        ni = model.nodes[e.ni]; nj = model.nodes[e.nj]
        _, L, c, s = _ke_truss(mat.E, sec.A, ni.x, ni.y, nj.x, nj.y)
        edofs = np.array([dof[(e.ni,"ux")], dof[(e.ni,"uy")], dof[(e.nj,"ux")], dof[(e.nj,"uy")]], dtype=int)
        ue = u[edofs]
        axial_def = np.array([-c, -s, c, s], dtype=float) @ ue
        N = (mat.E*sec.A/L) * axial_def
        elem_axial[eid] = float(N)

    # Joint force extraction hook: sum of member forces + applied loads (supports won't be ~0)
    nodal_sum: Dict[str, list[float]] = {nid:[0.0,0.0] for nid in model.nodes}
    for eid, N in elem_axial.items():
        e = model.elements[eid]
        ni = model.nodes[e.ni]; nj = model.nodes[e.nj]
        dx, dy = nj.x-ni.x, nj.y-ni.y
        L = float(np.hypot(dx,dy))
        c, s = dx/L, dy/L
        fi = (-N*c, -N*s)
        fj = ( N*c,  N*s)
        nodal_sum[e.ni][0] += fi[0]; nodal_sum[e.ni][1] += fi[1]
        nodal_sum[e.nj][0] += fj[0]; nodal_sum[e.nj][1] += fj[1]

    for nid, (Fx, Fy, _) in loadcase.nodal_forces.items():
        nodal_sum[nid][0] += Fx
        nodal_sum[nid][1] += Fy

    nodal_sum_out = {nid:(vals[0], vals[1]) for nid, vals in nodal_sum.items()}
    return TrussResult2D(u=u, reactions=reactions, elem_axial=elem_axial, nodal_equilibrium_sum=nodal_sum_out)
