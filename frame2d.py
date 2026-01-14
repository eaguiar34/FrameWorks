from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from core.model import Model2D
from core.loads import LoadCase2D
from solver.member_loads import LocalMemberLoad, consistent_nodal_loads_local, to_local_from_global_vec

@dataclass
class FrameResult2D:
    u: np.ndarray
    reactions: Dict[Tuple[str, str], float]
    end_forces_local: Dict[str, np.ndarray]  # [Ni,Vi,Mi,Nj,Vj,Mj]
    member_loads_local: Dict[str, List[LocalMemberLoad]]
    max_disp: float

def _frame_mats(E: float, A: float, I: float, xi: float, yi: float, xj: float, yj: float):
    dx, dy = xj-xi, yj-yi
    L = float(np.hypot(dx, dy))
    if L <= 0.0:
        raise ValueError("Zero-length frame element.")
    c, s = dx/L, dy/L

    EA_L = E*A / L
    EI = E*I
    kL = np.array([
        [ EA_L,    0.0,        0.0,    -EA_L,   0.0,        0.0],
        [ 0.0,   12*EI/L**3,  6*EI/L**2, 0.0,  -12*EI/L**3,  6*EI/L**2],
        [ 0.0,    6*EI/L**2,  4*EI/L,    0.0,   -6*EI/L**2,  2*EI/L],
        [-EA_L,   0.0,        0.0,     EA_L,   0.0,        0.0],
        [ 0.0,  -12*EI/L**3, -6*EI/L**2, 0.0,   12*EI/L**3, -6*EI/L**2],
        [ 0.0,    6*EI/L**2,  2*EI/L,    0.0,   -6*EI/L**2,  4*EI/L],
    ], dtype=float)

    T = np.array([
        [ c,  s, 0,  0,  0, 0],
        [-s,  c, 0,  0,  0, 0],
        [ 0,  0, 1,  0,  0, 0],
        [ 0,  0, 0,  c,  s, 0],
        [ 0,  0, 0, -s,  c, 0],
        [ 0,  0, 0,  0,  0, 1],
    ], dtype=float)

    kG = T.T @ kL @ T
    return L, c, s, kL, kG, T

def solve_frame_2d(model: Model2D, loadcase: LoadCase2D) -> FrameResult2D:
    dof = model.dof_map_frame()
    ndof = 3*len(model.nodes)
    K = np.zeros((ndof, ndof), dtype=float)
    F = np.zeros(ndof, dtype=float)

    for nid, (Fx, Fy, Mz) in loadcase.nodal_forces.items():
        F[dof[(nid,"ux")]] += Fx
        F[dof[(nid,"uy")]] += Fy
        F[dof[(nid,"rz")]] += Mz

    cache = {}
    member_loads_local: Dict[str, List[LocalMemberLoad]] = {}

    for eid, e in model.elements.items():
        if e.etype != "FRAME2D":
            continue
        mat = model.materials[e.material]
        sec = model.sections[e.section]
        ni = model.nodes[e.ni]; nj = model.nodes[e.nj]
        L, c, s, kL, kG, T = _frame_mats(mat.E, sec.A, sec.Izz, ni.x, ni.y, nj.x, nj.y)
        cache[eid] = (L, c, s, kL, T, mat, sec)

        edofs = [
            dof[(e.ni,"ux")], dof[(e.ni,"uy")], dof[(e.ni,"rz")],
            dof[(e.nj,"ux")], dof[(e.nj,"uy")], dof[(e.nj,"rz")],
        ]
        for a in range(6):
            for b in range(6):
                K[edofs[a], edofs[b]] += kG[a,b]

        loads_here: List[LocalMemberLoad] = []
        for ml in loadcase.member_loads.get(eid, []):
            if ml.coord == "LOCAL":
                loads_here.append(LocalMemberLoad(kind=ml.kind, direction=ml.direction, value=float(ml.value), a=float(ml.a), b=ml.b))
            else:
                # GLOBAL
                if ml.kind in ("UDL","POINT"):
                    gx = float(ml.value) if ml.direction == "X" else 0.0
                    gy = float(ml.value) if ml.direction == "Y" else 0.0
                    lx, ly = to_local_from_global_vec(gx=gx, gy=gy, c=c, s=s)
                    if abs(lx) > 0:
                        loads_here.append(LocalMemberLoad(kind=ml.kind, direction="X", value=lx, a=float(ml.a), b=ml.b))
                    if abs(ly) > 0:
                        loads_here.append(LocalMemberLoad(kind=ml.kind, direction="Y", value=ly, a=float(ml.a), b=ml.b))
                elif ml.kind == "MOMENT":
                    # In 2D, global/local Z are the same axis
                    loads_here.append(LocalMemberLoad(kind="MOMENT", direction="Z", value=float(ml.value), a=float(ml.a), b=None))
                else:
                    raise ValueError("Unknown member load kind")

        # self weight (global Y full span)
        if loadcase.include_self_weight:
            if mat.gamma is None:
                raise ValueError(f"Self-weight requested but material '{mat.name}' has gamma=None.")
            qy_sw_global = mat.gamma * sec.A * float(loadcase.gravity_y)
            lx, ly = to_local_from_global_vec(gx=0.0, gy=qy_sw_global, c=c, s=s)
            if abs(lx) > 0:
                loads_here.append(LocalMemberLoad(kind="UDL", direction="X", value=lx, a=0.0, b=None))
            if abs(ly) > 0:
                loads_here.append(LocalMemberLoad(kind="UDL", direction="Y", value=ly, a=0.0, b=None))

        member_loads_local[eid] = loads_here

        feL = consistent_nodal_loads_local(L=L, loads_local=loads_here, n_gauss=4)
        if np.any(np.abs(feL) > 0):
            feG = T.T @ feL
            for i, gdof in enumerate(edofs):
                F[gdof] += feG[i]

    fixed: List[int] = []
    for spt in model.supports:
        if spt.ux_fixed: fixed.append(dof[(spt.node_id,"ux")])
        if spt.uy_fixed: fixed.append(dof[(spt.node_id,"uy")])
        if spt.rz_fixed: fixed.append(dof[(spt.node_id,"rz")])
    fixed = sorted(set(fixed))
    free = [i for i in range(ndof) if i not in fixed]
    if not free:
        raise ValueError("No free DOFs.")

    Kff = K[np.ix_(free, free)]
    Ff = F[free]
    cond = np.linalg.cond(Kff)
    if not np.isfinite(cond) or cond > 1e12:
        raise ValueError(f"Frame unstable/ill-conditioned (cond={cond:.2e}).")

    uf = np.linalg.solve(Kff, Ff)
    u = np.zeros(ndof, dtype=float)
    u[free] = uf

    R = K @ u - F
    inv = {v:k for k,v in dof.items()}
    reactions: Dict[Tuple[str,str], float] = {}
    for idx in fixed:
        nid, dofname = inv[idx]
        reactions[(nid, dofname)] = float(R[idx])

    end_forces_local: Dict[str, np.ndarray] = {}
    for eid, e in model.elements.items():
        if e.etype != "FRAME2D":
            continue
        L, c, s, kL, T, mat, sec = cache[eid]
        edofs = np.array([
            dof[(e.ni,"ux")], dof[(e.ni,"uy")], dof[(e.ni,"rz")],
            dof[(e.nj,"ux")], dof[(e.nj,"uy")], dof[(e.nj,"rz")],
        ], dtype=int)
        ueG = u[edofs]
        ueL = T @ ueG
        fL = kL @ ueL
        feL = consistent_nodal_loads_local(L=L, loads_local=member_loads_local.get(eid, []), n_gauss=4)
        end_forces_local[eid] = fL - feL

    return FrameResult2D(
        u=u,
        reactions=reactions,
        end_forces_local=end_forces_local,
        member_loads_local=member_loads_local,
        max_disp=float(np.max(np.abs(u))) if u.size else 0.0
    )
