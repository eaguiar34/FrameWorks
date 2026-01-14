from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
from solver.member_loads import LocalMemberLoad

@dataclass
class ElementDiagrams:
    x: np.ndarray
    N: np.ndarray
    V: np.ndarray
    M: np.ndarray

@dataclass
class DiagramEnvelope:
    x: np.ndarray
    N_max: np.ndarray
    N_min: np.ndarray
    V_max: np.ndarray
    V_min: np.ndarray
    M_max: np.ndarray
    M_min: np.ndarray
    controls: Dict[str, Dict[str, str]]  # {"N_max": {"combo": "..."} ...}

def diagrams_from_end_forces_and_loads(
    L: float,
    end_forces_local: np.ndarray,
    loads_local: Optional[List[LocalMemberLoad]] = None,
    n_points: int = 200,
) -> ElementDiagrams:
    loads_local = loads_local or []
    Ni, Vi, Mi, Nj, Vj, Mj = [float(v) for v in end_forces_local.tolist()]
    x = np.linspace(0.0, float(L), int(max(2, n_points)))

    qx = np.zeros_like(x)
    qy = np.zeros_like(x)

    for ld in loads_local:
        if ld.kind != "UDL":
            continue
        a = max(0.0, min(L, float(ld.a)))
        b = float(L) if ld.b is None else max(0.0, min(L, float(ld.b)))
        if b <= a:
            continue
        mask = (x >= a) & (x <= b)
        if ld.direction == "X":
            qx[mask] += float(ld.value)
        elif ld.direction == "Y":
            qy[mask] += float(ld.value)

    pts = [(max(0.0, min(L, float(ld.a))), ld.direction, float(ld.value))
           for ld in loads_local if ld.kind == "POINT"]
    moms = [(max(0.0, min(L, float(ld.a))), float(ld.value))
            for ld in loads_local if ld.kind == "MOMENT" and ld.direction == "Z"]

    N = np.zeros_like(x)
    V = np.zeros_like(x)
    M = np.zeros_like(x)
    N[0] = Ni
    V[0] = Vi
    M[0] = Mi

    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        N[i] = N[i-1] - qx[i-1] * dx
        V[i] = V[i-1] - qy[i-1] * dx

        # point forces
        for xp, d, val in pts:
            if x[i-1] < xp <= x[i] + 1e-12:
                if d == "X":
                    N[i] -= val
                elif d == "Y":
                    V[i] -= val

        M[i] = M[i-1] + 0.5*(V[i-1] + V[i]) * dx

        # point moments cause a jump in M
        for xm, mm in moms:
            if x[i-1] < xm <= x[i] + 1e-12:
                M[i] += mm

    return ElementDiagrams(x=x, N=N, V=V, M=M)

def envelope_from_combo_diagrams(combo_diagrams: Dict[str, ElementDiagrams]) -> DiagramEnvelope:
    """Compute max/min envelopes across combos and track controlling combo per x for N/V/M."""
    combos = list(combo_diagrams.keys())
    if not combos:
        raise ValueError("No combo diagrams.")

    x = combo_diagrams[combos[0]].x
    stackN = np.vstack([combo_diagrams[c].N for c in combos])
    stackV = np.vstack([combo_diagrams[c].V for c in combos])
    stackM = np.vstack([combo_diagrams[c].M for c in combos])

    iNmax = np.argmax(stackN, axis=0); iNmin = np.argmin(stackN, axis=0)
    iVmax = np.argmax(stackV, axis=0); iVmin = np.argmin(stackV, axis=0)
    iMmax = np.argmax(stackM, axis=0); iMmin = np.argmin(stackM, axis=0)

    env = DiagramEnvelope(
        x=x,
        N_max=stackN[iNmax, np.arange(len(x))],
        N_min=stackN[iNmin, np.arange(len(x))],
        V_max=stackV[iVmax, np.arange(len(x))],
        V_min=stackV[iVmin, np.arange(len(x))],
        M_max=stackM[iMmax, np.arange(len(x))],
        M_min=stackM[iMmin, np.arange(len(x))],
        controls={
            "N_max": {"combo": ",".join(sorted(set(combos[i] for i in iNmax)))},
            "N_min": {"combo": ",".join(sorted(set(combos[i] for i in iNmin)))},
            "V_max": {"combo": ",".join(sorted(set(combos[i] for i in iVmax)))},
            "V_min": {"combo": ",".join(sorted(set(combos[i] for i in iVmin)))},
            "M_max": {"combo": ",".join(sorted(set(combos[i] for i in iMmax)))},
            "M_min": {"combo": ",".join(sorted(set(combos[i] for i in iMmin)))},
        }
    )
    return env

def controlling_combo_extremes(combo_diagrams: Dict[str, ElementDiagrams]) -> Dict[str, Tuple[str, float]]:
    """Return controlling combo for global extreme values along member for N,V,M (max abs, max, min)."""
    out: Dict[str, Tuple[str, float]] = {}
    for key, arr_name in [("N","N"),("V","V"),("M","M")]:
        best_abs = ("", -np.inf)
        best_max = ("", -np.inf)
        best_min = ("", np.inf)
        for cname, di in combo_diagrams.items():
            arr = getattr(di, arr_name)
            vmax = float(np.max(arr)); vmin = float(np.min(arr))
            vabs = float(np.max(np.abs(arr)))
            if vabs > best_abs[1]:
                best_abs = (cname, vabs)
            if vmax > best_max[1]:
                best_max = (cname, vmax)
            if vmin < best_min[1]:
                best_min = (cname, vmin)
        out[f"{key}_max_abs"] = best_abs
        out[f"{key}_max"] = best_max
        out[f"{key}_min"] = best_min
    return out
