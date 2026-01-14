from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

@dataclass(frozen=True)
class SteelCheckAssumptions:
    phi_tension: float = 0.90
    phi_compression: float = 0.90
    phi_flexure: float = 0.90
    phi_shear: float = 1.00

def aisc_column_Fcr(E: float, Fy: float, KL_over_r: float) -> float:
    if E <= 0 or Fy <= 0 or KL_over_r <= 0:
        return float("nan")
    lambdac = (KL_over_r) * np.sqrt(Fy / (np.pi**2 * E))
    if not np.isfinite(lambdac):
        return float("nan")
    if lambdac <= 1.5:
        return (0.658 ** (lambdac**2)) * Fy
    return (0.877 / (lambdac**2)) * Fy

def steel_member_utilization(
    E: float,
    Fy: float,
    A: float,
    L: float,
    K: float,
    r: Optional[float],
    Szz: Optional[float],
    Avy: Optional[float],
    N_max: float,
    V_max: float,
    M_max: float,
    assum: SteelCheckAssumptions = SteelCheckAssumptions(),
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    phiTn = assum.phi_tension * Fy * A if A > 0 else float("nan")
    out["phiTn"] = phiTn
    out["util_tension"] = abs(N_max)/phiTn if np.isfinite(phiTn) and phiTn > 0 and N_max > 0 else 0.0

    if r is None or r <= 0:
        out["phiPn"] = float("nan")
        out["util_compression"] = float("nan")
        out["KLr"] = float("nan")
        out["Fcr"] = float("nan")
    else:
        KLr = (K*L)/r
        Fcr = aisc_column_Fcr(E=E, Fy=Fy, KL_over_r=KLr)
        Pn = Fcr * A
        phiPn = assum.phi_compression * Pn
        out["phiPn"] = phiPn
        out["KLr"] = KLr
        out["Fcr"] = Fcr
        out["util_compression"] = abs(N_max)/phiPn if np.isfinite(phiPn) and phiPn > 0 and N_max < 0 else 0.0

    if Szz is None or Szz <= 0:
        out["phiMn"] = float("nan")
        out["util_flexure"] = float("nan")
    else:
        phiMn = assum.phi_flexure * Fy * Szz
        out["phiMn"] = phiMn
        out["util_flexure"] = abs(M_max)/phiMn if np.isfinite(phiMn) and phiMn > 0 else float("nan")

    Av = Avy if (Avy is not None and Avy > 0) else A
    phiVn = assum.phi_shear * 0.6 * Fy * Av if Av and Av > 0 else float("nan")
    out["phiVn"] = phiVn
    out["util_shear"] = abs(V_max)/phiVn if np.isfinite(phiVn) and phiVn > 0 else float("nan")

    if np.isfinite(out.get("phiPn", float("nan"))) and out.get("phiPn", 0.0) > 0 and np.isfinite(out.get("phiMn", float("nan"))) and out.get("phiMn", 0.0) > 0:
        out["util_interaction_PM"] = abs(N_max)/out["phiPn"] + abs(M_max)/out["phiMn"]
    else:
        out["util_interaction_PM"] = float("nan")

    return out
