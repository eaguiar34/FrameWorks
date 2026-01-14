from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class BoltResult:
    forces: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    centroid: Tuple[float, float]
    J: float

def bolt_group_elastic(
    bolts_xy: List[Tuple[float, float]],
    Vx: float,
    Vy: float,
    Mz: float,
) -> BoltResult:
    """
    Elastic bolt group method (direct shear + torsional shear from moment).
    """
    if len(bolts_xy) < 1:
        raise ValueError("Need at least 1 bolt.")
    pts = np.array(bolts_xy, dtype=float)
    xbar = float(np.mean(pts[:,0]))
    ybar = float(np.mean(pts[:,1]))
    rel = pts - np.array([xbar, ybar], dtype=float)
    r2 = rel[:,0]**2 + rel[:,1]**2
    J = float(np.sum(r2))
    n = pts.shape[0]

    fx_d = np.full(n, Vx / n, dtype=float)
    fy_d = np.full(n, Vy / n, dtype=float)

    if abs(Mz) < 1e-12 or J < 1e-12:
        fx = fx_d
        fy = fy_d
        return BoltResult(forces=np.hypot(fx, fy), fx=fx, fy=fy, centroid=(xbar,ybar), J=J)

    rx = rel[:,0]
    ry = rel[:,1]
    r = np.sqrt(r2)
    tx = np.zeros(n, dtype=float)
    ty = np.zeros(n, dtype=float)
    mask = r > 0
    tx[mask] = -ry[mask] / r[mask]
    ty[mask] = rx[mask] / r[mask]

    Ft = (Mz * r) / J
    fx_t = Ft * tx
    fy_t = Ft * ty

    fx = fx_d + fx_t
    fy = fy_d + fy_t
    return BoltResult(forces=np.hypot(fx, fy), fx=fx, fy=fy, centroid=(xbar,ybar), J=J)
