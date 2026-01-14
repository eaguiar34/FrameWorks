from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class UnitSystem(str, Enum):
    KIP_IN_KSI = "kip-in-ksi"
    N_MM_MPA = "N-mm-MPa"

@dataclass(frozen=True)
class Units:
    system: UnitSystem

    @property
    def force(self) -> str:
        return "kip" if self.system == UnitSystem.KIP_IN_KSI else "N"

    @property
    def length(self) -> str:
        return "in" if self.system == UnitSystem.KIP_IN_KSI else "mm"

    @property
    def stress(self) -> str:
        return "ksi" if self.system == UnitSystem.KIP_IN_KSI else "MPa"

    @property
    def moment(self) -> str:
        return "kip-in" if self.system == UnitSystem.KIP_IN_KSI else "N-mm"

    @property
    def unit_weight(self) -> str:
        return f"{self.force}/{self.length}^3"

# Exact factors:
KIP_TO_N = 4448.2216152605
IN_TO_MM = 25.4
KSI_TO_MPA = KIP_TO_N / (IN_TO_MM**2)  # kip/in^2 -> N/mm^2 = MPa

def convert_force(val: float, src: UnitSystem, dst: UnitSystem) -> float:
    if src == dst:
        return float(val)
    if src == UnitSystem.KIP_IN_KSI and dst == UnitSystem.N_MM_MPA:
        return float(val) * KIP_TO_N
    if src == UnitSystem.N_MM_MPA and dst == UnitSystem.KIP_IN_KSI:
        return float(val) / KIP_TO_N
    raise ValueError("Unsupported unit conversion.")

def convert_length(val: float, src: UnitSystem, dst: UnitSystem) -> float:
    if src == dst:
        return float(val)
    if src == UnitSystem.KIP_IN_KSI and dst == UnitSystem.N_MM_MPA:
        return float(val) * IN_TO_MM
    if src == UnitSystem.N_MM_MPA and dst == UnitSystem.KIP_IN_KSI:
        return float(val) / IN_TO_MM
    raise ValueError("Unsupported unit conversion.")

def convert_stress(val: float, src: UnitSystem, dst: UnitSystem) -> float:
    if src == dst:
        return float(val)
    if src == UnitSystem.KIP_IN_KSI and dst == UnitSystem.N_MM_MPA:
        return float(val) * KSI_TO_MPA  # ksi -> MPa
    if src == UnitSystem.N_MM_MPA and dst == UnitSystem.KIP_IN_KSI:
        return float(val) / KSI_TO_MPA  # MPa -> ksi
    raise ValueError("Unsupported unit conversion.")

def convert_area(val: float, src: UnitSystem, dst: UnitSystem) -> float:
    if src == dst:
        return float(val)
    f = IN_TO_MM if (src == UnitSystem.KIP_IN_KSI and dst == UnitSystem.N_MM_MPA) else (1.0 / IN_TO_MM)
    return float(val) * (f**2)

def convert_inertia(val: float, src: UnitSystem, dst: UnitSystem) -> float:
    if src == dst:
        return float(val)
    f = IN_TO_MM if (src == UnitSystem.KIP_IN_KSI and dst == UnitSystem.N_MM_MPA) else (1.0 / IN_TO_MM)
    return float(val) * (f**4)

def convert_moment(val: float, src: UnitSystem, dst: UnitSystem) -> float:
    if src == dst:
        return float(val)
    return convert_force(val, src, dst) * (IN_TO_MM if (src == UnitSystem.KIP_IN_KSI and dst == UnitSystem.N_MM_MPA) else (1.0/IN_TO_MM))
