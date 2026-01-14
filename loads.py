from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Literal

CoordSystem = Literal["GLOBAL", "LOCAL"]
Direction = Literal["X", "Y", "Z"]
LoadKind = Literal["UDL", "POINT", "MOMENT"]

@dataclass(frozen=True)
class MemberLoad2D:
    kind: LoadKind
    coord: CoordSystem
    direction: Direction
    value: float
    a: float = 0.0
    b: Optional[float] = None

@dataclass
class LoadCase2D:
    name: str
    # Nodal loads: Fx,Fy,Mz (Mz about global z)
    nodal_forces: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    # Member loads (FRAME2D)
    member_loads: Dict[str, List[MemberLoad2D]] = field(default_factory=dict)

    include_self_weight: bool = False
    gravity_y: float = -1.0

    def add_nodal_force(self, node_id: str, Fx: float = 0.0, Fy: float = 0.0, Mz: float = 0.0) -> None:
        if node_id in self.nodal_forces:
            fx0, fy0, mz0 = self.nodal_forces[node_id]
            self.nodal_forces[node_id] = (fx0 + Fx, fy0 + Fy, mz0 + Mz)
        else:
            self.nodal_forces[node_id] = (Fx, Fy, Mz)

    def _add_member_load(self, element_id: str, ml: MemberLoad2D) -> None:
        self.member_loads.setdefault(element_id, []).append(ml)

    # Distributed loads (partial span [a,b])
    def add_member_udl_global_y(self, element_id: str, qy: float, a: float = 0.0, b: Optional[float] = None) -> None:
        self._add_member_load(element_id, MemberLoad2D("UDL","GLOBAL","Y",float(qy),float(a), None if b is None else float(b)))

    def add_member_udl_global_x(self, element_id: str, qx: float, a: float = 0.0, b: Optional[float] = None) -> None:
        self._add_member_load(element_id, MemberLoad2D("UDL","GLOBAL","X",float(qx),float(a), None if b is None else float(b)))

    def add_member_udl_local_y(self, element_id: str, qy: float, a: float = 0.0, b: Optional[float] = None) -> None:
        self._add_member_load(element_id, MemberLoad2D("UDL","LOCAL","Y",float(qy),float(a), None if b is None else float(b)))

    def add_member_udl_local_x(self, element_id: str, qx: float, a: float = 0.0, b: Optional[float] = None) -> None:
        self._add_member_load(element_id, MemberLoad2D("UDL","LOCAL","X",float(qx),float(a), None if b is None else float(b)))

    # Point forces
    def add_member_point_global_y(self, element_id: str, P: float, a: float) -> None:
        self._add_member_load(element_id, MemberLoad2D("POINT","GLOBAL","Y",float(P),float(a),None))

    def add_member_point_global_x(self, element_id: str, P: float, a: float) -> None:
        self._add_member_load(element_id, MemberLoad2D("POINT","GLOBAL","X",float(P),float(a),None))

    def add_member_point_local_y(self, element_id: str, P: float, a: float) -> None:
        self._add_member_load(element_id, MemberLoad2D("POINT","LOCAL","Y",float(P),float(a),None))

    def add_member_point_local_x(self, element_id: str, P: float, a: float) -> None:
        self._add_member_load(element_id, MemberLoad2D("POINT","LOCAL","X",float(P),float(a),None))

    # Point moment (couple) about Z at position a
    def add_member_point_moment(self, element_id: str, Mz: float, a: float, coord: CoordSystem = "LOCAL") -> None:
        self._add_member_load(element_id, MemberLoad2D("MOMENT",coord,"Z",float(Mz),float(a),None))

@dataclass
class LoadCombo2D:
    name: str
    factors: Dict[str, float]

def combine_loadcases(cases: Dict[str, LoadCase2D], combo: LoadCombo2D) -> LoadCase2D:
    out = LoadCase2D(combo.name)
    for lc_name, fac in combo.factors.items():
        lc = cases[lc_name]
        for nid, (fx, fy, mz) in lc.nodal_forces.items():
            out.add_nodal_force(nid, Fx=fac*fx, Fy=fac*fy, Mz=fac*mz)
        for eid, loads in lc.member_loads.items():
            for ml in loads:
                out._add_member_load(eid, MemberLoad2D(
                    kind=ml.kind, coord=ml.coord, direction=ml.direction,
                    value=fac*ml.value, a=ml.a, b=ml.b
                ))
        out.include_self_weight = out.include_self_weight or lc.include_self_weight
    return out
