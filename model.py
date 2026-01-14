from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Literal
import numpy as np

ElementType = Literal["TRUSS2D", "FRAME2D"]

@dataclass(frozen=True)
class Material:
    name: str
    E: float
    gamma: Optional[float] = None  # force/length^3 in chosen base units


@dataclass(frozen=True)
class Section2D:
    name: str
    A: float
    Izz: float
    # Optional properties for more realistic checks
    I_min: Optional[float] = None      # weak-axis inertia (for buckling); fallback to Izz if None
    Szz: Optional[float] = None        # section modulus (for flexure capacity); if None flexure check is skipped
    Avy: Optional[float] = None        # shear area in local y; if None we use A


@dataclass(frozen=True)
class Node2D:
    id: str
    x: float
    y: float

@dataclass(frozen=True)
class Support2D:
    node_id: str
    ux_fixed: bool = True
    uy_fixed: bool = True
    rz_fixed: bool = False

@dataclass(frozen=True)
class Element2D:
    id: str
    etype: ElementType
    ni: str
    nj: str
    material: str
    section: str
    group: str = "DEFAULT"
    K: float = 1.0

@dataclass
class Model2D:
    nodes: Dict[str, Node2D] = field(default_factory=dict)
    materials: Dict[str, Material] = field(default_factory=dict)
    sections: Dict[str, Section2D] = field(default_factory=dict)
    elements: Dict[str, Element2D] = field(default_factory=dict)
    supports: List[Support2D] = field(default_factory=list)

    def add_node(self, n: Node2D) -> None:
        if n.id in self.nodes:
            raise ValueError(f"Duplicate node id: {n.id}")
        self.nodes[n.id] = n

    def add_material(self, m: Material) -> None:
        if m.name in self.materials:
            raise ValueError(f"Duplicate material name: {m.name}")
        self.materials[m.name] = m

    def add_section(self, s: Section2D) -> None:
        if s.name in self.sections:
            raise ValueError(f"Duplicate section name: {s.name}")
        self.sections[s.name] = s

    def add_element(self, e: Element2D) -> None:
        if e.id in self.elements:
            raise ValueError(f"Duplicate element id: {e.id}")
        if e.ni not in self.nodes or e.nj not in self.nodes:
            raise ValueError(f"Element {e.id} references missing node(s).")
        if e.material not in self.materials:
            raise ValueError(f"Element {e.id} references missing material: {e.material}")
        if e.section not in self.sections:
            raise ValueError(f"Element {e.id} references missing section: {e.section}")
        self.elements[e.id] = e

    def add_support(self, s: Support2D) -> None:
        if s.node_id not in self.nodes:
            raise ValueError(f"Support references missing node: {s.node_id}")
        self.supports.append(s)

    def node_order(self) -> List[str]:
        return list(self.nodes.keys())

    def dof_map_truss(self) -> Dict[Tuple[str, str], int]:
        ids = self.node_order()
        m: Dict[Tuple[str, str], int] = {}
        for i, nid in enumerate(ids):
            m[(nid, "ux")] = 2*i
            m[(nid, "uy")] = 2*i + 1
        return m

    def dof_map_frame(self) -> Dict[Tuple[str, str], int]:
        ids = self.node_order()
        m: Dict[Tuple[str, str], int] = {}
        for i, nid in enumerate(ids):
            m[(nid, "ux")] = 3*i
            m[(nid, "uy")] = 3*i + 1
            m[(nid, "rz")] = 3*i + 2
        return m

    def merge_close_nodes(self, tol: float) -> None:
        if tol <= 0:
            return
        bins: Dict[tuple[int,int], str] = {}
        remap: Dict[str, str] = {}
        new_nodes: Dict[str, Node2D] = {}

        for nid, n in self.nodes.items():
            key = (int(round(n.x/tol)), int(round(n.y/tol)))
            if key not in bins:
                bins[key] = nid
                new_nodes[nid] = n
                remap[nid] = nid
            else:
                remap[nid] = bins[key]

        if all(remap[k] == k for k in remap):
            return

        seen = set()
        new_elems: Dict[str, Element2D] = {}
        for eid, e in self.elements.items():
            ni = remap[e.ni]
            nj = remap[e.nj]
            if ni == nj:
                continue
            key = (e.etype, min(ni, nj), max(ni, nj), e.material, e.section, e.group)
            if key in seen:
                continue
            seen.add(key)
            new_elems[eid] = Element2D(
                id=eid, etype=e.etype, ni=ni, nj=nj,
                material=e.material, section=e.section, group=e.group, K=e.K
            )

        new_supports: List[Support2D] = [
            Support2D(
                node_id=remap[s.node_id],
                ux_fixed=s.ux_fixed, uy_fixed=s.uy_fixed, rz_fixed=s.rz_fixed
            ) for s in self.supports
        ]

        self.nodes = new_nodes
        self.elements = new_elems
        self.supports = new_supports

    def element_length(self, eid: str) -> float:
        e = self.elements[eid]
        ni = self.nodes[e.ni]
        nj = self.nodes[e.nj]
        return float(np.hypot(nj.x - ni.x, nj.y - ni.y))
