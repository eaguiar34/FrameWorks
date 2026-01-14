from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import ezdxf
from core.model import Model2D, Node2D, Element2D, Material, Section2D

@dataclass
class LayerMap:
    etype: str       # "TRUSS2D" or "FRAME2D"
    material: str
    section: str
    group: str = "DEFAULT"
    K: float = 1.0

def _insunits_scale(doc_units: int, target_units: str) -> float:
    """
    Return coordinate scale factor to convert from DXF $INSUNITS to target_units ("in" or "mm").
    Common AutoCAD codes: 1=inches, 4=millimeters.
    """
    if doc_units in (None, 0):
        return 1.0  # unitless
    if doc_units == 1:  # inches
        return 1.0 if target_units == "in" else 25.4
    if doc_units == 4:  # mm
        return 1.0 if target_units == "mm" else (1.0 / 25.4)
    return 1.0  # other units not handled yet

def import_model_from_dxf(
    file_bytes: bytes,
    layer_map: Dict[str, LayerMap],
    materials: Dict[str, Material],
    sections: Dict[str, Section2D],
    snap_tol: float = 1e-3,
    target_length_unit: str = "in",  # "in" or "mm"
) -> Tuple[Model2D, Optional[int], float]:
    doc = ezdxf.readstring(file_bytes.decode("utf-8", errors="ignore"))
    msp = doc.modelspace()
    insunits = doc.header.get("$INSUNITS", 0)
    insunits_int = int(insunits) if isinstance(insunits, (int, float)) else 0
    scale = _insunits_scale(insunits_int, target_length_unit)

    model = Model2D()
    for m in materials.values():
        model.add_material(m)
    for s in sections.values():
        model.add_section(s)

    bins: Dict[tuple[int,int], str] = {}
    coords: Dict[str, Tuple[float,float]] = {}
    next_node = 1

    def get_node_id(x: float, y: float) -> str:
        nonlocal next_node
        xs = float(x) * scale
        ys = float(y) * scale
        key = (int(round(xs/snap_tol)), int(round(ys/snap_tol)))
        if key in bins:
            return bins[key]
        nid = f"N{next_node}"
        next_node += 1
        bins[key] = nid
        coords[nid] = (xs, ys)
        return nid

    segments: List[Tuple[str, Tuple[float,float], Tuple[float,float]]] = []

    for e in msp:
        layer = getattr(e.dxf, "layer", "")
        if layer not in layer_map:
            continue
        if e.dxftype() == "LINE":
            p1 = (float(e.dxf.start.x), float(e.dxf.start.y))
            p2 = (float(e.dxf.end.x), float(e.dxf.end.y))
            segments.append((layer, p1, p2))
        elif e.dxftype() == "LWPOLYLINE":
            pts = [(float(x), float(y)) for x, y, *_ in e.get_points()]
            for a,b in zip(pts[:-1], pts[1:]):
                segments.append((layer, a, b))

    for _, p1, p2 in segments:
        get_node_id(*p1); get_node_id(*p2)

    for nid, (x,y) in coords.items():
        model.add_node(Node2D(nid, x, y))

    next_elem = 1
    for layer, p1, p2 in segments:
        ni = get_node_id(*p1)
        nj = get_node_id(*p2)
        lm = layer_map[layer]
        eid = f"E{next_elem}"
        next_elem += 1
        model.add_element(Element2D(
            id=eid, etype=lm.etype, ni=ni, nj=nj,
            material=lm.material, section=lm.section, group=lm.group, K=lm.K
        ))
    return model, insunits_int, scale
