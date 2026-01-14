import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.units import UnitSystem, Units
from core.model import Material, Section2D, Support2D
from core.loads import LoadCase2D, LoadCombo2D, combine_loadcases
from core.checks import truss_buckling_utilization
from solver.truss2d import solve_truss_2d
from solver.frame2d import solve_frame_2d
from solver.diagrams import diagrams_from_end_forces_and_loads, envelope_from_combo_diagrams, controlling_combo_extremes
from io.dxf_import import import_model_from_dxf, LayerMap

st.set_page_config(page_title="Practical Structural Analysis", layout="wide")
st.title("Practical Structural Analysis (2D Truss + 2D Frame)")
st.caption("Load patterns + combination envelopes + controlling combo, with member point loads and point moments.")

# ---------- Units ----------
u_choice = st.sidebar.selectbox("Unit system", [UnitSystem.KIP_IN_KSI.value, UnitSystem.N_MM_MPA.value], index=0)
units = Units(UnitSystem(u_choice))
st.sidebar.info(f"Base units: force={units.force}, length={units.length}, stress={units.stress}, moment={units.moment}")

# ---------- Material ----------
st.sidebar.subheader("Material (active)")
if units.system == UnitSystem.KIP_IN_KSI:
    default_E = 29000.0
    default_gamma = 0.000283  # kip/in^3
else:
    default_E = 200000.0
    default_gamma = 77e-6     # N/mm^3

E_in = st.sidebar.number_input(f"E ({units.stress})", value=float(default_E), step=float(default_E/100))
gamma_in = st.sidebar.number_input(f"gamma ({units.force}/{units.length}^3)", value=float(default_gamma), format="%.8f")
steel = Material(name="STEEL", E=float(E_in), gamma=float(gamma_in))
materials = {steel.name: steel}

# ---------- Sections ----------
st.sidebar.subheader("Section library")
st.sidebar.write("Upload CSV columns: name,A,Izz (optional: I_min,Szz,Avy).")
sec_csv = st.sidebar.file_uploader("Sections CSV", type=["csv"])
sections = {}
if sec_csv:
    df_sec = pd.read_csv(sec_csv)
    req = {"name","A","Izz"}
    if not req.issubset(set(df_sec.columns)):
        st.sidebar.error("CSV must have columns: name,A,Izz (optional I_min,Szz,Avy).")
    else:
        for _, row in df_sec.iterrows():
            def opt(col):
                return float(row[col]) if (col in df_sec.columns and pd.notna(row.get(col))) else None
            sections[str(row["name"])] = Section2D(
                name=str(row["name"]),
                A=float(row["A"]),
                Izz=float(row["Izz"]),
                I_min=opt("I_min"),
                Szz=opt("Szz"),
                Avy=opt("Avy")
            )
if not sections:
    sections = {"GEN1": Section2D(name="GEN1", A=1.0, Izz=100.0, I_min=50.0, Szz=None, Avy=None)}
sec_names = list(sections.keys())
sec_truss = st.sidebar.selectbox("Section for TRUSS layer", sec_names, index=0)
sec_frame = st.sidebar.selectbox("Section for FRAME layer", sec_names, index=0)

# ---------- DXF Import ----------
st.sidebar.subheader("DXF import layer mapping")
st.sidebar.write("Use layer names TRUSS and FRAME.")
layer_map = {
    "TRUSS": LayerMap(etype="TRUSS2D", material=steel.name, section=sec_truss, group="TRUSS", K=1.0),
    "FRAME": LayerMap(etype="FRAME2D", material=steel.name, section=sec_frame, group="FRAME", K=1.0),
}
snap_tol = st.sidebar.number_input(f"Snap tolerance ({units.length})", min_value=1e-9, value=1e-3, step=1e-3, format="%.6f")

uploaded = st.file_uploader("Upload DXF", type=["dxf"])
if not uploaded:
    st.info("Upload a DXF with LINE/LWPOLYLINE members on layers TRUSS and/or FRAME.")
    st.stop()

target_len = "in" if units.system == UnitSystem.KIP_IN_KSI else "mm"
model, insunits_code, scale_applied = import_model_from_dxf(
    uploaded.getvalue(),
    layer_map=layer_map,
    materials=materials,
    sections=sections,
    snap_tol=snap_tol,
    target_length_unit=target_len
)
model.merge_close_nodes(tol=snap_tol)

# ---------- Summary ----------
st.subheader("Import + model summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Nodes", len(model.nodes))
c2.metric("Elements", len(model.elements))
c3.metric("Unit system", units.system.value)
c4.metric("DXF $INSUNITS", int(insunits_code))
c5.metric("Scale applied", float(scale_applied))

node_ids = list(model.nodes.keys())
frame_eids = [eid for eid, e in model.elements.items() if e.etype == "FRAME2D"]
truss_eids = [eid for eid, e in model.elements.items() if e.etype == "TRUSS2D"]
has_frame = len(frame_eids) > 0
has_truss = len(truss_eids) > 0

# ---------- Supports ----------
st.subheader("Supports")
if len(node_ids) < 2:
    st.error("Need at least 2 nodes.")
    st.stop()

col1, col2, col3 = st.columns(3)
pin_node = col1.selectbox("Pinned node (ux, uy fixed)", node_ids, index=0)
roller_node = col2.selectbox("Roller node (uy fixed)", node_ids, index=len(node_ids)-1)
fix_rz = col3.checkbox("Fix rotation at pinned node (frame)", value=False)

model.supports = []
model.add_support(Support2D(pin_node, ux_fixed=True, uy_fixed=True, rz_fixed=fix_rz))
model.add_support(Support2D(roller_node, ux_fixed=False, uy_fixed=True, rz_fixed=False))

# ---------- Geometry plot ----------
def plot_geometry(u_frame=None, scale=1.0):
    fig = plt.figure()
    ax = plt.gca()
    for eid, e in model.elements.items():
        ni = model.nodes[e.ni]
        nj = model.nodes[e.nj]
        ax.plot([ni.x, nj.x], [ni.y, nj.y])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"x ({units.length})")
    ax.set_ylabel(f"y ({units.length})")
    if u_frame is not None:
        dof = model.dof_map_frame()
        for eid, e in model.elements.items():
            if e.etype != "FRAME2D":
                continue
            ni = model.nodes[e.ni]
            nj = model.nodes[e.nj]
            ui = np.array([u_frame[dof[(e.ni,'ux')]], u_frame[dof[(e.ni,'uy')]]], dtype=float)
            uj = np.array([u_frame[dof[(e.nj,'ux')]], u_frame[dof[(e.nj,'uy')]]], dtype=float)
            ax.plot([ni.x + scale*ui[0], nj.x + scale*uj[0]], [ni.y + scale*ui[1], nj.y + scale*uj[1]])
    st.pyplot(fig, clear_figure=True)

st.subheader("Geometry")
plot_geometry()

# ---------- Session State: Load patterns & combos ----------
def _init_state():
    if "loadcases" not in st.session_state:
        st.session_state.loadcases = {
            "D": {"nodal": [], "member": [], "self_weight": True},
            "L": {"nodal": [], "member": [], "self_weight": False},
        }
    if "combos" not in st.session_state:
        st.session_state.combos = [
            {"combo": "1.4D", "D": 1.4, "L": 0.0},
            {"combo": "1.2D+1.6L", "D": 1.2, "L": 1.6},
            {"combo": "D+L", "D": 1.0, "L": 1.0},
        ]

_init_state()

st.subheader("Load patterns (load cases)")
st.caption("Define load cases (patterns) first, then combinations below. Loads persist in your session.")

colA, colB = st.columns([1, 2])
with colA:
    lc_names = list(st.session_state.loadcases.keys())
    new_name = st.text_input("New load case name", value="")
    if st.button("Add load case"):
        nm = new_name.strip()
        if nm and nm not in st.session_state.loadcases:
            st.session_state.loadcases[nm] = {"nodal": [], "member": [], "self_weight": False}
            # extend combos with new factor column
            for row in st.session_state.combos:
                row[nm] = 0.0
            st.rerun()

    del_name = st.selectbox("Delete load case", ["(none)"] + lc_names)
    if st.button("Delete selected load case"):
        if del_name != "(none)" and del_name in st.session_state.loadcases:
            del st.session_state.loadcases[del_name]
            for row in st.session_state.combos:
                row.pop(del_name, None)
            st.rerun()

with colB:
    active_lc = st.selectbox("Active load case to edit", list(st.session_state.loadcases.keys()), index=0)
    st.session_state.loadcases[active_lc]["self_weight"] = st.checkbox(
        f"Include self-weight in {active_lc}",
        value=bool(st.session_state.loadcases[active_lc].get("self_weight", False))
    )

# Add loads to active load case
st.markdown("### Add loads to active load case")
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Nodal load**")
    n_node = st.selectbox("Node", node_ids, key="nl_node")
    Fx = st.number_input(f"Fx ({units.force})", value=0.0, step=1.0, key="nl_Fx")
    Fy = st.number_input(f"Fy ({units.force})", value=0.0, step=1.0, key="nl_Fy")
    Mz = st.number_input(f"Mz ({units.moment})", value=0.0, step=1.0, key="nl_Mz")
    if st.button("Add nodal load"):
        st.session_state.loadcases[active_lc]["nodal"].append({"node": n_node, "Fx": Fx, "Fy": Fy, "Mz": Mz})
        st.rerun()

with c2:
    st.markdown("**Member load (FRAME only)**")
    if not frame_eids:
        st.info("No FRAME elements in the DXF.")
    else:
        e_sel = st.selectbox("Element", frame_eids, key="ml_eid")
        L_sel = model.element_length(e_sel)
        kind = st.selectbox("Type", ["UDL", "POINT", "MOMENT"], index=0, key="ml_kind")
        coord = st.selectbox("Coordinates", ["GLOBAL", "LOCAL"], index=0, key="ml_coord")
        if kind == "MOMENT":
            direction = "Z"
            val_units = units.moment
        else:
            direction = st.selectbox("Direction", ["X","Y"], index=1, key="ml_dir")
            val_units = (f"{units.force}/{units.length}" if kind == "UDL" else units.force)
        value = st.number_input(f"Value ({val_units})", value=0.0, step=1.0, key="ml_val")

        if kind == "UDL":
            a = st.number_input(f"Start a ({units.length})", value=0.0, step=float(max(L_sel/10, 1e-6)), key="ml_a")
            b = st.number_input(f"End b ({units.length})", value=float(L_sel), step=float(max(L_sel/10, 1e-6)), key="ml_b")
        else:
            a = st.number_input(f"Location a ({units.length})", value=float(L_sel/2), step=float(max(L_sel/10, 1e-6)), key="ml_a2")
            b = None

        if st.button("Add member load"):
            st.session_state.loadcases[active_lc]["member"].append({
                "eid": e_sel, "kind": kind, "coord": coord, "dir": direction, "value": value, "a": a, "b": b
            })
            st.rerun()

# Show and delete loads
st.markdown("### Current loads in active load case")
lc_data = st.session_state.loadcases[active_lc]
nodal_df = pd.DataFrame(lc_data["nodal"]) if lc_data["nodal"] else pd.DataFrame(columns=["node","Fx","Fy","Mz"])
member_df = pd.DataFrame(lc_data["member"]) if lc_data["member"] else pd.DataFrame(columns=["eid","kind","coord","dir","value","a","b"])
cN, cM = st.columns(2)
with cN:
    st.write("Nodal loads")
    st.dataframe(nodal_df, use_container_width=True, height=200)
    if len(nodal_df) > 0:
        del_idx = st.number_input("Delete nodal load row #", min_value=1, max_value=len(nodal_df), value=1, step=1)
        if st.button("Delete nodal load"):
            lc_data["nodal"].pop(int(del_idx)-1)
            st.rerun()
with cM:
    st.write("Member loads")
    st.dataframe(member_df, use_container_width=True, height=200)
    if len(member_df) > 0:
        del_idx2 = st.number_input("Delete member load row #", min_value=1, max_value=len(member_df), value=1, step=1)
        if st.button("Delete member load"):
            lc_data["member"].pop(int(del_idx2)-1)
            st.rerun()

# ---------- Combination table editor ----------
st.subheader("Combination table editor")
st.caption("Edit factors and add/remove combo rows. Factor columns correspond to your load cases.")

combo_df = pd.DataFrame(st.session_state.combos)
# Ensure all LC columns exist
for lcname in st.session_state.loadcases.keys():
    if lcname not in combo_df.columns:
        combo_df[lcname] = 0.0
# Place 'combo' first
cols = ["combo"] + [c for c in combo_df.columns if c != "combo"]
combo_df = combo_df[cols]

edited = st.data_editor(combo_df, num_rows="dynamic", use_container_width=True)
if st.button("Save combinations"):
    st.session_state.combos = edited.fillna(0.0).to_dict(orient="records")
    st.success("Saved combinations.")

# ---------- Solve across combos + envelopes ----------
st.subheader("Solve + envelopes")
mode = st.selectbox("Analysis type", ["Auto", "Truss", "Frame"], index=0)

def build_loadcases() -> dict:
    out = {}
    for name, data in st.session_state.loadcases.items():
        lc = LoadCase2D(name)
        for r in data["nodal"]:
            lc.add_nodal_force(r["node"], Fx=float(r["Fx"]), Fy=float(r["Fy"]), Mz=float(r["Mz"]))
        # member loads
        for r in data["member"]:
            eid = r["eid"]
            kind = r["kind"]
            coord = r["coord"]
            d = r["dir"]
            val = float(r["value"])
            a = float(r["a"])
            b = None if (r.get("b") is None or (isinstance(r.get("b"), float) and np.isnan(r.get("b")))) else float(r.get("b"))
            if kind == "UDL":
                if coord == "GLOBAL" and d == "Y":
                    lc.add_member_udl_global_y(eid, qy=val, a=a, b=b)
                elif coord == "GLOBAL" and d == "X":
                    lc.add_member_udl_global_x(eid, qx=val, a=a, b=b)
                elif coord == "LOCAL" and d == "Y":
                    lc.add_member_udl_local_y(eid, qy=val, a=a, b=b)
                elif coord == "LOCAL" and d == "X":
                    lc.add_member_udl_local_x(eid, qx=val, a=a, b=b)
            elif kind == "POINT":
                if coord == "GLOBAL" and d == "Y":
                    lc.add_member_point_global_y(eid, P=val, a=a)
                elif coord == "GLOBAL" and d == "X":
                    lc.add_member_point_global_x(eid, P=val, a=a)
                elif coord == "LOCAL" and d == "Y":
                    lc.add_member_point_local_y(eid, P=val, a=a)
                elif coord == "LOCAL" and d == "X":
                    lc.add_member_point_local_x(eid, P=val, a=a)
            elif kind == "MOMENT":
                lc.add_member_point_moment(eid, Mz=val, a=a, coord=coord)
        lc.include_self_weight = bool(data.get("self_weight", False))
        out[name] = lc
    return out

if st.button("Solve all combos"):
    cases = build_loadcases()

    combos = []
    for row in st.session_state.combos:
        name = str(row.get("combo","")).strip()
        if not name:
            continue
        factors = {lc: float(row.get(lc, 0.0)) for lc in cases.keys()}
        combos.append(LoadCombo2D(name=name, factors=factors))

    if not combos:
        st.error("No valid combos.")
        st.stop()

    mode_eff = "FRAME2D" if (mode == "Auto" and has_frame) or mode == "Frame" else "TRUSS2D"


if mode_eff == "TRUSS2D":
        results = {}
        for cb in combos:
            lc_combo = combine_loadcases(cases, cb)
            results[cb.name] = solve_truss_2d(model, lc_combo)
        st.success(f"Solved {len(results)} combos (TRUSS).")

        # Controlling combo by max |N| per truss member
        rows = []
        for eid in truss_eids:
            best = ("", -np.inf)
            for cname, res in results.items():
                v = abs(float(res.elem_axial.get(eid, 0.0)))
                if v > best[1]:
                    best = (cname, v)
            rows.append({"eid": eid, "max_abs_N": best[1], "controlling_combo": best[0]})
        st.dataframe(pd.DataFrame(rows).sort_values("max_abs_N", ascending=False), use_container_width=True)

else:
        # FRAME: solve each combo, then compute envelopes and governing combo per element.
        results = {}
        for cb in combos:
            lc_combo = combine_loadcases(cases, cb)
            results[cb.name] = solve_frame_2d(model, lc_combo)
        st.success(f"Solved {len(results)} combos (FRAME).")

        if not frame_eids:
            st.info("No frame elements.")
            st.stop()

        # --- Multi-element governing table ---
        st.markdown("### Governing combos across ALL frame elements")
        st.caption("For each element: compute max/min and max-absolute N, V, M across combos (based on diagrams).")

        n_pts = st.slider("Diagram resolution (points per element)", min_value=50, max_value=400, value=150, step=25)

        rows = []
        for eid in frame_eids:
            Lm = model.element_length(eid)
            combo_diagrams = {}
            for cname, res in results.items():
                fL = res.end_forces_local[eid]
                loads_local = res.member_loads_local.get(eid, [])
                combo_diagrams[cname] = diagrams_from_end_forces_and_loads(L=Lm, end_forces_local=fL, loads_local=loads_local, n_points=int(n_pts))

            ctrl = controlling_combo_extremes(combo_diagrams)
            rows.append({
                "eid": eid,
                "N_max_abs": ctrl["N_max_abs"][1], "N_max_abs_combo": ctrl["N_max_abs"][0],
                "N_max": ctrl["N_max"][1], "N_max_combo": ctrl["N_max"][0],
                "N_min": ctrl["N_min"][1], "N_min_combo": ctrl["N_min"][0],
                "V_max_abs": ctrl["V_max_abs"][1], "V_max_abs_combo": ctrl["V_max_abs"][0],
                "V_max": ctrl["V_max"][1], "V_max_combo": ctrl["V_max"][0],
                "V_min": ctrl["V_min"][1], "V_min_combo": ctrl["V_min"][0],
                "M_max_abs": ctrl["M_max_abs"][1], "M_max_abs_combo": ctrl["M_max_abs"][0],
                "M_max": ctrl["M_max"][1], "M_max_combo": ctrl["M_max"][0],
                "M_min": ctrl["M_min"][1], "M_min_combo": ctrl["M_min"][0],
            })

        gov_df = pd.DataFrame(rows).sort_values("M_max_abs", ascending=False)
        st.dataframe(gov_df, use_container_width=True, height=320)

        # CSV export

        # --- Structure-level governing summary ---
        st.markdown("### Structure-level governing summary (entire model)")
        # Find the single governing element+combo for N/V/M max-abs based on gov_df table
        def _pick_max_abs(qcol, ccol):
            i = int(gov_df[qcol].values.argmax()) if len(gov_df) else 0
            return (str(gov_df.iloc[i]["eid"]), str(gov_df.iloc[i][ccol]), float(gov_df.iloc[i][qcol]))

        eidN, comboN, valN = _pick_max_abs("N_max_abs", "N_max_abs_combo")
        eidV, comboV, valV = _pick_max_abs("V_max_abs", "V_max_abs_combo")
        eidM, comboM, valM = _pick_max_abs("M_max_abs", "M_max_abs_combo")

        # Governing max displacement and governing reaction (max abs)
        disp_best = ("", -np.inf)
        react_best = ("", "", "", -np.inf)  # combo,node,dof,val
        for cname, res in results.items():
            if float(res.max_disp) > disp_best[1]:
                disp_best = (cname, float(res.max_disp))
            for (nid, dof), rv in res.reactions.items():
                v = abs(float(rv))
                if v > react_best[3]:
                    react_best = (cname, str(nid), str(dof), float(rv))

        cA, cB, cC, cD = st.columns(4)
        cA.metric("Governing |N|", f"{valN:.4g}", help=f"Element {eidN} in combo {comboN}")
        cB.metric("Governing |V|", f"{valV:.4g}", help=f"Element {eidV} in combo {comboV}")
        cC.metric("Governing |M|", f"{valM:.4g}", help=f"Element {eidM} in combo {comboM}")
        cD.metric("Governing max displacement", f"{disp_best[1]:.4g}", help=f"Combo {disp_best[0]} (max |u| over all DOFs)")

        st.caption(f"Governing reaction (max abs): combo={react_best[0]}, node={react_best[1]}, dof={react_best[2]}, value={react_best[3]:.6g}")

        # --- Per-combo exports: reactions + end forces ---
        st.markdown("### Exports (per-combo reactions & end forces)")

        reac_rows = []
        end_rows = []
        for cname, res in results.items():
            for (nid, dof), rv in res.reactions.items():
                reac_rows.append({"combo": cname, "node": nid, "dof": dof, "R": float(rv)})
            for eid, f in res.end_forces_local.items():
                end_rows.append({
                    "combo": cname, "eid": eid,
                    "Ni": float(f[0]), "Vi": float(f[1]), "Mi": float(f[2]),
                    "Nj": float(f[3]), "Vj": float(f[4]), "Mj": float(f[5]),
                })

        reactions_all_df = pd.DataFrame(reac_rows)
        endforces_all_df = pd.DataFrame(end_rows)

        st.write("Reactions (all combos)")
        st.dataframe(reactions_all_df, use_container_width=True, height=220)
        st.write("End forces (all combos, local)")
        st.dataframe(endforces_all_df, use_container_width=True, height=220)

        from report.export_data import build_excel_bytes, build_zip_of_csv_bytes

        # Excel workbook with tabs
        sheets = {
            "Governing_ByElement": gov_df,
            "Reactions_All": reactions_all_df,
            "EndForces_All": endforces_all_df,
        }

        # Add one sheet per combo (optional but handy)
        for cname in results.keys():
            df_r = reactions_all_df[reactions_all_df["combo"] == cname].drop(columns=["combo"])
            df_e = endforces_all_df[endforces_all_df["combo"] == cname].drop(columns=["combo"])
            sheets[f"R_{cname}"] = df_r
            sheets[f"EF_{cname}"] = df_e

        xlsx_bytes = build_excel_bytes(sheets)
        st.download_button("Download Excel workbook (tabs)", data=xlsx_bytes, file_name="analysis_exports.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Zip of CSVs
        files = {
            "governing_by_element.csv": gov_df,
            "reactions_all.csv": reactions_all_df,
            "endforces_all.csv": endforces_all_df,
        }
        for cname in results.keys():
            files[f"reactions_{cname}.csv"] = reactions_all_df[reactions_all_df["combo"] == cname].drop(columns=["combo"])
            files[f"endforces_{cname}.csv"] = endforces_all_df[endforces_all_df["combo"] == cname].drop(columns=["combo"])

        zip_bytes = build_zip_of_csv_bytes(files)
        st.download_button("Download ZIP of CSVs", data=zip_bytes, file_name="analysis_csvs.zip", mime="application/zip")

        csv_bytes = gov_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download governing table (CSV)", data=csv_bytes, file_name="governing_combos.csv", mime="text/csv")

        # --- Selected element envelopes + controlling combo ---
        st.markdown("### Envelopes for a selected frame element")
        eid_plot = st.selectbox("Element for envelope plots", frame_eids)
        Lm = model.element_length(eid_plot)

        combo_diagrams = {}
        for cname, res in results.items():
            fL = res.end_forces_local[eid_plot]
            loads_local = res.member_loads_local.get(eid_plot, [])
            combo_diagrams[cname] = diagrams_from_end_forces_and_loads(L=Lm, end_forces_local=fL, loads_local=loads_local, n_points=250)

        env = envelope_from_combo_diagrams(combo_diagrams)
        ctrl = controlling_combo_extremes(combo_diagrams)

        st.markdown("#### Controlling combo summary (selected element)")
        ctrl_rows = [{"quantity": k, "combo": v[0], "value": v[1]} for k, v in ctrl.items()]
        st.dataframe(pd.DataFrame(ctrl_rows), use_container_width=True)

        st.markdown("#### Envelope diagrams (max/min across combos)")
        def plot_env(x, y_max, y_min, ylabel):
            fig = plt.figure()
            ax = plt.gca()
            ax.plot(x, y_max)
            ax.plot(x, y_min)
            ax.set_xlabel(f"x ({units.length})")
            ax.set_ylabel(ylabel)
            return fig

        figN = plot_env(env.x, env.N_max, env.N_min, f"N ({units.force})")
        st.pyplot(figN, clear_figure=True)
        figV = plot_env(env.x, env.V_max, env.V_min, f"V ({units.force})")
        st.pyplot(figV, clear_figure=True)
        figM = plot_env(env.x, env.M_max, env.M_min, f"M ({units.moment})")
        st.pyplot(figM, clear_figure=True)

        # --- PDF export (summary + selected element plot sheet) ---
        from report.reporting import make_summary_pdf, ReportMeta
        from report.plot_export import fig_to_png_bytes

        st.markdown("### Export report (PDF)")
        page_size = st.selectbox("PDF page size", ["LETTER","A4"], index=0)
        top_n = st.number_input("Rows in summary table", min_value=5, max_value=200, value=30, step=5)

        # create a combined figure image for selected element plots
        comb_fig = plt.figure(figsize=(8, 10))
        ax1 = comb_fig.add_subplot(3,1,1); ax2 = comb_fig.add_subplot(3,1,2); ax3 = comb_fig.add_subplot(3,1,3)
        ax1.plot(env.x, env.N_max); ax1.plot(env.x, env.N_min); ax1.set_ylabel(f"N ({units.force})")
        ax2.plot(env.x, env.V_max); ax2.plot(env.x, env.V_min); ax2.set_ylabel(f"V ({units.force})")
        ax3.plot(env.x, env.M_max); ax3.plot(env.x, env.M_min); ax3.set_ylabel(f"M ({units.moment})"); ax3.set_xlabel(f"x ({units.length})")
        comb_fig.tight_layout()
        plot_png = fig_to_png_bytes(comb_fig)

        meta = ReportMeta(
            title="Structural Analysis Summary Report",
            units_note=f"{units.system.value} (force={units.force}, length={units.length}, stress={units.stress})",
            project_note=f"Selected element for plots: {eid_plot}",
            page_size=page_size
        )
        pdf_bytes = make_summary_pdf(meta, gov_df, top_n=int(top_n), element_plot_png=plot_png)
        st.download_button("Download PDF report", data=pdf_bytes, file_name="analysis_report.pdf", mime="application/pdf")
