# Practical Structural Analysis (Streamlit + Python) — v6

This upgrade adds “project deliverables” outputs and structure-level governance:

## New upgrades
- **Structure-level governing summary**
  - Governing **|N|, |V|, |M|** across the entire model (which element + combo)
  - Governing **max displacement** (which combo)
  - Governing **reaction** (max absolute reaction across combos)
- **Per-combo exports**
  - Reactions and end forces for **every combo**
  - Download as:
    - **Excel workbook with tabs** (one sheet per combo + aggregate sheets)
    - **ZIP of CSVs** (one file per combo + aggregates)

## Existing features
- Load patterns (load cases) stored in Streamlit session
- Combination table editor with dynamic rows/columns
- Solve across combos
- Member envelopes + controlling combos (per element)
- Member loads: UDL (partial), point loads, point moments; local/global X/Y/Z support

## Run
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests
```bash
pytest -q
```

## Disclaimer
Educational / preliminary tool. Always verify with hand checks and code-compliant workflows.
