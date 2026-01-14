from __future__ import annotations
from io import BytesIO
from typing import Dict
import zipfile
import pandas as pd

def build_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Return an .xlsx workbook (bytes) with one sheet per dataframe."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = str(name)[:31]  # Excel limit
            df.to_excel(writer, sheet_name=safe, index=False)
    return buf.getvalue()

def build_zip_of_csv_bytes(files: Dict[str, pd.DataFrame]) -> bytes:
    """Return a zip (bytes) containing CSV files."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fname, df in files.items():
            csv = df.to_csv(index=False).encode("utf-8")
            z.writestr(fname, csv)
    return buf.getvalue()
