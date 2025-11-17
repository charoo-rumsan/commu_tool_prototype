import os
import csv
import polars as pl
from openpyxl import load_workbook

def extract_headers(file_path: str):
    """
    Extract only headers (column names) from CSV or Excel files (.csv, .xlsx, .xls).
    Uses Polars for maximum speed and minimal memory usage.
    """
    ext = os.path.splitext(file_path)[1].lower()

    # ---------- CSV / TXT ----------
    if ext in [".csv", ".txt"]:
        try:
            # Fastest: Lazy scan with schema extraction (no row data read)
            return pl.scan_csv(file_path).collect_schema().names()
        except Exception:
            # Fallback: basic csv.reader
            with open(file_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                return next(reader)

    # ---------- XLSX / XLSM ----------
    elif ext in [".xlsx", ".xlsm"]:
        try:
            # New Polars Excel reader (very fast)
            return pl.read_excel(file_path, n_rows=0).columns
        except Exception:
            # Fallback to openpyxl read-only mode
            wb = load_workbook(file_path, read_only=True, data_only=True)
            ws = wb.active
            headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
            wb.close()
            return headers

    # ---------- XLS ----------
    elif ext == ".xls":
        try:
            # Try Polars Excel reader first (if supported)
            return pl.read_excel(file_path, n_rows=0).columns
        except Exception:
            # Fallback to pandas for legacy .xls (Excel 97-2003)
            import pandas as pd
            df = pd.read_excel(file_path, nrows=0, engine="xlrd")
            return list(df.columns)

    else:
        raise ValueError(f"Unsupported file type: {ext}")
