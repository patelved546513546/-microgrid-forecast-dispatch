import pandas as pd
from pathlib import Path
import re

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def _find_header_row(csv_path: Path, max_lines=100):
    # Look for a row that contains "consumption category" and "date"
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if i > max_lines: break
            low = line.lower()
            if ("consumption category" in low) and ("date" in low):
                return i
    return None

def _find_time_cols(df):
    # Accept 0:30, 1:00, 01:00, 0:30:00
    pat = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*$")
    return [c for c in df.columns if pat.match(str(c))]

def load_ausgrid_wide(file_path: str, aggregate="sum_all", keep_customers=None):
    """
    Works with '2012-2013 Solar home electricity data v2.csv' (wide format).
    Returns: DATETIME, CONSUMPTION_KW, SOLAR_KW, NET_LOAD_KW (30‑min).
    """
    csv_path = Path(file_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Detect the real header line (skip the banner row)
    hdr = _find_header_row(csv_path)
    if hdr is None:
        raise ValueError("Could not find header row with 'Consumption Category' and 'Date' in the first ~100 lines.")

    df = pd.read_csv(csv_path, header=hdr, low_memory=False, encoding="utf-8-sig")
    # Drop empty "Unnamed" cols if any
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # Map important columns robustly
    cmap = {_normalize(c): c for c in df.columns}
    cust_col = cmap.get("customer") or cmap.get("customer id") or None
    date_col = cmap.get("date") or cmap.get("day") or cmap.get("reading date")
    cat_col  = cmap.get("consumption category") or cmap.get("consumption_category") or cmap.get("category")

    if date_col is None or cat_col is None:
        raise ValueError(f"After header detection, 'Date' or 'Consumption Category' still missing. Columns: {list(df.columns)[:20]}")

    # Optional customer filter
    if keep_customers is not None and cust_col is None:
        raise ValueError("You asked to filter by customers but this file has no 'Customer' column.")
    if keep_customers is not None:
        df = df[df[cust_col].isin(keep_customers)] if isinstance(keep_customers, (list, tuple, set)) else df[df[cust_col]==keep_customers]

    # Melt wide times to long
    tcols = _find_time_cols(df)
    if not tcols:
        raise ValueError(f"No half-hour time columns detected. Example headers: {list(df.columns)[:20]}")

    id_vars = [date_col, cat_col] + ([cust_col] if cust_col else [])
    long = df.melt(id_vars=id_vars, value_vars=tcols, var_name="HHMM", value_name="kwh")
    if long.empty:
        raise ValueError("No rows after melting; check that time columns actually contain numbers.")

    # Parse datetime (day-first in this dataset)
    long["DATE"] = pd.to_datetime(long[date_col], dayfirst=True, errors="coerce")
    hhmm = long["HHMM"].astype(str).str.strip()
    # Add :00 seconds if missing
    hhmm = hhmm.str.replace(r"^(\d{1,2}:\d{2})$", r"\1:00", regex=True)
    long["DATETIME"] = long["DATE"] + pd.to_timedelta(hhmm)
    long["kwh"] = pd.to_numeric(long["kwh"], errors="coerce").fillna(0.0)

    # Pivot categories (GC, CL, GG)
    idx = ["DATETIME"] + ([cust_col] if cust_col else [])
    pv = long.pivot_table(index=idx, columns=cat_col, values="kwh", aggfunc="sum").reset_index()
    # Map category codes -> columns
    pv["GC_KWH"] = pv["GC"] if "GC" in pv else 0.0
    pv["CL_KWH"] = pv["CL"] if "CL" in pv else 0.0
    pv["GG_KWH"] = pv["GG"] if "GG" in pv else 0.0

    pv["CONSUMPTION_KWH"] = pv["GC_KWH"] + pv["CL_KWH"]
    pv["SOLAR_KWH"] = pv["GG_KWH"]
    pv["CONSUMPTION_KW"] = pv["CONSUMPTION_KWH"] * 2.0
    pv["SOLAR_KW"] = pv["SOLAR_KWH"] * 2.0
    pv["NET_LOAD_KW"] = (pv["CONSUMPTION_KW"] - pv["SOLAR_KW"]).clip(lower=0.0)

    # Aggregate
    if aggregate == "per_customer" and cust_col:
        return pv.rename(columns={cust_col:"CUSTOMER_ID"})[
            ["CUSTOMER_ID","DATETIME","CONSUMPTION_KW","SOLAR_KW","NET_LOAD_KW"]
        ].sort_values(["CUSTOMER_ID","DATETIME"]).reset_index(drop=True)

    if cust_col:
        g = pv.groupby("DATETIME", as_index=False)[["CONSUMPTION_KW","SOLAR_KW","NET_LOAD_KW"]].sum()
    else:
        g = pv[["DATETIME","CONSUMPTION_KW","SOLAR_KW","NET_LOAD_KW"]].copy()

    return g.sort_values("DATETIME").reset_index(drop=True)