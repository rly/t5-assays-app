import time
import pandas as pd
from app.services.sheets_service import get_sheets_from_folder, read_sheet

# In-memory cache: {data_source_key: (df, timestamp)}
_cache: dict[str, tuple[pd.DataFrame, float]] = {}
CACHE_TTL = 300  # 5 minutes


def _get_cached(key: str) -> pd.DataFrame | None:
    if key in _cache:
        df, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return df.copy()
    return None


def _set_cache(key: str, df: pd.DataFrame):
    _cache[key] = (df.copy(), time.time())


def clear_cache():
    _cache.clear()


def load_data(data_source_type: str, sheet_id: str | None = None) -> pd.DataFrame:
    """Load and return a DataFrame based on the data source type."""
    cache_key = f"{data_source_type}:{sheet_id or ''}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    if data_source_type == "veev_peitho_merge":
        df = _load_peitho_merge()
    elif data_source_type == "veev_parg_merge":
        df = _load_parg_merge()
    elif data_source_type == "single_sheet" and sheet_id:
        df = read_sheet(sheet_id)
    else:
        df = pd.DataFrame()

    # Normalize column names
    df.columns = df.columns.str.replace("\u65b0", "\u00b7s", regex=False)

    _set_cache(cache_key, df)
    return df


def _load_peitho_merge() -> pd.DataFrame:
    sheets = get_sheets_from_folder()

    sheet1_name = "PIETHOS_AI-docking_V2_F-converted"
    sheet2_name = "VEEV_MacroD_PEITHO_SPR_03132025_04302025_05072025"

    if sheet1_name not in sheets or sheet2_name not in sheets:
        raise ValueError(f"Required sheets not found: {sheet1_name}, {sheet2_name}")

    df1 = read_sheet(sheets[sheet1_name]["id"])
    df2 = read_sheet(sheets[sheet2_name]["id"])

    df1 = df1.drop_duplicates()

    df = pd.merge(df1, df2, left_on="Name", right_on="IDNUMBER", how="outer", suffixes=("_AI_Bind", "_SPR"))

    # Rearrange columns
    if "Chi2_ndof_RU2" in df.columns:
        col = df.pop("Chi2_ndof_RU2")
        df.insert(1, "Chi2_ndof_RU2", col)

    if "RMSE_RU" in df.columns:
        col = df.pop("RMSE_RU")
        df.insert(2, "RMSE_RU", col)

    if "VEEV - Binding Score" in df.columns:
        col = df.pop("VEEV - Binding Score")
        df.insert(3, "VEEV - AI Binding Score", col)

    # Move KA/KD columns
    ka_kd_cols = [c for c in df.columns if c.startswith(("kA", "kD", "KA", "KD"))]
    insert_pos = 4
    for c in ka_kd_cols:
        col_data = df.pop(c)
        df.insert(insert_pos, c, col_data)
        insert_pos += 1

    if "IDNUMBER" in df.columns:
        col = df.pop("IDNUMBER")
        df.insert(insert_pos, "IDNUMBER", col)

    df = df.sort_values(by="Chi2_ndof_RU2", ascending=True, na_position="last")
    return df


def _load_parg_merge() -> pd.DataFrame:
    sheets = get_sheets_from_folder()

    sheet1_name = "VEEV_MacroD_PARG_AI_Bind_09082025"
    sheet2_name = "VEEV_MacroD_PARG_Fluor_Pol_07292025"

    if sheet1_name not in sheets or sheet2_name not in sheets:
        raise ValueError(f"Required sheets not found: {sheet1_name}, {sheet2_name}")

    df1 = read_sheet(sheets[sheet1_name]["id"])
    df2 = read_sheet(sheets[sheet2_name]["id"])

    df1 = df1.drop_duplicates()

    # Transform PARG names: "PARG 1" -> "PARG001"
    if "Name" in df1.columns:
        df1["Name"] = df1["Name"].str.replace(
            r"PARG (\d+)", lambda m: f"PARG{int(m.group(1)):03d}", regex=True
        )

    # Rename binding score
    df1.rename(columns={"VEEV - Binding Score": "VEEV - AI Binding Score"}, inplace=True)

    # Handle duplicate PARG Number columns
    if "PARG Number.1" in df2.columns:
        df2.rename(columns={"PARG Number.1": "PARG Number FP"}, inplace=True)

    df = pd.merge(df1, df2, left_on="Name", right_on="PARG Number FP", how="outer", suffixes=("_AI_Bind", "_FP"))

    # Rearrange columns
    if "FP binding (uM)" in df.columns:
        col = df.pop("FP binding (uM)")
        df.insert(1, "FP binding (uM)", col)

    if "PARG Number FP" in df.columns and "PARG Number" in df.columns:
        col_fp = df.pop("PARG Number FP")
        col_parg = df.pop("PARG Number")
        df.insert(3, "PARG Number FP", col_fp)
        df.insert(4, "PARG Number", col_parg)

    # Sort by FP binding with numeric handling
    df["_sort_key"] = pd.to_numeric(df.get("FP binding (uM)"), errors="coerce")
    df = df.sort_values(by="_sort_key", ascending=True, na_position="last")
    df = df.drop(columns=["_sort_key"])

    return df
