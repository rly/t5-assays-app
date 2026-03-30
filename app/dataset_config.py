"""Configuration for pre-configured merged datasets and transform functions.

Merge configs define how to combine two Google Sheets into a single table.
Transform functions handle complex column rearrangements that can't be expressed declaratively.
"""
import pandas as pd


# ---------------------------------------------------------------------------
# Transform functions (referenced by name in MERGE_CONFIGS)
# ---------------------------------------------------------------------------

def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def _normalize_parg_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'PARG 1' -> 'PARG001' format."""
    if "Name" in df.columns:
        df = df.copy()
        df["Name"] = df["Name"].str.replace(
            r"PARG (\d+)", lambda m: f"PARG{int(m.group(1)):03d}", regex=True
        )
    return df


def _rename_binding_score(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"VEEV - Binding Score": "VEEV - AI Binding Score"})


def _rename_parg_number_dup(df: pd.DataFrame) -> pd.DataFrame:
    if "PARG Number.1" in df.columns:
        return df.rename(columns={"PARG Number.1": "PARG Number FP"})
    return df


def _transform_peitho_merge(df: pd.DataFrame) -> pd.DataFrame:
    """Post-merge column rearrangement for PEITHO SPR + AI docking."""
    if "Chi2_ndof_RU2" in df.columns:
        col = df.pop("Chi2_ndof_RU2")
        df.insert(1, "Chi2_ndof_RU2", col)

    if "RMSE_RU" in df.columns:
        col = df.pop("RMSE_RU")
        df.insert(2, "RMSE_RU", col)

    if "VEEV - Binding Score" in df.columns:
        col = df.pop("VEEV - Binding Score")
        df.insert(3, "VEEV - AI Binding Score", col)

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


def _transform_parg_merge(df: pd.DataFrame) -> pd.DataFrame:
    """Post-merge column rearrangement for PARG FP + AI binding."""
    if "FP binding (uM)" in df.columns:
        col = df.pop("FP binding (uM)")
        df.insert(1, "FP binding (uM)", col)

    if "PARG Number FP" in df.columns and "PARG Number" in df.columns:
        col_fp = df.pop("PARG Number FP")
        col_parg = df.pop("PARG Number")
        df.insert(3, "PARG Number FP", col_fp)
        df.insert(4, "PARG Number", col_parg)

    df["_sort_key"] = pd.to_numeric(df.get("FP binding (uM)"), errors="coerce")
    df = df.sort_values(by="_sort_key", ascending=True, na_position="last")
    df = df.drop(columns=["_sort_key"])
    return df


# ---------------------------------------------------------------------------
# Registry: maps string names -> callable transforms
# ---------------------------------------------------------------------------

TRANSFORM_REGISTRY: dict[str, callable] = {
    "drop_duplicates": _drop_duplicates,
    "normalize_parg_names": _normalize_parg_names,
    "rename_binding_score": _rename_binding_score,
    "rename_parg_number_dup": _rename_parg_number_dup,
    "transform_peitho_merge": _transform_peitho_merge,
    "transform_parg_merge": _transform_parg_merge,
}


# ---------------------------------------------------------------------------
# Merge configurations
# ---------------------------------------------------------------------------

MERGE_CONFIGS: dict[str, dict] = {
    "veev_peitho_merge": {
        "display_name": "VEEV MacroD PEITHO SPR + AI Docking",
        "description": "PEITHO SPR assay merged with AI docking predictions for VEEV macrodomain",
        "sheets": [
            {"name": "PIETHOS_AI-docking_V2_F-converted", "alias": "ai_bind"},
            {"name": "VEEV_MacroD_PEITHO_SPR_03132025_04302025_05072025", "alias": "spr"},
        ],
        "join": {
            "left_on": "Name",
            "right_on": "IDNUMBER",
            "how": "outer",
            "suffixes": ("_AI_Bind", "_SPR"),
        },
        "pre_merge_transforms": {
            "ai_bind": ["drop_duplicates"],
        },
        "post_merge_transform": "transform_peitho_merge",
        "default_filters": {"Chi2_ndof_RU2": 10.0, "RMSE_RU": 10.0},
    },
    "veev_parg_merge": {
        "display_name": "VEEV MacroD PARG FP + AI Binding",
        "description": "PARG Fluorescence Polarization assay merged with AI binding predictions for VEEV macrodomain",
        "sheets": [
            {"name": "VEEV_MacroD_PARG_AI_Bind_09082025", "alias": "ai_bind"},
            {"name": "VEEV_MacroD_PARG_Fluor_Pol_07292025", "alias": "fp"},
        ],
        "join": {
            "left_on": "Name",
            "right_on": "PARG Number FP",
            "how": "outer",
            "suffixes": ("_AI_Bind", "_FP"),
        },
        "pre_merge_transforms": {
            "ai_bind": ["drop_duplicates", "normalize_parg_names", "rename_binding_score"],
            "fp": ["rename_parg_number_dup"],
        },
        "post_merge_transform": "transform_parg_merge",
        "default_filters": {},
    },
}
