"""Data loading service: loads single sheets and pre-configured merged datasets.

Uses dataset_config.py for merge definitions and transform functions.
Results are cached in memory with a 5-minute TTL.
"""
import time

import pandas as pd

from app.dataset_config import MERGE_CONFIGS, TRANSFORM_REGISTRY
from app.services.sheets_service import get_sheets_from_folder, read_sheet

# In-memory cache: {key: (df, timestamp)}
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


def load_dataset(dataset_key: str) -> pd.DataFrame:
    """Load a dataset by key. Handles both merge configs and single sheets."""
    cached = _get_cached(dataset_key)
    if cached is not None:
        return cached

    if dataset_key in MERGE_CONFIGS:
        df = _load_merge(dataset_key)
    else:
        # Treat as a single sheet — key is the sheet name
        sheets = get_sheets_from_folder()
        if dataset_key in sheets:
            df = read_sheet(sheets[dataset_key]["id"])
        else:
            raise ValueError(f"Dataset not found: {dataset_key}")

    # Normalize column names (replace Chinese character with ·s)
    df.columns = df.columns.str.replace("\u65b0", "\u00b7s", regex=False)

    _set_cache(dataset_key, df)
    return df


def _load_merge(config_key: str) -> pd.DataFrame:
    """Load and merge sheets according to a merge config."""
    config = MERGE_CONFIGS[config_key]
    sheets = get_sheets_from_folder()
    join = config["join"]

    # Load each sheet
    dfs = {}
    for sheet_def in config["sheets"]:
        name = sheet_def["name"]
        alias = sheet_def["alias"]
        if name not in sheets:
            raise ValueError(f"Sheet '{name}' not found in Google Drive folder")
        dfs[alias] = read_sheet(sheets[name]["id"])

    # Apply pre-merge transforms
    for alias, transform_names in config.get("pre_merge_transforms", {}).items():
        for t_name in transform_names:
            if t_name in TRANSFORM_REGISTRY:
                dfs[alias] = TRANSFORM_REGISTRY[t_name](dfs[alias])

    # Merge (assumes exactly 2 sheets)
    aliases = [s["alias"] for s in config["sheets"]]
    df = pd.merge(
        dfs[aliases[0]], dfs[aliases[1]],
        left_on=join["left_on"], right_on=join["right_on"],
        how=join["how"], suffixes=join["suffixes"],
    )

    # Apply post-merge transform
    post_transform = config.get("post_merge_transform")
    if post_transform and post_transform in TRANSFORM_REGISTRY:
        df = TRANSFORM_REGISTRY[post_transform](df)

    return df


def get_all_datasets() -> list[dict]:
    """Return metadata for all available datasets (merges + single sheets).

    Returns list of dicts with keys: key, display_name, type, description, default_filters.
    """
    datasets = []

    # Pre-configured merges first
    for key, config in MERGE_CONFIGS.items():
        datasets.append({
            "key": key,
            "display_name": config["display_name"],
            "type": "merge",
            "description": config.get("description", ""),
            "default_filters": config.get("default_filters", {}),
        })

    # Single sheets from Google Drive
    try:
        sheets = get_sheets_from_folder()
        for name, info in sheets.items():
            # Skip sheets that are part of a merge config (they're already represented)
            merge_sheet_names = set()
            for config in MERGE_CONFIGS.values():
                for s in config["sheets"]:
                    merge_sheet_names.add(s["name"])

            datasets.append({
                "key": name,
                "display_name": name,
                "type": "merge-source" if name in merge_sheet_names else "sheet",
                "description": "",
                "default_filters": _infer_default_filters(name),
            })
    except Exception:
        pass

    return datasets


def _infer_default_filters(sheet_name: str) -> dict:
    """Infer default filters based on sheet naming conventions.

    Sheets with SPR data typically have Chi2_ndof_RU2 and RMSE_RU columns.
    """
    if "_SPR_" in sheet_name:
        return {"Chi2_ndof_RU2": 10.0, "RMSE_RU": 10.0}
    return {}
