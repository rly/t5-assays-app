import pandas as pd


def apply_filters(df: pd.DataFrame, chi2_max: float = 1e9, rmse_max: float = 1e9) -> pd.DataFrame:
    """Apply Chi2 and RMSE filters to a DataFrame. NaN values pass through."""
    result = df.copy()

    if "Chi2_ndof_RU2" in result.columns and chi2_max < 1e9:
        result["Chi2_ndof_RU2"] = pd.to_numeric(result["Chi2_ndof_RU2"], errors="coerce")
        result = result[(result["Chi2_ndof_RU2"] < chi2_max) | result["Chi2_ndof_RU2"].isna()]

    if "RMSE_RU" in result.columns and rmse_max < 1e9:
        result["RMSE_RU"] = pd.to_numeric(result["RMSE_RU"], errors="coerce")
        result = result[(result["RMSE_RU"] < rmse_max) | result["RMSE_RU"].isna()]

    return result
