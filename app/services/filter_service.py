"""Generic column-based filtering for DataFrames.

Applies "less than" filters to numeric columns. NaN values pass through.
"""
import pandas as pd


def apply_filters(df: pd.DataFrame, filters: dict[str, float]) -> pd.DataFrame:
    """Apply column filters to a DataFrame.

    Args:
        df: Input DataFrame
        filters: Dict of {column_name: max_value}. Only rows where column < max_value are kept.
                 NaN values in filtered columns pass through (OR logic).

    Returns:
        Filtered copy of the DataFrame.
    """
    if not filters:
        return df

    result = df.copy()
    for col, max_val in filters.items():
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
            result = result[(result[col] < max_val) | result[col].isna()]

    return result
