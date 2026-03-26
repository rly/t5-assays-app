import json
import numpy as np
import pandas as pd
import plotly.express as px


def generate_peitho_plots(df: pd.DataFrame) -> list[str]:
    """Generate 6 correlation plots for PEITHO data. Returns list of Plotly JSON strings."""
    # Filter for good values
    filtered = df.copy()
    if "RMSE_RU" in filtered.columns:
        filtered["RMSE_RU"] = pd.to_numeric(filtered["RMSE_RU"], errors="coerce")
        filtered = filtered[filtered["RMSE_RU"] < 10]
    if "Chi2_ndof_RU2" in filtered.columns:
        filtered["Chi2_ndof_RU2"] = pd.to_numeric(filtered["Chi2_ndof_RU2"], errors="coerce")
        filtered = filtered[filtered["Chi2_ndof_RU2"] < 10]

    plots = []

    correlation_plots = [
        ("KD_M", "VEEV - AI Binding Score"),
        ("kD[1/s]", "VEEV - AI Binding Score"),
        ("KD_M", "LogP"),
    ]

    for y_col, x_col in correlation_plots:
        fig_json = _make_correlation_plot(filtered, x_col, y_col)
        plots.append(fig_json)

    return plots


def _make_correlation_plot(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    """Create a scatter plot with OLS trendline and return Plotly JSON."""
    if x_col not in df.columns or y_col not in df.columns:
        return _empty_plot(f"Column not found: {x_col} or {y_col}")

    plot_df = df[[x_col, y_col]].copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna()
    plot_df = plot_df[plot_df[y_col] > 0]

    if len(plot_df) == 0:
        return _empty_plot(f"No valid data for {y_col} vs {x_col}")

    corr = np.corrcoef(plot_df[x_col], plot_df[y_col])[0, 1]

    # Short axis labels for titles
    short_names = {
        "VEEV - AI Binding Score": "AI Binding",
        "Hydrogen bonds donors": "H-bond donors",
        "Hydrogen bonds acceptors": "H-bond acceptors",
        "kA[1/(M\u00b7s)]": "kA",
        "kD[1/s]": "kD",
    }
    sx = short_names.get(x_col, x_col)
    sy = short_names.get(y_col, y_col)
    title = f"{sy} vs {sx} (r={corr:.2f})"

    fig = px.scatter(
        plot_df, x=x_col, y=y_col,
        trendline="ols", trendline_options=dict(log_y=True),
        title=title,
    )
    fig.update_layout(
        xaxis_title=x_col, yaxis_title=y_col,
        height=300, margin=dict(l=40, r=20, t=40, b=40),
        title_font_size=13,
    )

    return json.dumps(fig, cls=_PlotlyEncoder)


def generate_custom_plot(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    """Generate a custom scatter plot for user-selected columns."""
    if x_col not in df.columns or y_col not in df.columns:
        return _empty_plot(f"Column not found: {x_col} or {y_col}")

    plot_df = df[[x_col, y_col]].copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna()

    if len(plot_df) == 0:
        return _empty_plot(f"No valid numeric data for {x_col} vs {y_col}")

    corr = np.corrcoef(plot_df[x_col], plot_df[y_col])[0, 1] if len(plot_df) > 1 else 0
    fig = px.scatter(
        plot_df, x=x_col, y=y_col,
        trendline="ols",
        title=f"{y_col} vs {x_col} (r={corr:.2f})",
    )

    fig.update_layout(height=400, title_font_size=13, margin=dict(l=60, r=20, t=50, b=50))
    return json.dumps(fig, cls=_PlotlyEncoder)


def _empty_plot(message: str) -> str:
    """Return a minimal Plotly figure JSON with a message."""
    fig = px.scatter(title=message)
    fig.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40))
    return json.dumps(fig, cls=_PlotlyEncoder)


class _PlotlyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_plotly_json"):
            return obj.to_plotly_json()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
