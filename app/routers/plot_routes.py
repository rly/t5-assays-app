"""Routes for Plotly correlation plots and custom scatter plots."""
from fastapi import APIRouter, Depends, Form
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models import User
from app.services.merge_service import load_data
from app.services.filter_service import apply_filters
from app.services.plot_service import generate_peitho_plots, generate_custom_plot

router = APIRouter()


@router.get("/plots", response_class=HTMLResponse)
async def get_plots(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    prefs = user.preferences
    if not prefs:
        return HTMLResponse("<p>No data source selected.</p>")

    df = load_data(prefs.data_source_type)

    if prefs.data_source_type == "veev_peitho_merge":
        df_filtered = apply_filters(df, prefs.chi2_max, prefs.rmse_max)
        plots_json = generate_peitho_plots(df_filtered)

        html_parts = ['<div class="plots-row">']
        for i, pj in enumerate(plots_json):
            html_parts.append(f'''
                <div class="plot-wrapper">
                    <div id="plot-{i}"></div>
                    <script>
                        (function() {{
                            setTimeout(function() {{
                                var el = document.getElementById('plot-{i}');
                                if (!el) return;
                                var fig = {pj};
                                Plotly.newPlot(el, fig.data, fig.layout, {{
                                    responsive: true,
                                    displayModeBar: false
                                }});
                            }}, 50);
                        }})();
                    </script>
                </div>
            ''')
        html_parts.append('</div>')

        # Add custom plot controls
        columns = df_filtered.columns.tolist()
        html_parts.append(_custom_plot_form(columns))

        return HTMLResponse("\n".join(html_parts))
    else:
        # Generic: show custom plot controls only
        columns = df.columns.tolist()
        return HTMLResponse(_custom_plot_form(columns))


@router.post("/plots/custom", response_class=HTMLResponse)
async def custom_plot(
    x_col: str = Form(...), y_col: str = Form(...),
    user: User = Depends(get_current_user), db: Session = Depends(get_db),
):
    prefs = user.preferences
    df = load_data(prefs.data_source_type)
    if prefs.data_source_type == "veev_peitho_merge":
        df = apply_filters(df, prefs.chi2_max, prefs.rmse_max)

    plot_json = generate_custom_plot(df, x_col, y_col)

    return HTMLResponse(f'''
        <div id="custom-plot-result" style="width:100%; min-height:420px;"></div>
        <script>
            (function() {{
                // Use setTimeout to ensure the previous plot is fully purged from DOM
                setTimeout(function() {{
                    var el = document.getElementById('custom-plot-result');
                    if (!el) return;
                    var fig = {plot_json};
                    Plotly.newPlot(el, fig.data, fig.layout, {{
                        responsive: true,
                        displayModeBar: false
                    }});
                }}, 50);
            }})();
        </script>
    ''')


def _custom_plot_form(columns: list[str]) -> str:
    options = "\n".join(f'<option value="{c}">{c}</option>' for c in columns)
    return f'''
        <details class="custom-plot-panel">
            <summary>Custom Plot</summary>
            <form class="custom-plot-form" hx-post="/plots/custom" hx-target="#custom-plot-output" hx-swap="innerHTML">
                <select name="x_col"><option disabled selected>X axis</option>{options}</select>
                <select name="y_col"><option disabled selected>Y axis</option>{options}</select>
                <button type="submit">Plot</button>
            </form>
            <div id="custom-plot-output"></div>
        </details>
    '''
