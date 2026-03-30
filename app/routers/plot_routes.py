"""Routes for Plotly correlation plots and custom scatter plots."""
import json

from fastapi import APIRouter, Depends, Form, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models import User, DatasetSelection
from app.services.merge_service import load_dataset
from app.services.filter_service import apply_filters
from app.services.plot_service import generate_peitho_plots, generate_custom_plot

router = APIRouter()


def _load_filtered(db: Session, user: User, dataset_key: str):
    """Load a dataset with per-user filters applied."""
    df = load_dataset(dataset_key)
    sel = db.query(DatasetSelection).filter(
        DatasetSelection.user_id == user.id,
        DatasetSelection.dataset_key == dataset_key,
    ).first()
    if sel and sel.filters_json:
        filters = json.loads(sel.filters_json)
        if filters:
            df = apply_filters(df, filters)
    return df


@router.get("/plots", response_class=HTMLResponse)
async def get_plots(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    dataset_key: str = Query(""),
):
    if not dataset_key:
        return HTMLResponse("")

    try:
        df = _load_filtered(db, user, dataset_key)
    except Exception:
        return HTMLResponse("<p>Error loading data for plots.</p>")

    html_parts = []

    # Show preset correlation plots for PEITHO merge
    if dataset_key == "veev_peitho_merge":
        plots_json = generate_peitho_plots(df)
        html_parts.append('<div class="plots-row">')
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

    # Custom plot controls
    columns = df.columns.tolist()
    html_parts.append(_custom_plot_form(columns, dataset_key))

    return HTMLResponse("\n".join(html_parts))


@router.post("/plots/custom", response_class=HTMLResponse)
async def custom_plot(
    x_col: str = Form(...), y_col: str = Form(...),
    dataset_key: str = Form(""),
    user: User = Depends(get_current_user), db: Session = Depends(get_db),
):
    if not dataset_key:
        return HTMLResponse("<p>No dataset selected.</p>")

    df = _load_filtered(db, user, dataset_key)
    plot_json = generate_custom_plot(df, x_col, y_col)

    return HTMLResponse(f'''
        <div id="custom-plot-result" style="width:100%; min-height:420px;"></div>
        <script>
            (function() {{
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


def _custom_plot_form(columns: list[str], dataset_key: str) -> str:
    options = "\n".join(f'<option value="{c}">{c}</option>' for c in columns)
    return f'''
        <details class="custom-plot-panel">
            <summary>Custom Plot</summary>
            <form class="custom-plot-form" hx-post="/plots/custom" hx-target="#custom-plot-output" hx-swap="innerHTML">
                <input type="hidden" name="dataset_key" value="{dataset_key}">
                <select name="x_col"><option disabled selected>X axis</option>{options}</select>
                <select name="y_col"><option disabled selected>Y axis</option>{options}</select>
                <button type="submit">Plot</button>
            </form>
            <div id="custom-plot-output"></div>
        </details>
    '''
