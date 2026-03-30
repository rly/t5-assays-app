"""Routes for the main page, dataset selector, viewing, filtering, and preferences."""
import json

from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models import User, UserPreference, DatasetSelection, Message, Conversation
from app.config import MODEL_MAPPING, MODEL_DISPLAY_NAMES
from app.utils import encrypt_api_key, decrypt_api_key
from app.services.merge_service import load_dataset, get_all_datasets
from app.services.filter_service import apply_filters
from app.services.chat_service import get_or_create_conversation, get_chat_messages

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_prefs(db: Session, user: User) -> UserPreference:
    """Get or create user preferences."""
    if not user.preferences:
        prefs = UserPreference(user_id=user.id)
        db.add(prefs)
        db.commit()
        db.refresh(prefs)
        return prefs
    return user.preferences


def get_user_api_key(prefs: UserPreference) -> str | None:
    """Get decrypted OpenRouter API key from user preferences."""
    if prefs.openrouter_api_key_encrypted:
        return decrypt_api_key(prefs.openrouter_api_key_encrypted)
    return None


def get_conversation_stats(db: Session, conv_id: int) -> tuple[int, float]:
    """Return (total_tokens, total_cost) for a conversation."""
    tokens = db.query(func.sum(Message.tokens_used)).filter(Message.conversation_id == conv_id).scalar() or 0
    cost = db.query(func.sum(Message.cost)).filter(Message.conversation_id == conv_id).scalar() or 0
    return tokens, cost


def _ensure_dataset_selections(db: Session, user: User):
    """Auto-populate DatasetSelection rows for any new datasets the user hasn't seen."""
    existing_keys = {ds.dataset_key for ds in user.dataset_selections}
    all_datasets = get_all_datasets()

    for ds in all_datasets:
        if ds["key"] not in existing_keys:
            selection = DatasetSelection(
                user_id=user.id,
                dataset_key=ds["key"],
                dataset_type=ds["type"],
                display_name=ds["display_name"],
                provided_to_ai=True,
                filters_json=json.dumps(ds.get("default_filters", {})),
            )
            db.add(selection)

    db.commit()
    db.refresh(user)


def _get_datasets_for_template(db: Session, user: User, prefs: UserPreference) -> list[dict]:
    """Build the dataset list for rendering the selector template."""
    _ensure_dataset_selections(db, user)

    datasets = []
    for sel in user.dataset_selections:
        filters = json.loads(sel.filters_json) if sel.filters_json else {}
        filter_parts = [f"{col} < {val}" for col, val in filters.items()]
        filter_summary = ", ".join(filter_parts) if filter_parts else ""

        # Get row count (cached data)
        row_count = None
        try:
            df = load_dataset(sel.dataset_key)
            if filters:
                df = apply_filters(df, filters)
            row_count = len(df)
        except Exception:
            pass

        datasets.append({
            "key": sel.dataset_key,
            "display_name": sel.display_name,
            "type": sel.dataset_type,
            "description": "",
            "provided_to_ai": sel.provided_to_ai,
            "filter_summary": filter_summary,
            "row_count": row_count,
        })

    # Sort: merges first, then sheets alphabetically
    datasets.sort(key=lambda d: (0 if d["type"] == "merge" else 1, d["display_name"]))
    return datasets


def _get_ai_dataset_info(db: Session, user: User) -> tuple[int, str]:
    """Return (count, comma-separated names) of datasets provided to AI."""
    provided = [ds for ds in user.dataset_selections if ds.provided_to_ai]
    count = len(provided)
    names = ", ".join(ds.display_name for ds in provided)
    return count, names


def _build_filter_panel(dataset_key: str, filters: dict) -> str:
    """Build an inline filter panel for the viewed dataset."""
    from html import escape

    # Detect which columns are filterable by checking the dataset
    try:
        df = load_dataset(dataset_key)
        import pandas as pd
        numeric_cols = []
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                numeric_cols.append(col)
    except Exception:
        numeric_cols = list(filters.keys())

    if not numeric_cols and not filters:
        return ""

    # Show filters for columns that already have a filter, plus a few common ones
    shown_cols = list(filters.keys())
    common_filter_cols = ["Chi2_ndof_RU2", "RMSE_RU", "KD_M", "FP binding (uM)"]
    for col in common_filter_cols:
        if col in numeric_cols and col not in shown_cols:
            shown_cols.append(col)

    if not shown_cols:
        return ""

    inputs = []
    for col in shown_cols:
        val = filters.get(col, "")
        val_str = str(val) if val else ""
        inputs.append(
            f'<label class="filter-label">{escape(col)} &lt; '
            f'<input type="number" name="filter_{escape(col)}" value="{val_str}" step="any" placeholder="no filter">'
            f'</label>'
        )

    inputs_html = "\n".join(inputs)
    return f'''
        <div class="filter-panel">
            <form class="filter-form" hx-post="/datasets/{dataset_key}/filters"
                  hx-target="#filter-reload-target" hx-swap="innerHTML">
                {inputs_html}
                <button type="submit" class="btn-sm secondary">Apply</button>
                <button type="button" class="btn-sm outline secondary"
                        onclick="this.closest('form').querySelectorAll('input[type=number]').forEach(i=>i.value=''); this.closest('form').requestSubmit();">
                    Clear
                </button>
            </form>
            <small class="filter-note">Filtered data is shown in the table and provided to the AI model.</small>
            <div id="filter-reload-target"></div>
        </div>
    '''


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def index(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Main application page."""
    prefs = get_prefs(db, user)
    datasets = _get_datasets_for_template(db, user, prefs)
    ai_count, ai_names = _get_ai_dataset_info(db, user)

    conv = get_or_create_conversation(db, user.id)
    messages = get_chat_messages(db, conv.id)
    conv_tokens, conv_cost = get_conversation_stats(db, conv.id)

    # Get viewing dataset info + filters
    viewing_key = prefs.viewing_dataset_key
    viewing_display_name = ""
    viewing_filter_html = ""
    if viewing_key:
        for ds in datasets:
            if ds["key"] == viewing_key:
                viewing_display_name = ds["display_name"]
                break
        sel = db.query(DatasetSelection).filter(
            DatasetSelection.user_id == user.id,
            DatasetSelection.dataset_key == viewing_key,
        ).first()
        filters = json.loads(sel.filters_json) if sel and sel.filters_json else {}
        viewing_filter_html = _build_filter_panel(viewing_key, filters)

    # Get selected model display name
    selected_model_name = MODEL_DISPLAY_NAMES.get(prefs.selected_model, "Select model")

    return templates.TemplateResponse(request, "index.html", {
        "user": user,
        "prefs": prefs,
        "models": MODEL_MAPPING,
        "datasets": datasets,
        "viewing_key": viewing_key,
        "viewing_display_name": viewing_display_name,
        "viewing_filter_html": viewing_filter_html,
        "ai_dataset_count": ai_count,
        "ai_dataset_names": ai_names,
        "selected_model_name": selected_model_name,
        "api_key": get_user_api_key(prefs) or "",
        "messages": messages,
        "model_label": MODEL_DISPLAY_NAMES.get(prefs.selected_model, "AI"),
        "conv_tokens": f"{conv_tokens:,}" if conv_tokens else "",
        "conv_cost": f"{conv_cost:.4f}" if conv_cost else "",
    })


@router.get("/partials/dataset-selector", response_class=HTMLResponse)
async def dataset_selector_partial(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """HTMX partial: refresh the dataset selector table."""
    prefs = get_prefs(db, user)
    datasets = _get_datasets_for_template(db, user, prefs)
    return templates.TemplateResponse(request, "partials/dataset_selector.html", {
        "datasets": datasets,
        "viewing_key": prefs.viewing_dataset_key,
    })


@router.post("/datasets/{key:path}/view", response_class=HTMLResponse)
async def view_dataset(key: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Toggle viewing a dataset. If already viewing, hide it."""
    prefs = get_prefs(db, user)

    # Toggle: if already viewing this dataset, stop viewing
    if prefs.viewing_dataset_key == key:
        prefs.viewing_dataset_key = None
        db.commit()
        return HTMLResponse('''
            <p class="empty-state">Select a dataset above and click "View" to see the data table.</p>
            <script>
                window.APP_CONFIG.viewingDataset = "";
                var plots = document.getElementById("plots-section");
                if (plots) plots.innerHTML = "";
            </script>
        ''')

    prefs.viewing_dataset_key = key
    db.commit()

    # Find display name and filters
    display_name = key
    filters = {}
    sel = db.query(DatasetSelection).filter(
        DatasetSelection.user_id == user.id,
        DatasetSelection.dataset_key == key,
    ).first()
    if sel:
        display_name = sel.display_name
        filters = json.loads(sel.filters_json) if sel.filters_json else {}

    filter_html = _build_filter_panel(key, filters)

    return HTMLResponse(f'''
        <div class="section-header">
            <h3>{display_name}</h3>
            <span id="row-count" class="badge"></span>
        </div>
        {filter_html}
        <div id="data-grid" style="height: 500px; width: 100%;"></div>
        <script>
            window.APP_CONFIG.viewingDataset = "{key}";
            loadData("{key}");
            var plotsSection = document.getElementById("plots-section");
            if (plotsSection) {{
                if (!document.getElementById("plots-container")) {{
                    plotsSection.innerHTML = '<h3>Data Summary</h3><div id="plots-container"></div>';
                }}
            }}
            loadPlots("{key}");
        </script>
    ''')


@router.post("/datasets/{key:path}/provide", response_class=HTMLResponse)
async def toggle_provide(key: str, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Toggle whether a dataset is provided to the AI."""
    sel = db.query(DatasetSelection).filter(
        DatasetSelection.user_id == user.id,
        DatasetSelection.dataset_key == key,
    ).first()
    if sel:
        sel.provided_to_ai = not sel.provided_to_ai
        db.commit()
        db.refresh(user)

    # Re-render the selector + update the AI badge via OOB
    prefs = get_prefs(db, user)
    datasets = _get_datasets_for_template(db, user, prefs)
    ai_count, ai_names = _get_ai_dataset_info(db, user)

    selector_html = templates.TemplateResponse(request, "partials/dataset_selector.html", {
        "datasets": datasets,
        "viewing_key": prefs.viewing_dataset_key,
    }).body.decode()

    badge_oob = f'<span id="ai-dataset-count" class="ai-dataset-badge" title="{ai_names}" hx-swap-oob="true">{ai_count} dataset{"s" if ai_count != 1 else ""} provided to AI</span>'

    return HTMLResponse(selector_html + badge_oob)


@router.post("/datasets/{key:path}/filters", response_class=HTMLResponse)
async def update_filters(key: str, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update per-dataset filters from form data and reload the data view."""
    form = await request.form()
    sel = db.query(DatasetSelection).filter(
        DatasetSelection.user_id == user.id,
        DatasetSelection.dataset_key == key,
    ).first()
    if sel:
        # Build filters dict from form — only include non-empty values
        filters = {}
        for field_name, value in form.items():
            if field_name.startswith("filter_") and value.strip():
                col_name = field_name[7:]  # strip "filter_" prefix
                try:
                    filters[col_name] = float(value)
                except ValueError:
                    pass
        sel.filters_json = json.dumps(filters)
        db.commit()

    # Reload the data view with updated filters
    return HTMLResponse(f'''
        <script>
            loadData("{key}");
            loadPlots("{key}");
        </script>
    ''')


@router.post("/prefs/model")
async def change_model(model: str = Form(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update selected AI model."""
    prefs = get_prefs(db, user)
    prefs.selected_model = model
    db.commit()
    return HTMLResponse("")


@router.post("/prefs/apikey")
async def change_apikey(openrouter_key: str = Form(""), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Save or clear the user's OpenRouter API key (encrypted at rest)."""
    prefs = get_prefs(db, user)
    prefs.openrouter_api_key_encrypted = encrypt_api_key(openrouter_key) if openrouter_key else None
    db.commit()
    return HTMLResponse("")
