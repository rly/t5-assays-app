"""Routes for the main page, sidebar, data source/filter changes, and user preferences."""
from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models import User, UserPreference, Message
from app.config import MODEL_MAPPING, MODEL_DISPLAY_NAMES
from app.utils import encrypt_api_key, decrypt_api_key
from app.services.sheets_service import get_sheets_from_folder
from app.services.merge_service import load_data
from app.services.filter_service import apply_filters
from app.services.chat_service import get_or_create_conversation, get_chat_messages

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


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
    tokens = db.query(func.sum(Message.tokens_used)).filter(
        Message.conversation_id == conv_id,
    ).scalar() or 0
    cost = db.query(func.sum(Message.cost)).filter(
        Message.conversation_id == conv_id,
    ).scalar() or 0
    return tokens, cost


def _get_row_counts(prefs) -> tuple[int, int]:
    """Return (total_rows, filtered_rows) for the current data source."""
    try:
        df = load_data(prefs.data_source_type)
        total = len(df)
        if prefs.data_source_type == "veev_peitho_merge":
            filtered_df = apply_filters(df, prefs.chi2_max, prefs.rmse_max)
            return total, len(filtered_df)
        return total, total
    except Exception:
        return 0, 0


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Main application page."""
    prefs = get_prefs(db, user)
    sheets = get_sheets_from_folder() if prefs.data_source_type == "single_sheet" else {}

    conv = get_or_create_conversation(db, user.id, prefs.data_source_type)
    messages = get_chat_messages(db, conv.id)
    conv_tokens, conv_cost = get_conversation_stats(db, conv.id)

    # Compute row counts for filter display
    total_rows, filtered_rows = _get_row_counts(prefs)

    return templates.TemplateResponse(request, "index.html", {
        "user": user,
        "prefs": prefs,
        "models": MODEL_MAPPING,
        "sheets": sheets,
        "selected_sheet_id": None,
        "api_key": get_user_api_key(prefs) or "",
        "messages": messages,
        "model_label": MODEL_DISPLAY_NAMES.get(prefs.selected_model, "AI"),
        "conv_tokens": f"{conv_tokens:,}" if conv_tokens else "",
        "conv_cost": f"{conv_cost:.4f}" if conv_cost else "",
        "total_rows": total_rows,
        "filtered_rows": filtered_rows,
    })


@router.get("/partials/sidebar", response_class=HTMLResponse)
async def sidebar_partial(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Sidebar partial for HTMX reload after data source change."""
    prefs = get_prefs(db, user)
    sheets = get_sheets_from_folder() if prefs.data_source_type == "single_sheet" else {}

    total_rows, filtered_rows = _get_row_counts(prefs)

    return templates.TemplateResponse(request, "partials/sidebar.html", {
        "prefs": prefs,
        "models": MODEL_MAPPING,
        "sheets": sheets,
        "selected_sheet_id": None,
        "api_key": get_user_api_key(prefs) or "",
        "total_rows": total_rows,
        "filtered_rows": filtered_rows,
    })


@router.post("/data/source")
async def change_source(source: str = Form(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Switch data source (PEITHO merge, PARG merge, or single sheet)."""
    prefs = get_prefs(db, user)
    prefs.data_source_type = source
    db.commit()
    return HTMLResponse('<div hx-trigger="load" hx-swap-oob="true"></div>')


@router.post("/data/filters")
async def change_filters(chi2_max: float = Form(10.0), rmse_max: float = Form(10.0),
                         user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update Chi2/RMSE filter thresholds."""
    prefs = get_prefs(db, user)
    prefs.chi2_max = chi2_max
    prefs.rmse_max = rmse_max
    db.commit()
    return HTMLResponse('<div hx-trigger="load" hx-swap-oob="true"></div>')


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
