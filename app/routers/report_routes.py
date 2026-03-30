"""Routes for generating comprehensive analysis reports."""
from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models import User
from app.config import MODEL_DISPLAY_NAMES
from app.services.chat_service import (
    get_api_key, is_model_allowed, get_or_create_conversation,
    get_chat_messages, save_message, build_system_prompt,
    run_agent_chat, cleanup_response, MAX_RECENT_MESSAGES,
)
from app.routers.data_routes import get_user_api_key, get_conversation_stats
from app.routers.chat_routes import (
    _load_provided_datasets, _system_msg, _build_response_html, _build_session_oob,
)

router = APIRouter(prefix="/report")

REPORT_PROMPT = """Generate a comprehensive analysis report for the provided datasets. Structure the report as follows:

## 1. Dataset Overview
- For each dataset: name, number of rows/columns, key columns, and what assay/experiment it represents.

## 2. Key Findings
- Top 10 binding candidates (ranked by the most relevant binding metric for each dataset).
- Identify compounds that appear across multiple datasets.

## 3. Assay Comparison (if multiple datasets)
- Where do different assays agree on strong binders?
- Where do they disagree? Flag compounds with conflicting results.
- Correlation between AI-predicted binding scores and experimental results.

## 4. Data Quality
- Flag any data quality issues (missing values, outliers, suspicious values).
- Note columns with high proportions of missing data.

## 5. Recommendations
- Top 5 compounds to prioritize for further study, with justification.
- Suggested follow-up experiments.

Use the run_python tool to compute all statistics. Be specific — include compound names, values, and rankings.
Present the report in clean Markdown with tables where appropriate."""


@router.post("/generate", response_class=HTMLResponse)
async def generate_report(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Generate a comprehensive analysis report across all provided datasets."""
    prefs = user.preferences
    if not prefs:
        return _system_msg("No preferences found.")

    user_key = get_user_api_key(prefs) if prefs else None
    api_key = get_api_key(user_key)
    model_id = prefs.selected_model

    if not api_key:
        return _system_msg("Please provide an OpenRouter API key.")
    if not is_model_allowed(model_id, user_key):
        return _system_msg("This model requires your own API key.")

    try:
        datasets = _load_provided_datasets(db, user)
        if not datasets:
            return _system_msg("No datasets provided to AI. Check the AI checkbox next to datasets above.")

        conv = get_or_create_conversation(db, user.id)
        existing_messages = get_chat_messages(db, conv.id)

        save_message(db, conv.id, "user", REPORT_PROMPT)

        system_prompt = build_system_prompt(datasets)

        result = await run_agent_chat(
            api_key, model_id, system_prompt,
            existing_messages, REPORT_PROMPT, datasets, conv.summary,
        )

        full_response = cleanup_response(result["content"])
        usage_info = result["usage"]
        tool_steps = result.get("tool_steps", [])
        elapsed = result.get("elapsed", 0)

        save_message(db, conv.id, "assistant", full_response,
                     model_used=model_id, tokens=usage_info.get("total_tokens"), cost=usage_info.get("cost"))

        conv_tokens, conv_cost = get_conversation_stats(db, conv.id)

        # Show user prompt + AI response
        return HTMLResponse(f'''
            <div class="chat-msg chat-user">
                <div class="chat-msg-header">You</div>
                <div class="chat-msg-body">Generate comprehensive analysis report</div>
            </div>
            {_build_response_html(model_id, full_response, usage_info, tool_steps, elapsed)}
            {_build_session_oob(conv_tokens, conv_cost)}
        ''')
    except Exception as e:
        return _system_msg(f"Report generation failed: {str(e)}")
