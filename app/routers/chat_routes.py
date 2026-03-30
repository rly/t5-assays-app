"""Routes for AI chat: send messages, summarize, clear history."""
import json
from html import escape

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models import User, Message, DatasetSelection
from app.config import MODEL_DISPLAY_NAMES
from app.services.merge_service import load_dataset
from app.services.filter_service import apply_filters
from app.services.chat_service import (
    get_api_key, is_model_allowed, get_or_create_conversation,
    get_chat_messages, save_message, build_system_prompt,
    run_agent_chat, generate_summary_prompt, summarize_conversation,
    cleanup_response, MAX_RECENT_MESSAGES,
)
from app.routers.data_routes import get_user_api_key, get_conversation_stats

router = APIRouter(prefix="/chat")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_provided_datasets(db: Session, user: User) -> dict[str, "pd.DataFrame"]:
    """Load all datasets the user has toggled 'Provide to AI', with filters applied."""
    import pandas as pd
    datasets = {}
    selections = db.query(DatasetSelection).filter(
        DatasetSelection.user_id == user.id,
        DatasetSelection.provided_to_ai == True,
    ).all()

    for sel in selections:
        try:
            df = load_dataset(sel.dataset_key)
            filters = json.loads(sel.filters_json) if sel.filters_json else {}
            if filters:
                df = apply_filters(df, filters)
            # Use a clean variable-friendly name
            datasets[sel.dataset_key] = df
        except Exception:
            pass

    return datasets


def _system_msg(text: str) -> HTMLResponse:
    """Return a yellow system/warning message bubble."""
    return HTMLResponse(f'<div class="chat-msg chat-system"><div class="chat-msg-body">{text}</div></div>')


def _build_usage_html(usage_info: dict, elapsed: float = 0) -> str:
    if not usage_info:
        return ""
    pt = usage_info.get("prompt_tokens", 0)
    ct = usage_info.get("completion_tokens", 0)
    cost = usage_info.get("cost")
    if cost is None:
        cost = usage_info.get("total_cost")
    estimated = usage_info.get("cost_estimated", False)
    if cost is not None and cost > 0:
        prefix = "~" if estimated else ""
        cost_str = f" ({prefix}${cost:.4f})"
    elif cost is not None and cost == 0:
        cost_str = " (free)"
    else:
        cost_str = ""
    time_str = f" {elapsed:.1f}s" if elapsed else ""
    return f'<div class="chat-usage">{pt:,} in, {ct:,} out{cost_str}{time_str}</div>'


def _build_tool_steps_html(tool_steps: list[dict]) -> str:
    if not tool_steps:
        return ""
    steps_html = ""
    i = 0
    step_num = 0
    while i < len(tool_steps):
        step = tool_steps[i]
        if step["type"] == "call":
            step_num += 1
            code = step.get("args", "")
            try:
                args = json.loads(code) if isinstance(code, str) else code
                if isinstance(args, dict) and "code" in args:
                    code = args["code"]
            except (json.JSONDecodeError, TypeError):
                pass
            output = ""
            if i + 1 < len(tool_steps) and tool_steps[i + 1]["type"] == "return":
                output = tool_steps[i + 1].get("output", "")
                i += 1
            steps_html += f'''
                <div class="tool-step">
                    <div class="tool-step-label">Tool call {step_num}: run_python</div>
                    <details><summary>Show code</summary><pre><code>{escape(code)}</code></pre></details>
                    <div class="tool-step-output"><strong>Output:</strong><pre>{escape(output)}</pre></div>
                </div>
            '''
        i += 1
    if not steps_html:
        return ""
    return f'''
        <details class="tool-steps-container">
            <summary>Show {step_num} tool call{"s" if step_num != 1 else ""}</summary>
            {steps_html}
        </details>
    '''


def _build_response_html(model_id: str, content: str, usage_info: dict,
                         tool_steps: list[dict] = None, elapsed: float = 0) -> str:
    model_label = MODEL_DISPLAY_NAMES.get(model_id, model_id)
    usage_html = _build_usage_html(usage_info, elapsed)
    tool_html = _build_tool_steps_html(tool_steps or [])
    has_tools = any(s.get("type") == "call" for s in (tool_steps or []))
    computed_badge = '<span class="computed-badge">Computed via Python</span>' if has_tools else ""
    return f'''
        <div class="chat-msg chat-assistant">
            <div class="chat-msg-header">{model_label}{computed_badge}{" &middot; " + usage_html if usage_html else ""}</div>
            {tool_html}
            <div class="chat-msg-body"><div class="markdown-content">{content}</div></div>
        </div>
    '''


def _build_session_oob(conv_tokens: int, conv_cost: float) -> str:
    cost_str = f" (~${conv_cost:.4f})" if conv_cost else ""
    return f'<small id="session-cost" hx-swap-oob="true">Chat: ~{conv_tokens:,} tokens{cost_str}</small>'


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/history", response_class=HTMLResponse)
async def chat_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    prefs = user.preferences
    if not prefs:
        return HTMLResponse("")
    conv = get_or_create_conversation(db, user.id)
    messages = get_chat_messages(db, conv.id)

    html_parts = []
    for msg in messages:
        if msg["role"] == "user":
            role_label = "You"
            usage_html = ""
        else:
            model_id = msg.get("model_used") or prefs.selected_model
            role_label = MODEL_DISPLAY_NAMES.get(model_id, model_id or "AI")
            tokens = msg.get("tokens_used")
            usage_html = f' &middot; <span class="chat-usage">{tokens:,} tokens</span>' if tokens else ""
        css_class = f"chat-{msg['role']}"
        content = msg["content"]
        body = f'<div class="markdown-content">{content}</div>' if msg["role"] == "assistant" else content
        html_parts.append(f'''
            <div class="chat-msg {css_class}">
                <div class="chat-msg-header">{role_label}{usage_html}</div>
                <div class="chat-msg-body">{body}</div>
            </div>
        ''')
    return HTMLResponse("\n".join(html_parts))


async def _handle_chat(db, conv, existing_messages, message, api_key, model_id, datasets):
    """Run the agentic AI chat with multiple datasets."""
    save_message(db, conv.id, "user", message)

    if len(existing_messages) > MAX_RECENT_MESSAGES and not conv.summary:
        older = existing_messages[:-MAX_RECENT_MESSAGES]
        summary = await summarize_conversation(older, api_key, model_id)
        if summary:
            conv.summary = summary
            db.commit()

    system_prompt = build_system_prompt(datasets)

    result = await run_agent_chat(
        api_key, model_id, system_prompt,
        existing_messages, message, datasets, conv.summary,
    )

    full_response = cleanup_response(result["content"])
    usage_info = result["usage"]
    tool_steps = result.get("tool_steps", [])
    elapsed = result.get("elapsed", 0)

    save_message(db, conv.id, "assistant", full_response,
                 model_used=model_id, tokens=usage_info.get("total_tokens"), cost=usage_info.get("cost"))

    conv_tokens, conv_cost = get_conversation_stats(db, conv.id)
    return full_response, usage_info, tool_steps, conv_tokens, conv_cost, elapsed


@router.post("/send", response_class=HTMLResponse)
async def send_message(request: Request, message: str = Form(...),
                       user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    prefs = user.preferences
    if not prefs:
        return _system_msg("No preferences found. Please log out and log back in.")

    user_key = get_user_api_key(prefs) if prefs else None
    api_key = get_api_key(user_key)
    model_id = prefs.selected_model

    if not api_key:
        return _system_msg("Please provide an OpenRouter API key in the AI Model settings.")
    if not is_model_allowed(model_id, user_key):
        return _system_msg("This model requires your own API key. Please enter it or select a free model.")

    try:
        datasets = _load_provided_datasets(db, user)
        if not datasets:
            return _system_msg("No datasets provided to AI. Check the AI checkbox next to datasets in the selector above.")

        conv = get_or_create_conversation(db, user.id)
        existing_messages = get_chat_messages(db, conv.id)

        full_response, usage_info, tool_steps, conv_tokens, conv_cost, elapsed = await _handle_chat(
            db, conv, existing_messages, message, api_key, model_id, datasets,
        )

        return HTMLResponse(
            _build_response_html(model_id, full_response, usage_info, tool_steps, elapsed)
            + _build_session_oob(conv_tokens, conv_cost)
        )
    except Exception as e:
        return _system_msg(f"Something went wrong: {str(e)}")


@router.post("/summarize", response_class=HTMLResponse)
async def summarize(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    prefs = user.preferences
    if not prefs:
        return HTMLResponse("")

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
            return _system_msg("No datasets provided to AI.")

        prompt = generate_summary_prompt(list(datasets.keys()))
        conv = get_or_create_conversation(db, user.id)
        existing_messages = get_chat_messages(db, conv.id)

        full_response, usage_info, tool_steps, conv_tokens, conv_cost, elapsed = await _handle_chat(
            db, conv, existing_messages, prompt, api_key, model_id, datasets,
        )

        return HTMLResponse(f'''
            <div class="chat-msg chat-user">
                <div class="chat-msg-header">You</div>
                <div class="chat-msg-body">{prompt}</div>
            </div>
            {_build_response_html(model_id, full_response, usage_info, tool_steps, elapsed)}
            {_build_session_oob(conv_tokens, conv_cost)}
        ''')
    except Exception as e:
        return _system_msg(f"Something went wrong: {str(e)}")


@router.post("/clear", response_class=HTMLResponse)
async def clear_chat(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = get_or_create_conversation(db, user.id)
    db.query(Message).filter(Message.conversation_id == conv.id).delete()
    conv.summary = None
    db.commit()
    return HTMLResponse('<small id="session-cost" hx-swap-oob="true"></small>')
