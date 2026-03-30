"""AI chat service: agent creation, tool calling, conversation management.

Uses Pydantic AI with OpenRouter to provide an agentic chat that can execute
Python code against the dataset via the run_python tool.
"""
import json
import time
from dataclasses import dataclass

import httpx
import pandas as pd
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from sqlalchemy.orm import Session

from app.config import settings, MODEL_MAPPING, FREE_MODELS
from app.models import Conversation, Message
from app.services.sandbox_service import execute_code

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
MAX_RECENT_MESSAGES = 6

# Cache of model pricing: {model_id: {"prompt": float, "completion": float}}
_model_pricing: dict[str, dict[str, float]] = {}


async def _ensure_pricing_loaded():
    """Fetch model pricing from OpenRouter if not cached."""
    if _model_pricing:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(OPENROUTER_MODELS_URL)
            if resp.status_code == 200:
                for m in resp.json().get("data", []):
                    p = m.get("pricing", {})
                    try:
                        _model_pricing[m["id"]] = {
                            "prompt": float(p.get("prompt", 0)),
                            "completion": float(p.get("completion", 0)),
                        }
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass


def estimate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimate cost from cached pricing. Returns None if pricing unavailable."""
    pricing = _model_pricing.get(model_id)
    if not pricing:
        return None
    return pricing["prompt"] * prompt_tokens + pricing["completion"] * completion_tokens


def get_api_key(user_key: str | None) -> str:
    """Return the user's key or the default key."""
    return user_key or settings.openrouter_default_api_key


def is_model_allowed(model_id: str, user_key: str | None) -> bool:
    """Check if the model can be used with the given key."""
    if model_id in FREE_MODELS:
        return True
    return bool(user_key)


def get_or_create_conversation(db: Session, user_id: int) -> Conversation:
    """Get or create the single conversation for a user."""
    conv = db.query(Conversation).filter(Conversation.user_id == user_id).first()
    if not conv:
        conv = Conversation(user_id=user_id)
        db.add(conv)
        db.commit()
        db.refresh(conv)
    return conv


def save_message(db: Session, conversation_id: int, role: str, content: str, model_used: str = None, tokens: int = None, cost: float = None):
    msg = Message(
        conversation_id=conversation_id, role=role, content=content,
        model_used=model_used, tokens_used=tokens, cost=cost,
    )
    db.add(msg)
    db.commit()


def get_chat_messages(db: Session, conversation_id: int) -> list[dict]:
    msgs = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at).all()
    return [{"role": m.role, "content": m.content, "tokens_used": m.tokens_used, "model_used": m.model_used} for m in msgs]


def cleanup_response(text: str) -> str:
    """Post-process AI response to fix common formatting issues."""
    lines = text.split("\n")
    result = []
    in_code_block = False
    unfenced_code_lines = []

    def flush_unfenced():
        if unfenced_code_lines:
            result.append("```python")
            result.extend(unfenced_code_lines)
            result.append("```")
            unfenced_code_lines.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_unfenced()
            in_code_block = not in_code_block
            result.append(line)
            continue
        if in_code_block:
            result.append(line)
            continue
        is_code_like = (
            stripped.startswith(("import ", "from ", "df[", "df.", "top", "print(", "for ", "if ", "result"))
            and not stripped.startswith(("import**", "importantly", "from the", "from a"))
            and ("=" in stripped or "(" in stripped or stripped.startswith("import "))
        )
        if is_code_like:
            unfenced_code_lines.append(line)
        else:
            flush_unfenced()
            result.append(line)

    flush_unfenced()
    return "\n".join(result)


@dataclass
class ChatDeps:
    """Dependencies passed to the agent tools at runtime."""
    datasets: dict[str, pd.DataFrame]  # {name: DataFrame}


def _describe_dataset(name: str, df: pd.DataFrame) -> str:
    """Generate column summary for a single dataset, with human-readable descriptions."""
    from app.column_descriptions import COLUMN_DESCRIPTIONS

    col_descriptions = []
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        human_desc = COLUMN_DESCRIPTIONS.get(col, "")
        label = f"{col} — {human_desc}" if human_desc else col
        try:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().sum() > non_null * 0.5 and numeric.notna().sum() > 0:
                desc = f"  - {label} (numeric, {non_null} non-null, min={numeric.min():.4g}, max={numeric.max():.4g}, mean={numeric.mean():.4g})"
            else:
                raise ValueError()
        except (ValueError, TypeError):
            n_unique = df[col].nunique()
            sample = df[col].dropna().head(3).tolist()
            desc = f"  - {label} (text, {non_null} non-null, {n_unique} unique, e.g. {sample})"
        col_descriptions.append(desc)

    var_name = name.replace(" ", "_").replace("-", "_")
    return (
        f'Dataset "{name}" (variable: {var_name}, {len(df)} rows x {len(df.columns)} cols):\n'
        + "\n".join(col_descriptions)
    )


def build_system_prompt(datasets: dict[str, pd.DataFrame]) -> str:
    """Build the system prompt with summaries for all provided datasets."""
    dataset_descriptions = []
    for name, df in datasets.items():
        dataset_descriptions.append(_describe_dataset(name, df))

    data_context = "\n\n".join(dataset_descriptions)

    # Build variable list for the prompt
    var_list = []
    for name in datasets:
        var_name = name.replace(" ", "_").replace("-", "_")
        var_list.append(f'  - {var_name} (or datasets["{name}"])')

    if len(datasets) == 1:
        access_note = "For convenience, the single dataset is also available as `df`.\n"
    else:
        access_note = ""

    return (
        "You are a helpful data analyst assistant for alphaviral macrodomain assay data. "
        "You help scientists identify promising small molecule therapeutic candidates.\n\n"
        "IMPORTANT FORMATTING RULES:\n"
        "- Always format your responses in Markdown.\n"
        "- Use bullet points, numbered lists, and bold text for readability.\n"
        "- Keep explanations concise and focused on actionable insights.\n"
        "- When showing data or results, use fenced code blocks: ```\\n...\\n```\n\n"
        "COMPUTATION:\n"
        "- You have a `run_python` tool that executes Python code.\n"
        f"- Available datasets ({len(datasets)}):\n" + "\n".join(var_list) + "\n"
        f"{access_note}"
        "- ALWAYS use the run_python tool to access, query, or analyze data. You can see column summaries below but NOT the raw data.\n"
        "- Use print() in your code to produce output.\n"
        "- You can call the tool multiple times to do multi-step analysis.\n"
        "- NEVER guess, fabricate, or estimate data values. Always compute them using run_python.\n"
        "- If your code produces an error, fix it and try again.\n\n"
        f"Data context:\n{data_context}\n\n"
        "Answer questions accurately. Highlight key findings and actionable insights."
    )


def create_agent(api_key: str, model_id: str) -> Agent[ChatDeps, str]:
    """Create a Pydantic AI agent configured for OpenRouter."""
    model = OpenRouterModel(
        model_id,
        provider=OpenRouterProvider(api_key=api_key),
    )

    agent = Agent(
        model,
        deps_type=ChatDeps,
        output_type=str,
    )

    @agent.tool
    def run_python(ctx: RunContext[ChatDeps], code: str) -> str:
        """Execute Python code against the provided datasets.
        Available variables: each dataset as a named variable (e.g., VEEV_PEITHO_SPR), plus a 'datasets' dict.
        If only one dataset is provided, 'df' is also available.
        Use print() to output results. Only pandas, numpy, math, statistics, re, datetime are available."""
        result = execute_code(code, ctx.deps.datasets)
        if result["success"]:
            output = result.get("output", "").strip()
            return output if output else "(no output — make sure to use print())"
        else:
            error = result.get("error", "Unknown error")
            error_lines = error.strip().split("\n")
            return f"Error: {error_lines[-1]}"

    return agent


async def run_agent_chat(
    api_key: str, model_id: str, system_prompt: str,
    conversation_messages: list[dict], user_message: str,
    datasets: dict[str, pd.DataFrame],
    conversation_summary: str | None = None,
) -> dict:
    """Run the agent with tool calling. Returns {"content": str, "usage": dict, "tool_steps": list, "elapsed": float}."""
    await _ensure_pricing_loaded()

    agent = create_agent(api_key, model_id)

    # Build message history for the agent
    # Pydantic AI expects its own message format, but we store simple role/content pairs.
    # We'll pass conversation context via the system prompt instead of message_history,
    # since our stored messages don't include tool call details.
    history_text = ""
    previous = conversation_messages[:]
    if len(previous) > MAX_RECENT_MESSAGES:
        recent = previous[-MAX_RECENT_MESSAGES:]
        if conversation_summary:
            history_text = f"[Previous conversation summary: {conversation_summary}]\n\n"
        for m in recent:
            role = "User" if m["role"] == "user" else "Assistant"
            history_text += f"{role}: {m['content']}\n\n"
    else:
        for m in previous:
            role = "User" if m["role"] == "user" else "Assistant"
            history_text += f"{role}: {m['content']}\n\n"

    # Combine system prompt with history context
    full_system_prompt = system_prompt
    if history_text:
        full_system_prompt += f"\n\nConversation history:\n{history_text}"

    # Run the agent with timing
    start_time = time.monotonic()

    result = await agent.run(
        user_message,
        deps=ChatDeps(datasets=datasets),
        instructions=full_system_prompt,
    )

    elapsed = time.monotonic() - start_time

    content = result.output
    usage = result.usage()

    usage_info = {
        "prompt_tokens": usage.input_tokens or 0,
        "completion_tokens": usage.output_tokens or 0,
        "total_tokens": (usage.input_tokens or 0) + (usage.output_tokens or 0),
    }

    # Estimate cost
    pt = usage_info["prompt_tokens"]
    ct = usage_info["completion_tokens"]
    if pt or ct:
        est = estimate_cost(model_id, pt, ct)
        if est is not None:
            usage_info["cost"] = est
            usage_info["cost_estimated"] = True

    # Extract tool call steps from message history
    tool_steps = []
    for msg in result.all_messages():
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if part.part_kind == "tool-call":
                    tool_steps.append({
                        "type": "call",
                        "tool": part.tool_name,
                        "args": part.args if isinstance(part.args, str) else json.dumps(part.args),
                    })
                elif part.part_kind == "tool-return":
                    tool_steps.append({
                        "type": "return",
                        "tool": part.tool_name,
                        "output": str(part.content),
                    })

    return {"content": content, "usage": usage_info, "tool_steps": tool_steps, "elapsed": elapsed}




def generate_summary_prompt(dataset_names: list[str]) -> str:
    """Generate a summary prompt based on the provided dataset names."""
    if not dataset_names:
        return "No datasets are provided. Please select datasets to provide to the AI first."
    names = ", ".join(dataset_names)
    return f"Summarize the provided datasets ({names}). Highlight key findings, patterns, and contrasts between different assay results."


async def summarize_conversation(messages: list[dict], api_key: str, model_id: str) -> str | None:
    """Summarize a list of messages into a brief summary."""
    if not messages:
        return None

    conversation_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    prompt = (
        "Summarize the following conversation concisely, preserving key information, "
        "questions asked, and conclusions reached:\n\n" + conversation_text
    )

    try:
        model = OpenRouterModel(
            model_id,
            provider=OpenRouterProvider(api_key=api_key),
        )
        agent = Agent(model)
        result = await agent.run(prompt)
        return result.output
    except Exception:
        return None
