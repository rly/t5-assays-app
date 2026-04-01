"""End-to-end tests for the AI agent and tool calling pipeline.

These tests hit a real LLM via OpenRouter. They are slow (~30-60s each)
and non-deterministic. Run manually, not in CI.

    OPENROUTER_API_KEY=sk-... uv run pytest tests/test_agent_e2e.py -v
"""

import pytest

from app.services.chat_service import run_agent_chat

pytestmark = [pytest.mark.llm, pytest.mark.network]

TIMEOUT = 300


# ── helpers ──────────────────────────────────────────────────────────


async def _ask(api_key, model_id, system_prompt, datasets, message, history=None):
    """Send a message through run_agent_chat and return the result dict."""
    return await run_agent_chat(
        api_key=api_key,
        model_id=model_id,
        system_prompt=system_prompt,
        conversation_messages=history or [],
        user_message=message,
        datasets=datasets,
    )


def _tool_names(result):
    """Extract list of tool names from tool_steps."""
    return [s["tool"] for s in result["tool_steps"] if s["type"] == "call"]


# ── tests ────────────────────────────────────────────────────────────


@pytest.mark.timeout(TIMEOUT)
async def test_simple_data_question(api_key, model_id, system_prompt, sample_datasets):
    """Agent answers a column-listing question. May or may not use run_python."""
    result = await _ask(
        api_key, model_id, system_prompt, sample_datasets,
        "What columns are in the Test_Assay dataset? Just list them.",
    )

    assert result["content"], "Response should not be empty"
    content_lower = result["content"].lower()
    assert any(
        col.lower() in content_lower
        for col in ["Compound", "Structure", "KD_M", "AI_Binding_Score"]
    ), f"Response should mention column names. Got: {result['content'][:300]}"
    assert result["usage"]["total_tokens"] > 0


@pytest.mark.timeout(TIMEOUT)
async def test_tool_using_question(api_key, model_id, system_prompt, sample_datasets):
    """Agent uses run_python to compute a numeric answer from the data."""
    result = await _ask(
        api_key, model_id, system_prompt, sample_datasets,
        "What is the mean KD_M value in the Test_Assay dataset? Use run_python to compute it.",
    )

    assert result["content"], "Response should not be empty"
    assert result["tool_steps"], "Should have tool_steps (run_python expected)"
    assert "run_python" in _tool_names(result), "Should use run_python tool"
    assert any(c.isdigit() for c in result["content"]), "Response should contain numeric results"


@pytest.mark.timeout(TIMEOUT)
async def test_descriptor_tool_call(api_key, model_id, system_prompt, sample_datasets):
    """Agent uses compute_descriptors when asked about molecular properties."""
    result = await _ask(
        api_key, model_id, system_prompt, sample_datasets,
        "Compute molecular descriptors (MW, LogP, etc.) for the top 3 binders by KD_M. "
        "Use the compute_descriptors tool.",
    )

    assert result["content"], "Response should not be empty"
    tools_used = _tool_names(result)
    assert "compute_descriptors" in tools_used, (
        f"Should use compute_descriptors. Tools used: {tools_used}"
    )

    returns = [
        s for s in result["tool_steps"]
        if s["type"] == "return" and s["tool"] == "compute_descriptors"
    ]
    assert returns, "Should have a compute_descriptors return"
    output = returns[0]["output"]
    assert "MW" in output or "molecular_weight" in output.lower(), (
        f"Descriptor output should contain MW. Got: {output[:300]}"
    )


@pytest.mark.timeout(TIMEOUT)
async def test_pubchem_tool_call(api_key, model_id, system_prompt, sample_datasets):
    """Agent uses lookup_pubchem when asked to look up a compound."""
    result = await _ask(
        api_key, model_id, system_prompt, sample_datasets,
        "Look up Aspirin in PubChem using the lookup_pubchem tool. "
        "Report its CID and IUPAC name.",
    )

    assert result["content"], "Response should not be empty"
    tools_used = _tool_names(result)
    assert "lookup_pubchem" in tools_used, (
        f"Should use lookup_pubchem. Tools used: {tools_used}"
    )
    assert "2244" in result["content"], (
        f"Response should mention aspirin CID 2244. Got: {result['content'][:500]}"
    )


@pytest.mark.timeout(TIMEOUT)
async def test_multi_tool_chain(api_key, model_id, system_prompt, sample_datasets):
    """Agent chains multiple tools in a single response."""
    result = await _ask(
        api_key, model_id, system_prompt, sample_datasets,
        "First use run_python to find the top 3 binders by KD_M and get their SMILES. "
        "Then use compute_descriptors on those SMILES. "
        "Then use predict_admet on those SMILES. "
        "Report all results.",
    )

    assert result["content"], "Response should not be empty"
    tools_used = _tool_names(result)

    assert "run_python" in tools_used, f"Should use run_python. Tools: {tools_used}"
    chem_tools = {"compute_descriptors", "predict_admet"}
    used_chem = chem_tools.intersection(tools_used)
    assert len(used_chem) >= 1, (
        f"Should use at least one of {chem_tools}. Tools: {tools_used}"
    )


@pytest.mark.timeout(TIMEOUT)
async def test_critic_runs(api_key, model_id, system_prompt, sample_datasets):
    """Critic layer runs when the agent uses tools and returns a structured verdict."""
    result = await _ask(
        api_key, model_id, system_prompt, sample_datasets,
        "Use run_python to compute the min and max KD_M in the dataset.",
    )

    assert result["tool_steps"], "Should have tool_steps to trigger critic"

    critique = result.get("critique")
    if critique is None:
        pytest.skip("Critic returned None (model may have failed to produce valid JSON)")

    assert "verdict" in critique, f"Critique should have 'verdict'. Got: {critique}"
    valid_verdicts = {"Pass", "Minor issues", "Significant issues"}
    assert critique["verdict"] in valid_verdicts, (
        f"Verdict should be one of {valid_verdicts}. Got: {critique['verdict']}"
    )


@pytest.mark.timeout(TIMEOUT)
async def test_empty_dataset(api_key, model_id):
    """Agent handles an empty DataFrame gracefully."""
    import pandas as pd
    from app.services.chat_service import build_system_prompt

    empty_datasets = {"Empty_Sheet": pd.DataFrame(columns=["Compound", "Structure", "KD_M"])}
    prompt = build_system_prompt(empty_datasets)

    result = await _ask(
        api_key, model_id, prompt, empty_datasets,
        "What compounds are in the dataset?",
    )

    assert result["content"], "Response should not be empty"
    content_lower = result["content"].lower()
    assert any(
        word in content_lower for word in ["empty", "no ", "0 row", "no data", "no compound"]
    ), f"Response should mention empty/no data. Got: {result['content'][:300]}"


@pytest.mark.timeout(TIMEOUT)
async def test_conversation_context(api_key, model_id, system_prompt, sample_datasets):
    """Agent references prior conversation context in a follow-up message."""
    result1 = await _ask(
        api_key, model_id, system_prompt, sample_datasets,
        "Which compound has the lowest (best) KD_M value? Use run_python. Just give the name.",
    )

    assert result1["content"], "First response should not be empty"

    history = [
        {"role": "user", "content": "Which compound has the lowest (best) KD_M value? Use run_python. Just give the name."},
        {"role": "assistant", "content": result1["content"]},
    ]

    result2 = await _ask(
        api_key, model_id, system_prompt, sample_datasets,
        "What is the AI_Binding_Score for that compound? Use run_python.",
        history=history,
    )

    assert result2["content"], "Second response should not be empty"
    assert any(c.isdigit() for c in result2["content"]), (
        "Follow-up response should contain a numeric score"
    )
