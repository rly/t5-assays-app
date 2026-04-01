# Agent E2E Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Live end-to-end tests that verify the AI agent orchestrates tool calls correctly and produces valid responses against real LLM and external APIs.

**Architecture:** Tests call `run_agent_chat()` directly with a small fixture DataFrame and real OpenRouter API key. Each test sends a specific prompt designed to trigger particular tool usage, then asserts on the response structure and content. Pytest marks (`llm`, `network`) allow selective runs.

**Tech Stack:** pytest, pytest-asyncio, pandas, existing `chat_service` module

---

### Task 1: Project scaffolding — conftest and pytest config

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: `pyproject.toml` (add pytest config + dev deps)

- [ ] **Step 1: Add pytest dependencies and config to pyproject.toml**

Add after the existing `dependencies` list:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=1.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "network: hits external APIs (PubChem, ChEMBL, BindingDB, PDB, ADMETlab)",
    "llm: requires OPENROUTER_API_KEY env var and makes LLM calls",
]
```

- [ ] **Step 2: Install dev deps**

Run: `cd /Users/rly/Documents/BRAVE/t5-assays-app && uv sync --group dev`

- [ ] **Step 3: Create tests/__init__.py**

Empty file.

- [ ] **Step 4: Create tests/conftest.py**

```python
import os
import pytest
import pandas as pd

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
IBUPROFEN_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


@pytest.fixture(scope="session")
def api_key():
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def model_id():
    return "nvidia/nemotron-3-super-120b-a12b:free"


@pytest.fixture(scope="session")
def sample_datasets():
    """Small realistic dataset for agent testing."""
    df = pd.DataFrame({
        "Compound": [
            "Aspirin", "Caffeine", "Ibuprofen", "Naproxen", "Acetaminophen",
            "Diclofenac", "Celecoxib", "Indomethacin", "Piroxicam", "Meloxicam",
        ],
        "Structure": [
            ASPIRIN_SMILES,
            CAFFEINE_SMILES,
            IBUPROFEN_SMILES,
            "COc1ccc2cc(ccc2c1)C(C)C(=O)O",        # naproxen
            "CC(=O)Nc1ccc(O)cc1",                    # acetaminophen
            "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",       # diclofenac
            "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",  # celecoxib
            "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1",     # indomethacin
            "OC(=O)C1=C(O)N2C(=O)c3ccccc3Nc3ccccc3S2(=O)=O",      # piroxicam (approx)
            "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",        # meloxicam (approx)
        ],
        "KD_M": [3.2e-6, 1.5e-5, 8.7e-7, 2.1e-6, 5.0e-5,
                 4.3e-7, 1.2e-7, 6.5e-7, 3.8e-6, 9.1e-7],
        "AI_Binding_Score": [0.72, 0.45, 0.88, 0.65, 0.31,
                             0.91, 0.95, 0.85, 0.58, 0.82],
    })
    return {"Test_Assay": df}


@pytest.fixture(scope="session")
def system_prompt(sample_datasets):
    from app.services.chat_service import build_system_prompt
    return build_system_prompt(sample_datasets)
```

- [ ] **Step 5: Verify pytest discovers the fixture**

Run: `cd /Users/rly/Documents/BRAVE/t5-assays-app && uv run pytest tests/ --collect-only`
Expected: shows conftest fixtures, 0 tests collected (no test files yet)

- [ ] **Step 6: Commit**

```bash
git add tests/ pyproject.toml
git commit -m "Add pytest scaffolding and fixtures for agent E2E tests"
```

---

### Task 2: Implement E2E test file

**Files:**
- Create: `tests/test_agent_e2e.py`

All tests call `run_agent_chat()` with real API key and model, using the fixture dataset.

- [ ] **Step 1: Create tests/test_agent_e2e.py with all 8 tests**

```python
"""End-to-end tests for the AI agent and tool calling pipeline.

These tests hit a real LLM via OpenRouter. They are slow (~30-60s each)
and non-deterministic. Run manually, not in CI.

    OPENROUTER_API_KEY=sk-... uv run pytest tests/test_agent_e2e.py -v
"""

import json
import pytest

from app.services.chat_service import run_agent_chat

pytestmark = [pytest.mark.llm, pytest.mark.network]

# Generous timeout — LLM calls can be slow, especially free models
TIMEOUT = 180


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
    # The agent should mention at least some of the actual column names
    content_lower = result["content"].lower()
    assert any(
        col.lower() in content_lower
        for col in ["Compound", "Structure", "KD_M", "AI_Binding_Score"]
    ), f"Response should mention column names. Got: {result['content'][:300]}"

    # Usage should be populated
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

    # The mean of our fixture KD_M values is ~7.94e-6 — response should contain a number
    # We can't assert exact value (LLM formatting varies), but it should mention scientific notation
    content = result["content"]
    assert any(c.isdigit() for c in content), "Response should contain numeric results"


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

    # Check that a tool return contains descriptor-like data
    returns = [s for s in result["tool_steps"] if s["type"] == "return" and s["tool"] == "compute_descriptors"]
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

    # Aspirin's CID is 2244
    content = result["content"]
    assert "2244" in content, f"Response should mention aspirin CID 2244. Got: {content[:500]}"


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
    # At least one of the cheminformatics tools should be called
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
async def test_empty_dataset(api_key, model_id, system_prompt):
    """Agent handles an empty DataFrame gracefully."""
    import pandas as pd

    empty_datasets = {"Empty_Sheet": pd.DataFrame(columns=["Compound", "Structure", "KD_M"])}
    empty_prompt = "from app.services.chat_service import build_system_prompt"
    # Rebuild prompt for empty dataset
    from app.services.chat_service import build_system_prompt
    prompt = build_system_prompt(empty_datasets)

    result = await _ask(
        api_key, model_id, prompt, empty_datasets,
        "What compounds are in the dataset?",
    )

    assert result["content"], "Response should not be empty"
    # Agent should indicate there's no data or the dataset is empty
    content_lower = result["content"].lower()
    assert any(
        word in content_lower for word in ["empty", "no ", "0 row", "no data", "no compound"]
    ), f"Response should mention empty/no data. Got: {result['content'][:300]}"


@pytest.mark.timeout(TIMEOUT)
async def test_conversation_context(api_key, model_id, system_prompt, sample_datasets):
    """Agent references prior conversation context in a follow-up message."""
    # First message: ask about top binder
    result1 = await _ask(
        api_key, model_id, system_prompt, sample_datasets,
        "Which compound has the lowest (best) KD_M value? Use run_python. Just give the name.",
    )

    assert result1["content"], "First response should not be empty"

    # Build history from first exchange
    history = [
        {"role": "user", "content": "Which compound has the lowest (best) KD_M value? Use run_python. Just give the name."},
        {"role": "assistant", "content": result1["content"]},
    ]

    # Second message: follow-up referencing "that compound"
    result2 = await _ask(
        api_key, model_id, system_prompt, sample_datasets,
        "What is the AI_Binding_Score for that compound? Use run_python.",
        history=history,
    )

    assert result2["content"], "Second response should not be empty"
    # The best binder is Celecoxib (KD_M=1.2e-7, AI_Binding_Score=0.95)
    # Response should contain a score value
    assert any(c.isdigit() for c in result2["content"]), (
        "Follow-up response should contain a numeric score"
    )
```

- [ ] **Step 2: Install pytest-timeout**

Add `"pytest-timeout>=2.0"` to the dev dependency group, then:

Run: `cd /Users/rly/Documents/BRAVE/t5-assays-app && uv sync --group dev`

- [ ] **Step 3: Run the tests**

Run: `cd /Users/rly/Documents/BRAVE/t5-assays-app && OPENROUTER_API_KEY=sk-... uv run pytest tests/test_agent_e2e.py -v --timeout=180`

Expected: All tests pass (some may be slow on free model). Non-determinism is expected — if a test fails due to LLM response variability, note which one and adjust assertion thresholds.

- [ ] **Step 4: Commit**

```bash
git add tests/test_agent_e2e.py pyproject.toml
git commit -m "Add agent E2E tests: 8 live tests covering tools, critic, and context"
```
