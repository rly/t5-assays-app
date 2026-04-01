# AI Agent & Tool Call Testing Plan

**Goal:** Live end-to-end validation that tools produce correct results against real APIs and that the agent orchestrates them properly.

**Runner:** pytest + pytest-asyncio. Tests hit real services — expect ~30-60s per external API test. Run manually, not in CI.

**Location:** `tests/` directory, one file per test group.

---

## Test fixtures (conftest.py)

| Fixture | Purpose |
|---------|---------|
| `sample_datasets` | Dict of 1-2 small DataFrames (~10 rows) with realistic columns: Compound, Structure (SMILES), KD_M, AI Binding Score. Reuse across all tests. |
| `agent_deps` | `ChatDeps(datasets=sample_datasets)` |
| `api_key` | Load from `OPENROUTER_API_KEY` env var; skip suite if missing |
| `model_id` | Default to cheapest model in `MODEL_MAPPING` (Nemotron free) |
| `agent` | Call `create_agent(api_key, model_id)` once per session |
| `known_smiles` | Dict of reliable test SMILES: aspirin, caffeine, ibuprofen — compounds guaranteed to exist in PubChem/ChEMBL |

---

## 1. Sandbox code execution — `tests/test_sandbox.py`

Tests `execute_code()` from `sandbox_service.py` directly (no LLM).

| # | Test | What to verify |
|---|------|---------------|
| 1.1 | Print scalar | `print(2+2)` returns `"4"` in stdout |
| 1.2 | Dataset access | Code reads `datasets["test"]` and prints shape — output matches fixture |
| 1.3 | Pandas operations | Filter, groupby, describe on fixture data — output is valid |
| 1.4 | RDKit available | `from rdkit import Chem; print(Chem.MolFromSmiles("CCO") is not None)` returns `True` |
| 1.5 | Numpy available | `import numpy as np; print(np.mean([1,2,3]))` returns `2.0` |
| 1.6 | Blocked import | `import subprocess` raises error / returns error string |
| 1.7 | Blocked import (socket) | `import socket` raises error |
| 1.8 | Timeout | Infinite loop (`while True: pass`) returns error within ~35s |
| 1.9 | Syntax error | `def foo(` returns meaningful error message |
| 1.10 | Runtime error | `1/0` returns ZeroDivisionError in output |

---

## 2. Local RDKit tools — `tests/test_rdkit_tools.py`

Test tool functions directly with known SMILES. No network, no LLM.

### compute_descriptors

| # | Test | What to verify |
|---|------|---------------|
| 2.1 | Aspirin descriptors | MW ~180.16, LogP ~1.2, HBD=1, HBA=4, Lipinski=Pass |
| 2.2 | Invalid SMILES | Entry has `error` field, doesn't crash the batch |
| 2.3 | With names | Names propagate to output records |
| 2.4 | Batch (10 compounds) | Returns 10 records, all have expected keys |
| 2.5 | Empty list | Returns empty results, no crash |

### cluster_by_scaffold

| # | Test | What to verify |
|---|------|---------------|
| 2.6 | 5 related compounds | Returns cluster assignments, at least 1 cluster |
| 2.7 | Scaffold extraction | Murcko scaffolds are valid SMILES |
| 2.8 | Single compound | Returns 1 cluster with 1 member |

### compute_tanimoto_matrix

| # | Test | What to verify |
|---|------|---------------|
| 2.9 | 3 compounds | Returns 3x3 matrix, diagonal is 1.0, symmetric |
| 2.10 | Identical pair | Similarity = 1.0 for duplicate SMILES |
| 2.11 | Diverse pair | Aspirin vs caffeine similarity < 0.5 |

### predict_admet

| # | Test | What to verify |
|---|------|---------------|
| 2.12 | Aspirin ADMET | Returns expected keys: gi_absorption, bbb, pains, brenk, lipinski |
| 2.13 | Known PAINS hit | Compound with known PAINS alert is flagged |
| 2.14 | Batch + summary | Summary counts match individual predictions |

---

## 3. PubChem tools — `tests/test_pubchem.py`

Live HTTP calls. Mark all with `@pytest.mark.network`.

| # | Test | What to verify |
|---|------|---------------|
| 3.1 | lookup_pubchem (aspirin by name) | Returns CID 2244, has IUPAC name, MW ~180 |
| 3.2 | lookup_pubchem (by SMILES) | Aspirin SMILES resolves to CID 2244 |
| 3.3 | lookup_pubchem (by CID) | CID 2244 returns aspirin data |
| 3.4 | lookup_pubchem (invalid) | Returns error/empty, no crash |
| 3.5 | search_by_substructure | Benzene SMARTS `c1ccccc1` returns >0 hits with valid CIDs |
| 3.6 | search_by_similarity | Aspirin SMILES at 90% threshold returns hits including aspirin itself |
| 3.7 | get_pubchem_bioassays | CID 2244 returns >0 bioassay records |
| 3.8 | get_pubchem_bioassays (obscure CID) | Returns empty list or valid results, no crash |

---

## 4. ChEMBL tools — `tests/test_chembl.py`

Live HTTP calls. Mark `@pytest.mark.network`.

| # | Test | What to verify |
|---|------|---------------|
| 4.1 | search_chembl (aspirin SMILES) | Returns results with chembl_id, similarity, max_phase |
| 4.2 | search_chembl (novel SMILES) | Returns empty or results, no crash |
| 4.3 | get_chembl_activities (CHEMBL25) | Aspirin ChEMBL ID returns activity records with target, type, value |
| 4.4 | get_chembl_activities (invalid ID) | Returns error/empty, no crash |

---

## 5. BindingDB & PDB tools — `tests/test_external.py`

Live HTTP calls. Mark `@pytest.mark.network`.

| # | Test | What to verify |
|---|------|---------------|
| 5.1 | search_bindingdb (aspirin) | Returns binding records or empty (BindingDB may not have aspirin — just verify no crash) |
| 5.2 | search_pdb ("VEEV macrodomain") | Returns PDB entries with IDs, titles, resolution |
| 5.3 | search_pdb (gibberish) | Returns empty results, no crash |
| 5.4 | get_pdb_ligands (known PDB ID) | Use a PDB ID from 5.2; returns ligand records with names, SMILES |
| 5.5 | get_pdb_ligands (invalid ID) | Returns error/empty, no crash |
| 5.6 | predict_admet_ml (aspirin) | ADMETlab returns predictions with expected property keys |
| 5.7 | predict_admet_ml (invalid SMILES) | Returns error for bad entry, doesn't crash batch |

---

## 6. Critic layer — `tests/test_critic.py`

Requires LLM call. Mark `@pytest.mark.llm`.

| # | Test | What to verify |
|---|------|---------------|
| 6.1 | Correct response | Pass a faithful response + matching tool output. Verdict should be "Pass" |
| 6.2 | Hallucinated numbers | Response claims "KD = 5 nM" but tool output says "KD = 500 nM". Verdict should flag issue |
| 6.3 | No tool steps | `run_critic()` with empty tool_steps returns None (skipped) |
| 6.4 | Malformed input | Garbage tool_steps string — returns None, no crash |

---

## 7. Full agent flow — `tests/test_agent_e2e.py`

End-to-end: user message in, structured response out. Requires LLM. Mark `@pytest.mark.llm`.

| # | Test | What to verify |
|---|------|---------------|
| 7.1 | Simple data question | "What columns are in the dataset?" — response mentions actual column names from fixture |
| 7.2 | Tool-using question | "What is the mean KD_M?" — tool_steps is non-empty, response contains a number consistent with fixture data |
| 7.3 | Descriptor tool call | "Compute descriptors for the top 3 binders" — tool_steps includes compute_descriptors call |
| 7.4 | PubChem tool call | "Look up aspirin in PubChem" — tool_steps includes lookup_pubchem, response mentions CID or IUPAC name |
| 7.5 | Multi-tool chain | "Compute descriptors then predict ADMET for top 5" — tool_steps includes both tool calls |
| 7.6 | Critic runs | Any tool-using question — critique field is present with verdict |
| 7.7 | Empty dataset | Pass empty DataFrame — agent responds gracefully, no crash |
| 7.8 | Conversation context | Send 2 messages sequentially — second response references first exchange |

---

## 8. Chat service persistence — `tests/test_chat_service.py`

Uses test DB (SQLite in-memory). No LLM needed.

| # | Test | What to verify |
|---|------|---------------|
| 8.1 | Create conversation | `get_or_create_conversation()` returns conversation with user_id |
| 8.2 | Idempotent creation | Calling twice for same user returns same conversation |
| 8.3 | Save + retrieve messages | Save user + assistant messages, `get_chat_messages()` returns both in order |
| 8.4 | Message fields | Saved message has correct role, content, model_used, tokens_used, cost |
| 8.5 | Clear chat | After clear, `get_chat_messages()` returns empty list |
| 8.6 | Conversation summary | Set summary text, verify it persists and is retrievable |

---

## Running the tests

```bash
# All tests (local + network + LLM)
OPENROUTER_API_KEY=sk-... uv run pytest tests/ -v

# Only local tests (sandbox + RDKit, fast)
uv run pytest tests/test_sandbox.py tests/test_rdkit_tools.py tests/test_chat_service.py -v

# Only network tests (PubChem, ChEMBL, etc.)
OPENROUTER_API_KEY=sk-... uv run pytest tests/ -m network -v

# Only LLM tests (agent + critic)
OPENROUTER_API_KEY=sk-... uv run pytest tests/ -m llm -v
```

---

## Pytest marks (register in pyproject.toml)

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "network: hits external APIs (PubChem, ChEMBL, BindingDB, PDB, ADMETlab)",
    "llm: requires OpenRouter API key and makes LLM calls",
]
```

---

## Dependencies to add

```
pytest
pytest-asyncio
```
