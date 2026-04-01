# T5 Assays Data Assistant

Web application for exploring alphaviral macrodomain assay data with AI-powered analysis. Built for the BRAVE research group to make multi-modal assay data accessible to non-technical scientists.

## Features

- **Data browsing** -- Interactive AG Grid table with sorting, filtering, and column pinning. Loads data from Google Sheets via service account.
- **Data merging** -- Merges PEITHO (SPR + AI docking) and PARG (FP + AI docking) datasets automatically.
- **Correlation plots** -- Plotly scatter plots with OLS trendlines for key assay comparisons.
- **AI chat** -- Ask questions about the data using OpenRouter models. The AI can execute Python code (pandas/numpy/RDKit) against the dataset to compute answers.
- **Tool calling** -- Pydantic AI handles the agentic loop: the model calls `run_python` and PubChem tools, sees results, and can iterate.
- **Cheminformatics** -- RDKit runs inside the sandboxed Python environment for local molecular analysis. Async `httpx` calls to the PubChem REST API run in the main process as dedicated agent tools for live database queries (no `pubchempy` dependency).
- **Cost tracking** -- Per-message and per-conversation token counts and estimated costs.
- **Authentication** -- Simple email/password login with encrypted session cookies.

## Tech Stack

- **Backend**: FastAPI + Jinja2 + HTMX
- **Data table**: AG Grid (Theming API, via CDN)
- **Charts**: Plotly.js
- **AI**: Pydantic AI + OpenRouter (supports tool calling)
- **Database**: SQLite via SQLAlchemy
- **CSS**: Pico CSS

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Copy `.env.example` to `.env` and fill in credentials:
```bash
cp .env.example .env
```

Required variables:
- `SECRET_KEY` -- random string for session signing and API key encryption
- `GOOGLE_CREDENTIALS_JSON` -- Google service account JSON (single line)
- `GOOGLE_DRIVE_FOLDER_ID` -- Drive folder containing the Google Sheets
- `OPENROUTER_DEFAULT_API_KEY` -- default key for free models

3. Add a user:
```bash
uv run python -m app.manage add-user user@example.com
```

4. Run the app:
```bash
uv run uvicorn app.main:app --reload
```

5. Open http://localhost:8000 and log in.

## User Management

```bash
uv run python -m app.manage add-user alice@example.com
uv run python -m app.manage list-users
```

## Project Structure

```
app/
  main.py              # FastAPI app, middleware, lifespan
  config.py            # Settings (from .env), model list, pricing
  database.py          # SQLite + SQLAlchemy setup
  models.py            # ORM: User, UserPreference, Conversation, Message
  auth.py              # bcrypt passwords, signed session cookies
  dependencies.py      # FastAPI auth dependency
  utils.py             # Shared encryption utilities
  manage.py            # CLI for user management
  routers/
    auth_routes.py     # Login/logout
    data_routes.py     # Main page, sidebar, preferences
    api_routes.py      # JSON data for AG Grid
    plot_routes.py     # Plotly plots
    chat_routes.py     # AI chat (send, summarize, clear)
  services/
    sheets_service.py  # Google Sheets API connection
    merge_service.py   # PEITHO/PARG merge logic
    filter_service.py  # Chi2/RMSE data filters
    plot_service.py    # Plotly figure generation
    chat_service.py    # Pydantic AI agent, OpenRouter, conversation management
    sandbox_service.py # Restricted Python code execution
  templates/           # Jinja2 HTML templates
  static/              # CSS + JS
```

## Cheminformatics Integration

### RDKit (local, no internet required)

RDKit is pre-loaded in the sandboxed `run_python` environment. The AI can use it directly in Python code alongside pandas for molecular analysis of the `Structure` (SMILES) column.

**Example questions:**

- *"Which compounds pass Lipinski's Rule of Five?"*
- *"Compute molecular weight, LogP, and TPSA for all compounds and show the distribution."*
- *"Find all compounds containing a benzimidazole scaffold."*
- *"Cluster the top 20 hits by Tanimoto fingerprint similarity and show a heatmap."*
- *"Which compounds have the highest SPR binding affinity AND are drug-like (MW < 500, LogP < 5)?"*
- *"Plot LogP vs TPSA for all compounds, coloured by whether they pass the Rule of Five."*
- *"What is the most common ring system among the active compounds?"*

### PubChem Tools (live database queries)

Four agent tools query the PubChem REST API from the main process. The sandbox blocks all network access, so only these trusted tools can reach the internet.

| Tool | Description |
|---|---|
| `lookup_pubchem` | Fetch metadata (IUPAC name, synonyms, MW, LogP, TPSA, CID) for a list of SMILES, names, or CIDs (max 20 — CID resolution is sequential, one request per identifier) |
| `search_pubchem_by_substructure` | Find all PubChem compounds containing a SMARTS substructure |
| `search_pubchem_by_similarity` | Find analogs by Tanimoto similarity threshold (0–100%) |
| `get_pubchem_bioassays` | Retrieve bioassay activity history for a compound CID |

**Example questions:**

- *"What are the IUPAC names and known trade names for our top 5 hits?"*
- *"Have any of our active compounds been previously reported in PubChem bioassays?"*
- *"Find commercially available analogs of our best SPR hit with ≥ 85% Tanimoto similarity."*
- *"Search PubChem for all known compounds containing the macrodomain-binding scaffold from compound X."*
- *"Which of our hits have been tested in antiviral assays before? Pull their bioassay history."*
- *"Cross-reference our top 10 compounds with PubChem — do any have known toxicity flags or PAINS alerts in the bioassay data?"*

### Cheminformatics Agent Tools (local, no internet required)

Four dedicated agent tools run RDKit in the main process for fast, structured cheminformatics analysis. These are faster and more reliable than writing equivalent RDKit code inside `run_python`.

| Tool | Description |
|---|---|
| `compute_descriptors` | MW, LogP, TPSA, HBD, HBA, RotBonds, QED, and Lipinski Ro5 pass/fail for up to 500 SMILES |
| `cluster_by_scaffold` | Murcko scaffold decomposition + Butina fingerprint clustering; returns cluster ID and scaffold SMILES per compound |
| `compute_tanimoto_matrix` | Pairwise Morgan fingerprint Tanimoto similarity matrix for up to 100 compounds |
| `predict_admet` | Rule-based ADMET prediction: GI absorption, BBB permeability, P-gp substrate, CYP inhibition (1A2/2C9/2C19/2D6/3A4), PAINS alerts, Brenk alerts |

**Example questions:**

- *"Run ADMET predictions on the top 20 hits and flag any PAINS or Brenk alerts."*
- *"How many distinct chemical series are in the top 50 binders? Cluster by scaffold."*
- *"Compute descriptors for all compounds and show the QED distribution."*
- *"Which top hits are predicted to be BBB-permeable?"*
- *"Show a Tanimoto similarity heatmap for the top 15 compounds."*

**`predict_admet` details:**

| Property | Method |
|---|---|
| GI absorption | Veber rules: RotBonds ≤ 10 AND TPSA ≤ 140 |
| BBB permeability | MW < 450, TPSA < 90, LogP ∈ [0,5], HBD ≤ 3 |
| P-gp substrate | MW > 400 OR TPSA > 75 |
| CYP1A2 inhibitor | SMARTS: pyrrole, indole, aniline, furan |
| CYP2C9 inhibitor | SMARTS: carboxylic acid / sulfonic acid + aromatic |
| CYP2C19 inhibitor | SMARTS: imidazole, pyridine |
| CYP2D6 inhibitor | SMARTS: basic N within 2 bonds of aromatic ring |
| CYP3A4 inhibitor | MW > 400 AND ≥ 3 aromatic rings |
| PAINS alerts | RDKit FilterCatalog (PAINS_A/B/C) |
| Brenk alerts | RDKit FilterCatalog (Brenk) |
| Lipinski Ro5 | MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10 |

### Architecture

```
Agent (LLM)
├── run_python tool                 → sandboxed subprocess (RDKit, pandas, numpy — no network)
├── compute_descriptors             → main process (RDKit)
├── cluster_by_scaffold             → main process (RDKit)
├── compute_tanimoto_matrix         → main process (RDKit)
├── predict_admet                   → main process (RDKit + FilterCatalog)
├── lookup_pubchem                  → main process → PubChem REST API
├── search_pubchem_by_substructure  → main process → PubChem REST API
├── search_pubchem_by_similarity    → main process → PubChem REST API
└── get_pubchem_bioassays           → main process → PubChem REST API
```

Typical workflow: the agent uses `run_python` to extract SMILES or names from the dataset, then calls a cheminformatics or PubChem tool with those values.

## Docker

```bash
docker build -t t5-assays .
docker run -p 8000:8000 --env-file .env t5-assays
```

## Deployment (Railway / Render)

1. Push to GitHub
2. Connect the repo on Railway or Render
3. Set all environment variables from `.env.example`
4. Deploy

The app runs on port 8000 by default.
