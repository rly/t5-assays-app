# T5 Assays Data Assistant

Web application for exploring alphaviral macrodomain assay data with AI-powered analysis. Built for the BRAVE research group to make multi-modal assay data accessible to non-technical scientists.

## Features

- **Data browsing** -- Interactive AG Grid table with sorting, filtering, and column pinning. Loads data from Google Sheets via service account.
- **Data merging** -- Merges PEITHO (SPR + AI docking) and PARG (FP + AI docking) datasets automatically.
- **Correlation plots** -- Plotly scatter plots with OLS trendlines for key assay comparisons.
- **AI chat** -- Ask questions about the data using OpenRouter models. The AI can execute Python code (pandas/numpy) against the dataset to compute answers.
- **Tool calling** -- Pydantic AI handles the agentic loop: the model calls `run_python`, sees results, and can iterate.
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
