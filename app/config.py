from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    secret_key: str = "change-me"
    debug: bool = False

    google_credentials_json: str = "{}"
    google_drive_folder_id: str = ""

    openrouter_default_api_key: str = ""

    database_url: str = "sqlite:///data/assays.db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

MODEL_MAPPING = {
    "Nemotron 3 Super 120B (free)": "nvidia/nemotron-3-super-120b-a12b:free",
    "GPT-OSS 20B Nitro ($)": "openai/gpt-oss-20b:nitro",
    "Gemini 3.1 Flash Lite ($)": "google/gemini-3.1-flash-lite-preview",
    "GPT-5.4 ($$)": "openai/gpt-5.4",
    "Claude Sonnet 4.6 ($$$)": "anthropic/claude-sonnet-4.6",
}

FREE_MODELS = {"nvidia/nemotron-3-super-120b-a12b:free"}

# Reverse mapping: model_id -> display name
MODEL_DISPLAY_NAMES = {v: k for k, v in MODEL_MAPPING.items()}
