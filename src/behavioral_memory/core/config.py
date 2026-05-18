"""Configuration via environment variables using pydantic-settings.

All settings can be overridden via environment variables or a .env file.
The framework is model-agnostic — LLM and embedding models are passed
as constructor arguments, not configured here.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global settings for the behavioral-memory framework."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- PostgreSQL + pgvector ---
    vector_store_url: str = "postgresql+psycopg://localhost:5432/behavioral_memory"
    vector_store_collection: str = "validated_traces"

    # --- Retrieval tuning ---
    few_shot_k: int = 3
    max_prompt_tokens: int = 3500
    similarity_dedup_threshold: float = 0.95

    # --- Langfuse (optional) ---
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Feedback loop ---
    feedback_score_name: str = "quality"
    feedback_positive_threshold: float = 1.0
    feedback_poll_interval: int = 60

    # --- Sandbox ---
    sandbox_timeout_seconds: int = 30

    @property
    def langfuse_enabled(self) -> bool:
        return bool(
            self.langfuse_secret_key
            and self.langfuse_public_key
            and not self.langfuse_secret_key.startswith("sk-lf-...")
        )
