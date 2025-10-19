import os
from typing import Optional


class Settings:
    """Application configuration loaded from environment variables.

    Priority:
    - If USE_SQLITE is truthy, use SQLite regardless of Deta config.
    - Else, if DETA_PROJECT_KEY exists, use Deta Base.
    - Fallback to SQLite in-memory if nothing configured.
    """

    def __init__(self) -> None:
        self.use_sqlite: bool = self._get_bool(os.getenv("USE_SQLITE", "false"))
        self.sqlite_path: str = os.getenv("SQLITE_PATH", "./data/guardian.db")

        self.deta_project_key: Optional[str] = os.getenv("DETA_PROJECT_KEY")
        self.deta_base_name: str = os.getenv("DETA_BASE_NAME", "emotion_logs")

    @staticmethod
    def _get_bool(value: str) -> bool:
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "y", "on"}


settings = Settings()


