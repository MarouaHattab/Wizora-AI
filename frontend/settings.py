import os
import secrets
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontendSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).resolve().parents[1] / ".env", extra="ignore")

    FRONTEND_BACKEND_URL: str = "http://127.0.0.1:8000"
    FRONTEND_HOST: str = "127.0.0.1"
    FRONTEND_PORT: int = Field(default=7860, ge=1, le=65535)
    FRONTEND_TIMEOUT_SECONDS: float = Field(default=600, gt=0, allow_inf_nan=False)
    FRONTEND_BROWSER_SECRET: str | None = Field(default=None, min_length=32)
    MAX_UPLOAD_BYTES: int = Field(default=268435456, gt=0)

    @field_validator("FRONTEND_BROWSER_SECRET", mode="before")
    @classmethod
    def empty_secret(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("FRONTEND_BACKEND_URL")
    @classmethod
    def backend_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        try:
            parsed.port
        except ValueError:
            raise ValueError("Backend URL has an invalid port") from None
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Backend URL must be an HTTP(S) URL without credentials, query, or fragment")
        return value.rstrip("/")


@lru_cache(maxsize=1)
def get_frontend_settings() -> FrontendSettings:
    return FrontendSettings()


def browser_secret(settings: FrontendSettings | None = None) -> str:
    configured = (settings or get_frontend_settings()).FRONTEND_BROWSER_SECRET
    if configured:
        return configured
    path = Path(__file__).resolve().parents[1] / "runtime" / "gradio-browser.secret"
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = secrets.token_urlsafe(48)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        saved = path.read_text(encoding="ascii").strip()
        if len(saved) < 32:
            raise RuntimeError("Stored browser secret is invalid; configure FRONTEND_BROWSER_SECRET") from None
        return saved
    with os.fdopen(descriptor, "w", encoding="ascii") as stream:
        stream.write(candidate)
    return candidate
