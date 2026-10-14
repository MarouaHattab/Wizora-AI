from pathlib import Path

import pytest
from dotenv import dotenv_values


@pytest.fixture(autouse=True)
def backend_environment(monkeypatch):
    template = Path(__file__).resolve().parents[1] / ".env.example"
    for name, value in dotenv_values(template).items():
        monkeypatch.setenv(name, value or "")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
