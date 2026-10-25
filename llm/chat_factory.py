import math
import os
from collections.abc import Mapping
from importlib import import_module
from urllib.parse import urlsplit

from pydantic import SecretStr

from .provider_types import LLMEnums

_ADAPTERS = {
    "OpenAICompatibleProvider": ("llm.providers.compatible", "OpenAICompatibleProvider"),
    "NvidiaNimProvider": ("llm.providers.nvidia_nim", "NvidiaNimProvider"),
    "OllamaProvider": ("llm.providers.ollama", "OllamaProvider"),
    "GoogleGenAIProvider": ("llm.providers.google_genai", "GoogleGenAIProvider"),
    "GroqProvider": ("llm.providers.groq", "GroqProvider"),
}

# Credentials and endpoints belong to each provider rather than the global
# legacy selection, so individually configured agents keep their own routing.
_PROVIDERS = {
    "OPENAI_COMPATIBLE": (
        "OpenAICompatibleProvider",
        "OPENAI_COMPATIBLE_API_KEY",
        "OPENAI_COMPATIBLE_BASE_URL",
        "https://api.openai.com/v1",
    ),
    "NVIDIA_NIM": (
        "NvidiaNimProvider",
        "NVIDIA_NIM_API_KEY",
        "NVIDIA_NIM_BASE_URL",
        "https://integrate.api.nvidia.com/v1",
    ),
    "GROQ": (
        "GroqProvider",
        "GROQ_API_KEY",
        "GROQ_BASE_URL",
        "https://api.groq.com/openai/v1",
    ),
    "GOOGLE_GENAI": (
        "GoogleGenAIProvider",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    "OLLAMA": (
        "OllamaProvider",
        "OLLAMA_API_KEY",
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    ),
}


def __getattr__(name):
    if name not in _ADAPTERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _ADAPTERS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def _adapter(name):
    return globals()[name] if name in globals() else __getattr__(name)


class LLMProviderFactory:
    def __init__(self, config):
        self.config = config

    def create(
        self,
        provider: str,
        model_id: str = None,
        model_temperature: float = 0.2,
        timeout_seconds: float | None = None,
    ):
        if not isinstance(provider, str) or provider.strip().upper() not in {
            item.value for item in LLMEnums
        }:
            raise ValueError("LLM provider must be one of " + ", ".join(_PROVIDERS))
        provider = provider.strip().upper()
        if provider == "OPENAI_COMPATIBLE" and model_id is None:
            model_id = self._setting("OPENAI_COMPATIBLE_MODEL_ID")
            model_temperature = self._setting("OPENAI_COMPATIBLE_TEMPERATURE", model_temperature)
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"A nonempty model_id is required for {provider}")
        model_id = model_id.strip()
        temperature = self._finite_number(model_temperature, "Model temperature", minimum=0)
        if provider != "OLLAMA" and temperature > 2:
            raise ValueError(f"Model temperature must be at most 2 for {provider}")
        timeout = self._finite_number(
            self._setting("LLM_TIMEOUT_SECONDS", 60)
            if timeout_seconds is None
            else timeout_seconds,
            "LLM timeout",
            minimum=0,
            exclusive=True,
        )
        adapter, credential_name, endpoint_name, default_endpoint = _PROVIDERS[provider]
        endpoint = self._endpoint(endpoint_name, default_endpoint)
        credential = self._credential(credential_name, required=provider != "OLLAMA")
        if provider == "GOOGLE_GENAI":
            model_id = model_id.removeprefix("models/")
            if model_id.lower().startswith(("gemini-1.0", "gemini-1.5", "gemini-2.0")):
                raise ValueError(
                    "The configured GOOGLE_GENAI model is retired; choose a supported model explicitly"
                )
        if provider == "OLLAMA":
            endpoint = endpoint.rstrip("/").removesuffix("/v1")
        options = {
            "model_name": model_id,
            "temperature": temperature,
            "api_key": credential,
            "base_url": endpoint,
            "timeout_seconds": timeout,
            "max_tokens": self._integer("LLM_MAX_OUTPUT_TOKENS", 2048, minimum=1),
            "max_retries": self._integer("LLM_MAX_RETRIES", 2, minimum=0, maximum=3),
            "max_concurrency": self._integer("LLM_MAX_CONCURRENCY", 2, minimum=1),
            "retry_max_wait_seconds": self._finite_number(
                self._setting("LLM_RETRY_MAX_WAIT_SECONDS", 10),
                "LLM retry maximum wait",
                minimum=0,
                exclusive=True,
            ),
        }
        if provider == "OLLAMA":
            options["num_ctx"] = self._integer("OLLAMA_NUM_CTX", 16384, minimum=2048)
        return _adapter(adapter)(**options)

    def _setting(self, name, default=None):
        return (
            self.config.get(name, default)
            if isinstance(self.config, Mapping)
            else getattr(self.config, name, default)
        )

    def _credential(self, name, *, required=True):
        value = self._setting(name)
        if value is None:
            value = os.getenv(name)
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if not isinstance(value, str) or not value.strip():
            if required:
                raise ValueError(f"{name} is required for the selected LLM provider")
            value = "ollama"
        return SecretStr(value.strip())

    def _endpoint(self, name, default):
        value = self._setting(name, default)
        if not isinstance(value, str):
            raise ValueError(f"{name} must be an HTTP(S) URL")
        value = value.strip()
        try:
            parts = urlsplit(value)
            valid = (
                parts.scheme in {"http", "https"}
                and bool(parts.hostname)
                and parts.username is None
                and parts.password is None
                and not parts.query
                and not parts.fragment
            )
            parts.port
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(
                f"{name} must be an HTTP(S) URL without credentials, query or fragment"
            )
        return value

    def _integer(self, name, default, *, minimum, maximum=None):
        value = self._setting(name, default)
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        try:
            parsed = int(value)
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"{name} must be an integer") from None
        if parsed != value or parsed < minimum or (maximum is not None and parsed > maximum):
            raise ValueError(f"{name} is outside its supported integer range")
        return parsed

    @staticmethod
    def _finite_number(value, label, *, minimum, exclusive=False):
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{label} must be a finite number") from None
        if not math.isfinite(parsed) or parsed < minimum or (exclusive and parsed == minimum):
            raise ValueError(f"{label} is outside its supported finite range")
        return parsed
